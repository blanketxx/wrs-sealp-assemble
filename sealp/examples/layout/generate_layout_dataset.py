"""布局数据集生成 (复用原始 evaluate_layout, 不改动任何现有脚本)。

复用 global search / random search / sample_collision_free_xy / evaluate_layout
生成 layout samples, 可行与不可行样本都保存 (不可行作为 feasibility classifier 的
负样本)。每条样本写为一行 jsonl。

用法示例 (Tower / Totem / YuanChair 统一入口):
    # Tower (默认 asmdef)
    python -m sealp.examples.layout.generate_layout_dataset \
        --output-jsonl sealp/examples/layout/_output/layout_dataset_v2_live.jsonl \
        --gen-resume --gen-samples 1000 --gen-seeds 0,1,2,3,4 \
        --gen-station-mode center_continuous

    # Totem
    python -m sealp.examples.layout.generate_layout_dataset \
        --output-jsonl sealp/examples/layout/_output/totem_layout_v1.jsonl \
        --overwrite --gen-samples 200 --gen-seeds 0,1 \
        --gen-station-mode center_continuous \
        --asmdef sealp/assembly_sequence/_demo_output/totem.asmdef \
        --grasp-dir sealp/examples/grasp/totem_grasp \
        --part-order base_plate,post_br,post_fr,post_bl,post_fl,middle_plate,top_cross

    # YuanChair
    python -m sealp.examples.layout.generate_layout_dataset \
        --output-jsonl sealp/examples/layout/_output/yuanchair_layout_v1.jsonl \
        --overwrite --gen-samples 200 --gen-seeds 0,1 \
        --gen-station-mode center_continuous --gen-assembly-type chair \
        --asmdef sealp/assembly_sequence/_demo_output/yuanchair.asmdef \
        --grasp-dir sealp/examples/grasp/yuanchair_grasp \
        --part-order seat,leg_bl,leg_br,leg_fl,leg_fr

装配站采样模式 (--gen-station-mode):
    center_continuous (默认) 连续可行域 + 中心优先 radial-shell 采样 (中心到外围渐进);
    uniform_continuous       连续可行域纯均匀采样 (对照);
    grid3x3                  原始 3x3 网格中心 center-first round-robin (与 global 基线对齐)。

其它相关参数:
    --gen-center-bias F        center_continuous: 中心采样概率 (其余 1-F 全域均匀), 默认 0.6;
    --gen-center-sigma-frac F  中心区尺度 / radial-shell 初始内环半径占比, 默认 0.25;
    --gen-resume               安全续采并追加: 扫描已落盘记录, 从每个 seed 的下一条继续;
    --append / --gen-append    显式仅追加 (不恢复进度);
    --overwrite                显式覆盖目标文件;
    --gen-threads N            限制 OMP/MKL/OpenBLAS 线程数, 适合低 CPU 长跑;
    --gen-fsync-every N        每 N 条做一次 fsync (默认 1, 最安全);
写入模式必须显式选择。包含 datasets/repro/frozen 的正式快照路径以及旧
layout_dataset_v2.jsonl 均拒绝写入。

说明: 其余参数(--asmdef/--config/--grasp-dir/--planner-obstacle-mode/... )
与 find_optimal_initial_layout_tower_strict_pycharm / _global 完全一致。
"""

from __future__ import annotations

import json
import os
import hashlib
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple


def _configure_thread_env_from_argv() -> None:
    """在导入 numpy 前设置 BLAS/OpenMP 线程上限。"""
    value = None
    for i, arg in enumerate(sys.argv):
        if arg == "--gen-threads" and i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
            break
        if arg.startswith("--gen-threads="):
            value = arg.split("=", 1)[1]
            break
    if value is None:
        return
    try:
        n_threads = max(1, int(value))
    except ValueError:
        return
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(n_threads)


_configure_thread_env_from_argv()

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
import find_optimal_initial_layout_tower_nsga2_v1 as nsga2
import find_optimal_initial_layout_tower_global as gmod

LayoutCandidate = fol.LayoutCandidate

# 数据生成配置
DCFG: Dict[str, object] = {
    "dataset_out": None,
    "gen_samples": 300,       # 每个 seed 的随机采样评估次数
    "jitter": 0,              # 对每个采样 layout 额外做多少个抖动变体 (增加近可行负样本)
    "jitter_sigma": 0.03,     # 抖动幅度 (米)
    "seeds": [0],
    "center_bias": 0.6,       # 连续站位采样的中心偏好 (center_continuous 用)
    # 装配站采样模式: center_continuous(默认) / uniform_continuous / grid3x3
    "station_mode": "center_continuous",
    "center_sigma_frac": 0.25,  # 中心区尺度 / radial-shell 初始内环半径占比
    "append": False,            # 仅追加写; 不自动恢复进度
    "overwrite": False,         # 显式覆盖写
    "resume": False,            # 扫描已有记录并按 seed 安全续采
    "write_mode_explicit": False,
    "threads": None,            # BLAS/OpenMP 线程上限
    "fsync_every": 1,           # 每 N 条强制同步到磁盘; 1=最安全
    "max_errors": 20,           # 单次运行最多容忍的候选评估异常
    "assembly_type": "",        # 任务类型标签 (空=从 asmdef 名推断); 跨任务泛化用
}


def _count_jsonl_records(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8-sig") as stream:
        return sum(1 for line in stream if line.strip())


def _validate_dataset_output_path(path: str) -> None:
    normalized = os.path.abspath(path).replace("\\", "/").lower()
    basename = os.path.basename(normalized)
    protected = (
        "/datasets/" in normalized
        or "repro" in basename
        or "frozen" in basename
        or basename == "layout_dataset_v2.jsonl"
    )
    if protected:
        raise ValueError(
            "拒绝把生成数据写入正式/历史实验数据路径: "
            f"{os.path.abspath(path)}。请使用 "
            "sealp/examples/layout/_output/layout_dataset_v2_live.jsonl "
            "或 generated_runs/<run_name>.jsonl。")


def _to_list(x) -> Optional[list]:
    if x is None:
        return None
    return np.asarray(x, dtype=float).reshape(-1).tolist()


# ------------------------------------------------------------
# 连续装配站采样 (取代固定 3x3 网格)。底层 _set_assembly_station 接受任意连续
# fixture_pos, _assembly_region_reject_reason 也对任意 pos 生效, 因此装配站可落在
# 桌面连续可行域的任意位置, 比离散网格更精细。neural 搜索脚本也复用这两个函数。
# ------------------------------------------------------------

def station_safe_bounds(searcher) -> Tuple[float, float, float, float]:
    """装配站可行 xy 范围 (收缩第一件 footprint, 与离散网格口径一致)。"""
    first_pid = searcher._first_part_id()
    if first_pid is not None and first_pid in searcher.rot_cands:
        fp = np.asarray(searcher.rot_cands[first_pid][0].footprint, dtype=float)
    else:
        fp = np.zeros(2)
    x_min, x_max = searcher.table_x_range
    y_min, y_max = searcher.table_y_range
    xs0, xs1 = x_min + fp[0] / 2.0, x_max - fp[0] / 2.0
    ys0, ys1 = y_min + fp[1] / 2.0, y_max - fp[1] / 2.0
    if xs0 > xs1:
        xs0, xs1 = x_min, x_max
    if ys0 > ys1:
        ys0, ys1 = y_min, y_max
    return float(xs0), float(xs1), float(ys0), float(ys1)


def station_reference_center(searcher) -> Tuple[float, float]:
    """连续站位的参考中心: x=可行域中点, y=双臂基座 y 中点(双臂可达最佳带)。

    与离散 center-first 同口径, 保证连续采样也"中心优先"。
    """
    xs0, xs1, ys0, ys1 = station_safe_bounds(searcher)
    ref_x = 0.5 * (xs0 + xs1)
    try:
        arm_ys = [float(xy[1]) for xy in searcher._arm_base_xy_map().values()]
        ref_y = float(np.mean(arm_ys)) if arm_ys else 0.5 * (ys0 + ys1)
    except Exception:
        ref_y = 0.5 * (ys0 + ys1)
    ref_y = float(np.clip(ref_y, ys0, ys1))
    return ref_x, ref_y


def sample_continuous_station(searcher, rng: np.random.Generator,
                              max_tries: int = 60,
                              center_bias: float = 0.6,
                              center_sigma_frac: float = 0.25):
    """在连续可行域内采样一个装配站, 返回 (region_id, rc, pos) 三元组或 None。

    region_id="cont", rc=(-1,-1) 表示连续装配站 (无网格坐标)。

    center_bias: 以该概率围绕"中心参考点"做截断高斯采样(利用双臂可达中心带),
                 其余概率在全可行域均匀采样(探索)。=0 则纯均匀(不偏中心)。
                 这样连续采样在小评估预算下的命中率不低于离散 center-first 3x3。
    """
    xs0, xs1, ys0, ys1 = station_safe_bounds(searcher)
    rx, ry = station_reference_center(searcher)
    sx = max((xs1 - xs0) * float(center_sigma_frac), 1e-4)
    sy = max((ys1 - ys0) * float(center_sigma_frac), 1e-4)
    for _ in range(int(max_tries)):
        if rng.random() < float(center_bias):
            x = float(np.clip(rng.normal(rx, sx), xs0, xs1))
            y = float(np.clip(rng.normal(ry, sy), ys0, ys1))
        else:
            x = float(rng.uniform(xs0, xs1))
            y = float(rng.uniform(ys0, ys1))
        pos = np.array([x, y, float(searcher.table_top_z)], dtype=float)
        if searcher._assembly_region_reject_reason(pos) is None:
            return ("cont", (-1, -1), pos)
    return None


def sample_station_by_mode(searcher, rng: np.random.Generator, mode: str,
                           progress: float = 0.0,
                           center_bias: float = 0.6,
                           center_sigma_frac: float = 0.25,
                           inner_frac: float = 0.25,
                           max_tries: int = 60):
    """按 station_mode 在连续可行域采样一个装配站, 返回 (region_id, rc, pos) 或 None。

    mode:
      - "center_continuous": 中心优先的 **radial shell(方案B)** 连续采样。
          progress (0->1, 采集进度) 控制中心圆盘半径上限 r_max, 前期只采内环(中心),
          后期逐步扩到中/外环, 保证"中心优先 + 全局覆盖"。同时保留 (1-center_bias)
          比例的全域 uniform, 避免分布过窄。
      - "uniform_continuous": 全可行域纯均匀采样 (对照组)。
      - 其它值: 退回 center_continuous。

    grid3x3 由主循环单独处理 (使用离散网格中心 round-robin), 不走本函数。
    """
    xs0, xs1, ys0, ys1 = station_safe_bounds(searcher)
    rx, ry = station_reference_center(searcher)
    half_x = 0.5 * (xs1 - xs0)
    half_y = 0.5 * (ys1 - ys0)
    # 半径上限随进度线性扩张: 前期内环, 后期覆盖整域。
    r_max = float(np.clip(inner_frac + float(progress) * (1.0 - inner_frac), inner_frac, 1.0))

    for _ in range(int(max_tries)):
        if mode == "uniform_continuous" or rng.random() >= float(center_bias):
            x = float(rng.uniform(xs0, xs1))
            y = float(rng.uniform(ys0, ys1))
        else:
            # area-uniform disk of radius r_max, 再按半宽缩放到矩形域 (各向异性)。
            rad = r_max * float(np.sqrt(rng.random()))
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            x = float(np.clip(rx + rad * half_x * np.cos(theta), xs0, xs1))
            y = float(np.clip(ry + rad * half_y * np.sin(theta), ys0, ys1))
        pos = np.array([x, y, float(searcher.table_top_z)], dtype=float)
        if searcher._assembly_region_reject_reason(pos) is None:
            return ("cont", (-1, -1), pos)
    return None


def _assembly_id(searcher) -> str:
    asmdef_path = str(getattr(searcher, "asmdef_path", "") or "")
    return os.path.splitext(os.path.basename(asmdef_path))[0] if asmdef_path else "unknown"


def _generation_signature(searcher) -> str:
    """标识会改变采样分布/标签口径的配置, 防止错误续接到另一组实验。"""
    payload = {
        "schema": 2,
        "assembly_id": _assembly_id(searcher),
        "station_mode": str(DCFG["station_mode"]),
        "center_bias": float(DCFG["center_bias"]),
        "center_sigma_frac": float(DCFG["center_sigma_frac"]),
        "jitter": int(DCFG["jitter"]),
        "jitter_sigma": float(DCFG["jitter_sigma"]),
        "part_order": list(searcher.part_order),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _stable_sample_id(signature: str, seed: int, sample_index: int) -> int:
    """生成兼容旧 schema 的稳定整数 ID, resume 重跑不会产生新 ID。"""
    text = f"{signature}:{int(seed)}:{int(sample_index)}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 62) - 1)


def _scan_resume_state(out_path: str, signature: str) -> Tuple[Dict[int, int], int, int]:
    """扫描已完成记录。若末次崩溃留下坏行, 备份原文件并保留所有有效行。"""
    counts: Dict[int, int] = {}
    feasible = 0
    valid_records: List[Dict] = []
    invalid_lines = 0
    if not os.path.isfile(out_path):
        return counts, feasible, 0

    with open(out_path, "r", encoding="utf-8-sig") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                print(f"[dataset] WARN: 忽略损坏 JSONL 行 #{line_no}")
                continue
            valid_records.append(rec)
            if str(rec.get("generation_signature", "")) != signature:
                continue
            sd = int(rec.get("seed", 0))
            idx = int(rec.get("sample_index", -1))
            counts[sd] = max(counts.get(sd, 0), idx + 1)
            feasible += int(bool(rec.get("l2_pass", False)))

    if invalid_lines:
        backup = f"{out_path}.corrupt-{int(time.time())}.bak"
        shutil.copy2(out_path, backup)
        tmp_path = out_path + ".repair.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fout:
            for rec in valid_records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(tmp_path, out_path)
        print(f"[dataset] repaired={invalid_lines} bad line(s), backup={backup}")
    return counts, feasible, len(valid_records)


def _durable_write(fout, record: Dict, write_count: int, fsync_every: int) -> None:
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    fout.flush()
    if fsync_every > 0 and write_count % fsync_every == 0:
        os.fsync(fout.fileno())


def _pose_candidates_for_part(searcher, pid: str, selected_tag: Optional[str],
                              selected_rot: Optional[str]) -> List[Dict]:
    """Serialize stable-pose candidates for generator schema (optional field)."""
    cands = list(searcher.rot_cands.get(pid, []) or [])
    out: List[Dict] = []
    for c in cands:
        fp = _to_list(getattr(c, "footprint", [0, 0]))
        rot = np.asarray(getattr(c, "rotmat", np.eye(3)), dtype=float).reshape(-1).tolist()
        out.append({
            "pose_id": str(getattr(c, "tag", "unknown")),
            "pose_tag": str(getattr(c, "tag", "unknown")),
            "rot_name": str(getattr(c, "rot_name", "unknown")),
            "rotmat": rot,
            "footprint": fp,
            "support_area": float(fp[0] * fp[1]) if len(fp) >= 2 else 0.0,
            "support_area_ratio": 1.0,
            "center_of_mass_height": float(getattr(c, "extent", [0, 0, 0])[2]) * 0.5
            if hasattr(c, "extent") else 0.0,
            "stable_probability": 1.0,
            "grasp_total": float(getattr(c, "grasp_total", 0)),
            "topdown_count": float(getattr(c, "topdown_count", 0)),
        })
    if not out and selected_tag is not None:
        out.append({
            "pose_id": str(selected_tag),
            "pose_tag": str(selected_tag),
            "rot_name": str(selected_rot or "unknown"),
            "rotmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "footprint": [0.05, 0.05],
            "support_area": 0.0025,
            "support_area_ratio": 1.0,
            "center_of_mass_height": 0.01,
            "stable_probability": 1.0,
            "grasp_total": 0.0,
            "topdown_count": 0.0,
        })
    return out[:8]


def sample_from_candidate(searcher, cand: LayoutCandidate, seed: int,
                          region: Tuple[str, Tuple[int, int], np.ndarray],
                          sample_index: int, generation_signature: str) -> Dict:
    """把一个已评估的 LayoutCandidate + 静态元信息序列化成一条 dataset sample。"""
    part_order = list(searcher.part_order)
    first_pid = searcher._first_part_id() if searcher.preassemble_first_part else None
    parent_map = searcher._part_parent_map()

    parts: List[Dict] = []
    for idx, pid in enumerate(part_order):
        rc0 = searcher.rot_cands.get(pid, [None])[0]
        extent = _to_list(getattr(rc0, "extent", [0, 0, 0])) if rc0 is not None else [0, 0, 0]
        footprint = _to_list(getattr(rc0, "footprint", [0, 0])) if rc0 is not None else [0, 0]
        gp, gr = searcher.world_poses.get(pid, (np.zeros(3), np.eye(3)))
        try:
            gc = searcher._grasp_collection(pid)
            grasp_total = int(len(gc)) if gc is not None else 0
        except Exception:
            grasp_total = 0
        staging_xy = None
        if pid in cand.xy:
            staging_xy = [float(cand.xy[pid][0]), float(cand.xy[pid][1])]
        # 任务无关几何派生量 (供跨零件/跨任务泛化, 不依赖 part_id)
        ext = np.asarray(extent, dtype=float)
        volume = float(ext[0] * ext[1] * ext[2]) if ext.size >= 3 else 0.0
        long_side = float(ext.max()) if ext.size else 0.0
        short_side = float(ext.min()) if ext.size else 0.0
        aspect_ratio = float(long_side / short_side) if short_side > 1e-9 else 0.0
        thinness = float(short_side / long_side) if long_side > 1e-9 else 0.0
        parts.append({
            "part_id": pid,
            "order_index": idx,
            "is_first": bool(pid == first_pid),
            "extent": extent,
            "footprint": footprint,
            "volume": volume,
            "aspect_ratio": aspect_ratio,
            "thinness": thinness,
            "goal_pos": _to_list(gp),
            "goal_rotmat": np.asarray(gr, dtype=float).reshape(-1).tolist(),
            "parent": parent_map.get(pid),
            "topdown_count": int(searcher.topdown_identity_counts.get(pid, 0)),
            "grasp_total": grasp_total,
            "staging_xy": staging_xy,
            "pose_tag": cand.pose_tag.get(pid),
            "rot_name": cand.rot_name.get(pid),
            "grasp_count": int(cand.grasp_counts.get(pid, 0)),
            "arm_choice": cand.arm_choice.get(pid),
            "per_part_dist": float(cand.per_part_dist.get(pid, 0.0)),
            "per_part_manip": float(cand.per_part_manip.get(pid, 0.0)),
            "per_part_rot_angle": float(cand.per_part_rot_angle.get(pid, 0.0)),
            "pose_candidates": _pose_candidates_for_part(
                searcher, pid,
                cand.pose_tag.get(pid), cand.rot_name.get(pid)),
            "target_pose_index": 0,
            "target_rotation_index": 0,
        })

    # ---- 任务级元信息 (跨任务泛化用; 单 asmdef = 单任务) ----
    assembly_id = _assembly_id(searcher)
    assembly_type = str(DCFG.get("assembly_type", "") or "") or assembly_id.split("_")[0] or "unknown"

    # ---- 装配站到桌面中心距离 (任务无关全局约束特征) ----
    st_pos = np.asarray(_to_list(getattr(cand, "assembly_station_pos", region[2])) or [0, 0, 0],
                        dtype=float)
    tab_cx = 0.5 * (float(searcher.table_x_range[0]) + float(searcher.table_x_range[1]))
    tab_cy = 0.5 * (float(searcher.table_y_range[0]) + float(searcher.table_y_range[1]))
    station_dist_center = float(np.hypot(st_pos[0] - tab_cx, st_pos[1] - tab_cy))

    base = {
        "sample_id": _stable_sample_id(generation_signature, seed, sample_index),
        "sample_index": int(sample_index),
        "generation_signature": generation_signature,
        "sampler_version": 2,
        "seed": int(seed),
        "task_id": assembly_id,
        "assembly_id": assembly_id,
        "assembly_type": assembly_type,
        "num_parts": len(part_order),
        "assembly_region_id": str(getattr(cand, "assembly_region_id", region[0])),
        "assembly_region_rc": list(getattr(cand, "assembly_region_rc", region[1])),
        "assembly_grid": int(searcher.assembly_grid),
        "assembly_station_pos": _to_list(getattr(cand, "assembly_station_pos", region[2])),
        "station_distance_to_center": station_dist_center,
        "table_x_range": list(map(float, searcher.table_x_range)),
        "table_y_range": list(map(float, searcher.table_y_range)),
        "table_top_z": float(searcher.table_top_z),
        "part_order": part_order,
        "parts": parts,
        "l2_pass": bool(getattr(cand, "l2_pass", False)),
        "l3_pass": bool(getattr(cand, "l3_pass", False)),
        "layout_score": float(getattr(cand, "layout_score", 0.0)) if getattr(cand, "l2_pass", False) else 0.0,
        "grasp_score_norm": float(getattr(cand, "grasp_score_norm", 0.0)),
        "manip_score_norm": float(getattr(cand, "manip_score_norm", 0.0)),
        "dist_score_norm": float(getattr(cand, "dist_score_norm", 0.0)),
        "rot_score_norm": float(getattr(cand, "rot_score_norm", 0.0)),
        "spatial_score_norm": float(getattr(cand, "spatial_score_norm", 0.0)),
        "fail_reason": str(getattr(cand, "fail_reason", "")),
        "fail_part": getattr(cand, "fail_part", None),
        "fail_detail": dict(getattr(cand, "fail_detail", {}) or {}),
    }
    # Lazy: layout_learning.generator_* pulls torch; BSFS/experiments only need
    # ``_pose_candidates_for_part`` from this module and must not require torch.
    from layout_learning.generator_dataset import sample_schema_extensions
    base.update(sample_schema_extensions(base))
    return base


class DataCollectingSearcher(gmod.GlobalLayoutSearcher):
    """覆写 random_search 为"数据采集循环": 评估大量 layout 并写 jsonl。"""

    def random_search(self, n_samples, seed, max_resample_layout=80, verbose=True,
                      enable_l3=False, l3_top_k=3, l3_obstacle_mode="staging_aware",
                      require_l3=True) -> Optional[LayoutCandidate]:
        out_path = str(DCFG["dataset_out"])
        if not out_path:
            raise RuntimeError("必须提供 --dataset-out。")
        _validate_dataset_output_path(out_path)
        if not bool(DCFG["write_mode_explicit"]):
            raise ValueError(
                "必须显式选择写入模式: --append / --overwrite / "
                "--gen-resume（兼容别名: --gen-append）。")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        seeds = [int(s) for s in DCFG["seeds"]] or [int(seed)]
        n_per_seed = int(DCFG["gen_samples"])
        jitter = int(DCFG["jitter"])
        jsigma = float(DCFG["jitter_sigma"])
        mode = str(DCFG["station_mode"])
        center_bias = float(DCFG["center_bias"])
        center_sigma_frac = float(DCFG["center_sigma_frac"])
        resume = bool(DCFG["resume"])
        append = bool(DCFG["append"]) or resume
        fsync_every = max(1, int(DCFG["fsync_every"]))
        max_errors = max(1, int(DCFG["max_errors"]))
        signature = _generation_signature(self)
        xs0, xs1, ys0, ys1 = station_safe_bounds(self)
        rx, ry = station_reference_center(self)

        resume_counts: Dict[int, int] = {}
        existing_feasible = 0
        existing_total = 0
        if resume:
            resume_counts, existing_feasible, existing_total = _scan_resume_state(
                out_path, signature)

        # grid3x3: 预计算离散网格中心 (center-first 排序), round-robin 使用。
        grid_regions: List[Tuple[str, Tuple[int, int], np.ndarray]] = []
        if mode == "grid3x3":
            grid_regions = self._order_regions_center_first(
                self._assembly_region_candidates(), verbose=False)

        print("\n========== Layout Dataset Generation ==========")
        print(f"dataset_out   = {os.path.abspath(out_path)}")
        print(f"write_mode    = {'resume/append' if resume else 'append' if append else 'overwrite'}")
        print(f"existing_rows = {_count_jsonl_records(out_path)}")
        print(f"seeds         = {seeds}")
        print(f"gen_samples   = {n_per_seed} per seed")
        print(f"station_mode  = {mode}")
        print(f"station bounds = x[{xs0:.3f},{xs1:.3f}] y[{ys0:.3f},{ys1:.3f}] (连续可行域)")
        print(f"ref center    = ({rx:.3f}, {ry:.3f})  # x=可行域中点, y=双臂基座y中点")
        if mode == "center_continuous":
            print(f"center_bias   = {center_bias}  center_sigma_frac = {center_sigma_frac}  "
                  f"(radial-shell: 半径随进度由内向外扩张)")
        elif mode == "grid3x3":
            print(f"grid centers  = {len(grid_regions)} (3x3 center-first round-robin)")
        print(f"jitter        = {jitter} variants x sigma={jsigma}m")
        print(f"max_evals     = {nsga2._CFG.get('max_evals')}")
        print(f"signature     = {signature}")
        print(f"durability    = flush each sample, fsync every {fsync_every}")
        if resume:
            print(f"resume state  = {resume_counts or '(new run)'}")

        t0 = time.time()
        n_written = 0
        n_feasible = existing_feasible
        n_errors = 0
        best: Optional[LayoutCandidate] = None
        dists: List[float] = []  # 每个采纳站位到参考中心的距离 (监控采样分布)
        first_pid = self._first_part_id() if self.preassemble_first_part else None

        gi = sum(resume_counts.values())  # grid3x3 round-robin 指针
        open_mode = "a" if append else "w"
        error_path = out_path + ".errors.jsonl"
        with open(out_path, open_mode, encoding="utf-8") as fout, \
                open(error_path, "a", encoding="utf-8") as ferr:
            for sd in seeds:
                evaluated = int(resume_counts.get(sd, 0)) if resume else 0
                if evaluated >= n_per_seed:
                    print(f"[dataset] seed={sd} already complete ({evaluated}/{n_per_seed}), skip")
                    continue
                attempts = 0
                max_attempts = max(1, n_per_seed - evaluated) * 4 + 50
                print(f"[dataset] seed={sd} start at sample_index={evaluated}")
                while evaluated < n_per_seed and attempts < max_attempts:
                    if self._eval_budget_exhausted():
                        print("[dataset] 停止: 达到 --global-max-evals。")
                        break
                    attempts += 1
                    # 每个 sample_index 使用独立确定性随机流。中断后无需保存 RNG state。
                    rng = np.random.default_rng(
                        np.random.SeedSequence([int(sd), int(evaluated), int(attempts)]))
                    # 按 station_mode 采样一个装配站
                    if mode == "grid3x3":
                        if not grid_regions:
                            break
                        region = grid_regions[gi % len(grid_regions)]
                        gi += 1
                    else:
                        progress = evaluated / max(1, n_per_seed)
                        region = sample_station_by_mode(
                            self, rng, mode, progress=progress,
                            center_bias=center_bias, center_sigma_frac=center_sigma_frac,
                            inner_frac=center_sigma_frac)
                    if region is None:
                        continue
                    st = np.asarray(region[2], dtype=float)
                    dists.append(float(np.hypot(st[0] - rx, st[1] - ry)))
                    self._set_region_from_tuple(region)
                    xy = None
                    for _ in range(max_resample_layout):
                        xy = self.sample_collision_free_xy(rng)
                        if xy is not None:
                            break
                    if xy is None:
                        continue

                    base_variants = [xy]
                    for _ in range(jitter):
                        v = nsga2._copy_xy(xy)
                        for pid in list(v.keys()):
                            if pid == first_pid:
                                continue
                            v[pid] = self._clip_xy_for_part(
                                pid, np.asarray(v[pid]) + rng.normal(0, jsigma, size=2))
                        base_variants.append(v)

                    for variant in base_variants:
                        if evaluated >= n_per_seed or self._eval_budget_exhausted():
                            break
                        te = time.time()
                        sample_index = evaluated
                        try:
                            cand = self._evaluate_gene(variant, region)
                        except KeyboardInterrupt:
                            raise
                        except Exception as exc:
                            n_errors += 1
                            error_rec = {
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "seed": int(sd),
                                "sample_index": int(sample_index),
                                "generation_signature": signature,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            _durable_write(ferr, error_rec, n_errors, 1)
                            print(f"[dataset] WARN: eval error {n_errors}/{max_errors}: "
                                  f"{type(exc).__name__}: {exc}")
                            if n_errors >= max_errors:
                                raise RuntimeError(
                                    f"连续采集累计 {n_errors} 个评估异常; "
                                    f"详情见 {error_path}") from exc
                            continue
                        rec = sample_from_candidate(
                            self, cand, sd, region, sample_index, signature)
                        rec["eval_time"] = float(time.time() - te)
                        rec["station_mode"] = mode
                        n_written += 1
                        _durable_write(fout, rec, n_written, fsync_every)
                        evaluated += 1
                        if bool(getattr(cand, "l2_pass", False)):
                            n_feasible += 1
                            if best is None or cand.layout_score > best.layout_score:
                                best = cand
                        if verbose and n_written % 20 == 0:
                            bs = float(best.layout_score) if best is not None else 0.0
                            total_relevant = sum(resume_counts.values()) + n_written
                            print(f"[dataset] written={n_written} feasible={n_feasible} "
                                  f"({100.0*n_feasible/max(total_relevant,1):.0f}%) "
                                  f"evals={self._nsga_eval_count} best={bs:.4f}")
                fout.flush()
                os.fsync(fout.fileno())

        d_arr = np.asarray(dists, dtype=float) if dists else np.zeros(1)
        print("\n========== Dataset Summary ==========")
        print(f"station_mode     = {mode}")
        print(f"new samples      = {n_written}")
        print(f"existing records = {existing_total if resume else 0}")
        relevant_total = sum(resume_counts.values()) + n_written
        print(f"feasible samples = {n_feasible} "
              f"({100.0 * n_feasible / max(relevant_total,1):.1f}%)")
        print(f"eval errors      = {n_errors}")
        print(f"station dist->center: mean={float(d_arr.mean()):.3f} "
              f"min={float(d_arr.min()):.3f} max={float(d_arr.max()):.3f} (m)")
        print(f"best layout_score= {float(best.layout_score) if best is not None else 0.0:.4f}")
        print(f"real evaluations = {self._nsga_eval_count}")
        print(f"cache hits       = {self._nsga_cache_hits}")
        print(f"wall time        = {time.time() - t0:.1f}s")
        print(f"[OK] dataset -> {out_path}")
        return best


def _consume_dataset_args() -> None:
    v = fast._consume_extra_value("--dataset-out")
    alias = fast._consume_extra_value("--output-jsonl")
    if v is not None and alias is not None and os.path.abspath(v) != os.path.abspath(alias):
        raise ValueError("--dataset-out 与 --output-jsonl 指向不同文件")
    v = alias if alias is not None else v
    if v is not None:
        DCFG["dataset_out"] = v
    v = fast._consume_extra_value("--gen-samples")
    if v is not None:
        DCFG["gen_samples"] = int(v)
    v = fast._consume_extra_value("--gen-jitter")
    if v is not None:
        DCFG["jitter"] = int(v)
    v = fast._consume_extra_value("--gen-jitter-sigma")
    if v is not None:
        DCFG["jitter_sigma"] = float(v)
    v = fast._consume_extra_value("--gen-seeds")
    if v is not None:
        DCFG["seeds"] = [int(s) for s in v.replace(",", " ").split()]
    v = fast._consume_extra_value("--gen-center-bias")
    if v is not None:
        DCFG["center_bias"] = float(v)
    v = fast._consume_extra_value("--gen-station-mode")
    if v is not None:
        m = str(v).strip().lower()
        if m not in ("center_continuous", "uniform_continuous", "grid3x3"):
            print(f"[dataset] WARN: 未知 --gen-station-mode '{v}', 回退 center_continuous。")
            m = "center_continuous"
        DCFG["station_mode"] = m
    v = fast._consume_extra_value("--gen-center-sigma-frac")
    if v is not None:
        DCFG["center_sigma_frac"] = float(v)
    append_flag = (
        fast._consume_extra_flag("--gen-append")
        or fast._consume_extra_flag("--append"))
    overwrite_flag = fast._consume_extra_flag("--overwrite")
    resume_flag = fast._consume_extra_flag("--gen-resume")
    selected_modes = sum(bool(value) for value in (
        append_flag, overwrite_flag, resume_flag))
    if selected_modes > 1:
        raise ValueError("--append/--overwrite/--gen-resume 只能选择一个")
    if append_flag:
        DCFG["append"] = True
        DCFG["write_mode_explicit"] = True
    if overwrite_flag:
        DCFG["overwrite"] = True
        DCFG["append"] = False
        DCFG["write_mode_explicit"] = True
    if resume_flag:
        DCFG["resume"] = True
        DCFG["append"] = True
        DCFG["write_mode_explicit"] = True
    v = fast._consume_extra_value("--gen-threads")
    if v is not None:
        DCFG["threads"] = max(1, int(v))
    v = fast._consume_extra_value("--gen-fsync-every")
    if v is not None:
        DCFG["fsync_every"] = max(1, int(v))
    v = fast._consume_extra_value("--gen-max-errors")
    if v is not None:
        DCFG["max_errors"] = max(1, int(v))
    v = fast._consume_extra_value("--gen-assembly-type")
    if v is not None:
        DCFG["assembly_type"] = str(v).strip()


def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = DataCollectingSearcher


def main() -> None:
    nsga2._enforce_l3_default_off()
    nsga2._enforce_l3_skip_middle_plate()
    gmod._consume_global_args()
    _consume_dataset_args()

    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    if DCFG["dataset_out"] is None:
        DCFG["dataset_out"] = os.path.join(
            gmod.fol.SEALP_ROOT, "examples", "layout", "_output",
            "layout_dataset_v2_live.jsonl")

    print("[dataset] config:")
    for k, val in DCFG.items():
        print(f"    {k:16s} = {val}")

    _patch_module()
    wall_t0 = time.perf_counter()
    try:
        fol.main()
    except KeyboardInterrupt:
        print("\n[dataset] 用户中断；已完成样本均已安全落盘。"
              "使用完全相同命令并加/保留 --gen-resume 即可续跑。")
    finally:
        print(f"[dataset] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")


if __name__ == "__main__":
    main()
