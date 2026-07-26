#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Heatmap + Surrogate + PSO Tower Initial Layout Search
========================================================

这是在 ``find_optimal_initial_layout_tower_strict_pycharm_fast.py`` 之上的
**搜索策略改造版**，不修改任何现有脚本。

它复用 ``FastWeightedInitialLayoutSearcher``(自带 IK 缓存、grasp 抽样、
"已装件最终位姿永远是障碍"的可执行性修复)以及原脚本 ``fol.main()`` 的
命令行解析 / 构造 / 保存逻辑；只重写 ``random_search``，把"纯随机采样"换成:

1. 可行性热力图(feasibility heatmap) —— 先验
   - 对每个装配区 × 每个可动零件, 在桌面粗网格上做"单件可抓取性"检查
     (该零件单独放在某格, 只把环境 + 已装第一件当障碍, 机器人 HOME, 跑
     reason_common_gids, 记可行 grasp 数)。
   - 这是"必要非充分"条件: 热力图=0 的格执行时必死, 高分格才值得精算。
   - 结果缓存到 .pkl, 下次直接加载(真·离线)。
   - 真正搜索时, 不再全桌面均匀随机, 而是**按热力图分数加权**采样高分区。

2. surrogate 代理模型 —— 省算(可选, sklearn 缺失自动跳过)
   - 收集已精确评估过的 (布局特征, 是否可行, score), 训练
     RandomForest 分类器(可行?) + 回归器(score?)。
   - 在 PSO 提议新粒子时, 先用代理模型预测; 预测"几乎不可行"的粒子
     直接给惩罚分跳过精算, 省下昂贵的 evaluate_layout。
   - **安全保证**: 全局最优(gbest)只会来自真实 evaluate_layout 的可行解,
     代理模型只用于"跳过/排序", 绝不直接接受布局, 不破坏可执行性。

3. PSO 局部优化 —— 搜索驱动
   - 在最有希望的装配区内, 以可动零件的连续 xy 向量为决策变量,
     fitness = 真实 evaluate_layout 的 layout_score(不可行则惩罚, 惩罚分
     用热力图先验做 shaping, 给粒子一个朝可行区移动的梯度)。
   - 离散选择(姿态 / 左右臂 / 装配区)交给评估器内部自行枚举挑选,
     PSO 只管连续 xy。
   - 粒子初始化自"热力图加权采样 + 已找到的最优可行解"。

最终一律走真实 evaluate_layout(并可选 --enable-l3 全流程动态避障)把关,
因此找到的布局保证"能跑"。

运行(命令行参数与原脚本完全一致, 额外开关见下):
    python -m sealp.examples.layout.find_optimal_initial_layout_tower_heatmap_pso \
        --n-samples 30 --planner-obstacle-mode none --assembly-grid 3

额外 CLI 开关(均可选):
    --hm-grid N         热力图每轴格数(默认 9)
    --hm-cap  N         热力图阶段每位姿 grasp 上限(默认 200, 越小越快越粗)
    --hm-top-regions N  只在热力图最有希望的前 N 个装配区里做 PSO(默认 2)
    --pso-particles N   PSO 粒子数(默认 12)
    --pso-iters N       PSO 迭代代数(默认 8)
    --no-heatmap        关闭热力图(退回均匀随机采样)
    --no-pso            关闭 PSO(只做热力图加权随机)
    --no-surrogate      关闭代理模型
    --rebuild-heatmap   忽略 .pkl 缓存, 强制重算热力图
"""
from __future__ import annotations

import copy
import hashlib
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 复用 fast 版(IK 缓存 + 可执行性修复)和原版(CLI / main / 保存)。
import find_optimal_initial_layout_tower_strict as fol
import find_optimal_initial_layout_tower_strict_pycharm_fast as fast
from find_optimal_initial_layout_tower_strict import (
    HOME_JV,
    LayoutCandidate,
    NORM_GRASP_MEAN_TARGET,
    PickPlacePlanner,
)

# 换手验证(L3 与动画执行对齐): middle_plate 在动画里走双臂换手, L3 也用换手验证。
import wrs.manipulation.handover_regrasp as horeg
from sealp.assembly_sequence.mating_axis import assembly_mating_dirs

# L3 里用换手(而非单臂)验证的零件。
# 默认【空】= 全部单臂: 之前观察到 middle_plate 的"换手"其实退化成单臂(同一抓取在
# 取料位和目标位都成立), 两只手并没真正交接; 而换手图搜索又慢, 所以默认取消换手。
# 想恢复 middle_plate 换手验证: 命令行加 --l3-handover-middle-plate。
L3_HANDOVER_PART_IDS = frozenset()
# 换手 hopg 目录(与动画端 DEFAULT_HANDOVER_DIR 一致): sealp/examples/grasp/tower_handover
HANDOVER_DIR = os.path.join(os.path.dirname(_THIS_DIR), "grasp", "tower_handover")
L3_HANDOVER_HOPG = {"middle_plate": "middle_plate_hopg.pickle"}


class _QuietHandoverPlanner(horeg.HandoverPlanner):
    """禁掉换手规划过程中的图形弹窗(布局搜索是无界面批量跑)。"""

    def show_graph(self):
        return


def _duplicate_grasp_collection(gc):
    try:
        return gc.copy()
    except Exception:
        return copy.deepcopy(gc)

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


# ============================================================
# 可调默认参数(也可被 CLI 覆盖)
# ============================================================

HM_GRID = 7              # 热力图每轴格数
HM_CAP = 120             # 热力图阶段每位姿 grasp 上限(小=快=粗)
HM_TOP_REGIONS = 2       # 只在前 N 个最有希望的装配区做 PSO
HM_SHARPEN = 2.0         # 热力图加权采样的锐化指数(越大越偏向高分格)
# 热力图是"廉价先验", 不需要精确分。下面三个开关把单格成本压到 1~2 次
# reason_common_gids (否则 = 旋转候选数 × 双臂 ≈ 24 次/格, 整体会卡死)。
HM_MAX_ROT = 1           # 每格只试前 N 个旋转候选(默认 1=identity)
HM_PREFERRED_ARM_ONLY = True   # 每格只试首选臂(_arm_order 的第一只)
HM_GOOD_ENOUGH = 5       # 某候选已达到这么多可行 grasp 就提前停(够当先验)

SEED_FRAC = 0.5          # n_samples 里多少比例用于"热力图加权随机"播种
PSO_PARTICLES = 12       # PSO 粒子数
PSO_ITERS = 8            # PSO 迭代代数
PSO_W = 0.6              # 惯性权重
PSO_C1 = 1.5             # 个体学习因子
PSO_C2 = 1.5             # 群体学习因子

SURROGATE_ENABLE = True
SURROGATE_MIN_DATA = 40  # 训练代理模型所需的最少样本数
SURROGATE_MIN_POS = 6    # 至少要有这么多可行样本才训练
SURROGATE_SKIP_PROB = 0.12  # 代理预测可行概率低于此值则跳过精算
SURROGATE_RETRAIN_EVERY = 2  # PSO 每隔几代用最新数据重训一次代理(0=只在开头训一次)

SAMPLE_TRIES_PER_PART = 14   # 碰撞感知采样: 每个零件最多重采几次以避开已放件

INFEASIBLE_BASE = -1.0   # 不可行布局的基础惩罚分(可行分约在 [0,1])

# 默认【开启】"topdown 抓取不足时强制立起" 这条硬约束。
#   这条规则并非针对 middle_plate 写死, 而是按【抓取库数据】触发: 任何零件若
#   topdown(-Z) 抓取数 < topdown_min(默认10), 就禁止平放(identity), 必须选站立/
#   侧立姿态。原因很物理——薄板/无顶抓取的件平放在桌上, 侧边只有 1~2cm 且紧贴
#   桌面, 夹爪手指会插进桌子, 现实里根本抓不了(staging_aware 排除了桌面 mesh,
#   所以这种平放会"假性通过" L2 抓取检查, 必须靠这条硬约束挡掉)。
#   想关闭(允许平放): 命令行加 --disable-upright-preference。
DEFAULT_DISABLE_UPRIGHT_PREFERENCE = False

# 运行期由 CLI 覆盖
_CFG = {
    "hm_grid": HM_GRID,
    "hm_cap": HM_CAP,
    "hm_top_regions": HM_TOP_REGIONS,
    "pso_particles": PSO_PARTICLES,
    "pso_iters": PSO_ITERS,
    "use_heatmap": True,
    "use_pso": True,
    "use_surrogate": SURROGATE_ENABLE,
    "rebuild_heatmap": False,
}


# ============================================================
# L3 落位验证: 与 execute_layout_sequence_visual 同口径的运动参数
#   (这样 l3_pass=true 的布局, 动画端用同样的多方向+两臂尝试一定能跑通)
# ============================================================
L3_APPROACH_DIST = 0.0
L3_PICK_DEPART_DIST = 0.02
L3_PLACE_APPROACH_DIST = 0.02
L3_PLACE_DEPART_DIST = 0.05
L3_LINEAR_GRANULARITY = 0.04


def _l3_unit_vec(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _l3_motion_candidate_kwargs(asm, pid: str, goal_rotmat):
    """与 execute_layout_sequence_visual._plan_candidates 同口径: 只试装配定义的插接轴。

    以前这里跟执行脚本一样轮流试纯 Z 和四个水平偏置方向。那些方向是凭空猜的世界方向,
    与装配体如何装配无关, 每个失败方向都要把整套 grasp 的直线段 IK 重算一遍。asmdef 已经
    写明了插接轴, 只试它即可; 该步没声明时回退到"从上方放下、向上撤离"。
    """
    approach, depart = assembly_mating_dirs(asm, pid, goal_rotmat)
    if approach is None or depart is None:
        approach, depart = _l3_unit_vec([0, 0, -1]), _l3_unit_vec([0, 0, 1])
        tag = "topdown_default"
    else:
        tag = "asmdef_axis"
    return [(tag, dict(
        pick_depart_direction=_l3_unit_vec([0, 0, 1]),
        pick_depart_distance=L3_PICK_DEPART_DIST,
        place_approach_direction_list=[_l3_unit_vec(approach)],
        place_approach_distance_list=[L3_PLACE_APPROACH_DIST],
        place_depart_direction_list=[_l3_unit_vec(depart)],
        place_depart_distance_list=[L3_PLACE_DEPART_DIST],
    ))]


# ============================================================
# 搜索器
# ============================================================

class HeatmapPSOSearcher(fast.FastWeightedInitialLayoutSearcher):
    """热力图先验 + 代理模型 + PSO 的布局搜索器。"""

    # 强制某些件的旋转候选(rot_name)。由 main() 按 --mp-force-rot 注入。
    # _precompute_rot_candidates(基类)用 getattr(self, "force_rot_name", {}) 读取。
    force_rot_name: Dict[str, str] = {}

    # ----- 基础辅助 -----

    def _movable_parts(self) -> List[str]:
        first = self._first_part_id() if self.preassemble_first_part else None
        return [
            p for p in self.part_order
            if p != first and p in self.world_poses and p in self.staging_models
        ]

    def _backup_arms(self):
        for arm in (self.robot.lft_arm, self.robot.rgt_arm):
            try:
                arm.backup_state()
            except Exception:
                pass

    def _restore_arms(self):
        for arm in (self.robot.lft_arm, self.robot.rgt_arm):
            try:
                arm.restore_state()
            except Exception:
                try:
                    arm.goto_given_conf(HOME_JV)
                except Exception:
                    pass

    def _hm_planner_obstacles(self) -> List:
        """热力图单件检查的障碍: 环境 + 已装第一件最终位姿(经可执行性过滤)。"""
        obs = list(self.env_obs)
        first = self._first_part_id() if self.preassemble_first_part else None
        if first is not None and first in self.goal_models:
            obs.append(self.goal_models[first])
        return self._planner_obstacles(obs)

    def _single_part_feasibility(self, pid: str, xy: np.ndarray, cap: int) -> int:
        """单件可抓取性: 返回该格上(多姿态/双臂取最大)可行 grasp 数, 0=不可抓。

        只把环境 + 已装第一件当障碍(忽略其它可动件), 故为必要非充分条件。
        """
        gc = self._grasp_collection(pid)
        if gc is None or len(gc) == 0:
            return 0
        gp, gr = self.world_poses[pid]
        planner_obs = self._hm_planner_obstacles()

        cands = self.rot_cands.get(pid, [])[:max(1, HM_MAX_ROT)]
        arms = self._arm_order(pid)
        if HM_PREFERRED_ARM_ONLY:
            arms = arms[:1]

        best = 0
        for cand in cands:
            self._apply_staging_pose(pid, xy, cand)
            if self._robot_home_collision_reason(active_pids=[pid]):
                continue
            sp = self.staging_models[pid].pos.copy()
            sr = self.staging_models[pid].rotmat.copy()
            for arm_tag in arms:
                arm = self.robot.rgt_arm if arm_tag == "rgt" else self.robot.lft_arm
                planner = PickPlacePlanner(robot=arm)
                try:
                    gids = planner.reason_common_gids(
                        grasp_collection=gc,
                        goal_pose_list=[(sp, sr), (gp, gr)],
                        obstacle_list=planner_obs,
                    )
                except Exception:
                    gids = []
                if len(gids) > best:
                    best = len(gids)
                if best >= HM_GOOD_ENOUGH:
                    return best  # 够当先验, 提前停
        return best

    def _part_bounds(self, pid: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        cand0 = self.rot_cands[pid][0]
        return self._xy_bounds_for_part_and_cand(pid, cand0)

    # ----- 热力图 -----

    def _heatmap_cache_path(self) -> str:
        out_dir = os.path.join(_THIS_DIR, "_output")
        key = "|".join([
            ",".join(self.part_order),
            str(_CFG["hm_grid"]),
            str(_CFG["hm_cap"]),
            f"{self.table_x_range[0]:.3f}:{self.table_x_range[1]:.3f}",
            f"{self.table_y_range[0]:.3f}:{self.table_y_range[1]:.3f}",
            str(self.assembly_grid),
            str(self.planner_obstacle_mode),
        ])
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
        return os.path.join(out_dir, f"heatmap_cache_{h}.pkl")

    def build_heatmaps(self, regions) -> Dict[str, Any]:
        """返回 {region_id: {"pos","rc","parts":{pid:{"grid","xs","ys","bx","by"}},
                              "promise": float}}。带 .pkl 缓存。"""
        cache_path = self._heatmap_cache_path()
        if not _CFG["rebuild_heatmap"] and os.path.isfile(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                print(f"[heatmap] loaded cache: {cache_path}")
                return data
            except Exception as e:
                print(f"[heatmap] cache load failed ({e}), rebuilding...")

        grid = int(_CFG["hm_grid"])
        cap = int(_CFG["hm_cap"])
        movable = self._movable_parts()

        # 热力图阶段临时用较小 grasp cap 提速(精算阶段恢复)。
        old_cap = fast.MAX_GRASPS_PER_POSE
        fast.MAX_GRASPS_PER_POSE = cap

        print("\n========== Build Feasibility Heatmaps ==========")
        print(f"regions={len(regions)}  movable_parts={movable}  grid={grid}x{grid}  cap={cap}")
        print(f"per-cell cost: max_rot={HM_MAX_ROT} preferred_arm_only={HM_PREFERRED_ARM_ONLY} "
              f"good_enough={HM_GOOD_ENOUGH}  "
              f"(<= {len(regions) * len(movable) * grid * grid} reason_common_gids 调用)")
        t0 = time.time()
        data: Dict[str, Any] = {}

        self._backup_arms()
        try:
            for (region_id, rc, pos) in regions:
                self._set_assembly_station(pos, region_id=region_id, rc=rc)
                fast._pose_cache_reset_stats()
                parts_hm: Dict[str, Any] = {}
                region_promise = 0.0

                for pid in movable:
                    (bx0, bx1), (by0, by1) = self._part_bounds(pid)
                    xs = np.linspace(bx0, bx1, grid)
                    ys = np.linspace(by0, by1, grid)
                    g = np.zeros((grid, grid), dtype=float)
                    pt0 = time.time()
                    for ix, xv in enumerate(xs):
                        for iy, yv in enumerate(ys):
                            g[ix, iy] = float(self._single_part_feasibility(
                                pid, np.array([xv, yv], dtype=float), cap))
                        # 逐行进度, 避免看起来卡死
                        print(f"    [{region_id}/{pid}] row {ix + 1}/{grid} "
                              f"({time.time() - pt0:.1f}s)", flush=True)
                    parts_hm[pid] = {
                        "grid": g, "xs": xs, "ys": ys,
                        "bx": (bx0, bx1), "by": (by0, by1),
                    }
                    frac_ok = float(np.mean(g > 0))
                    region_promise += frac_ok
                    print(f"  region {region_id} part {pid:14s}: "
                          f"feasible_cells={int(np.sum(g > 0))}/{grid * grid} "
                          f"({frac_ok * 100:.0f}%)  max_grasps={int(g.max())}")

                data[region_id] = {
                    "pos": np.asarray(pos, dtype=float).copy(),
                    "rc": tuple(rc),
                    "parts": parts_hm,
                    "promise": region_promise,
                }
                print(f"  region {region_id} promise(sum frac feasible) = {region_promise:.3f}")
        finally:
            self._restore_arms()
            fast.MAX_GRASPS_PER_POSE = old_cap

        print(f"[heatmap] built in {time.time() - t0:.1f}s")
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"[heatmap] cached -> {cache_path}")
        except Exception as e:
            print(f"[heatmap] cache save failed: {e}")
        return data

    @staticmethod
    def _ascii_heatmap(g: np.ndarray) -> List[str]:
        """把一张热力图渲染成 ASCII, 方便组会"看见"先验。行=x, 列=y。"""
        mx = float(g.max())
        chars = " .:-=+*#%@"
        out = []
        for ix in range(g.shape[0]):
            row = []
            for iy in range(g.shape[1]):
                v = g[ix, iy]
                if mx <= 0:
                    row.append(" ")
                else:
                    row.append(chars[min(len(chars) - 1, int(v / mx * (len(chars) - 1)))])
            out.append("".join(row))
        return out

    def _print_heatmap_preview(self, heatmaps: Dict[str, Any], region_id: str) -> None:
        reg = heatmaps.get(region_id)
        if not reg:
            return
        print(f"\n[heatmap preview] region={region_id}  (行=x 列=y, 字符越满越易抓)")
        for pid, hm in reg["parts"].items():
            print(f"  {pid}:  max_grasps={int(hm['grid'].max())}")
            for line in self._ascii_heatmap(hm["grid"]):
                print(f"    |{line}|")

    # ----- 热力图加权采样 -----

    def _sample_xy_from_heatmap(self, rng: np.random.Generator,
                                region_hm: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """按热力图分数加权, 给每个可动件采一个 xy; 第一件交给 evaluate 内部处理。"""
        xy: Dict[str, np.ndarray] = {}
        for pid, hm in region_hm["parts"].items():
            g = hm["grid"]
            xs, ys = hm["xs"], hm["ys"]
            w = np.power(np.maximum(g, 0.0), HM_SHARPEN).ravel()
            if w.sum() <= 0:
                # 该件全图不可抓 -> 整个区域无望
                return None
            idx = int(rng.choice(len(w), p=w / w.sum()))
            ix, iy = divmod(idx, g.shape[1])
            # 在该格内加一点抖动(半个格宽)
            dx = (xs[1] - xs[0]) * 0.5 if len(xs) > 1 else 0.0
            dy = (ys[1] - ys[0]) * 0.5 if len(ys) > 1 else 0.0
            x = float(np.clip(xs[ix] + rng.uniform(-dx, dx), hm["bx"][0], hm["bx"][1]))
            y = float(np.clip(ys[iy] + rng.uniform(-dy, dy), hm["by"][0], hm["by"][1]))
            xy[pid] = np.array([x, y], dtype=float)
        return xy

    def _sample_xy_collision_aware(self, rng: np.random.Generator,
                                   region_hm: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """碰撞感知顺序采样: 按 part_order 逐件从热力图加权采样, 采一个就锁定其
        占位, 后续件避开已放件(含已装第一件)。显著降低 pair_collision 失败率。

        返回 None 表示某件在多次重采后仍无法避开已放件 -> 放弃该布局。
        """
        movable = self._movable_parts()
        # 先把第一件放到装配区占位, 让后续件也避开它。
        self._apply_first_part_as_assembled()
        first = self._first_part_id() if self.preassemble_first_part else None
        placed_ids: List[str] = [first] if (first and first in self.staging_models) else []

        xy: Dict[str, np.ndarray] = {}
        for pid in movable:
            hm = region_hm["parts"].get(pid)
            if hm is None:
                return None
            g, xs, ys = hm["grid"], hm["xs"], hm["ys"]
            w = np.power(np.maximum(g, 0.0), HM_SHARPEN).ravel()
            if w.sum() <= 0:
                return None  # 该件全图不可抓 -> 此区无望
            w = w / w.sum()
            cand0 = self.rot_cands[pid][0]
            dx = (xs[1] - xs[0]) * 0.5 if len(xs) > 1 else 0.0
            dy = (ys[1] - ys[0]) * 0.5 if len(ys) > 1 else 0.0

            placed_ok = False
            for _ in range(SAMPLE_TRIES_PER_PART):
                idx = int(rng.choice(len(w), p=w))
                ix, iy = divmod(idx, g.shape[1])
                x = float(np.clip(xs[ix] + rng.uniform(-dx, dx), hm["bx"][0], hm["bx"][1]))
                y = float(np.clip(ys[iy] + rng.uniform(-dy, dy), hm["by"][0], hm["by"][1]))
                self._apply_staging_pose(pid, np.array([x, y], dtype=float), cand0)
                hit = False
                for q in placed_ids:
                    if self.staging_models[pid].is_mcdwith(self.staging_models[q]):
                        hit = True
                        break
                if not hit:
                    xy[pid] = np.array([x, y], dtype=float)
                    placed_ids.append(pid)
                    placed_ok = True
                    break
            if not placed_ok:
                return None
        return xy

    def _heatmap_value_at(self, region_hm: Dict[str, Any], pid: str, xy: np.ndarray) -> float:
        hm = region_hm["parts"].get(pid)
        if hm is None:
            return 0.0
        g, xs, ys = hm["grid"], hm["xs"], hm["ys"]
        ix = int(np.clip(np.searchsorted(xs, xy[0]) - 0, 0, g.shape[0] - 1))
        iy = int(np.clip(np.searchsorted(ys, xy[1]) - 0, 0, g.shape[1] - 1))
        return float(g[ix, iy])

    def _layout_promise(self, region_hm: Dict[str, Any], xy: Dict[str, np.ndarray]) -> float:
        """不可行布局的热力图先验(0~1), 用于 PSO 惩罚分 shaping。"""
        vals = []
        for pid, p in xy.items():
            hm = region_hm["parts"].get(pid)
            if hm is None:
                continue
            mx = float(hm["grid"].max())
            if mx > 0:
                vals.append(self._heatmap_value_at(region_hm, pid, p) / mx)
        return float(np.mean(vals)) if vals else 0.0

    # ----- 代理模型 -----

    def _features(self, region_hm: Dict[str, Any], movable: List[str],
                  xy: Dict[str, np.ndarray]) -> List[float]:
        feats: List[float] = []
        for pid in movable:
            (bx0, bx1), (by0, by1) = region_hm["parts"][pid]["bx"], region_hm["parts"][pid]["by"]
            p = xy[pid]
            feats.append((float(p[0]) - bx0) / max(bx1 - bx0, 1e-6))
            feats.append((float(p[1]) - by0) / max(by1 - by0, 1e-6))
            mx = float(region_hm["parts"][pid]["grid"].max())
            feats.append(self._heatmap_value_at(region_hm, pid, p) / mx if mx > 0 else 0.0)
        return feats

    def _maybe_train_surrogate(self, X, yc, ys):
        if not (_HAS_SK and _CFG["use_surrogate"]):
            return None
        if len(X) < SURROGATE_MIN_DATA or int(np.sum(yc)) < SURROGATE_MIN_POS:
            return None
        try:
            clf = RandomForestClassifier(n_estimators=120, max_depth=None,
                                         random_state=0, n_jobs=-1)
            clf.fit(np.asarray(X), np.asarray(yc).astype(int))
            reg = None
            pos = [i for i, v in enumerate(yc) if v]
            if len(pos) >= SURROGATE_MIN_POS:
                reg = RandomForestRegressor(n_estimators=120, random_state=0, n_jobs=-1)
                reg.fit(np.asarray([X[i] for i in pos]), np.asarray([ys[i] for i in pos]))
            return {"clf": clf, "reg": reg}
        except Exception as e:
            print(f"[surrogate] train failed: {e}")
            return None

    # ----- 精确评估封装(顺便累积数据集) -----

    def _exact_eval(self, xy: Dict[str, np.ndarray]) -> Tuple[bool, float, LayoutCandidate]:
        cand = LayoutCandidate(xy={k: np.asarray(v, dtype=float).copy() for k, v in xy.items()})
        ok = bool(self.evaluate_layout(cand))
        return ok, (float(cand.layout_score) if ok else INFEASIBLE_BASE), cand

    # ----- PSO -----

    def _pso(self, rng, region_id, region_hm, movable, seed_layouts, dataset):
        """在固定装配区内对可动件 xy 做 PSO。seed_layouts: 已找到的可行 xy 列表。"""
        n_part = int(_CFG["pso_particles"])
        n_iter = int(_CFG["pso_iters"])
        dim = 2 * len(movable)

        lo = np.zeros(dim)
        hi = np.zeros(dim)
        for i, pid in enumerate(movable):
            (bx0, bx1), (by0, by1) = self._part_bounds(pid)
            lo[2 * i], hi[2 * i] = bx0, bx1
            lo[2 * i + 1], hi[2 * i + 1] = by0, by1
        span = np.maximum(hi - lo, 1e-6)

        def xy_to_vec(xy):
            v = np.zeros(dim)
            for i, pid in enumerate(movable):
                v[2 * i], v[2 * i + 1] = xy[pid]
            return v

        def vec_to_xy(v):
            return {pid: np.array([v[2 * i], v[2 * i + 1]], dtype=float)
                    for i, pid in enumerate(movable)}

        # 初始化粒子: 优先用已找到的可行解, 其余用碰撞感知顺序采样。
        pos = np.zeros((n_part, dim))
        for k in range(n_part):
            if k < len(seed_layouts):
                pos[k] = np.clip(xy_to_vec(seed_layouts[k]), lo, hi)
            else:
                s = None
                for _ in range(6):
                    s = self._sample_xy_collision_aware(rng, region_hm)
                    if s is not None:
                        break
                pos[k] = np.clip(xy_to_vec(s), lo, hi) if s else rng.uniform(lo, hi)
        vel = rng.uniform(-0.25, 0.25, size=(n_part, dim)) * span

        # 代理模型放进 holder, 以便 PSO 过程中周期性重训(用 holder 让闭包能看到最新模型)。
        sur_holder = {"m": self._maybe_train_surrogate(dataset["X"], dataset["yc"], dataset["ys"])}
        if sur_holder["m"] is not None:
            print(f"[surrogate] active at start (train={len(dataset['X'])} samples, "
                  f"pos={int(np.sum(dataset['yc']))})")

        pbest_pos = pos.copy()
        pbest_val = np.full(n_part, -np.inf)
        gbest_val = -np.inf
        gbest_cand: Optional[LayoutCandidate] = None
        gbest_vec = pos[0].copy()
        n_exact = 0
        n_skip = 0

        def fitness(v):
            nonlocal n_exact, n_skip
            xy = vec_to_xy(v)
            # 代理模型预筛: 预测几乎不可行就跳过精算, 只给 shaping 惩罚分。
            sur = sur_holder["m"]
            if sur is not None:
                try:
                    feat = np.asarray([self._features(region_hm, movable, xy)])
                    prob = float(sur["clf"].predict_proba(feat)[0][1])
                except Exception:
                    prob = 1.0
                if prob < SURROGATE_SKIP_PROB:
                    n_skip += 1
                    return INFEASIBLE_BASE + 0.04 * self._layout_promise(region_hm, xy), None
            ok, val, cand = self._exact_eval(xy)
            n_exact += 1
            dataset["X"].append(self._features(region_hm, movable, xy))
            dataset["yc"].append(1 if ok else 0)
            dataset["ys"].append(val if ok else 0.0)
            if not ok:
                val = INFEASIBLE_BASE + 0.05 * self._layout_promise(region_hm, xy)
                cand = None
            return val, cand

        for it in range(n_iter):
            for k in range(n_part):
                val, cand = fitness(pos[k])
                if val > pbest_val[k]:
                    pbest_val[k] = val
                    pbest_pos[k] = pos[k].copy()
                if cand is not None and val > gbest_val:
                    gbest_val = val
                    gbest_cand = cand
                    gbest_vec = pos[k].copy()
            # 速度/位置更新(围绕 gbest, 若还没有可行 gbest 则围绕全局 pbest)
            anchor = gbest_vec if gbest_cand is not None else pbest_pos[int(np.argmax(pbest_val))]
            r1 = rng.random((n_part, dim))
            r2 = rng.random((n_part, dim))
            vel = (PSO_W * vel
                   + PSO_C1 * r1 * (pbest_pos - pos)
                   + PSO_C2 * r2 * (anchor[None, :] - pos))
            vel = np.clip(vel, -0.5 * span, 0.5 * span)
            pos = np.clip(pos + vel, lo, hi)

            # 周期性用最新数据重训代理(让它在数据够了之后真正开始 skip 省算)。
            if SURROGATE_RETRAIN_EVERY > 0 and (it + 1) % SURROGATE_RETRAIN_EVERY == 0:
                m = self._maybe_train_surrogate(dataset["X"], dataset["yc"], dataset["ys"])
                if m is not None:
                    was_none = sur_holder["m"] is None
                    sur_holder["m"] = m
                    if was_none:
                        print(f"    [surrogate] now active (train={len(dataset['X'])} "
                              f"samples, pos={int(np.sum(dataset['yc']))})")

            print(f"  [PSO][{region_id}] iter {it + 1}/{n_iter}: "
                  f"gbest={gbest_val if gbest_cand else float('nan'):.4f} "
                  f"exact={n_exact} skip={n_skip} "
                  f"surrogate={'on' if sur_holder['m'] is not None else 'off'}")

        return gbest_cand, n_exact, n_skip

    # ----- L3: 与动画执行严格对齐(多方向 + 两臂单臂尝试) -----

    def _l3_plan_part(self, layout, step_idx, pid, arm_tag, sp, sr, gp, gr, gc, obs,
                      lft_transport, rgt_transport, verbose=True, placement_obs=None) -> bool:
        """重写父类钩子: 用与 execute_layout_sequence_visual 完全相同的方式验证落位。

        关键: 父类默认只试【指定臂】, 比动画(两臂)更严, 会误杀动画其实能跑通的布局。
        这里改成同口径:
            - preferred 臂 + 另一只臂;
            - 每只臂沿 asmdef 声明的插接轴做一次强制直线取放;
            - place approach 距离 0.02(与动画一致, 比父类 0.05 短, 更贴近实际);
            - 任一手臂规划成功即判通过。
        动画端在这之后还有一层无强制直线段的兜底, 所以这里通过的布局动画端一定能跑通。

        L3_HANDOVER_PART_IDS 里的零件(默认空)仍走换手验证。
        """
        if pid in L3_HANDOVER_PART_IDS:
            return self._l3_validate_handover(
                layout, step_idx, pid, sp, sr, gp, gr, gc, obs, verbose
            )

        if placement_obs is None:
            placement_obs = obs

        gp_a = np.asarray(gp, dtype=float)
        gr_a = np.asarray(gr, dtype=float)
        arm_order = [arm_tag, "rgt" if arm_tag == "lft" else "lft"]
        last_err = "no plan"

        for at in arm_order:
            transport = rgt_transport if at == "rgt" else lft_transport
            for motion_tag, mk in _l3_motion_candidate_kwargs(self.asm, pid, gr_a):
                obj_cm = fol.make_collision_model(self.asm.model_path(pid),
                                                  cdprim_type=self.cdprim_type)
                obj_cm.pos = np.asarray(sp, dtype=float).copy()
                obj_cm.rotmat = np.asarray(sr, dtype=float).copy()
                obj_cm._sealp_part_id = pid
                obj_cm._sealp_role = "l3_moving_object"
                try:
                    try:
                        res = transport.plan(
                            obj_cmodel=obj_cm,
                            grasp_collection=gc,
                            goal_pose_list=[(gp_a, gr_a)],
                            obstacle_list=obs,
                            grasp_obstacle_list=placement_obs,
                            approach_distance=L3_APPROACH_DIST,
                            depart_distance=L3_PICK_DEPART_DIST,
                            linear_granularity=L3_LINEAR_GRANULARITY,
                            **mk,
                        )
                    except TypeError:
                        res = transport.plan(
                            obj_cmodel=obj_cm,
                            grasp_collection=gc,
                            goal_pose_list=[(gp_a, gr_a)],
                            obstacle_list=obs,
                            approach_distance=L3_APPROACH_DIST,
                            depart_distance=L3_PICK_DEPART_DIST,
                            linear_granularity=L3_LINEAR_GRANULARITY,
                            **mk,
                        )
                except Exception as e:
                    last_err = f"{at} motion={motion_tag}: {type(e).__name__}: {e!r}"
                    continue
                if bool(getattr(res, "success", False)):
                    if verbose:
                        print(f"  [OK] step={step_idx} pid={pid:14s} arm={at} "
                              f"motion={motion_tag} (L3 == anime)")
                    return True
                last_err = (f"{at} motion={motion_tag}: "
                            f"{getattr(res, 'error_msg', '') or 'no plan'}")

        layout.l3_fail_reason = (f"L3 step={step_idx} {pid}: "
                                 f"all arm/motion candidates failed; last={last_err}")
        if verbose:
            print(f"  [FAIL] {layout.l3_fail_reason}")
        return False

    def _l3_validate_handover(self, layout, step_idx, pid, sp, sr, gp, gr, gc, obs,
                              verbose=True) -> bool:
        """用双臂换手验证某零件的全流程(取->换手->放), 与动画端逻辑一致。"""
        hopg_fp = os.path.join(HANDOVER_DIR, L3_HANDOVER_HOPG.get(pid, f"{pid}_hopg.pickle"))
        if not os.path.isfile(hopg_fp):
            layout.l3_fail_reason = f"L3 step={step_idx} {pid}: hopg missing: {hopg_fp}"
            if verbose:
                print(f"  [FAIL] {layout.l3_fail_reason}")
            return False

        start_pose = (np.asarray(sp, dtype=float).copy(), np.asarray(sr, dtype=float).copy())
        goal_pose = (np.asarray(gp, dtype=float).copy(), np.asarray(gr, dtype=float).copy())

        obj_cm = fol.make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
        obj_cm.pos = start_pose[0].copy()
        obj_cm.rotmat = start_pose[1].copy()
        obj_cm._sealp_part_id = pid
        obj_cm._sealp_role = "l3_handover_object"

        preferred = (layout.arm_choice or {}).get(pid)
        pairs = [
            ("lft", "rgt", self.robot.lft_arm, self.robot.rgt_arm),
            ("rgt", "lft", self.robot.rgt_arm, self.robot.lft_arm),
        ]
        if preferred == "rgt":
            pairs.reverse()

        last_err = "no handover path"
        for sender_tag, receiver_tag, sender_arm, receiver_arm in pairs:
            try:
                planner = _QuietHandoverPlanner(
                    obj_cmodel=obj_cm,
                    sender_robot=sender_arm,
                    receiver_robot=receiver_arm,
                    sender_reference_gc=gc,
                    receiver_reference_gc=_duplicate_grasp_collection(gc),
                )
                planner.add_hopg_collection_from_disk(hopg_fp)
                motion_list = planner.plan_by_obj_poses(
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    obstacle_list=obs,
                    toggle_dbg=False,
                )
            except Exception as e:
                last_err = (f"handover {sender_tag}->{receiver_tag}: "
                            f"exception {type(e).__name__}: {e!r}")
                continue

            if motion_list:
                if verbose:
                    print(f"  [OK/handover] step={step_idx} {pid} "
                          f"{sender_tag}->{receiver_tag} segs={len(motion_list)}")
                return True
            last_err = f"handover {sender_tag}->{receiver_tag}: no path"

        layout.l3_fail_reason = f"L3 step={step_idx} {pid}: {last_err}"
        if verbose:
            print(f"  [FAIL] {layout.l3_fail_reason}")
        return False

    # ----- 主搜索 -----

    def random_search(self,
                      n_samples: int,
                      seed: int,
                      max_resample_layout: int = 80,
                      verbose: bool = True,
                      enable_l3: bool = False,
                      l3_top_k: int = 3,
                      l3_obstacle_mode: str = "mesh",
                      require_l3: bool = True) -> Optional[LayoutCandidate]:
        if not _CFG["use_heatmap"] and not _CFG["use_pso"]:
            # 完全退回 fast/父类行为。
            return super().random_search(
                n_samples=n_samples, seed=seed, max_resample_layout=max_resample_layout,
                verbose=verbose, enable_l3=enable_l3, l3_top_k=l3_top_k,
                l3_obstacle_mode=l3_obstacle_mode, require_l3=require_l3,
            )

        rng = np.random.default_rng(seed)
        regions = self._assembly_region_candidates()
        movable = self._movable_parts()

        print("\n========== Heatmap + Surrogate + PSO Layout Search ==========")
        print(f"n_samples={n_samples} seed={seed}")
        print(f"heatmap={_CFG['use_heatmap']} pso={_CFG['use_pso']} "
              f"surrogate={_CFG['use_surrogate'] and _HAS_SK}")
        print(f"movable parts = {movable}")
        print(f"part_order    = {self.part_order}")

        # ---------- Phase 1: 热力图 ----------
        heatmaps = self.build_heatmaps(regions) if _CFG["use_heatmap"] else {}

        # 按 promise 排序装配区
        if heatmaps:
            ranked = sorted(regions, key=lambda r: heatmaps.get(r[0], {}).get("promise", 0.0),
                            reverse=True)
            print("\n[region ranking by heatmap promise]")
            for (rid, rc, pos) in ranked:
                print(f"  region {rid} rc={rc} promise={heatmaps.get(rid, {}).get('promise', 0.0):.3f}")
            self._print_heatmap_preview(heatmaps, ranked[0][0])
        else:
            ranked = list(regions)

        top_regions = ranked[:max(1, int(_CFG["hm_top_regions"]))]

        feasible: List[LayoutCandidate] = []
        dataset = {"X": [], "yc": [], "ys": []}
        t0 = time.time()

        # ---------- Phase 2: 热力图加权随机播种 ----------
        n_seed = int(n_samples * SEED_FRAC) if _CFG["use_pso"] else n_samples
        n_seed = max(n_seed, len(top_regions) * 2)
        print(f"\n========== Phase 2: heatmap-biased seeding ({n_seed} samples) ==========")
        for i in range(n_seed):
            rid, rc, pos = top_regions[i % len(top_regions)]
            self._set_assembly_station(pos, region_id=rid, rc=rc)
            region_hm = heatmaps.get(rid)
            xy = None
            if region_hm is not None:
                # 碰撞感知顺序采样, 多试几次拿到不重叠布局
                for _ in range(6):
                    xy = self._sample_xy_collision_aware(rng, region_hm)
                    if xy is not None:
                        break
            else:
                for _ in range(max_resample_layout):
                    xy = self.sample_collision_free_xy(rng)
                    if xy is not None:
                        break
            if xy is None:
                continue
            ok, val, cand = self._exact_eval(xy)
            if heatmaps:
                dataset["X"].append(self._features(heatmaps[rid], movable, xy))
                dataset["yc"].append(1 if ok else 0)
                dataset["ys"].append(val if ok else 0.0)
            if ok:
                feasible.append(cand)
                if verbose:
                    print(f"#{i:03d} L2_OK score={cand.layout_score:.4f} region={cand.assembly_region_id} "
                          f"counts={cand.grasp_counts}")
            elif verbose:
                print(f"#{i:03d} FAIL {cand.fail_reason[:80]}")

        print(f"[seed] feasible={len(feasible)}/{n_seed}  elapsed={time.time() - t0:.1f}s")

        # ---------- Phase 3+4: PSO(在最优区) ----------
        if _CFG["use_pso"] and heatmaps:
            # 选 PSO 的区: 优先"已找到可行解最多/最好"的区, 否则 promise 最高区。
            if feasible:
                from collections import Counter
                best_region = Counter(c.assembly_region_id for c in feasible).most_common(1)[0][0]
            else:
                best_region = top_regions[0][0]
            reg_tuple = next((r for r in regions if r[0] == best_region), top_regions[0])
            self._set_assembly_station(reg_tuple[2], region_id=reg_tuple[0], rc=reg_tuple[1])
            region_hm = heatmaps[best_region]

            seed_layouts = [c.xy for c in feasible if c.assembly_region_id == best_region]
            print(f"\n========== Phase 3/4: PSO in region {best_region} "
                  f"({len(seed_layouts)} seeds) ==========")
            gbest, n_exact, n_skip = self._pso(rng, best_region, region_hm, movable,
                                               seed_layouts, dataset)
            if gbest is not None:
                feasible.append(gbest)
            print(f"[PSO] done. exact_evals={n_exact} surrogate_skips={n_skip}")

        # ---------- 汇总 ----------
        if not feasible:
            print("\n[FAIL] 没有找到任何 L2 可行布局。")
            return None
        feasible.sort(key=lambda c: c.layout_score, reverse=True)
        best = feasible[0]

        print("\n========== Search Summary ==========")
        print(f"total wall          = {time.time() - t0:.1f}s")
        print(f"feasible layouts    = {len(feasible)}")
        print(f"dataset size        = {len(dataset['X'])} (pos={int(np.sum(dataset['yc'])) if dataset['yc'] else 0})")
        print(f"[BEST] score={best.layout_score:.4f} region={best.assembly_region_id} "
              f"rc={best.assembly_region_rc}")
        print(f"  grasp_counts={best.grasp_counts}")
        print(f"  arm_choice  ={best.arm_choice}")
        print(f"  pose_tag    ={best.pose_tag}")

        # ---------- 可选 L3 ----------
        if enable_l3:
            print("\n========== L3 full-process validation ==========")
            k = min(int(l3_top_k), len(feasible))
            for rank, cand in enumerate(feasible[:k], start=1):
                print(f"[L3] rank {rank}/{k} score={cand.layout_score:.4f}")
                if self.validate_full_sequence_l3(cand, obstacle_mode=l3_obstacle_mode, verbose=True):
                    print(f"[OK] L3 passed rank={rank}")
                    return cand
                print(f"[NO] L3 failed rank={rank}: {cand.l3_fail_reason}")
            if require_l3:
                print("\n[FAIL] L3 top-k 全部未通过。")
                return None
            print("\n[WARN] L3 全失败, require_l3=False, 回退 L2 最优。")

        return best


# ============================================================
# Entry point
# ============================================================

def _patch_module() -> None:
    fol.WeightedInitialLayoutSearcher = HeatmapPSOSearcher


def main():
    # 抽出自定义 flag, 否则 fol._parse_args 会报 unrecognized arguments。
    _CFG["use_heatmap"] = not fast._consume_extra_flag("--no-heatmap")
    _CFG["use_pso"] = not fast._consume_extra_flag("--no-pso")
    _CFG["use_surrogate"] = not fast._consume_extra_flag("--no-surrogate")
    _CFG["rebuild_heatmap"] = fast._consume_extra_flag("--rebuild-heatmap")
    for name, key in (("--hm-grid", "hm_grid"), ("--hm-cap", "hm_cap"),
                      ("--hm-top-regions", "hm_top_regions"),
                      ("--pso-particles", "pso_particles"), ("--pso-iters", "pso_iters")):
        val = fast._consume_extra_value(name)
        if val is not None:
            try:
                _CFG[key] = int(val)
            except ValueError:
                print(f"[heatmap-pso] WARN: 无法解析 {name} '{val}', 用默认。")

    # 姿态硬约束: 默认【开启】"topdown 抓取不足必须立起"。这条规则按抓取库数据触发
    # (不是给 middle_plate 写死), 防止薄板被平放在桌上导致夹爪插桌、实际抓不到。
    # 兼容旧 flag: --keep-upright-preference 仍可显式保留(等价默认行为);
    # 想关闭(允许平放)用 --disable-upright-preference。
    fast._consume_extra_flag("--keep-upright-preference")  # 吞掉, 兼容旧命令
    if DEFAULT_DISABLE_UPRIGHT_PREFERENCE and "--disable-upright-preference" not in sys.argv:
        sys.argv.append("--disable-upright-preference")
        print("[heatmap-pso] auto-injected --disable-upright-preference (允许平放)")
    elif "--disable-upright-preference" in sys.argv:
        print("[heatmap-pso] --disable-upright-preference: 允许平放(薄板可能实际抓不到, 慎用)")
    else:
        print("[heatmap-pso] 姿态硬约束开启: topdown 抓取不足的件(如 middle_plate)必须立起, "
              "禁止平放(避免夹爪插桌抓不到)")

    # 换手已默认取消(L3 全部单臂验证)。如需恢复 middle_plate 换手验证, 加该 flag。
    if fast._consume_extra_flag("--l3-handover-middle-plate"):
        global L3_HANDOVER_PART_IDS
        L3_HANDOVER_PART_IDS = frozenset({"middle_plate"})
        print("[heatmap-pso] --l3-handover-middle-plate: L3 将对 middle_plate 走换手验证(较慢)")
    else:
        print("[heatmap-pso] 换手已取消: L3 全部走单臂验证(middle_plate 也单臂)")

    # 强制 middle_plate 旋转(用户显式要求): 默认 rot90_04(长边直立, 可被 handover 抓)。
    # --mp-force-rot none 关闭; --mp-force-rot <rot_name> 指定其它。
    mp_force_rot = fast._consume_extra_value("--mp-force-rot")
    if mp_force_rot is None:
        mp_force_rot = "rot90_04"
    if str(mp_force_rot).lower() in ("none", "off", ""):
        HeatmapPSOSearcher.force_rot_name = {}
        print("[heatmap-pso] --mp-force-rot none: 不强制 middle_plate 旋转")
    else:
        HeatmapPSOSearcher.force_rot_name = {"middle_plate": str(mp_force_rot)}
        print(f"[heatmap-pso] middle_plate 强制旋转 = {mp_force_rot} (长边直立)")

    print("[heatmap-pso] config:")
    for k, v in _CFG.items():
        print(f"    {k:16s} = {v}")
    print(f"[heatmap-pso] sklearn available = {_HAS_SK}")

    # 走 fast 的默认 flag 注入(order-x 默认关) + IK 缓存安装。
    fast._maybe_inject_default_flags()
    fast._install_ik_cache()
    fast._pose_cache_reset_stats()

    _patch_module()

    wall_t0 = time.perf_counter()
    try:
        fol.main()
    finally:
        print(f"[heatmap-pso] wall-clock total = {time.perf_counter() - wall_t0:.3f}s")
        try:
            fast._print_ik_cache_report()
        except Exception:
            pass


if __name__ == "__main__":
    main()
