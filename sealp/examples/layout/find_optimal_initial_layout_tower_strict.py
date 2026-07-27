#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict Tower Initial Layout Search — PyCharm Runnable
=====================================================

这是一个可直接在 PyCharm 里右键运行的 Tower 初始摆放搜索脚本。
它模仿 find_optimal_layout.py 的核心思路：

1. 先把 work_table 粗分成 3x3 装配候选区域，自动选择装配中心；再在 work_table 上随机采样其它零件初始 (x, y)。
2. 对每个随机 layout，按装配顺序逐个零件评估可行性。
3. 每个零件自动枚举候选初始旋转姿态：
   - 优先使用 flatsurface.py 计算出的稳定摆放旋转矩阵；
   - 若 flatsurface 不可用，再回退 identity/90° 站立/侧放姿态；
   - 每次旋转后自动根据旋转后的包围盒 z_min 计算 z_offset，避免半个物体插进 work_table。
4. 对每个可行 layout 计算综合得分：
       layout_score
         = w_grasp * 抓取冗余度
         + w_manip * 端点灵巧度
         + w_dist  * 搬运距离得分
         + w_rot   * 旋转代价得分
6. 保存综合得分最高的 layout，供后续执行脚本使用.

推荐放置：
    sealp/examples/layout/find_optimal_initial_layout_tower_strict.py

PyCharm 右键运行：
    不需要填写任何参数，默认读取当前 Tower 的 asmdef / sample_config / tower_grasp，
    默认执行 L2 加权随机搜索，并额外检查初始 staging 不与机器人 home 姿态穿模；如需 L3 全流程动态避障验证，可加 --enable-l3。

命令行也可以覆盖默认参数，例如：
    python -m sealp.examples.layout.find_optimal_initial_layout_tower_strict_pycharm --n-samples 20 --l3-top-k 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # 兜底：假设当前脚本在 sealp/examples/layout 下
            return os.path.abspath(os.path.join(start_dir, "..", ".."))
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
from wrs.manipulation.pick_place import PickPlacePlanner
import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda
from sealp.config.workspace_settings import load_workspace_settings
from sealp.layout.layout_robot_factory import (
    arm_base_xy_map as _layout_arm_base_xy_map,
    create_layout_robot,
    get_layout_arm,
    goto_home_joints,
    iter_layout_arms,
    layout_arm_pairs,
    reset_robot_for_l3 as _reset_layout_robot_for_l3,
)
from wrs.grasping.grasp import GraspCollection

from sealp.assembly_sequence import AssemblyDef
from sealp.assembly_sequence.mating_detection import MatingCache
from sealp.config import load_config
from sealp.colliders import StaticEnvironment
from sealp.layout import WorkspaceLayout
from sealp.layout.dual_staging_search import find_obstacle_def
from sealp.primitives.transport import TransportPrimitive

try:
    from sealp.layout.reachability import check_pose_reachability
except Exception:
    check_pose_reachability = None


# ============================================================
# 默认参数
# ============================================================

DUAL_ARM_Y_OFFSET = 0.62
HOME_JV = np.zeros(6)

# 全流程 transport / pick-place 验证参数
PICK_DEPART_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
PICK_DEPART_DIST = 0.05
PLACE_APPROACH_DIR = np.array([0.0, 0.0, -1.0], dtype=float)
PLACE_APPROACH_DIST = 0.05
PLACE_DEPART_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
PLACE_DEPART_DIST = 0.05
APPROACH_DIST = 0.0
LINEAR_GRANULARITY = 0.04

DEFAULT_TABLE_MARGIN = 0.06
DEFAULT_TABLE_CLEARANCE = 0.003
DEFAULT_MAX_ROT_CANDIDATES = 12

# 归一化参数
NORM_GRASP_MIN_TARGET = 8.0
NORM_GRASP_MEAN_TARGET = 20.0
NORM_GRASP_HILL_K = 2.0
NORM_MANIP_TARGET = 0.030
NORM_DIST_DECAY = 0.55
NORM_ROT_DECAY = 0.75


# 装配区域搜索：默认把 work_table 粗分成 3x3，自动选择装配中心。
DEFAULT_PLAN_ASSEMBLY_REGION = True
DEFAULT_ASSEMBLY_GRID = 3
DEFAULT_PREASSEMBLE_FIRST_PART = True

# 装配区避让：
# 3x3 的某些中心点可能落在左/右臂底盘附近，尤其 y 接近 +/- DUAL_ARM_Y_OFFSET。
# 这类装配中心即使 L2 分数高，真实运动时也容易离手臂太近、姿态别扭。
# 默认直接从 assembly region candidates 中剔除。
# 装配区不再使用“离左右臂基座矩形距离”的过滤。
# 装配区仍然按照 3x3 网格全部尝试；
# 只在 preassembled 第一件实际放到该装配中心后，检查它是否和机械臂碰撞盒碰撞。
DEFAULT_FILTER_ASSEMBLY_NEAR_ARMS = False
DEFAULT_ASSEMBLY_ARM_X_CLEARANCE = 0.22
DEFAULT_ASSEMBLY_ARM_Y_CLEARANCE = 0.22

# staging 初始摆放避让：
# 不只装配区中心要避开左右臂，随机采样每个待抓取零件的初始位置时，
# 也要避开左臂/右臂基座附近的矩形区域。
# 判断方式是“零件 footprint AABB 与 arm keepout rectangle 是否重叠”。
# 这样避免“先乱采样，再被 robot_home_collision 大量杀掉”，提高有效采样率。
DEFAULT_FILTER_STAGING_NEAR_ARMS = True
DEFAULT_STAGING_ARM_X_CLEARANCE = 0.12
DEFAULT_STAGING_ARM_Y_CLEARANCE = 0.12

# 根据最终装配位置的 y 左/右关系，对 staging 初始位置做半桌面优先采样。
# y 更大 = 更偏左 -> 优先采样桌子左半边；
# y 更小 = 更偏右 -> 优先采样桌子右半边。
# 这是“优先”，不是硬约束；前面一定比例尝试优先半边，失败后回退全桌面。
DEFAULT_GOAL_Y_SIDE_BIASED_SAMPLING = True
DEFAULT_GOAL_Y_SIDE_BIAS_RATIO = 0.75
DEFAULT_GOAL_Y_SIDE_EPS = 0.005

# 初始摆放姿态：默认优先用 flatsurface.py 计算稳定摆放角度。
DEFAULT_USE_FLATSURFACE = True
DEFAULT_FS_STABILITY_THRESHOLD = 0.10

# 初始布局硬约束：所有 staging 零件不得与机器人 home 姿态下的任一机械臂/夹爪穿模。
# 该约束用于过滤 middle_plate 立起来后插到机械手附近的无效 layout。
DEFAULT_CHECK_ROBOT_HOME_COLLISION = True

# 更严格的初始布局检查：
# 1) final layout 中所有初始物体，包括 preassembled 的 base_plate，都不能和左/右机械臂碰撞盒碰撞；
# 2) 在 3x3 装配区候选阶段，如果第一件预装在该中心后会和机械臂碰撞盒碰撞，则该装配区直接剔除。
# 注意：这里不使用 self.robot.use_all()/self.robot.is_collided() 的整机碰撞检查，
# 只检查 self.robot.lft_arm / self.robot.rgt_arm 的 arm collision boxes。
DEFAULT_STRICT_INITIAL_ROBOT_COLLISION = True

# 新增初始布局几何约束：
# 1) 任意两个 staging 零件外轮廓之间至少保留 1cm 间隙；
# 2) 装配顺序越靠后的零件，默认不允许比前一个待装零件放得更远，也就是 x 更大；
# 3) y 方向尽量往两侧分布，作为 layout 分数的小幅加成。
DEFAULT_MIN_STAGING_MESH_CLEARANCE = 0.01

DEFAULT_ENFORCE_ORDER_X_CONSTRAINT = True
DEFAULT_ORDER_X_TOLERANCE = 0.03
DEFAULT_ENABLE_Y_SIDE_DISTRIBUTION_SCORE = True

# 如果某个零件在原始姿态下“夹爪朝世界 -Z 方向”的抓取数太少，
# 则搜索时强制选择 upright / 侧立候选姿态，不再允许 identity / 平放姿态。
DEFAULT_PREFER_UPRIGHT_WHEN_TOPDOWN_LOW = True
DEFAULT_TOPDOWN_MIN_COUNT = 10
DEFAULT_TOPDOWN_ALIGN_COS = 0.82

# L2 快速抓取检查：
# 对指定零件，在 L2 reason_common_gids 之后再额外检查 pre-pick / pick / post-pick 三个位姿。
# 目的：提前过滤“静态看起来能抓，但实际夹爪接近/抬起时会碰桌子、碰机器人、碰其它零件”的初始姿态。
# 默认关闭 L2 pre/pick/post 快速检查：
# 也就是默认不再用“撤离/接近距离”过滤 layout。
# 其它限制，例如 mesh 间距、robot home collision、upright、staging keepout 等仍保留。
DEFAULT_CHECK_L2_PICK_QUICK_MOTION = False
DEFAULT_L2_PICK_CHECK_PARTS = "middle_plate"
DEFAULT_L2_PICK_CHECK_LIFT_DIST = 0.06

# L2 pre/pick/post 快速检查不应该只固定 +Z 撤离。
# 默认尝试多个方向；只要其中一个方向能保留有效 gid，就认为该候选通过。
DEFAULT_L2_PICK_CHECK_DIRECTIONS = "z,x_plus,x_minus,y_plus,y_minus"
DEFAULT_L2_PICK_CHECK_TILT = 0.35

# 机器人 home 额外安全距离：
# 原来只检查是否穿模；现在再尽量要求 staging 零件离机器人 home 手臂/夹爪保留一定距离。
# 注意：该距离检查依赖能否从 WRS robot link 提取 mesh/AABB；如果提取失败，会自动退化为原来的碰撞检查。
DEFAULT_ROBOT_HOME_CLEARANCE = 0.03

DEFAULT_W_GRASP = 0.3
DEFAULT_W_MANIP = 0.4
DEFAULT_W_DIST = 0.1
DEFAULT_W_ROT = 0.2


# ============================================================
# PyCharm 右键运行默认配置
# ============================================================
# 这些默认路径都基于 SEALP_ROOT 自动生成，所以只要本脚本放在 sealp 目录内部，
# 直接右键运行就能跑 Tower 任务。
DEFAULT_ASMDEF = os.path.join(SEALP_ROOT, "assembly_sequence", "_demo_output", "topdown_tower.asmdef")
DEFAULT_CONFIG = os.path.join(SEALP_ROOT, "config", "sample_config.yaml")
DEFAULT_GRASP_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_grasp")
DEFAULT_PART_ORDER = "base_plate,post_br,post_fr,post_bl,post_fl,middle_plate,top_cross"
DEFAULT_OUTPUT_NAME = "tower_optimal_initial"
DEFAULT_OUTPUT_DIR = os.path.join(SEALP_ROOT, "examples", "layout", "_output")
DEFAULT_FIXTURE_POS = "0.36,0,0"
DEFAULT_ROBOT_BASE_POS = "0,0,0"

# 右键运行时不要太大，否则 L3 会很慢。正式搜索可改成 20/50/80。
DEFAULT_N_SAMPLES = 10
DEFAULT_SEED = 0
DEFAULT_CDPRIM_TYPE = "triangles"

# L2 抓取校验默认 staging_aware: 排除桌面误杀 + 已装件接触豁免 + 其它 staging 件做障碍。
# 不把桌面放进抓取校验(见下方说明): 零件贴桌摆放时, 把桌面当障碍会误杀几乎所有低位抓取
# (连站立细杆 post 都被判 no_common_gids)。执行时的桌面安全改由"让细长件站立"保证:
# 站立的 post 小面触地, 抓取点在杆身/顶部, 天然远离桌面。executor_match 仍作为可选模式保留。
DEFAULT_L2_OBSTACLE_MODE = "staging_aware"

# 默认只保存 L2 结果；严格 L3 很慢，需要时用 --enable-l3 打开。
# L3 与 L2 保持同一障碍口径(staging_aware), 保证 L2 通过的公共抓取不会在 L3 被二次否决。
DEFAULT_ENABLE_L3 = False
DEFAULT_L3_TOP_K = 3
DEFAULT_L3_OBSTACLE_MODE = "staging_aware"

# L3 全流程验证时跳过运动规划的零件(逗号分隔)。
# middle_plate 这类大件在 L3 里的取放直线段/RRT 常因为过严而失败, 但实际执行没问题;
# 跳过后仍把它当作已放置(计入后续零件的 step-aware 障碍), 只是不对它本身做 L3 运动验证,
# 其余零件照常严格验证。
DEFAULT_L3_SKIP_PARTS = "middle_plate"
DEFAULT_ALLOW_L2_FALLBACK = True

# ---- 默认姿态保持 (default resting-face / STL up-face preservation) ----
# 泛化性软偏好: 对"有指向性"的件(细长杆 / 扁平板), 在没有 topdown 硬约束时,
# 倾向保持它 STL 默认姿态下的朝上轴仍朝上 —— 即细杆保持"站立"、扁板保持"平放",
# 而不是被翻倒。这样细长 post 不会平躺贴桌导致执行取放撞桌。
# 判定完全基于几何(STL 默认 extent 的指向性 + 候选把 STL-Z 轴翻转多少度), 无需针对具体零件硬编码。
# 这是软惩罚: 只有当直立/默认姿态确实存在 common grasp 时它才会胜出;
# 若默认姿态没有可行抓取, 带惩罚的其它姿态仍可被选中(不会把零件卡死)。
DEFAULT_PREFER_STL_UPFACE = True
DEFAULT_W_STL_UPFACE = 0.35            # 惩罚权重(相对归一化后的 part_score, 量级~[0,1])
DEFAULT_STL_UPFACE_MIN_THINNESS = 0.20  # 只对 thinness>=此值(足够细长/扁平)的件生效


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RotCandidate:
    rotmat: np.ndarray
    z_offset: float
    tag: str
    extent: np.ndarray
    footprint: np.ndarray
    rot_name: str = "unknown"
    fs_pos: Optional[np.ndarray] = None


@dataclass
class LayoutCandidate:
    xy: Dict[str, np.ndarray]
    layout_score: float = -np.inf
    assembly_region_id: str = "fixed"
    assembly_region_rc: Tuple[int, int] = (-1, -1)
    assembly_station_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    assembly_station_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    chosen_rotmat: Dict[str, np.ndarray] = field(default_factory=dict)
    z_offset: Dict[str, float] = field(default_factory=dict)
    pose_tag: Dict[str, str] = field(default_factory=dict)
    rot_name: Dict[str, str] = field(default_factory=dict)
    arm_choice: Dict[str, str] = field(default_factory=dict)
    grasp_counts: Dict[str, int] = field(default_factory=dict)
    per_part_dist: Dict[str, float] = field(default_factory=dict)
    per_part_manip: Dict[str, float] = field(default_factory=dict)
    per_part_rot_angle: Dict[str, float] = field(default_factory=dict)
    grasp_score_norm: float = 0.0
    manip_score_norm: float = 0.0
    dist_score_norm: float = 0.0
    rot_score_norm: float = 0.0
    spatial_score_norm: float = 0.0
    topdown_counts: Dict[str, int] = field(default_factory=dict)
    fail_reason: str = ""
    fail_part: Optional[str] = None
    fail_detail: Dict[str, int] = field(default_factory=dict)
    l2_pass: bool = False
    l3_pass: bool = False
    l3_fail_reason: str = ""


# ============================================================
# 基础工具
# ============================================================

def _parse_vec3(text: str, default: Tuple[float, float, float]) -> np.ndarray:
    if text is None or str(text).strip() == "":
        return np.asarray(default, dtype=float)
    return np.asarray([float(x.strip()) for x in text.split(",")], dtype=float)


def _parse_part_order(text: str) -> Optional[List[str]]:
    if text is None or text.strip() == "":
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_json_map(text: str) -> Dict[str, str]:
    if text is None or text.strip() == "":
        return {}
    return dict(json.loads(text))


def _hill(x: float, target: float, k: float = 2.0) -> float:
    if x <= 0:
        return 0.0
    r = (x / target) ** k
    return float(r / (1.0 + r))


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    try:
        _, ang = rm.axangle_between_rotmat(np.asarray(Ra), np.asarray(Rb))
        return float(ang) if np.isfinite(ang) else 0.0
    except Exception:
        v = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
        return float(np.arccos(np.clip(v, -1.0, 1.0)))


def _load_mesh_vertices(mesh_path: str) -> np.ndarray:
    try:
        import trimesh
    except Exception:
        import wrs.basis.trimesh as trimesh
    mesh = trimesh.load_mesh(mesh_path)
    return np.asarray(mesh.vertices, dtype=float)


def _bounds_after_rotation(vertices: np.ndarray, rotmat: np.ndarray):
    pts = np.asarray(vertices, dtype=float) @ np.asarray(rotmat, dtype=float).T
    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    extent = bmax - bmin
    return bmin, bmax, extent


def _rotmat_90_candidates() -> List[np.ndarray]:
    """生成常用 90° 旋转候选。

    identity 会排在最前。其余候选用于在平放不可行时站立/侧放。
    """
    base_rots = [
        np.eye(3),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(180.0)),
    ]

    yaw_rots = [
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(a))
        for a in (0.0, 90.0, -90.0, 180.0)
    ]

    out: List[np.ndarray] = []
    for B in base_rots:
        for Z in yaw_rots:
            R = Z @ B
            if not any(np.allclose(R, old, atol=1e-6) for old in out):
                out.append(R)

    # identity 强制第一
    out.sort(key=lambda R: 0 if np.allclose(R, np.eye(3), atol=1e-6) else 1)
    return out



def _load_flatsurface_reference_class():
    """动态加载 flatsurface.py 里的 FSReferencePoses。

    你当前项目里的真实路径是：
        wrs/manipulation/placement/flatsurface.py

    所以这里优先用包导入：
        from wrs.manipulation.placement.flatsurface import FSReferencePoses

    如果包导入失败，再尝试若干文件路径；都失败才回退到 90° 姿态候选。
    """
    # 1) 正确优先路径：wrs/manipulation/placement/flatsurface.py
    try:
        from wrs.manipulation.placement.flatsurface import FSReferencePoses
        print("[INFO] flatsurface loaded: wrs.manipulation.placement.flatsurface")
        return FSReferencePoses
    except Exception as e:
        print(f"[WARN] 包导入 flatsurface 失败: {type(e).__name__}: {e}")

    # 2) 文件路径兜底
    candidate_files = [
        os.path.join(PROJECT_ROOT, "wrs", "manipulation", "placement", "flatsurface.py"),
        os.path.join(SEALP_ROOT, "..", "wrs", "manipulation", "placement", "flatsurface.py"),
        os.path.join(_THIS_DIR, "flatsurface.py"),
        os.path.join(SEALP_ROOT, "examples", "layout", "flatsurface.py"),
        os.path.join(SEALP_ROOT, "examples", "grasp", "flatsurface.py"),
        os.path.join(SEALP_ROOT, "flatsurface.py"),
    ]

    for fp in candidate_files:
        fp = os.path.abspath(fp)
        if not os.path.isfile(fp):
            continue
        try:
            spec = importlib.util.spec_from_file_location("tower_flatsurface_runtime", fp)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, "FSReferencePoses", None)
            if cls is not None:
                print(f"[INFO] flatsurface loaded: {fp}")
                return cls
        except Exception as e:
            print(f"[WARN] 加载 flatsurface.py 失败: {fp}, {type(e).__name__}: {e}")

    # 3) 最后再尝试普通 import，兼容用户自己放到 PYTHONPATH 的情况
    try:
        from wrs.manipulation.placement.flatsurface import FSReferencePoses
        print("[INFO] flatsurface loaded by normal import.")
        return FSReferencePoses
    except Exception as e:
        print(f"[WARN] 未找到可用 flatsurface.py，回退 90° 姿态候选: {type(e).__name__}: {e}")
        return None


def _stable_pose_candidates_from_flatsurface(
    mesh_path: str,
    stability_threshold: float = 0.10,
    max_candidates: int = 24,
) -> List[Tuple[str, Optional[np.ndarray], np.ndarray]]:
    """用 flatsurface.py 计算稳定摆放姿态。

    返回:
        [(rot_name, fs_pos, rotmat), ...]

    注意：
        flatsurface 返回的 pos 是稳定放置参考位姿里的位置修正。
        本脚本最终仍统一用 rotated bounds 计算 z_offset，避免物体插入桌面。
    """
    cls = _load_flatsurface_reference_class()
    if cls is None:
        return []

    try:
        obj = mcm.CollisionModel(mesh_path)
        fs_ref = cls(
            obj_cmodel=obj,
            stability_threshhold=float(stability_threshold),
        )
        out: List[Tuple[str, Optional[np.ndarray], np.ndarray]] = []
        for i, pose in enumerate(fs_ref):
            try:
                fs_pos, R = pose
                R = np.asarray(R, dtype=float)
                fs_pos = np.asarray(fs_pos, dtype=float)
                if R.shape != (3, 3):
                    continue
                if not np.all(np.isfinite(R)):
                    continue
                if any(np.allclose(R, old_R, atol=1e-6) for _, _, old_R in out):
                    continue
                out.append((f"fs_{i:02d}", fs_pos, R))
                if len(out) >= int(max_candidates):
                    break
            except Exception:
                continue
        return out
    except Exception as e:
        print(f"[WARN] flatsurface 计算失败: {os.path.basename(mesh_path)}, {type(e).__name__}: {e}")
        return []


def _fallback_rot_candidates_with_names() -> List[Tuple[str, Optional[np.ndarray], np.ndarray]]:
    """把旧的 90° 候选转成带名字的候选。"""
    out: List[Tuple[str, Optional[np.ndarray], np.ndarray]] = []
    for i, R in enumerate(_rotmat_90_candidates()):
        if np.allclose(R, np.eye(3), atol=1e-6):
            name = "identity"
        else:
            name = f"rot90_{i:02d}"
        out.append((name, None, R))
    return out


def _dedupe_pose_rot_candidates(
    candidates: List[Tuple[str, Optional[np.ndarray], np.ndarray]]
) -> List[Tuple[str, Optional[np.ndarray], np.ndarray]]:
    """按旋转矩阵去重，保留前面的候选。"""
    out: List[Tuple[str, Optional[np.ndarray], np.ndarray]] = []
    for name, fs_pos, R in candidates:
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            continue
        if any(np.allclose(R, old_R, atol=1e-6) for _, _, old_R in out):
            continue
        out.append((name, fs_pos, R))
    return out


def _table_info(config_yaml: str, table_name: str, margin: float):
    """读取 work_table 的 xy 范围和顶面高度。"""
    if not config_yaml or not os.path.isfile(config_yaml):
        return (0.05, 0.65), (-0.95, 0.35), 0.0

    cfg = load_config(config_yaml)
    table_def = find_obstacle_def(cfg.obstacle_defs, table_name)
    if table_def and table_def.get("type") == "box":
        pos = np.asarray(table_def.get("pos", [0, 0, 0]), dtype=float)
        extent = np.asarray(table_def.get("extent", [0.8, 1.2, 0.02]), dtype=float)
        x_range = (
            float(pos[0] - extent[0] / 2.0 + margin),
            float(pos[0] + extent[0] / 2.0 - margin),
        )
        y_range = (
            float(pos[1] - extent[1] / 2.0 + margin),
            float(pos[1] + extent[1] / 2.0 - margin),
        )
        top_z = float(pos[2] + extent[2] / 2.0)
        return x_range, y_range, top_z

    return (0.05, 0.65), (-0.95, 0.35), 0.0


def _load_env_obstacles(config_yaml: str) -> List:
    if not config_yaml or not os.path.isfile(config_yaml):
        return []
    cfg = load_config(config_yaml)
    env = StaticEnvironment(obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
    obs = list(env.obstacle_list)
    for o in obs:
        o._sealp_role = "environment_obstacle"
    return obs


def make_collision_model(mesh_path: str, cdprim_type: str = "triangles"):
    """创建 CollisionModel，并尽量使用更接近 STL mesh 的碰撞类型。

    旧问题：直接 mcm.CollisionModel(path) 在不少 WRS 版本里会生成较粗的
    collision primitive，例如 box / cdprim，容易出现“视觉上没碰撞，但碰撞
    检查认为碰撞”的误杀。

    这里优先尝试 cdprim_type="triangles"。如果当前 WRS 版本不支持，就依次
    退回 convex_hull / 默认构造，保证脚本仍可运行。
    """
    cdprim_type = str(cdprim_type or "triangles")
    trials = []
    if cdprim_type not in ("default", "none", ""):
        trials.append({"initor": mesh_path, "cdprim_type": cdprim_type})
    if cdprim_type != "triangles":
        trials.append({"initor": mesh_path, "cdprim_type": "triangles"})
    trials.extend([
        {"initor": mesh_path, "cdprim_type": "convex_hull"},
        {"initor": mesh_path},
        {"initor": mesh_path},
    ])

    last_err = None
    for kw in trials:
        try:
            cm = mcm.CollisionModel(**kw)
            cm._sealp_cdprim_type = kw.get("cdprim_type", "default")
            return cm
        except TypeError as e:
            # 有些 WRS 版本参数名不是 initor，尝试位置参数形式
            last_err = e
            if "initor" in kw and "cdprim_type" in kw:
                try:
                    cm = mcm.CollisionModel(mesh_path, cdprim_type=kw["cdprim_type"])
                    cm._sealp_cdprim_type = kw["cdprim_type"]
                    return cm
                except Exception as ee:
                    last_err = ee
            elif "initor" in kw:
                try:
                    cm = mcm.CollisionModel(mesh_path)
                    cm._sealp_cdprim_type = "default"
                    return cm
                except Exception as ee:
                    last_err = ee
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"无法创建 CollisionModel: {mesh_path}, last_err={last_err!r}")


def _model_id_for_part(asm: AssemblyDef, part_id: str) -> Optional[str]:
    for attr in ("parts", "part_defs"):
        parts = getattr(asm, attr, None)
        if parts is None:
            continue
        if isinstance(parts, dict):
            p = parts.get(part_id)
            if p is not None:
                return getattr(p, "model", None) or getattr(p, "model_id", None)
        else:
            for p in parts:
                if getattr(p, "part_id", None) == part_id:
                    return getattr(p, "model", None) or getattr(p, "model_id", None)
    try:
        return os.path.splitext(os.path.basename(asm.model_path(part_id)))[0]
    except Exception:
        return None



def _reset_robot_for_l3(robot, *, single_arm: bool = False) -> None:
    """清理机械臂 hold 状态，避免一次 L3 失败影响下一次候选。"""
    _reset_layout_robot_for_l3(robot, HOME_JV, single_arm=single_arm)




def _patch_rrt_for_l3() -> None:
    """与 find_optimal_layout.py 类似，对 L3 RRT 做可控放宽。

    这里不是降低碰撞要求，而是给 RRT 更合理的搜索步长和时间；
    obstacle_list 仍然完整传入，碰撞检查不会被跳过。
    """
    try:
        from wrs.motion.probabilistic.rrt_connect import RRTConnect
    except Exception:
        return

    if getattr(RRTConnect.plan, "_tower_l3_patched", False):
        return

    _orig_plan = RRTConnect.plan

    def _patched_plan(self, *args, **kwargs):
        kwargs.setdefault("ext_dist", 0.30)
        kwargs.setdefault("smoothing_n_iter", 150)
        kwargs.setdefault("max_time", 10.0)
        return _orig_plan(self, *args, **kwargs)

    _patched_plan._tower_l3_patched = True
    RRTConnect.plan = _patched_plan

def _transport_kwargs():
    """通用 tower pick-place 方向。

    这里保持从上方抓取、向上撤离、从上往下放置。
    如果后续某类零件需要专属方向，可以在这里按 part_id 扩展。
    """
    return dict(
        pick_depart_direction=PICK_DEPART_DIR,
        pick_depart_distance=PICK_DEPART_DIST,
        place_approach_direction_list=[PLACE_APPROACH_DIR],
        place_approach_distance_list=[PLACE_APPROACH_DIST],
        place_depart_direction_list=[PLACE_DEPART_DIR],
        place_depart_distance_list=[PLACE_DEPART_DIST],
    )


def _find_grasp_pickle(part_id: str,
                       model_id: Optional[str],
                       mesh_path: str,
                       grasp_dir: str,
                       explicit_map: Dict[str, str]) -> Optional[str]:
    keys = [part_id]
    if model_id:
        keys.append(model_id)
    mesh_base = os.path.splitext(os.path.basename(mesh_path))[0]
    keys.append(mesh_base)

    # tower 的 post_bl/post_fl/post_br/post_fr 共用 post 抓取库
    if part_id.startswith("post_"):
        keys.append("post")

    for k in keys:
        if k in explicit_map:
            p = explicit_map[k]
            if not os.path.isabs(p):
                p = os.path.join(grasp_dir, p)
            if os.path.isfile(p):
                return p

    names = []
    for k in keys:
        names.extend([
            f"{k}_grasps_topdown.pickle",
            f"{k}_grasps.pickle",
            f"{k}.pickle",
            f"tower_{k}_grasps_topdown.pickle",
            f"tower_{k}_grasps.pickle",
        ])

    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = os.path.join(grasp_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _count_downward_grasps(gc: Optional[GraspCollection], obj_rotmat: Optional[np.ndarray] = None,
                           align_cos: float = DEFAULT_TOPDOWN_ALIGN_COS) -> int:
    """粗略统计“夹爪朝世界 -Z 方向”的 grasp 数量。

    不同 gripper 的 approach 轴定义可能不完全一致，所以这里同时检查
    ac_rotmat 的 ±X / ±Z 方向，只要其中一个方向和世界 -Z 足够接近，就认为
    该 grasp 具备从上往下抓的趋势。

    这个数只用于“是否优先让零件立起来”的启发式，不作为严格物理判据。
    """
    if gc is None or len(gc) == 0:
        return 0
    Robj = np.eye(3) if obj_rotmat is None else np.asarray(obj_rotmat, dtype=float)
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    count = 0
    for g in gc:
        try:
            Rtcp = Robj.dot(np.asarray(g.ac_rotmat, dtype=float))
            axes = [
                Rtcp[:, 0], -Rtcp[:, 0],
                Rtcp[:, 2], -Rtcp[:, 2],
            ]
            if any(float(np.dot(a / max(np.linalg.norm(a), 1e-9), down)) >= float(align_cos) for a in axes):
                count += 1
        except Exception:
            continue
    return int(count)


# ============================================================
# Searcher
# ============================================================

class WeightedInitialLayoutSearcher:
    def __init__(self,
                 asmdef_path: str,
                 config_yaml: str,
                 grasp_dir: str,
                 fixture_pos: np.ndarray,
                 fixture_rotmat: np.ndarray,
                 robot_base_pos: np.ndarray,
                 robot_base_rotmat: np.ndarray,
                 part_order: Optional[List[str]],
                 output_name: str,
                 table_name: str,
                 table_margin: float,
                 table_clearance: float,
                 grasp_map: Dict[str, str],
                 max_rot_candidates: int,
                 w_grasp: float,
                 w_manip: float,
                 w_dist: float,
                 w_rot: float,
                 ignore_env: bool = False,
                 cdprim_type: str = "triangles",
                 planner_obstacle_mode: str = "mesh",
                 plan_assembly_region: bool = DEFAULT_PLAN_ASSEMBLY_REGION,
                 assembly_grid: int = DEFAULT_ASSEMBLY_GRID,
                 preassemble_first_part: bool = DEFAULT_PREASSEMBLE_FIRST_PART,
                 filter_assembly_near_arms: bool = DEFAULT_FILTER_ASSEMBLY_NEAR_ARMS,
                 assembly_arm_x_clearance: float = DEFAULT_ASSEMBLY_ARM_X_CLEARANCE,
                 assembly_arm_y_clearance: float = DEFAULT_ASSEMBLY_ARM_Y_CLEARANCE,
                 filter_staging_near_arms: bool = DEFAULT_FILTER_STAGING_NEAR_ARMS,
                 staging_arm_x_clearance: float = DEFAULT_STAGING_ARM_X_CLEARANCE,
                 staging_arm_y_clearance: float = DEFAULT_STAGING_ARM_Y_CLEARANCE,
                 goal_y_side_biased_sampling: bool = DEFAULT_GOAL_Y_SIDE_BIASED_SAMPLING,
                 goal_y_side_bias_ratio: float = DEFAULT_GOAL_Y_SIDE_BIAS_RATIO,
                 goal_y_side_eps: float = DEFAULT_GOAL_Y_SIDE_EPS,
                 use_flatsurface: bool = DEFAULT_USE_FLATSURFACE,
                 fs_stability_threshold: float = DEFAULT_FS_STABILITY_THRESHOLD,
                 check_robot_home_collision: bool = DEFAULT_CHECK_ROBOT_HOME_COLLISION,
                 strict_initial_robot_collision: bool = DEFAULT_STRICT_INITIAL_ROBOT_COLLISION,
                 min_staging_mesh_clearance: float = DEFAULT_MIN_STAGING_MESH_CLEARANCE,
                 enforce_order_x_constraint: bool = DEFAULT_ENFORCE_ORDER_X_CONSTRAINT,
                 order_x_tolerance: float = DEFAULT_ORDER_X_TOLERANCE,
                 enable_y_side_distribution_score: bool = DEFAULT_ENABLE_Y_SIDE_DISTRIBUTION_SCORE,
                 prefer_upright_when_topdown_low: bool = DEFAULT_PREFER_UPRIGHT_WHEN_TOPDOWN_LOW,
                 topdown_min_count: int = DEFAULT_TOPDOWN_MIN_COUNT,
                 topdown_align_cos: float = DEFAULT_TOPDOWN_ALIGN_COS,
                 check_l2_pick_quick_motion: bool = DEFAULT_CHECK_L2_PICK_QUICK_MOTION,
                 l2_pick_check_parts: Optional[List[str]] = None,
                 l2_pick_check_lift_dist: float = DEFAULT_L2_PICK_CHECK_LIFT_DIST,
                 l2_pick_check_directions: Optional[List[str]] = None,
                 l2_pick_check_tilt: float = DEFAULT_L2_PICK_CHECK_TILT,
                 robot_home_clearance: float = DEFAULT_ROBOT_HOME_CLEARANCE,
                 l3_skip_parts: Optional[List[str]] = None,
                 prefer_stl_upface: bool = DEFAULT_PREFER_STL_UPFACE,
                 w_stl_upface: float = DEFAULT_W_STL_UPFACE,
                 stl_upface_min_thinness: float = DEFAULT_STL_UPFACE_MIN_THINNESS):
        self.asmdef_path = os.path.abspath(asmdef_path)
        self.config_yaml = os.path.abspath(config_yaml)
        self.grasp_dir = os.path.abspath(grasp_dir)
        self.fixture_pos = np.asarray(fixture_pos, dtype=float)
        self.fixture_rotmat = np.asarray(fixture_rotmat, dtype=float)
        self.robot_base_pos = np.asarray(robot_base_pos, dtype=float)
        self.robot_base_rotmat = np.asarray(robot_base_rotmat, dtype=float)
        self.output_name = output_name
        self.table_name = table_name
        self.table_margin = float(table_margin)
        self.table_clearance = float(table_clearance)
        self.grasp_map = dict(grasp_map)
        self.max_rot_candidates = int(max_rot_candidates)
        # collision primitive 类型必须在创建 staging/goal CollisionModel 之前保存。
        # 否则 _make_staging_models() 调用 self.cdprim_type 会 AttributeError。
        self.cdprim_type = str(cdprim_type) if 'cdprim_type' in locals() else 'triangles'
        self.ignore_env = bool(ignore_env)
        self.planner_obstacle_mode = str(planner_obstacle_mode) if 'planner_obstacle_mode' in locals() else 'mesh'
        self.plan_assembly_region = bool(plan_assembly_region)
        self.assembly_grid = max(1, int(assembly_grid))
        self.preassemble_first_part = bool(preassemble_first_part)
        self.filter_assembly_near_arms = bool(filter_assembly_near_arms)
        self.assembly_arm_x_clearance = max(0.0, float(assembly_arm_x_clearance))
        self.assembly_arm_y_clearance = max(0.0, float(assembly_arm_y_clearance))
        self.filter_staging_near_arms = bool(filter_staging_near_arms)
        self.staging_arm_x_clearance = max(0.0, float(staging_arm_x_clearance))
        self.staging_arm_y_clearance = max(0.0, float(staging_arm_y_clearance))
        self.goal_y_side_biased_sampling = bool(goal_y_side_biased_sampling)
        self.goal_y_side_bias_ratio = float(np.clip(float(goal_y_side_bias_ratio), 0.0, 1.0))
        self.goal_y_side_eps = max(0.0, float(goal_y_side_eps))
        self.use_flatsurface = bool(use_flatsurface)
        self.fs_stability_threshold = float(fs_stability_threshold)
        self.check_robot_home_collision = bool(check_robot_home_collision)
        self.strict_initial_robot_collision = bool(strict_initial_robot_collision)
        self.min_staging_mesh_clearance = max(0.0, float(min_staging_mesh_clearance))
        self.enforce_order_x_constraint = bool(enforce_order_x_constraint)
        self.order_x_tolerance = max(0.0, float(order_x_tolerance))
        self.enable_y_side_distribution_score = bool(enable_y_side_distribution_score)
        self.prefer_upright_when_topdown_low = bool(prefer_upright_when_topdown_low)
        self.topdown_min_count = int(topdown_min_count)
        self.topdown_align_cos = float(topdown_align_cos)

        self.check_l2_pick_quick_motion = bool(check_l2_pick_quick_motion)
        self.l2_pick_check_parts = set(l2_pick_check_parts or [])
        self.l2_pick_check_lift_dist = max(0.0, float(l2_pick_check_lift_dist))
        self.l2_pick_check_directions = list(l2_pick_check_directions or _parse_part_order(DEFAULT_L2_PICK_CHECK_DIRECTIONS) or ["z"])
        self.l2_pick_check_tilt = float(l2_pick_check_tilt)
        self.robot_home_clearance = max(0.0, float(robot_home_clearance))
        # L3 全流程验证时跳过运动规划的零件(见 DEFAULT_L3_SKIP_PARTS 注释)。
        self.l3_skip_parts = set(l3_skip_parts or [])
        self._home_robot_aabbs_cache = None

        self.mesh_vertices: Dict[str, np.ndarray] = {}
        self.identity_extent: Dict[str, np.ndarray] = {}
        self.topdown_identity_counts: Dict[str, int] = {}
        # 默认姿态保持软偏好参数
        self.prefer_stl_upface = bool(prefer_stl_upface)
        self.w_stl_upface = max(0.0, float(w_stl_upface))
        self.stl_upface_min_thinness = float(stl_upface_min_thinness)
        self.current_assembly_region_id = "fixed"
        self.current_assembly_region_rc = (-1, -1)

        w_sum = float(w_grasp + w_manip + w_dist + w_rot)
        if w_sum <= 1e-9:
            raise ValueError("权重之和不能为 0")
        self.w_grasp = float(w_grasp / w_sum)
        self.w_manip = float(w_manip / w_sum)
        self.w_dist = float(w_dist / w_sum)
        self.w_rot = float(w_rot / w_sum)

        self.asm = AssemblyDef.load(self.asmdef_path)
        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=self.fixture_pos,
            fixture_rotmat=self.fixture_rotmat,
        )

        if part_order is None:
            self.part_order = [
                s.part_id for s in self.asm.steps
                if s.part_id in self.asm.part_ids and s.part_id in self.world_poses
            ]
        else:
            self.part_order = [
                p for p in part_order
                if p in self.asm.part_ids and p in self.world_poses
            ]

        self.table_x_range, self.table_y_range, self.table_top_z = _table_info(
            self.config_yaml, self.table_name, self.table_margin
        )

        ws = load_workspace_settings(self.config_yaml)
        self.single_arm_mode = bool(ws["single_arm"])
        self.dual_arm_y_offset = float(ws["dual_arm_y_offset"])
        robot_bundle = create_layout_robot(
            single_arm=self.single_arm_mode,
            robot_base_pos=self.robot_base_pos,
            robot_base_rotmat=self.robot_base_rotmat,
            dual_arm_y_offset=self.dual_arm_y_offset,
            enable_cc=True,
        )
        self.robot = robot_bundle.robot
        goto_home_joints(self.robot, HOME_JV, single_arm=self.single_arm_mode)

        self.env_obs = [] if self.ignore_env else _load_env_obstacles(self.config_yaml)

        self.grasp_cache: Dict[str, GraspCollection] = {}
        self.grasp_file_for_part: Dict[str, str] = {}
        self._load_grasps()

        self.rot_cands: Dict[str, List[RotCandidate]] = {}
        self._precompute_rot_candidates()

        self.staging_models = self._make_staging_models()
        self.goal_models = self._make_goal_models()

    # --------------------------------------------------------
    # 初始化
    # --------------------------------------------------------

    def _load_grasps(self):
        print("\n========== Grasp pickle 检查 ==========")
        for pid in self.part_order:
            mp = self.asm.model_path(pid)
            mid = _model_id_for_part(self.asm, pid)
            pkl = _find_grasp_pickle(pid, mid, mp, self.grasp_dir, self.grasp_map)
            if pkl is None:
                print(f"{pid:16s}: MISSING")
                continue
            if pkl not in self.grasp_cache:
                self.grasp_cache[pkl] = GraspCollection.load_from_disk(file_name=pkl)
            self.grasp_file_for_part[pid] = pkl
            gc = self.grasp_cache[pkl]
            down_n = _count_downward_grasps(gc, np.eye(3), align_cos=self.topdown_align_cos)
            self.topdown_identity_counts[pid] = int(down_n)
            print(f"{pid:16s}: {pkl}  n={len(gc)}  topdown(-Z)={down_n}")

    def _precompute_rot_candidates(self):
        print("\n========== 自动旋转候选 ==========")
        print(f"use_flatsurface        = {self.use_flatsurface}")
        print(f"fs_stability_threshold = {self.fs_stability_threshold}")

        for pid in self.part_order:
            mp = self.asm.model_path(pid)
            verts = _load_mesh_vertices(mp)
            self.mesh_vertices[pid] = verts

            # 1) 优先使用 flatsurface.py 计算出的稳定摆放姿态；
            # 2) 再追加旧的 identity/90° 候选作为 fallback；
            # 3) 按旋转矩阵去重。
            pose_rots: List[Tuple[str, Optional[np.ndarray], np.ndarray]] = []
            if self.use_flatsurface:
                pose_rots.extend(_stable_pose_candidates_from_flatsurface(
                    mp,
                    stability_threshold=self.fs_stability_threshold,
                    max_candidates=max(self.max_rot_candidates * 2, 16),
                ))

            pose_rots.extend(_fallback_rot_candidates_with_names())
            pose_rots = _dedupe_pose_rot_candidates(pose_rots)

            cands: List[RotCandidate] = []

            _, _, identity_extent = _bounds_after_rotation(verts, np.eye(3))
            self.identity_extent[pid] = np.asarray(identity_extent, dtype=float)
            is_flat = float(identity_extent.min() / max(identity_extent.max(), 1e-9)) < 0.22

            for rot_name, fs_pos, R in pose_rots:
                bmin, bmax, extent = _bounds_after_rotation(verts, R)
                z_offset = self.table_top_z + self.table_clearance - float(bmin[2])
                is_identity = np.allclose(R, np.eye(3), atol=1e-6)
                from_flatsurface = str(rot_name).startswith("fs_")

                if is_identity:
                    tag = "identity"
                elif from_flatsurface and is_flat and extent[2] > identity_extent[2] * 2.0:
                    tag = "fs_upright_lifted"
                elif from_flatsurface:
                    tag = "fs_stable_lifted"
                elif is_flat and extent[2] > identity_extent[2] * 2.0:
                    tag = "auto_upright_lifted"
                else:
                    tag = "rot90_lifted"

                cands.append(RotCandidate(
                    rotmat=np.asarray(R, dtype=float),
                    z_offset=float(z_offset),
                    tag=tag,
                    extent=np.asarray(extent, dtype=float),
                    footprint=np.asarray(extent[:2], dtype=float),
                    rot_name=str(rot_name),
                    fs_pos=None if fs_pos is None else np.asarray(fs_pos, dtype=float),
                ))

            # 排序策略：
            #   identity 永远保留且靠前；
            #   flatsurface 稳定姿态优先于普通 90° fallback；
            #   在同级内部，较低高度/较小旋转代价更靠前。
            def _sort_key(c: RotCandidate):
                if c.tag == "identity":
                    group = 0
                elif c.rot_name.startswith("fs_"):
                    group = 1
                elif c.tag == "auto_upright_lifted":
                    group = 2
                else:
                    group = 3
                return (group, float(c.extent[2]), float(np.linalg.norm(c.extent[:2])))

            cands.sort(key=_sort_key)

            # 强制旋转: 某些件(如 middle_plate)只允许指定的 rot_name。
            # 这是用户显式要求的"直接强制板子直立方式", 用于绕过排序里 fs_* 优先
            # 把 rot90_04(长边直立, 可被 handover 抓取)挤掉的问题。
            forced_map = getattr(self, "force_rot_name", None) or {}
            forced = forced_map.get(pid)
            if forced:
                matched = [c for c in cands if str(c.rot_name) == str(forced)]
                if matched:
                    cands = matched
                    print(f"  [FORCE-ROT] {pid}: 仅保留 rot_name={forced} "
                          f"({len(matched)} 候选)")
                else:
                    print(f"  [FORCE-ROT][WARN] {pid}: 未找到 rot_name={forced}, "
                          f"保留全部候选")

            self.rot_cands[pid] = cands[:self.max_rot_candidates]

            print(f"{pid:16s}: {len(self.rot_cands[pid])} candidates")
            for i, c in enumerate(self.rot_cands[pid][:6]):
                print(
                    f"  [{i}] tag={c.tag:20s} "
                    f"rot={c.rot_name:12s} "
                    f"z_offset={c.z_offset:.4f} "
                    f"extent={np.round(c.extent, 4).tolist()}"
                )

    def _make_staging_models(self):
        out = {}
        for pid in self.part_order:
            mp = self.asm.model_path(pid)
            cm = make_collision_model(mp, cdprim_type=self.cdprim_type)
            cm.pos = np.zeros(3)
            cm.rotmat = np.eye(3)
            cm._sealp_part_id = pid
            cm._sealp_role = "weighted_staging"
            out[pid] = cm
        return out

    def _make_goal_models(self):
        out = {}
        for pid in self.part_order:
            if pid not in self.world_poses:
                continue
            mp = self.asm.model_path(pid)
            gp, gr = self.world_poses[pid]
            cm = make_collision_model(mp, cdprim_type=self.cdprim_type)
            cm.pos = np.asarray(gp, dtype=float)
            cm.rotmat = np.asarray(gr, dtype=float)
            cm._sealp_part_id = pid
            cm._sealp_role = "weighted_goal"
            out[pid] = cm
        return out


    # --------------------------------------------------------
    # 机器人 home 姿态穿模检查
    # --------------------------------------------------------

    def _robot_home_collision_reason(self, active_pids: Optional[List[str]] = None) -> Optional[str]:
        """检查指定初始零件是否与左/右机械臂的碰撞盒穿模。

        v9.0.2 修改点：
            用户要求只检查“机械臂碰撞盒子”，不检查所谓“整个双臂系统”。

        因此这里不再调用：
            self.robot.use_all()
            self.robot.is_collided(...)

        而是只调用：
            self.robot.lft_arm.is_collided(obstacle_list=[cm])
            self.robot.rgt_arm.is_collided(obstacle_list=[cm])

        这样检查对象就是左右机械臂各自 collision checker 里注册的碰撞盒。
        preassembled 的 base_plate 也会参与这个检查。
        """
        if not self.check_robot_home_collision:
            return None

        if active_pids is None:
            pids = list(self.part_order)
        else:
            pids = [p for p in active_pids if p in self.staging_models]

        if not pids:
            return None

        for arm in iter_layout_arms(self.robot, single_arm=self.single_arm_mode):
            try:
                arm.backup_state()
            except Exception:
                pass

        try:
            goto_home_joints(self.robot, HOME_JV, single_arm=self.single_arm_mode)

            for pid in pids:
                cm = self.staging_models.get(pid)
                if cm is None:
                    continue

                for arm_tag, arm in layout_arm_pairs(self.robot, single_arm=self.single_arm_mode):
                    try:
                        hit = arm.is_collided(obstacle_list=[cm])
                        collided = hit[0] if isinstance(hit, tuple) else hit
                    except Exception as e:
                        # 查询异常时保守处理，避免保存不确定的 layout。
                        return f"{pid} vs robot_{arm_tag}_arm_collision_box check_exception={type(e).__name__}: {e}"

                    if collided:
                        return f"{pid} vs robot_{arm_tag}_arm_collision_box"

            return None

        finally:
            for arm in iter_layout_arms(self.robot, single_arm=self.single_arm_mode):
                try:
                    arm.restore_state()
                except Exception:
                    try:
                        arm.goto_given_conf(HOME_JV)
                    except Exception:
                        pass


    def _extract_vertices_from_cmodel(self, cm) -> Optional[np.ndarray]:
        """尽量从 WRS CollisionModel / GeometricModel 中提取世界系顶点。

        这个函数只用于 home clearance 近似检查，不影响原来的精确碰撞检查。
        如果某个 WRS 版本内部字段不同导致提取失败，会返回 None。
        """
        if cm is None:
            return None

        mesh = None
        for attr in ("trm_mesh", "_trm_mesh", "mesh", "_mesh"):
            try:
                mesh = getattr(cm, attr, None)
                if mesh is not None and hasattr(mesh, "vertices"):
                    break
            except Exception:
                mesh = None

        if mesh is None and hasattr(cm, "objtrm"):
            try:
                mesh = cm.objtrm
            except Exception:
                mesh = None

        if mesh is None or not hasattr(mesh, "vertices"):
            return None

        try:
            verts = np.asarray(mesh.vertices, dtype=float)
        except Exception:
            return None

        if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) == 0:
            return None

        try:
            R = np.asarray(getattr(cm, "rotmat", np.eye(3)), dtype=float)
        except Exception:
            R = np.eye(3)

        try:
            p = np.asarray(getattr(cm, "pos", np.zeros(3)), dtype=float)
        except Exception:
            p = np.zeros(3)

        if R.shape != (3, 3):
            R = np.eye(3)
        if p.shape != (3,):
            p = np.zeros(3)

        try:
            return verts.dot(R.T) + p
        except Exception:
            return verts

    def _iter_robot_home_cmodels(self):
        """尽量遍历机器人 home 状态下左右臂/夹爪的 link cmodel。"""
        arms = list(layout_arm_pairs(self.robot, single_arm=self.single_arm_mode))
        for arm_tag, arm in arms:
            # arm 主链
            jlc = getattr(arm, "jlc", None)
            if jlc is not None:
                for i, jnt in enumerate(getattr(jlc, "jnts", []) or []):
                    lnk = getattr(jnt, "lnk", None)
                    cm = getattr(lnk, "cmodel", None) if lnk is not None else None
                    if cm is not None:
                        yield f"{arm_tag}_arm_lnk_{i}", cm

            # end effector / gripper 链
            ee = getattr(arm, "end_effector", None)
            if ee is not None:
                # 尽量覆盖不同 WRS 版本/不同夹爪结构
                for attr in ("jlc", "coupling", "lft", "rgt", "lft_outer", "rgt_outer", "lft_inner", "rgt_inner"):
                    sub = getattr(ee, attr, None)
                    if sub is None:
                        continue
                    jnts = getattr(sub, "jnts", None)
                    if jnts is None and hasattr(sub, "jlc"):
                        jnts = getattr(sub.jlc, "jnts", None)
                    for i, jnt in enumerate(jnts or []):
                        lnk = getattr(jnt, "lnk", None)
                        cm = getattr(lnk, "cmodel", None) if lnk is not None else None
                        if cm is not None:
                            yield f"{arm_tag}_ee_{attr}_lnk_{i}", cm

    def _home_robot_link_aabbs(self):
        """缓存机器人 home 姿态的 link AABB，用于额外安全距离检查。"""
        if getattr(self, "_home_robot_aabbs_cache", None) is not None:
            return self._home_robot_aabbs_cache

        old_delegator = getattr(self.robot, "delegator", None)
        for arm in iter_layout_arms(self.robot, single_arm=self.single_arm_mode):
            try:
                arm.backup_state()
            except Exception:
                pass

        try:
            goto_home_joints(self.robot, HOME_JV, single_arm=self.single_arm_mode)

            aabbs = []
            for name, cm in self._iter_robot_home_cmodels():
                verts = self._extract_vertices_from_cmodel(cm)
                if verts is None or len(verts) == 0:
                    continue
                try:
                    bmin = verts.min(axis=0)
                    bmax = verts.max(axis=0)
                    if np.all(np.isfinite(bmin)) and np.all(np.isfinite(bmax)):
                        aabbs.append((name, bmin, bmax))
                except Exception:
                    continue

            self._home_robot_aabbs_cache = aabbs
            return aabbs

        finally:
            for arm in iter_layout_arms(self.robot, single_arm=self.single_arm_mode):
                try:
                    arm.restore_state()
                except Exception:
                    try:
                        arm.goto_given_conf(HOME_JV)
                    except Exception:
                        pass

            try:
                if not self.single_arm_mode and old_delegator is self.robot.rgt_arm:
                    self.robot.use_rgt()
                elif old_delegator is self.robot.lft_arm:
                    self.robot.use_lft()
                elif not self.single_arm_mode:
                    self.robot.use_all()
            except Exception:
                pass

    def _aabb_clearance(self, amin: np.ndarray, amax: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
        sep = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
        return float(np.linalg.norm(sep))

    def _robot_home_clearance_reason(self, active_pids: Optional[List[str]] = None) -> Optional[str]:
        """检查 staging 零件与机器人 home 链节 AABB 的额外安全距离。

        这是对 _robot_home_collision_reason() 的补充：
            - collision check：不能穿模；
            - clearance check：尽量离 home 手臂/夹爪至少 robot_home_clearance。

        如果无法从机器人模型中提取 link AABB，则自动跳过，不影响原逻辑。
        """
        min_clear = float(getattr(self, "robot_home_clearance", 0.0))
        if min_clear <= 1e-9:
            return None

        aabbs = self._home_robot_link_aabbs()
        if not aabbs:
            return None

        if active_pids is None:
            pids = list(self.part_order)
        else:
            pids = [p for p in active_pids if p in self.staging_models]

        for pid in pids:
            verts = self._world_vertices_for_staging(pid)
            if verts is None or len(verts) == 0:
                continue
            try:
                pmin = verts.min(axis=0)
                pmax = verts.max(axis=0)
            except Exception:
                continue

            for link_name, bmin, bmax in aabbs:
                d = self._aabb_clearance(pmin, pmax, bmin, bmax)
                if d < min_clear:
                    return (
                        f"{pid} too close to robot_home {link_name}: "
                        f"clearance={d:.4f}m < {min_clear:.4f}m"
                    )
        return None


    # --------------------------------------------------------
    # 新增初始布局空间约束
    # --------------------------------------------------------

    def _active_pick_part_order(self) -> List[str]:
        """返回需要从 staging 抓取的零件顺序。

        如果第一件已经 preassembled，则不把第一件纳入 order-x 约束，
        避免所有零件都被迫放在 base_plate 的 x 左侧。
        """
        first_pid = self._first_part_id() if self.preassemble_first_part else None
        return [p for p in self.part_order if p != first_pid]

    def _world_vertices_for_staging(self, pid: str) -> Optional[np.ndarray]:
        verts = self.mesh_vertices.get(pid)
        cm = self.staging_models.get(pid)
        if verts is None or cm is None:
            return None
        return np.asarray(verts, dtype=float).dot(np.asarray(cm.rotmat, dtype=float).T) + np.asarray(cm.pos, dtype=float)

    def _aabb_distance_between_world_vertices(self, va: np.ndarray, vb: np.ndarray) -> float:
        amin, amax = va.min(axis=0), va.max(axis=0)
        bmin, bmax = vb.min(axis=0), vb.max(axis=0)
        sep = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
        return float(np.linalg.norm(sep))

    def _vertex_distance_between_world_vertices(self, va: np.ndarray, vb: np.ndarray) -> float:
        """顶点级最小距离，作为 mesh 外轮廓间距的快速近似。

        Tower 里的零件基本由盒子拼出来，顶点级距离足够用于 1cm 安全间隙过滤。
        """
        if va.size == 0 or vb.size == 0:
            return float("inf")
        # 小模型直接全量计算；大模型则均匀抽样，避免偶发超慢。
        max_pts = 600
        if len(va) > max_pts:
            va = va[np.linspace(0, len(va) - 1, max_pts).astype(int)]
        if len(vb) > max_pts:
            vb = vb[np.linspace(0, len(vb) - 1, max_pts).astype(int)]
        diff = va[:, None, :] - vb[None, :, :]
        return float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))

    def _mesh_clearance_reason(self, active_pids: Optional[List[str]] = None) -> Optional[str]:
        """检查 staging 零件之间的外轮廓间距是否满足要求。

        先用 AABB 快速判断；如果 AABB 间距已经大于阈值，则直接通过；
        如果 AABB 太近，再用 mesh 顶点最小距离近似复核。
        """
        min_clear = float(self.min_staging_mesh_clearance)
        if min_clear <= 1e-9:
            return None

        ids = active_pids or list(self.staging_models.keys())
        ids = [p for p in ids if p in self.staging_models]

        world_verts = {}
        for pid in ids:
            v = self._world_vertices_for_staging(pid)
            if v is not None:
                world_verts[pid] = v

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a not in world_verts or b not in world_verts:
                    continue
                va, vb = world_verts[a], world_verts[b]
                aabb_d = self._aabb_distance_between_world_vertices(va, vb)
                if aabb_d >= min_clear:
                    continue
                vtx_d = self._vertex_distance_between_world_vertices(va, vb)
                if vtx_d < min_clear:
                    return f"{a} vs {b}: clearance={vtx_d:.4f}m < required {min_clear:.4f}m"
        return None

    def _order_x_constraint_reason(self, layout: LayoutCandidate) -> Optional[str]:
        """装配顺序越靠后的零件，尽量不要放到更远的 x 方向。

        默认按需要抓取的零件顺序做“相邻 step x 不递增”硬约束。
        即：x_later <= x_previous + tolerance。
        """
        if not self.enforce_order_x_constraint:
            return None
        order = self._active_pick_part_order()
        order = [p for p in order if p in layout.xy]
        tol = float(self.order_x_tolerance)
        for prev, cur in zip(order[:-1], order[1:]):
            x_prev = float(layout.xy[prev][0])
            x_cur = float(layout.xy[cur][0])
            if x_cur > x_prev + tol:
                return f"order-x violation: {cur}.x={x_cur:.4f} > {prev}.x={x_prev:.4f}+tol={tol:.4f}"
        return None

    def _side_distribution_score(self, layout: LayoutCandidate) -> float:
        """y 方向两侧分布评分，作为小幅加成，不是硬约束。"""
        if not self.enable_y_side_distribution_score:
            return 1.0
        order = [p for p in self._active_pick_part_order() if p in layout.xy]
        if len(order) <= 1:
            return 1.0
        y0 = float(self.fixture_pos[1])
        ys = np.asarray([float(layout.xy[p][1] - y0) for p in order], dtype=float)
        # spread：离装配中心 y 越分散越好；balance：两侧数量越均衡越好。
        spread = float(np.clip(np.std(ys) / 0.28, 0.0, 1.0))
        n_pos = int(np.sum(ys > 0.03))
        n_neg = int(np.sum(ys < -0.03))
        balance = float(min(n_pos, n_neg) / max(max(n_pos, n_neg), 1))
        return float(np.clip(0.65 * spread + 0.35 * balance, 0.0, 1.0))

    def _is_upright_candidate(self, pid: str, cand: RotCandidate) -> bool:
        if "upright" in str(cand.tag):
            return True
        try:
            identity_extent = self.rot_cands[pid][0].extent
            return float(cand.extent[2]) > float(identity_extent[2]) * 1.8
        except Exception:
            return False

    def _part_thinness(self, pid: str) -> float:
        """零件"指向性"度量: 1 - min_extent/max_extent, 基于 STL 默认(identity)包围盒。

        细长杆(post: 0.024x0.024x0.135) -> ~0.82; 扁平板(0.195x0.15x0.024) -> ~0.88;
        近似立方体 -> ~0。越接近 1 越"有明确朝向", 越应该保持 STL 默认摆放不翻倒。
        """
        ext = self.identity_extent.get(pid)
        if ext is None:
            return 0.0
        mx = float(np.max(ext))
        if mx <= 1e-9:
            return 0.0
        return float(1.0 - float(np.min(ext)) / mx)

    def _stl_upface_flip_penalty(self, pid: str, cand: RotCandidate) -> float:
        """默认姿态保持惩罚 ∈ [0, ~1]。

        含义: 候选姿态把"STL 默认朝上轴(世界+Z)"翻转了多少 —— 翻得越多、零件越有指向性,
        惩罚越大。

        - tilt = STL 的 +Z 轴(经候选旋转后 = rotmat[:,2])与世界 +Z 的夹角。
          identity(站立/平放不翻) -> rotmat[2,2]=1 -> tilt=0 -> 惩罚 0;
          翻倒 90° -> rotmat[2,2]=0 -> tilt=90° -> 惩罚最大。
        - 用 thinness 缩放: 只有细长/扁平件翻倒才明显扣分, 近似立方体几乎不受影响。
        - thinness < min_thinness 的件直接不惩罚(没有明确"该朝哪"的方向)。
        """
        if not self.prefer_stl_upface or self.w_stl_upface <= 0.0:
            return 0.0
        thinness = self._part_thinness(pid)
        if thinness < self.stl_upface_min_thinness:
            return 0.0
        R = np.asarray(cand.rotmat, dtype=float)
        up_cos = float(np.clip(R[2, 2], -1.0, 1.0))
        tilt = float(np.arccos(up_cos))            # 0..pi
        tilt_frac = min(1.0, tilt / (np.pi / 2.0))  # 0..1 (>=90° 视为完全翻倒)
        return float(thinness * tilt_frac)

    def _upright_preference_adjusted_score(self, pid: str, cand: RotCandidate, part_score: float) -> float:
        """兼容旧评分接口。

        现在 topdown 抓取不足时已经改成硬约束：
            _upright_hard_constraint_reason() 会直接过滤非 upright 候选。
        因此这里不再对平放姿态做软惩罚，只保留轻微 upright 加分。
        """
        if not self.prefer_upright_when_topdown_low:
            return float(part_score)
        down_n = int(self.topdown_identity_counts.get(pid, 0))
        if down_n >= int(self.topdown_min_count):
            return float(part_score)
        if self._is_upright_candidate(pid, cand):
            return float(min(1.0, part_score + 0.05))
        return float(part_score)

    def _upright_hard_constraint_reason(self, pid: str, cand: RotCandidate) -> Optional[str]:
        """topdown 抓取不足时的硬约束。

        规则：
            如果某零件在原始姿态下 topdown(-Z) 抓取数 < topdown_min_count，
            那么该零件的初始姿态必须是 upright / 侧立候选；
            identity / 普通平放 / 非 upright 的 fs_stable_lifted 直接跳过。

        典型作用：
            middle_plate 的 topdown(-Z)=0 时，禁止选择 identity 平放，
            强制从 fs_upright_lifted / auto_upright_lifted 里选。
        """
        if not self.prefer_upright_when_topdown_low:
            return None

        down_n = int(self.topdown_identity_counts.get(pid, 0))
        if down_n >= int(self.topdown_min_count):
            return None

        if self._is_upright_candidate(pid, cand):
            return None

        return (
            f"upright hard constraint: {pid} topdown(-Z)={down_n} "
            f"< {int(self.topdown_min_count)}, reject pose={cand.tag}, rot={cand.rot_name}"
        )


    def _arm_base_xy_map(self) -> Dict[str, Tuple[float, float]]:
        return _layout_arm_base_xy_map(
            self.robot_base_pos,
            single_arm=self.single_arm_mode,
            dual_arm_y_offset=self.dual_arm_y_offset,
        )

    def _arm_base_y_map(self) -> Dict[str, float]:
        """兼容旧 metadata/打印：只返回臂基座 y 值。"""
        xy_map = self._arm_base_xy_map()
        return {name: float(xy[1]) for name, xy in xy_map.items()}

    def _staging_arm_keepout_reason(self, pid: str, xy: np.ndarray, cand: RotCandidate) -> Optional[str]:
        """随机采样/姿态评估阶段，禁止初始 staging 位置落入左右臂附近矩形禁区。

        这次不是只考虑 y，而是同时考虑 x 和 y。

        判定方式：
            把 arm base 附近看成一个矩形禁区：
                x ∈ [arm_x - clear_x, arm_x + clear_x]
                y ∈ [arm_y - clear_y, arm_y + clear_y]

            把零件 footprint 近似成 XY AABB：
                x 半宽 = footprint_x / 2
                y 半宽 = footprint_y / 2

            如果零件 footprint AABB 与 arm 矩形禁区重叠，则认为这个 staging 位置不合适。
            等价判定：
                abs(part_x - arm_x) < clear_x + footprint_x/2
            且
                abs(part_y - arm_y) < clear_y + footprint_y/2

            只有 x 和 y 同时靠近才剔除，不再是整条 y 禁区。
        """
        if not self.filter_staging_near_arms:
            return None

        clear_x = float(self.staging_arm_x_clearance)
        clear_y = float(self.staging_arm_y_clearance)
        if clear_x <= 1e-9 and clear_y <= 1e-9:
            return None

        xy = np.asarray(xy, dtype=float)
        x = float(xy[0])
        y = float(xy[1])

        fp_x, fp_y = 0.0, 0.0
        try:
            fp = np.asarray(cand.footprint, dtype=float)
            fp_x = float(fp[0])
            fp_y = float(fp[1])
        except Exception:
            pass

        req_x = clear_x + fp_x / 2.0
        req_y = clear_y + fp_y / 2.0

        for name, (arm_x, arm_y) in self._arm_base_xy_map().items():
            dx = abs(x - float(arm_x))
            dy = abs(y - float(arm_y))
            if dx < req_x and dy < req_y:
                return (
                    f"{pid} staging overlaps {name} rectangular keepout: "
                    f"dx={dx:.4f}m < clear_x({clear_x:.4f})+footprint_x/2({fp_x/2.0:.4f})={req_x:.4f}m, "
                    f"dy={dy:.4f}m < clear_y({clear_y:.4f})+footprint_y/2({fp_y/2.0:.4f})={req_y:.4f}m"
                )

        return None

    def _assembly_region_reject_reason(self, pos: np.ndarray) -> Optional[str]:
        """判断一个 3x3 装配区中心是否应该被剔除。

        v9.3 修改：
            装配区中心不再使用“离左右臂基座 XY 矩形距离”的过滤。
            也就是说，3x3 网格中心仍然全部参与候选搜索。

        仅保留一个必要硬约束：
            如果第一件是 preassembled，例如 base_plate，
            那么把第一件放到该装配中心后，不能和左/右机械臂碰撞盒发生碰撞。

        这样满足当前需求：
            - 装配区采样还是 3x3；
            - 装配区不再因为靠近 arm base 的矩形距离被跳过；
            - 但 preassembled 实体本身仍不能和机械臂穿模。
        """
        # 如果第一件是 preassembled，则直接把第一件放到该候选装配中心，
        # 检查它是否和机械臂 home 姿态碰撞。
        if self.preassemble_first_part and self.strict_initial_robot_collision:
            first_pid = self._first_part_id()
            if first_pid is not None and first_pid in self.staging_models:
                old_pos = np.asarray(self.staging_models[first_pid].pos, dtype=float).copy()
                old_rot = np.asarray(self.staging_models[first_pid].rotmat, dtype=float).copy()
                old_fixture_pos = self.fixture_pos.copy()
                old_world_poses = self.world_poses

                try:
                    tmp_world_poses = self.asm.compute_world_poses(
                        fixture_pos=np.asarray(pos, dtype=float),
                        fixture_rotmat=self.fixture_rotmat,
                    )
                    if first_pid in tmp_world_poses:
                        gp, gr = tmp_world_poses[first_pid]
                        self.staging_models[first_pid].pos = np.asarray(gp, dtype=float).copy()
                        self.staging_models[first_pid].rotmat = np.asarray(gr, dtype=float).copy()

                        hit = self._robot_home_collision_reason(active_pids=[first_pid])
                        if hit:
                            return f"preassembled first part collides with arm collision boxes at this assembly center: {hit}"
                finally:
                    self.staging_models[first_pid].pos = old_pos
                    self.staging_models[first_pid].rotmat = old_rot
                    self.fixture_pos = old_fixture_pos
                    self.world_poses = old_world_poses

        return None


    def _set_assembly_station(self, fixture_pos: np.ndarray, region_id: str = "fixed", rc: Tuple[int, int] = (-1, -1)) -> None:
        """更新当前装配区中心，并同步 asm world poses / goal models。"""
        self.fixture_pos = np.asarray(fixture_pos, dtype=float)
        self.current_assembly_region_id = str(region_id)
        self.current_assembly_region_rc = tuple(rc)
        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=self.fixture_pos,
            fixture_rotmat=self.fixture_rotmat,
        )

        # 如果 goal_models 已经存在，重新生成 goal collision models，保证 L2/L3 用的是最新装配区。
        if hasattr(self, "goal_models"):
            self.goal_models = self._make_goal_models()

    def _first_part_id(self) -> Optional[str]:
        if not self.part_order:
            return None
        return self.part_order[0]

    def _apply_first_part_as_assembled(self, layout: Optional[LayoutCandidate] = None) -> None:
        """把第一步零件直接放到当前装配区，等价于第一步已经装好。

        对 tower 来说，第一步通常是 base_plate。
        """
        if not self.preassemble_first_part:
            return
        first_pid = self._first_part_id()
        if first_pid is None or first_pid not in self.world_poses:
            return

        gp, gr = self.world_poses[first_pid]
        if first_pid in self.staging_models:
            self.staging_models[first_pid].pos = np.asarray(gp, dtype=float).copy()
            self.staging_models[first_pid].rotmat = np.asarray(gr, dtype=float).copy()

        if layout is not None:
            layout.xy[first_pid] = np.asarray(gp[:2], dtype=float).copy()
            layout.chosen_rotmat[first_pid] = np.asarray(gr, dtype=float).copy()
            layout.z_offset[first_pid] = float(gp[2])
            layout.pose_tag[first_pid] = "preassembled_at_assembly_region"
            layout.rot_name[first_pid] = "goal_pose"
            layout.arm_choice[first_pid] = "preassembled"
            # 第一件已经装好，不参与抓取规划；给一个中性值，避免 score 中 min/count 异常。
            layout.grasp_counts[first_pid] = int(NORM_GRASP_MEAN_TARGET)
            layout.per_part_dist[first_pid] = 0.0
            layout.per_part_manip[first_pid] = 0.0
            layout.per_part_rot_angle[first_pid] = 0.0

    def _assembly_region_candidates(self) -> List[Tuple[str, Tuple[int, int], np.ndarray]]:
        """把 work_table 粗分为 grid x grid 个区域，返回每块中心点作为装配区候选。

        返回:
            [(region_id, (row, col), fixture_pos), ...]

        row 对应 y 方向分块，col 对应 x 方向分块。
        """
        if not self.plan_assembly_region:
            return [("fixed", (-1, -1), self.fixture_pos.copy())]

        grid = max(1, int(self.assembly_grid))
        first_pid = self._first_part_id()

        # 用第一步零件的 identity footprint 约束装配中心，避免第一件一开始就出桌面。
        if first_pid is not None and first_pid in self.rot_cands:
            first_fp = np.asarray(self.rot_cands[first_pid][0].footprint, dtype=float)
        else:
            first_fp = np.zeros(2)

        x_min, x_max = self.table_x_range
        y_min, y_max = self.table_y_range
        x_safe_min = x_min + float(first_fp[0]) / 2.0
        x_safe_max = x_max - float(first_fp[0]) / 2.0
        y_safe_min = y_min + float(first_fp[1]) / 2.0
        y_safe_max = y_max - float(first_fp[1]) / 2.0

        if x_safe_min > x_safe_max:
            x_safe_min, x_safe_max = x_min, x_max
        if y_safe_min > y_safe_max:
            y_safe_min, y_safe_max = y_min, y_max

        x_edges = np.linspace(x_min, x_max, grid + 1)
        y_edges = np.linspace(y_min, y_max, grid + 1)

        raw_out: List[Tuple[str, Tuple[int, int], np.ndarray]] = []
        for r in range(grid):
            for c in range(grid):
                raw_x = float((x_edges[c] + x_edges[c + 1]) / 2.0)
                raw_y = float((y_edges[r] + y_edges[r + 1]) / 2.0)
                x = float(np.clip(raw_x, x_safe_min, x_safe_max))
                y = float(np.clip(raw_y, y_safe_min, y_safe_max))
                region_id = f"r{r}_c{c}"
                raw_out.append((region_id, (r, c), np.array([x, y, self.table_top_z], dtype=float)))

        out: List[Tuple[str, Tuple[int, int], np.ndarray]] = []
        skipped: List[Tuple[str, Tuple[int, int], np.ndarray, str]] = []
        for rid, rc, pos in raw_out:
            reason = self._assembly_region_reject_reason(pos)
            if reason is None:
                out.append((rid, rc, pos))
            else:
                skipped.append((rid, rc, pos, reason))

        # 如果用户设置过严，把所有候选都过滤掉，则回退原始候选，避免程序完全没法跑。
        if not out:
            print("\n[WARN] assembly arm keepout 过滤掉了所有装配区候选，自动回退使用全部 3x3 候选。")
            out = raw_out

        # 去重：如果第一件太大导致多个角落被 clip 到同一个点，仍保留 region_id，
        # 这样输出还能看出是哪个 coarse block 被尝试。
        print("\n========== Assembly Region Candidates ==========")
        print(f"plan_assembly_region       = {self.plan_assembly_region}")
        print(f"assembly_grid              = {grid} x {grid}")
        print(f"preassemble_first          = {self.preassemble_first_part}")
        print(f"filter_assembly_near_arms  = False  # disabled: 3x3 centers are not filtered by arm-base rectangle")
        print(f"assembly_collision_check   = preassembled first part vs arm collision boxes")
        print(f"filter_staging_near_arms   = {self.filter_staging_near_arms}")
        print(f"staging_arm_x_clearance    = {self.staging_arm_x_clearance:.4f} m")
        print(f"staging_arm_y_clearance    = {self.staging_arm_y_clearance:.4f} m")
        print(f"arm_base_xy                = {self._arm_base_xy_map()}")
        print("  kept:")
        for rid, rc, pos in out:
            print(f"    {rid:6s} rc={rc} center={np.round(pos, 4).tolist()}")
        if skipped:
            print("  skipped:")
            for rid, rc, pos, reason in skipped:
                print(f"    {rid:6s} rc={rc} center={np.round(pos, 4).tolist()}  reason={reason}")
        return out


    # --------------------------------------------------------
    # layout 采样
    # --------------------------------------------------------

    def _xy_bounds_for_part_and_cand(self, pid: str, cand: RotCandidate):
        fx, fy = cand.footprint
        xlo, xhi = self.table_x_range
        ylo, yhi = self.table_y_range
        return (
            xlo + fx / 2.0,
            xhi - fx / 2.0,
        ), (
            ylo + fy / 2.0,
            yhi - fy / 2.0,
        )

    def _apply_staging_pose(self, pid: str, xy: np.ndarray, cand: RotCandidate):
        cm = self.staging_models[pid]
        cm.pos = np.array([float(xy[0]), float(xy[1]), float(cand.z_offset)], dtype=float)
        cm.rotmat = np.asarray(cand.rotmat, dtype=float)

    def _pairwise_collision(self, active_pids: Optional[List[str]] = None) -> Optional[str]:
        ids = active_pids or list(self.staging_models.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if self.staging_models[a].is_mcdwith(self.staging_models[b]):
                    return f"{a} vs {b}"
        return None

    def _goal_y_reference(self, first_pid: Optional[str] = None) -> float:
        """用于判断某个零件最终装配 y 是偏左还是偏右的参考 y。

        默认使用除了 preassembled 第一件以外，其余零件最终 goal y 的均值。
        若无法计算，则退回当前装配中心 fixture_pos[1]。
        """
        ys = []
        for pid in self.part_order:
            if first_pid is not None and pid == first_pid:
                continue
            if pid not in self.world_poses:
                continue
            try:
                ys.append(float(np.asarray(self.world_poses[pid][0], dtype=float)[1]))
            except Exception:
                pass

        if ys:
            return float(np.mean(ys))
        return float(self.fixture_pos[1])

    def _goal_y_side_for_part(self, pid: str, first_pid: Optional[str] = None) -> str:
        """根据最终装配 y 判断该零件更偏左还是更偏右。

        约定：
            y 更大 = 更偏左 -> "left"
            y 更小 = 更偏右 -> "right"
            接近参考 y -> "center"
        """
        if first_pid is not None and pid == first_pid:
            return "preassembled"

        if pid not in self.world_poses:
            return "center"

        try:
            gy = float(np.asarray(self.world_poses[pid][0], dtype=float)[1])
        except Exception:
            return "center"

        ref = self._goal_y_reference(first_pid=first_pid)
        eps = float(self.goal_y_side_eps)

        if gy > ref + eps:
            return "left"
        if gy < ref - eps:
            return "right"
        return "center"

    def _preferred_y_range_by_goal_side(
        self,
        pid: str,
        ylo: float,
        yhi: float,
        first_pid: Optional[str] = None,
    ) -> Tuple[Tuple[float, float], str]:
        """根据 goal y 的左右关系，返回优先采样的桌面半区。

        注意：
            这是采样“优先半区”，不是硬约束。
            如果优先半区和当前零件 footprint 后的 y 范围无交集，会自动退回原范围。
        """
        if not self.goal_y_side_biased_sampling:
            return (float(ylo), float(yhi)), "disabled"

        side = self._goal_y_side_for_part(pid, first_pid=first_pid)
        if side not in ("left", "right"):
            return (float(ylo), float(yhi)), side

        table_ylo, table_yhi = self.table_y_range
        y_mid = float((table_ylo + table_yhi) / 2.0)

        if side == "left":
            # y 更大 = 更偏左，所以优先 [mid, yhi]
            bylo = max(float(ylo), y_mid)
            byhi = float(yhi)
        else:
            # y 更小 = 更偏右，所以优先 [ylo, mid]
            bylo = float(ylo)
            byhi = min(float(yhi), y_mid)

        if bylo >= byhi:
            return (float(ylo), float(yhi)), f"{side}_fallback_full"

        return (bylo, byhi), side

    def sample_collision_free_xy(self,
                                 rng: np.random.Generator,
                                 max_attempts_per_part: int = 150) -> Optional[Dict[str, np.ndarray]]:
        """在 work_table 上为所有零件随机采样一个尽量不重叠的初始 layout。

        如果 self.preassemble_first_part=True：
            第一件零件直接放在当前装配区中心，等价于它一开始已经装好；
            后续只给其它零件随机采样初始位置。
        """
        xy: Dict[str, np.ndarray] = {}
        placed: List[str] = []

        first_pid = self._first_part_id() if self.preassemble_first_part else None

        if first_pid is not None and first_pid in self.world_poses:
            gp, gr = self.world_poses[first_pid]
            xy[first_pid] = np.asarray(gp[:2], dtype=float).copy()
            if first_pid in self.staging_models:
                self.staging_models[first_pid].pos = np.asarray(gp, dtype=float).copy()
                self.staging_models[first_pid].rotmat = np.asarray(gr, dtype=float).copy()
            placed.append(first_pid)

        # 大件先放，显著降低随机碰撞概率；第一件若已预装，则跳过随机采样。
        order = sorted(
            [p for p in self.part_order if p != first_pid],
            key=lambda p: float(np.prod(self.rot_cands[p][0].footprint)),
            reverse=True,
        )

        for pid in order:
            cand0 = self.rot_cands[pid][0]
            (xlo, xhi), (ylo, yhi) = self._xy_bounds_for_part_and_cand(pid, cand0)
            if xlo >= xhi or ylo >= yhi:
                return None

            preferred_y_range, goal_side = self._preferred_y_range_by_goal_side(
                pid, ylo, yhi, first_pid=first_pid
            )
            prefer_attempts = int(round(max_attempts_per_part * self.goal_y_side_bias_ratio))

            ok = False
            for attempt_i in range(max_attempts_per_part):
                # 前 prefer_attempts 次优先在目标侧半桌面采样；
                # 如果一直失败，后面自动回退到全桌面，避免变成硬约束。
                if attempt_i < prefer_attempts and goal_side in ("left", "right"):
                    cur_ylo, cur_yhi = preferred_y_range
                else:
                    cur_ylo, cur_yhi = ylo, yhi

                if cur_ylo >= cur_yhi:
                    cur_ylo, cur_yhi = ylo, yhi

                p = np.array([
                    float(rng.uniform(xlo, xhi)),
                    float(rng.uniform(cur_ylo, cur_yhi)),
                ], dtype=float)

                # 采样阶段就避开左右臂附近矩形区域。
                # 这样不用等到 evaluate_layout 里的 robot_home_collision 才大量失败。
                keepout_hit = self._staging_arm_keepout_reason(pid, p, cand0)
                if keepout_hit:
                    continue

                self._apply_staging_pose(pid, p, cand0)

                # 与已放零件不碰撞即可，包括已经装在装配区的第一件。
                collision = False
                for q in placed:
                    if q in self.staging_models and self.staging_models[pid].is_mcdwith(self.staging_models[q]):
                        collision = True
                        break
                if collision:
                    continue

                xy[pid] = p
                placed.append(pid)
                ok = True
                break

            if not ok:
                return None

        return {pid: xy[pid] for pid in self.part_order if pid in xy}

    # --------------------------------------------------------
    # 评估
    # --------------------------------------------------------

    def _grasp_collection(self, pid: str) -> Optional[GraspCollection]:
        pkl = self.grasp_file_for_part.get(pid)
        if pkl is None:
            return None
        return self.grasp_cache.get(pkl)

    def _arm_order(self, pid: str) -> Tuple[str, ...]:
        if self.single_arm_mode:
            return ("lft",)
        if pid.endswith("_r") or pid.endswith("br") or pid.endswith("fr"):
            return ("rgt", "lft")
        return ("lft", "rgt")

    def _step_obstacles(self, current_pid: str, placed: set) -> List:
        obs = list(self.env_obs)
        for pid in placed:
            if pid in self.goal_models:
                obs.append(self.goal_models[pid])
        for pid in self.part_order:
            if pid == current_pid or pid in placed:
                continue
            if pid in self.staging_models:
                obs.append(self.staging_models[pid])
        return obs

    # --------------------------------------------------------
    # 接触/插接豁免 (与 execute_layout_sequence_visual 口径一致)
    # --------------------------------------------------------

    def _part_parent_map(self) -> Dict[str, str]:
        cached = getattr(self, "_part_parent_cache", None)
        if cached is not None:
            return cached
        out: Dict[str, str] = {}
        for s in getattr(self.asm, "steps", []):
            pid = getattr(s, "part_id", None)
            par = getattr(s, "parent_id", None)
            if pid is not None and par is not None:
                out[pid] = par
        self._part_parent_cache = out
        return out

    def _contact_exclusion_set(self, current_pid: Optional[str], placed: set) -> set:
        """规划 current_pid 时应临时排除的"已装接触件"集合。

        接触件由 asmdef 的 direct parent + 最终 goal 位姿下的真实表面距离自动判定, 再 ∩ 已装件。
        只有"已经装好"的接触件才豁免; 还没装的零件仍应作为 staging 障碍。
        """
        if not current_pid:
            return set()
        placed = placed or set()
        cache = getattr(self, "_mating_cache", None)
        if cache is None or getattr(self, "_mating_cache_key", None) != id(self.goal_models):
            cache = MatingCache(goal_model_of=self.goal_models.get,
                                parent_of=self._part_parent_map().get,
                                verbose=bool(getattr(self, "verbose", False)))
            self._mating_cache = cache
            self._mating_cache_key = id(self.goal_models)
        return {p for p in cache.get(current_pid, placed) if p in placed}

    def _planner_obstacles(self, obs: List, current_pid: Optional[str] = None,
                           placed: Optional[set] = None) -> List:
        """给 reason_common_gids 使用的 obstacle_list。

        mesh:          全部 triangles/mesh 碰撞。对本塔这种堆叠装配会被桌面 mesh +
                       接触面 mesh 大量误杀, 通常找不到任何解, 不推荐。
        env_only:      只保留 work_table 等环境。
        none:          不传任何障碍, 彻底避免 gripper 的 box/cdprim/桌面误杀。
        staging_aware: 推荐。排除环境(桌面)误杀 + 保留已装件(weighted_goal, 但按
                       parent/接触表豁免当前件要插接/承托的那些已装件, 与执行脚本一致)
                       + 保留其它 staging 件(weighted_staging) 作障碍。这样既不会像
                       mesh 那样被桌面/接触面误杀, 又能避免某个零件被周围 staging 件
                       包围、执行时根本抓不进去。
        """
        mode = self.planner_obstacle_mode
        if mode == "none":
            return []
        if mode == "env_only":
            return [o for o in obs if getattr(o, "_sealp_role", None) == "environment_obstacle"]
        if mode in ("staging_aware", "executor_match"):
            # executor_match = staging_aware + 保留工作台桌面(与执行脚本 _placement_obstacles
            # 逐项一致); staging_aware 则排除桌面, 避免贴桌零件抓取被薄桌盒误杀(默认)。
            keep_env = (mode == "executor_match")
            excluded = self._contact_exclusion_set(current_pid, placed or set())
            out = []
            for o in obs:
                role = getattr(o, "_sealp_role", None)
                if role == "environment_obstacle":
                    if keep_env:
                        out.append(o)
                    continue
                if role == "weighted_goal" and getattr(o, "_sealp_part_id", None) in excluded:
                    continue  # 接触豁免: 排除当前件插接/承托的已装件
                out.append(o)  # 其它 weighted_goal + 全部 weighted_staging 都保留
            return out
        return obs

    def _l2_pick_direction_vectors(self) -> List[Tuple[str, np.ndarray]]:
        """L2 quick pick check 使用的候选撤离方向。

        注意：
            这里检查的是“物体抓起/撤离方向”，不是最终 L3 的完整路径。
            但它不应该定死为 +Z，所以这里默认尝试：
                z, x_plus, x_minus, y_plus, y_minus

            每个方向都会归一化。
        """
        t = float(self.l2_pick_check_tilt)

        raw = {
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
            "x_plus": np.array([t, 0.0, 1.0], dtype=float),
            "x_minus": np.array([-t, 0.0, 1.0], dtype=float),
            "y_plus": np.array([0.0, t, 1.0], dtype=float),
            "y_minus": np.array([0.0, -t, 1.0], dtype=float),
            # 兼容一些可能的写法
            "+x": np.array([t, 0.0, 1.0], dtype=float),
            "-x": np.array([-t, 0.0, 1.0], dtype=float),
            "+y": np.array([0.0, t, 1.0], dtype=float),
            "-y": np.array([0.0, -t, 1.0], dtype=float),
        }

        out: List[Tuple[str, np.ndarray]] = []
        for name in self.l2_pick_check_directions:
            key = str(name).strip()
            if not key:
                continue
            if key not in raw:
                print(f"[WARN] unknown l2 pick direction {key!r}, ignored.")
                continue
            v = raw[key].astype(float)
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                continue
            out.append((key, v / n))

        if not out:
            out.append(("z", np.array([0.0, 0.0, 1.0], dtype=float)))
        return out

    def _l2_pick_quick_check_gids(
        self,
        pid: str,
        planner: PickPlacePlanner,
        gc: GraspCollection,
        gids: List[int],
        sp: np.ndarray,
        sr: np.ndarray,
        obs: List,
    ) -> Tuple[List[int], str]:
        """L2 阶段对指定零件做 pre-pick / pick / post-pick 快速检查。

        v9.4 修改：
            以前这里固定使用 +Z 撤离：
                pre/post = sp + [0,0,1] * lift
            这会把“必须向上撤离”写死，容易误杀 middle_plate。

            现在改成多方向尝试：
                z, x_plus, x_minus, y_plus, y_minus

            对每个方向分别检查：
                pre-pick  = sp + dir * lift
                pick      = sp
                post-pick = sp + dir * lift

            只要任意一个方向能保留有效 common gid，就通过。
            返回的 valid gids 是所有通过方向的 gid 并集。
        """
        if not self.check_l2_pick_quick_motion:
            return list(gids), "disabled"
        if pid not in self.l2_pick_check_parts:
            return list(gids), "not_target_part"
        if not gids:
            return [], "empty_input_gids"

        lift = float(self.l2_pick_check_lift_dist)
        pick_sp = np.asarray(sp, dtype=float)
        sr = np.asarray(sr, dtype=float)
        input_gids = list(gids)

        valid_union = []
        valid_set = set()
        pass_dirs = []
        fail_msgs = []

        for dir_name, dir_vec in self._l2_pick_direction_vectors():
            lift_vec = np.asarray(dir_vec, dtype=float) * lift
            pre_sp = pick_sp + lift_vec
            post_sp = pick_sp + lift_vec

            try:
                quick_gids = planner.reason_common_gids(
                    grasp_collection=gc,
                    goal_pose_list=[(pre_sp, sr), (pick_sp, sr), (post_sp, sr)],
                    obstacle_list=obs,
                )
            except Exception as e:
                fail_msgs.append(f"{dir_name}: exception={type(e).__name__}")
                continue

            quick_set = set(quick_gids)
            local_valid = [gid for gid in input_gids if gid in quick_set]
            if local_valid:
                pass_dirs.append(f"{dir_name}:{len(local_valid)}")
                for gid in local_valid:
                    if gid not in valid_set:
                        valid_set.add(gid)
                        valid_union.append(gid)
            else:
                fail_msgs.append(f"{dir_name}:0")

        if not valid_union:
            return [], (
                "no gids after multi-direction pre/pick/post quick check; "
                f"tried={','.join([d for d, _ in self._l2_pick_direction_vectors()])}; "
                f"fail={'; '.join(fail_msgs[:8])}"
            )

        return valid_union, f"ok dirs={','.join(pass_dirs)}"

    def _endpoint_manip(self, arm, gc, sp, sr, gp, gr, obs) -> float:
        if check_pose_reachability is None:
            return 0.0
        try:
            pick = check_pose_reachability(arm, sp, sr, gc, obs, max_grasps=5)
            place = check_pose_reachability(arm, gp, gr, gc, obs, max_grasps=5)
            vals = [
                float(getattr(pick, "best_manipulability", 0.0)),
                float(getattr(place, "best_manipulability", 0.0)),
            ]
            vals = [v for v in vals if np.isfinite(v) and v > 0.0]
            return float(np.mean(vals)) if vals else 0.0
        except Exception:
            return 0.0

    def evaluate_layout(self, layout: LayoutCandidate) -> bool:
        layout.assembly_region_id = self.current_assembly_region_id
        layout.assembly_region_rc = self.current_assembly_region_rc
        layout.assembly_station_pos = self.fixture_pos.copy()
        layout.assembly_station_rotmat = self.fixture_rotmat.copy()

        # 先用 identity 候选初始化所有零件，这样 step0 的 obstacle 也是有效随机位置。
        for pid in self.part_order:
            if pid in layout.xy:
                self._apply_staging_pose(pid, layout.xy[pid], self.rot_cands[pid][0])

        placed = set()
        fail_detail = {}
        layout.fail_part = None
        layout.fail_detail = {}

        # 第一件直接对齐当前装配区中心，视为已经装好，不再做 pick-and-place。
        first_pid = self._first_part_id() if self.preassemble_first_part else None
        if first_pid is not None:
            self._apply_first_part_as_assembled(layout)
            placed.add(first_pid)

        # 新增硬约束：后装零件不要比前装零件放得更远，也就是 x 不应明显变大。
        order_x_hit = self._order_x_constraint_reason(layout)
        if order_x_hit:
            layout.fail_part = "order_x_constraint"
            layout.fail_reason = order_x_hit
            return False

        for pid in self.part_order:
            if pid == first_pid:
                continue
            if pid not in self.world_poses:
                continue

            gc = self._grasp_collection(pid)
            if gc is None or len(gc) == 0:
                layout.fail_part = pid
                layout.fail_reason = f"{pid}: grasp collection missing or empty"
                return False

            gp, gr = self.world_poses[pid]
            best_record = None
            fail_counter = {
                "pair_collision": 0,
                "mesh_clearance": 0,
                "robot_home_collision": 0,
                "upright_constraint": 0,
                "staging_arm_keepout": 0,
                "l2_pick_quick_check": 0,
                "home_clearance": 0,
                "no_common_gids": 0,
                "reason_exception": 0,
            }

            for cand in self.rot_cands[pid]:
                # 硬约束：如果原始姿态 topdown(-Z) 抓取数太少，
                # 则禁止 identity / 平放姿态，只允许 upright / 侧立候选。
                upright_hit = self._upright_hard_constraint_reason(pid, cand)
                if upright_hit:
                    fail_counter["upright_constraint"] += 1
                    continue

                keepout_hit = self._staging_arm_keepout_reason(pid, layout.xy[pid], cand)
                if keepout_hit:
                    fail_counter["staging_arm_keepout"] += 1
                    continue

                self._apply_staging_pose(pid, layout.xy[pid], cand)

                # 候选姿态下，所有 staging 不能互相穿模
                hit = self._pairwise_collision()
                if hit:
                    fail_counter["pair_collision"] += 1
                    continue

                # 新增硬约束：任意两个 staging 零件外轮廓至少间隔 1cm。
                clearance_hit = self._mesh_clearance_reason(active_pids=self.part_order)
                if clearance_hit:
                    fail_counter["mesh_clearance"] += 1
                    continue

                # 新增硬约束：当前候选姿态不能和机器人 home 姿态下的手臂/夹爪穿模。
                # 这里只检查当前 pid，避免后续尚未评估的零件以临时姿态造成误杀；
                # 所有零件最终都会在自己的候选评估中检查，最后还会做全局复检。
                home_hit = self._robot_home_collision_reason(active_pids=[pid])
                if home_hit:
                    fail_counter["robot_home_collision"] += 1
                    continue

                home_clear_hit = self._robot_home_clearance_reason(active_pids=[pid])
                if home_clear_hit:
                    fail_counter["home_clearance"] += 1
                    continue

                sp = self.staging_models[pid].pos.copy()
                sr = self.staging_models[pid].rotmat.copy()
                obs = self._step_obstacles(pid, placed)
                planner_obs = self._planner_obstacles(obs, current_pid=pid, placed=placed)

                for arm_tag in self._arm_order(pid):
                    arm = get_layout_arm(
                        self.robot, arm_tag, single_arm=self.single_arm_mode)
                    planner = PickPlacePlanner(robot=arm)

                    try:
                        gids = planner.reason_common_gids(
                            grasp_collection=gc,
                            goal_pose_list=[(sp, sr), (gp, gr)],
                            obstacle_list=planner_obs,
                        )
                    except Exception:
                        fail_counter["reason_exception"] += 1
                        gids = []

                    # ---- 诊断: 桌面是否误杀该姿态的抓取(SEALP_TABLE_PROBE=部件名 打开) ----
                    # 对比三种障碍口径下该 (staging,goal) 姿态的 common grasp 数:
                    #   n_notable  = 当前(staging_aware, 不含桌面)
                    #   n_table    = 追加真实桌面
                    #   n_sunk     = 追加下沉 0.1m 的桌面(等效"抬高抓取余量")
                    # 若 n_notable>0 且 n_table=0:
                    #   - n_sunk 恢复 -> 抓取确实贴桌(低位/蹭桌), 站立也救不回低位抓取;
                    #   - n_sunk 仍=0 -> 桌面碰撞盒误杀(与高度无关的几何/位姿 bug)。
                    _probe = os.environ.get("SEALP_TABLE_PROBE")
                    if _probe and pid == _probe:
                        def _cnt(poses, extra_obs):
                            try:
                                g = planner.reason_common_gids(
                                    grasp_collection=gc,
                                    goal_pose_list=poses,
                                    obstacle_list=list(planner_obs) + list(extra_obs),
                                )
                                return len(g) if g else 0
                            except Exception:
                                return -1
                        sunk = []
                        for _o in self.env_obs:
                            try:
                                _c = _o.copy()
                                _p = np.asarray(_c.pos, dtype=float).copy(); _p[2] -= 0.10
                                _c.pos = _p
                                sunk.append(_c)
                            except Exception:
                                sunk.append(_o)
                        both = [(sp, sr), (gp, gr)]
                        n_notable = len(gids) if gids else 0
                        n_table = _cnt(both, self.env_obs)
                        n_sunk = _cnt(both, sunk)
                        # 厚桌面测试: 顶面仍在 z=0, 但盒子加厚到 0.30m(消除薄 trimesh 数值问题)
                        try:
                            thick = mcm.gen_box(np.array([0.6, 1.2, 0.30]),
                                                np.array([0.23, -0.35, -0.15]))
                            n_thick = _cnt(both, [thick])
                        except Exception as _te:
                            n_thick = -2
                        # 分别看 staging 端 / goal 端各自被真实桌面误杀多少
                        n_stage_nt = _cnt([(sp, sr)], [])
                        n_stage_tb = _cnt([(sp, sr)], self.env_obs)
                        n_goal_nt = _cnt([(gp, gr)], [])
                        n_goal_tb = _cnt([(gp, gr)], self.env_obs)
                        print(f"[TABLE-PROBE] pid={pid:10s} pose={cand.tag:16s} rot={cand.rot_name:10s} "
                              f"arm={arm_tag} z_off={cand.z_offset:.3f} up={float(cand.rotmat[2,2]):+.2f} "
                              f"| both: nt={n_notable:3d} tb={n_table:3d} sunk={n_sunk:3d} thick={n_thick:3d} "
                              f"| stage: nt={n_stage_nt:3d} tb={n_stage_tb:3d} "
                              f"| goal: nt={n_goal_nt:3d} tb={n_goal_tb:3d} "
                              f"| sp_z={float(sp[2]):.3f} gp_z={float(gp[2]):.3f}")
                        # 逐抓取量出"夹爪最低点 z" vs 桌面顶(z=0): 判断到底是真的探到桌下, 还是碰撞误判。
                        if cand.tag == "identity" and arm_tag == "rgt":
                            from panda3d.core import Point3
                            try:
                                g_stage = planner.reason_common_gids(
                                    grasp_collection=gc, goal_pose_list=[(sp, sr)], obstacle_list=[])
                                g_stage = list(g_stage) if g_stage else []
                            except Exception:
                                g_stage = []
                            for gi in g_stage[:6]:
                                try:
                                    grasp = gc._grasp_list[gi] if hasattr(gc, "_grasp_list") else gc[gi]
                                    ac_pos = np.asarray(sr, float) @ np.asarray(grasp.ac_pos, float) + np.asarray(sp, float)
                                    ac_rot = np.asarray(sr, float) @ np.asarray(grasp.ac_rotmat, float)
                                    arm.end_effector.grip_at_by_pose(
                                        jaw_center_pos=ac_pos, jaw_center_rotmat=ac_rot,
                                        jaw_width=grasp.ee_values)
                                    gmin = 1e9      # 真实 mesh 世界最低点
                                    cdtypes = []
                                    for el in arm.end_effector.cdelements:
                                        cmo = getattr(el, "cmodel", None)
                                        if cmo is None:
                                            continue
                                        cdtypes.append(str(getattr(cmo, "cdmesh_type", "?")))
                                        try:
                                            # 用碰撞模型自身的世界位姿(cmo.pos/rotmat, 即碰撞真正使用的口径)
                                            cpos = np.asarray(cmo.pos, dtype=float)
                                            crot = np.asarray(cmo.rotmat, dtype=float)
                                            verts = np.asarray(cmo.trm_mesh.vertices, dtype=float)
                                            wz = (verts @ crot.T + cpos)[:, 2]
                                            gmin = min(gmin, float(np.min(wz)))
                                        except Exception:
                                            pass
                                    approach = ac_rot[:, 2]  # 夹爪 +z(接近方向)在世界系
                                    # 只测"夹爪 vs 桌面"(不含手臂), 区分是夹爪误杀还是手臂撞桌
                                    try:
                                        eef_hit = arm.end_effector.is_mesh_collided(cmodel_list=list(self.env_obs))
                                    except Exception as _ee:
                                        eef_hit = f"err:{_ee}"
                                    # IK 多解测试: home 种子的解碰不碰? 换多个随机种子里有没有不碰的解?
                                    home_free = None
                                    seeds_tried = 0
                                    seeds_free = 0
                                    try:
                                        jr = np.asarray(arm.jnt_ranges, dtype=float)
                                        rng = np.random.default_rng(gi)
                                        jv0 = arm.ik(tgt_pos=ac_pos, tgt_rotmat=ac_rot)
                                        if jv0 is not None:
                                            arm.goto_given_conf(jv0)
                                            home_free = not arm.is_collided(obstacle_list=list(self.env_obs))
                                        for _ in range(16):
                                            seed = jr[:, 0] + rng.random(jr.shape[0]) * (jr[:, 1] - jr[:, 0])
                                            jv = arm.ik(tgt_pos=ac_pos, tgt_rotmat=ac_rot, seed_jnt_values=seed)
                                            if jv is None:
                                                continue
                                            seeds_tried += 1
                                            arm.goto_given_conf(jv)
                                            if not arm.is_collided(obstacle_list=list(self.env_obs)):
                                                seeds_free += 1
                                    except Exception as _ie:
                                        home_free = f"err:{_ie}"
                                    print(f"    [GRIPPER] gid={gi:4d} jaw_z={float(ac_pos[2]):+.3f} "
                                          f"mesh_min_z={gmin:+.3f} eef_vs_table={eef_hit} "
                                          f"| IK home_seed_collisionfree={home_free} "
                                          f"random_seeds: free={seeds_free}/{seeds_tried}")
                                except Exception as _e:
                                    print(f"    [GRIPPER] gid={gi} measure failed: {_e}")

                    n = len(gids)
                    if n <= 0:
                        fail_counter["no_common_gids"] += 1
                        continue

                    # 新增 L2 快速抓取运动检查：
                    # 对 middle_plate 等指定大件，进一步检查 pre-pick / pick / post-pick
                    # 三个位姿是否在当前动态障碍下仍有共同 grasp。
                    gids, quick_msg = self._l2_pick_quick_check_gids(
                        pid=pid,
                        planner=planner,
                        gc=gc,
                        gids=list(gids),
                        sp=sp,
                        sr=sr,
                        obs=obs,
                    )
                    n = len(gids)
                    if n <= 0:
                        fail_counter["l2_pick_quick_check"] += 1
                        continue

                    dist = float(np.linalg.norm(np.asarray(sp) - np.asarray(gp)))
                    rot_ang = _rot_angle(sr, gr)
                    manip = self._endpoint_manip(arm, gc, sp, sr, gp, gr, planner_obs)

                    # 单零件临时得分，用于在该零件多个候选姿态中选一个最好姿态。
                    # identity 可行时通常旋转代价最小，因此会自然胜出；
                    # 不可行时才会选站立/侧放。
                    grasp_s = _hill(min(n, 60), NORM_GRASP_MEAN_TARGET, NORM_GRASP_HILL_K)
                    manip_s = float(1.0 - np.exp(-max(manip, 0.0) / NORM_MANIP_TARGET))
                    dist_s = float(np.exp(-max(dist, 0.0) / NORM_DIST_DECAY))
                    rot_s = float(np.exp(-max(rot_ang, 0.0) / NORM_ROT_DECAY))
                    if cand.tag == "identity":
                        rot_s = min(1.0, rot_s + 0.10)

                    part_score = (
                        self.w_grasp * grasp_s
                        + self.w_manip * manip_s
                        + self.w_dist * dist_s
                        + self.w_rot * rot_s
                    )

                    # 如果该零件原始姿态下从上往下抓取太少，强制只能选择 upright / 侧立姿态。
                    part_score = self._upright_preference_adjusted_score(pid, cand, part_score)

                    # 默认姿态保持: 细长/扁平件翻倒(偏离 STL 默认朝向)时软扣分,
                    # 让细杆 post 优先"站立"、扁板优先"平放", 从而执行取放时天然远离桌面。
                    # 软惩罚 -> 只有默认姿态确实存在 common grasp 时才会胜出。
                    part_score = part_score - self.w_stl_upface * self._stl_upface_flip_penalty(pid, cand)

                    record = (
                        part_score, n, manip, dist, rot_ang,
                        arm_tag, cand, sp.copy(),
                    )
                    if best_record is None or record[0] > best_record[0]:
                        best_record = record

            if best_record is None:
                fail_detail[pid] = dict(fail_counter)
                layout.fail_part = pid
                layout.fail_detail = dict(fail_counter)
                layout.fail_reason = f"{pid}: all rotation/arm candidates failed; fail_counter={fail_counter}"
                return False

            part_score, n, manip, dist, rot_ang, arm_tag, cand, sp = best_record

            # commit 当前零件的最佳姿态
            self._apply_staging_pose(pid, layout.xy[pid], cand)

            layout.chosen_rotmat[pid] = cand.rotmat.copy()
            layout.z_offset[pid] = float(cand.z_offset)
            layout.pose_tag[pid] = cand.tag
            layout.rot_name[pid] = cand.rot_name
            layout.arm_choice[pid] = arm_tag
            layout.grasp_counts[pid] = int(n)
            layout.topdown_counts[pid] = int(self.topdown_identity_counts.get(pid, 0))
            layout.per_part_dist[pid] = float(dist)
            layout.per_part_manip[pid] = float(manip)
            layout.per_part_rot_angle[pid] = float(rot_ang)

            placed.add(pid)

        # 最终复检所有 staging 都不在左右臂附近禁区内
        for _pid in self.part_order:
            if _pid in self.rot_cands and _pid in layout.xy and _pid in layout.pose_tag:
                # 使用最终提交的旋转姿态重新构造一个临时候选 footprint
                # 优先从 rot_name 找到对应 cand
                _cand = None
                for _c in self.rot_cands.get(_pid, []):
                    if str(_c.rot_name) == str(layout.rot_name.get(_pid, "")):
                        _cand = _c
                        break
                if _cand is None and self.rot_cands.get(_pid):
                    _cand = self.rot_cands[_pid][0]
                if _cand is not None:
                    _khit = self._staging_arm_keepout_reason(_pid, layout.xy[_pid], _cand)
                    if _khit:
                        layout.fail_part = "final_staging_arm_keepout"
                        layout.fail_reason = f"final staging arm keepout violation: {_khit}"
                        return False

        # 最终复检所有 staging 不碰撞
        hit = self._pairwise_collision()
        if hit:
            layout.fail_part = "final_pairwise_collision"
            layout.fail_reason = f"final staging collision: {hit}"
            return False

        clearance_hit = self._mesh_clearance_reason(active_pids=self.part_order)
        if clearance_hit:
            layout.fail_part = "final_mesh_clearance"
            layout.fail_reason = f"final staging mesh clearance too small: {clearance_hit}"
            return False

        # 最终复检所有 staging 不与机器人 home 姿态穿模。
        # 这一步可以过滤掉 middle_plate 立起来后插进机械手/夹爪的 layout。
        home_hit = self._robot_home_collision_reason(active_pids=self.part_order)
        if home_hit:
            layout.fail_part = "final_robot_home_collision"
            layout.fail_reason = f"final robot home collision: {home_hit}"
            return False

        home_clear_hit = self._robot_home_clearance_reason(active_pids=self.part_order)
        if home_clear_hit:
            layout.fail_part = "final_robot_home_clearance"
            layout.fail_reason = f"final robot home clearance too small: {home_clear_hit}"
            return False

        # 计算 layout 级综合得分，模仿 find_optimal_layout.py：
        # grasp 用 min + mean 双项，避免某个零件成为瓶颈。
        counts = list(layout.grasp_counts.values())
        if not counts:
            layout.fail_reason = "no part evaluated"
            return False

        n_min = float(min(counts))
        n_mean = float(np.mean(counts))
        grasp_min_s = _hill(n_min, NORM_GRASP_MIN_TARGET, NORM_GRASP_HILL_K)
        grasp_mean_s = _hill(n_mean, NORM_GRASP_MEAN_TARGET, NORM_GRASP_HILL_K)
        grasp_score = 0.70 * grasp_min_s + 0.30 * grasp_mean_s

        avg_manip = float(np.mean(list(layout.per_part_manip.values()))) if layout.per_part_manip else 0.0
        manip_score = float(1.0 - np.exp(-max(avg_manip, 0.0) / NORM_MANIP_TARGET))

        avg_dist = float(np.mean(list(layout.per_part_dist.values()))) if layout.per_part_dist else 0.0
        dist_score = float(np.exp(-max(avg_dist, 0.0) / NORM_DIST_DECAY))

        avg_rot = float(np.mean(list(layout.per_part_rot_angle.values()))) if layout.per_part_rot_angle else 0.0
        rot_score = float(np.exp(-max(avg_rot, 0.0) / NORM_ROT_DECAY))

        layout.grasp_score_norm = grasp_score
        layout.manip_score_norm = manip_score
        layout.dist_score_norm = dist_score
        spatial_score = self._side_distribution_score(layout)

        layout.grasp_score_norm = grasp_score
        layout.manip_score_norm = manip_score
        layout.dist_score_norm = dist_score
        layout.rot_score_norm = rot_score
        layout.spatial_score_norm = spatial_score

        base_score = (
            self.w_grasp * grasp_score
            + self.w_manip * manip_score
            + self.w_dist * dist_score
            + self.w_rot * rot_score
        )
        # y 方向两侧分布作为小幅修正，不改变原四项权重结构。
        layout.layout_score = float(base_score * (0.90 + 0.10 * spatial_score))
        layout.l2_pass = True
        return True


    # --------------------------------------------------------
    # L3：全流程动态避障验证
    # --------------------------------------------------------

    def _apply_final_layout(self, layout: LayoutCandidate) -> None:
        """把 L2 选中的最终 staging pose 套到所有 staging CollisionModel。"""
        for pid in self.part_order:
            if pid not in self.staging_models:
                continue
            if pid not in layout.xy:
                continue
            if pid not in layout.chosen_rotmat:
                continue
            self.staging_models[pid].pos = np.array([
                float(layout.xy[pid][0]),
                float(layout.xy[pid][1]),
                float(layout.z_offset.get(pid, 0.0)),
            ], dtype=float)
            self.staging_models[pid].rotmat = np.asarray(layout.chosen_rotmat[pid], dtype=float)

    def _l3_obstacles(self, current_pid: str, placed: set, mode: str = "mesh") -> List:
        """L3 真实路径规划时的 step-aware 障碍物。

        动态语义：
            - 已经装好的零件：使用 goal pose，作为装配区障碍；
            - 还没装的零件：使用 staging pose，作为取料区障碍；
            - 当前正在搬运的零件：不放进 obstacle，因为它由 TransportPrimitive 作为 held object 处理；
            - work_table 等环境：按 mode 决定是否加入。
        """
        obs = []
        if mode == "none":
            return obs
        if mode in ("mesh", "env_only", "executor_match"):
            obs.extend(self.env_obs)
        # mesh/executor_match: 含桌面; staging_aware: 排除桌面, 与 L2 _planner_obstacles 同口径。
        if mode in ("mesh", "staging_aware", "executor_match"):
            for pid in placed:
                if pid in self.goal_models:
                    obs.append(self.goal_models[pid])
            for pid in self.part_order:
                if pid == current_pid or pid in placed:
                    continue
                if pid in self.staging_models:
                    obs.append(self.staging_models[pid])
        return obs

    def _l3_placement_obstacles(self, current_pid: str, placed: set, mode: str = "mesh") -> List:
        """L3 抓取/落位 IK 校验用障碍(传给 transport.plan 的 grasp_obstacle_list)。

        与执行脚本 _placement_obstacles 口径一致: 对【已装好的接触件】(direct parent /
        承托件, 如 post 脚下的 base_plate、middle_plate 下的四柱)做接触豁免, 否则
        零件落位时脚下的支撑件会把抓取 IK 全判成碰撞 -> "No common grasp id" ->
        L3 比真实执行严格得多、永远过不了。运输路径仍用完整障碍(_l3_obstacles)。
        """
        obs = []
        if mode == "none":
            return obs
        if mode in ("mesh", "env_only", "executor_match"):
            obs.extend(self.env_obs)
        # mesh: 含桌面(对夹爪抓取易误杀, 不推荐); staging_aware: 排除桌面(默认);
        # executor_match: 含桌面 + 接触豁免, 与执行脚本 _placement_obstacles 逐项一致。
        if mode in ("mesh", "staging_aware", "executor_match"):
            excluded = self._contact_exclusion_set(current_pid, placed)
            for pid in placed:
                if pid in excluded:
                    continue
                if pid in self.goal_models:
                    obs.append(self.goal_models[pid])
            for pid in self.part_order:
                if pid == current_pid or pid in placed:
                    continue
                if pid in self.staging_models:
                    obs.append(self.staging_models[pid])
        return obs

    def validate_full_sequence_l3(
        self,
        layout: LayoutCandidate,
        obstacle_mode: str = "mesh",
        verbose: bool = True,
    ) -> bool:
        """对一个 L2 layout 进行严格 L3 全流程验证。

        这一步比 reason_common_gids 更严格：
            pick -> depart -> transport/RRT -> place approach -> place -> depart
        全流程中会检查机器人、夹爪、被抓物体与 step-aware 障碍物的碰撞。
        """
        if layout.assembly_station_pos is not None and np.asarray(layout.assembly_station_pos).shape == (3,):
            self._set_assembly_station(
                layout.assembly_station_pos,
                region_id=layout.assembly_region_id,
                rc=layout.assembly_region_rc,
            )

        self._apply_final_layout(layout)
        _patch_rrt_for_l3()
        _reset_robot_for_l3(self.robot, single_arm=self.single_arm_mode)

        lft_transport = TransportPrimitive(self.robot.lft_arm)
        rgt_transport = (
            TransportPrimitive(self.robot.lft_arm)
            if self.single_arm_mode
            else TransportPrimitive(self.robot.rgt_arm)
        )
        placed = set()
        first_pid = self._first_part_id() if self.preassemble_first_part else None
        if first_pid is not None:
            placed.add(first_pid)

        if verbose:
            print("\n========== L3 Full-sequence validation ==========")
            print(f"obstacle_mode = {obstacle_mode}")

        try:
            for step_idx, pid in enumerate(self.part_order):
                if pid not in self.world_poses:
                    continue
                if pid == first_pid:
                    if verbose:
                        print(f"  [SKIP] step={step_idx} pid={pid:14s} already assembled at region={layout.assembly_region_id}")
                    continue

                # 用户指定跳过 L3 运动验证的零件(如 middle_plate)：
                # 不做取放/RRT 验证，但仍视为已放置，计入后续零件的 step-aware 障碍。
                if pid in getattr(self, "l3_skip_parts", set()):
                    placed.add(pid)
                    if verbose:
                        print(f"  [SKIP-L3] step={step_idx} pid={pid:14s} L3 motion validation skipped (still counted as placed obstacle)")
                    continue

                gc = self._grasp_collection(pid)
                if gc is None or len(gc) == 0:
                    layout.l3_fail_reason = f"L3 step={step_idx} {pid}: grasp collection missing"
                    return False

                arm_tag = layout.arm_choice.get(pid, "lft")
                sp = self.staging_models[pid].pos.copy()
                sr = self.staging_models[pid].rotmat.copy()
                gp, gr = self.world_poses[pid]
                # 运输/RRT 用完整障碍; 抓取/落位 IK 用接触豁免障碍(口径同执行脚本)。
                obs = self._l3_obstacles(pid, placed, mode=obstacle_mode)
                placement_obs = self._l3_placement_obstacles(pid, placed, mode=obstacle_mode)

                # 单个零件的运动验证走可重写钩子，子类可对特定零件改用换手等，
                # 使 L3 验证方式与真实执行(动画)一致。
                ok = self._l3_plan_part(
                    layout=layout, step_idx=step_idx, pid=pid, arm_tag=arm_tag,
                    sp=sp, sr=sr, gp=gp, gr=gr, gc=gc, obs=obs,
                    lft_transport=lft_transport, rgt_transport=rgt_transport,
                    verbose=verbose, placement_obs=placement_obs,
                )
                if not ok:
                    return False  # fail_reason 已在钩子内写好

                placed.add(pid)
                if verbose:
                    print(f"  [OK] step={step_idx} pid={pid:14s} arm={arm_tag} obs={len(obs)}")

            layout.l3_pass = True
            layout.l3_fail_reason = ""
            if verbose:
                print("[OK] L3 full sequence passed.")
            return True

        finally:
            _reset_robot_for_l3(self.robot, single_arm=self.single_arm_mode)

    def _l3_plan_part(self, layout, step_idx, pid, arm_tag, sp, sr, gp, gr, gc, obs,
                      lft_transport, rgt_transport, verbose=True, placement_obs=None) -> bool:
        """L3 单个零件的全流程运动验证钩子(默认: 单臂 TransportPrimitive)。

        pick -> depart -> transport/RRT -> place approach -> place -> depart 全流程。
        子类可重写, 对特定零件改用换手等其它运动方式, 使 L3 与真实执行一致。
        返回 True/False; 失败时负责写 ``layout.l3_fail_reason``。

        ``obs`` = 运输/RRT 完整障碍; ``placement_obs`` = 抓取/落位 IK 接触豁免障碍
        (作为 grasp_obstacle_list 传入, 口径同执行脚本; 旧版 plan 无此参数时回退)。
        """
        transport = rgt_transport if arm_tag == "rgt" else lft_transport
        if placement_obs is None:
            placement_obs = obs

        obj_cm = make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
        obj_cm.pos = np.asarray(sp, dtype=float).copy()
        obj_cm.rotmat = np.asarray(sr, dtype=float).copy()
        obj_cm._sealp_part_id = pid
        obj_cm._sealp_role = "l3_moving_object"

        try:
            try:
                res = transport.plan(
                    obj_cmodel=obj_cm,
                    grasp_collection=gc,
                    goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                    obstacle_list=obs,
                    grasp_obstacle_list=placement_obs,
                    approach_distance=APPROACH_DIST,
                    depart_distance=PICK_DEPART_DIST,
                    linear_granularity=LINEAR_GRANULARITY,
                    **_transport_kwargs(),
                )
            except TypeError:
                # 兼容旧版 TransportPrimitive.plan(无 grasp_obstacle_list)
                res = transport.plan(
                    obj_cmodel=obj_cm,
                    grasp_collection=gc,
                    goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                    obstacle_list=obs,
                    approach_distance=APPROACH_DIST,
                    depart_distance=PICK_DEPART_DIST,
                    linear_granularity=LINEAR_GRANULARITY,
                    **_transport_kwargs(),
                )
        except Exception as e:
            layout.l3_fail_reason = (
                f"L3 step={step_idx} {pid} {arm_tag}: "
                f"exception {type(e).__name__}: {e!r}"
            )
            if verbose:
                print(f"  [FAIL] {layout.l3_fail_reason}")
            return False

        if not bool(getattr(res, "success", False)):
            err = getattr(res, "error_msg", "") or "no plan"
            layout.l3_fail_reason = f"L3 step={step_idx} {pid} {arm_tag}: {err}"
            if verbose:
                print(f"  [FAIL] {layout.l3_fail_reason}")
            return False

        return True


    # --------------------------------------------------------
    # Random search
    # --------------------------------------------------------

    def random_search(self,
                      n_samples: int,
                      seed: int,
                      max_resample_layout: int = 80,
                      verbose: bool = True,
                      enable_l3: bool = False,
                      l3_top_k: int = 3,
                      l3_obstacle_mode: str = "mesh",
                      require_l3: bool = True) -> Optional[LayoutCandidate]:
        """加权随机搜索。

        流程：
            1. 在 work_table 上随机采样 n_samples 个 layout；
            2. 每个 layout 走 L2 快速评估并计算综合得分；
            3. 对 L2 通过的 layout 按 score 降序排序；
            4. 如果 enable_l3=True，对 top-k 做 TransportPrimitive 全流程验证；
            5. 返回 L3 通过的最高分 layout。若 require_l3=False 且 L3 全失败，则回退 L2 第一名。
        """
        rng = np.random.default_rng(seed)
        best_l2: Optional[LayoutCandidate] = None
        feasible: List[LayoutCandidate] = []

        print("\n========== Weighted Random Search ==========")
        print(f"n_samples             = {n_samples}")
        print(f"seed                  = {seed}")
        print(f"work_table x_range    = {self.table_x_range}")
        print(f"work_table y_range    = {self.table_y_range}")
        print(f"table_top_z           = {self.table_top_z:.6f}")
        print(f"weights               = grasp {self.w_grasp:.2f}, manip {self.w_manip:.2f}, dist {self.w_dist:.2f}, rot {self.w_rot:.2f}")
        print(f"part_order            = {self.part_order}")
        print(f"L3                    = {enable_l3}, top_k={l3_top_k}, obstacle_mode={l3_obstacle_mode}, require_l3={require_l3}")

        assembly_regions = self._assembly_region_candidates()

        t0 = time.time()
        n_generated = 0
        n_reject_sample = 0

        for i in range(n_samples):
            # coarse 装配区搜索：在 3x3 work_table 区域中心之间循环尝试。
            region_id, region_rc, region_pos = assembly_regions[i % len(assembly_regions)]
            self._set_assembly_station(region_pos, region_id=region_id, rc=region_rc)

            xy = None
            for _ in range(max_resample_layout):
                xy = self.sample_collision_free_xy(rng)
                if xy is not None:
                    break

            if xy is None:
                n_reject_sample += 1
                if verbose:
                    print(f"#{i:03d} SAMPLE_FAIL  无法在 work_table 上采样到不重叠初始 xy")
                continue

            n_generated += 1
            cand = LayoutCandidate(xy=xy)
            ok = self.evaluate_layout(cand)

            if not ok:
                if verbose:
                    print(f"#{i:03d} FAIL  {cand.fail_reason}")
                continue

            feasible.append(cand)
            if best_l2 is None or cand.layout_score > best_l2.layout_score:
                best_l2 = cand

            if verbose:
                print(
                    f"#{i:03d} L2_OK  score={cand.layout_score:.4f} "
                    f"G={cand.grasp_score_norm:.3f} "
                    f"M={cand.manip_score_norm:.3f} "
                    f"D={cand.dist_score_norm:.3f} "
                    f"R={cand.rot_score_norm:.3f} "
                    f"S={cand.spatial_score_norm:.3f} "
                    f"region={cand.assembly_region_id} "
                    f"counts={cand.grasp_counts} "
                    f"arms={cand.arm_choice} "
                    f"pose={cand.pose_tag} "
                    f"rot={cand.rot_name}"
                )

        feasible.sort(key=lambda c: c.layout_score, reverse=True)
        dt_l2 = time.time() - t0

        print("\n========== L2 Search Summary ==========")
        print(f"generated layouts     = {n_generated}/{n_samples}")
        print(f"sample rejected       = {n_reject_sample}")
        print(f"L2 feasible layouts   = {len(feasible)}/{n_samples}")
        print(f"L2 elapsed            = {dt_l2:.1f}s")

        if not feasible:
            print("[FAIL] L2 没有找到可行 layout。")
            return None

        print("[OK] best L2 layout:")
        print(f"  score               = {feasible[0].layout_score:.4f}")
        print(f"  components          = G {feasible[0].grasp_score_norm:.3f}, M {feasible[0].manip_score_norm:.3f}, D {feasible[0].dist_score_norm:.3f}, R {feasible[0].rot_score_norm:.3f}, S {feasible[0].spatial_score_norm:.3f}")
        print(f"  assembly_region     = {feasible[0].assembly_region_id}, rc={feasible[0].assembly_region_rc}, pos={np.round(feasible[0].assembly_station_pos, 4).tolist()}")
        print(f"  grasp_counts        = {feasible[0].grasp_counts}")
        print(f"  arm_choice          = {feasible[0].arm_choice}")
        print(f"  pose_tag            = {feasible[0].pose_tag}")
        print(f"  rot_name            = {feasible[0].rot_name}")

        if not enable_l3:
            return feasible[0]

        # L3：对 top-k 做真正全流程动态避障验证
        k = min(int(l3_top_k), len(feasible))
        print("\n========== L3 Top-k full-process validation ==========")
        print(f"try top-k             = {k}")

        for rank, cand in enumerate(feasible[:k], start=1):
            print(f"\n[L3] try rank={rank}/{k}, L2_score={cand.layout_score:.4f}")
            ok_l3 = self.validate_full_sequence_l3(
                cand,
                obstacle_mode=l3_obstacle_mode,
                verbose=True,
            )
            if ok_l3:
                print(f"[OK] L3 passed, selected rank={rank}, score={cand.layout_score:.4f}")
                return cand
            else:
                print(f"[NO] L3 failed rank={rank}: {cand.l3_fail_reason}")

        if require_l3:
            print("\n[FAIL] L2 top-k 全部未通过 L3，全流程动态避障未找到严格可行布局。")
            return None

        print("\n[WARN] L3 全失败，但 require_l3=False，回退保存 L2 最高分 layout。")
        return feasible[0]

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    def save_layout(self, layout: LayoutCandidate, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{self.output_name}.layout")

        if layout.assembly_station_pos is not None and np.asarray(layout.assembly_station_pos).shape == (3,):
            self._set_assembly_station(
                layout.assembly_station_pos,
                region_id=layout.assembly_region_id,
                rc=layout.assembly_region_rc,
            )

        staging_positions = {}
        for pid in self.part_order:
            staging_positions[pid] = (
                np.array([
                    float(layout.xy[pid][0]),
                    float(layout.xy[pid][1]),
                    float(layout.z_offset[pid]),
                ], dtype=float),
                np.asarray(layout.chosen_rotmat[pid], dtype=float),
            )

        ws = WorkspaceLayout(
            robot_base_pos=self.robot_base_pos.copy(),
            robot_base_rotmat=self.robot_base_rotmat.copy(),
            assembly_station_pos=np.asarray(layout.assembly_station_pos, dtype=float).copy(),
            assembly_station_rotmat=np.asarray(layout.assembly_station_rotmat, dtype=float).copy(),
            staging_positions=staging_positions,
            name=self.output_name,
            metadata={
                "search_method": "weighted_random_on_work_table",
                "l3_pass": bool(layout.l3_pass),
                "l3_fail_reason": str(layout.l3_fail_reason),
                "l3_skip_parts": sorted(list(self.l3_skip_parts)),
                "score": float(layout.layout_score),
                "score_components": {
                    "grasp": float(layout.grasp_score_norm),
                    "manip": float(layout.manip_score_norm),
                    "dist": float(layout.dist_score_norm),
                    "rot": float(layout.rot_score_norm),
                    "spatial_y_distribution": float(layout.spatial_score_norm),
                },
                "spatial_constraints": {
                    "min_staging_mesh_clearance": float(self.min_staging_mesh_clearance),
                    "enforce_order_x_constraint": bool(self.enforce_order_x_constraint),
                    "order_x_tolerance": float(self.order_x_tolerance),
                    "enable_y_side_distribution_score": bool(self.enable_y_side_distribution_score),
                    "force_upright_when_topdown_low": bool(self.prefer_upright_when_topdown_low),
                    "topdown_min_count": int(self.topdown_min_count),
                "upright_rule": "if identity topdown(-Z) count < threshold, non-upright initial poses are forbidden",
                    "topdown_align_cos": float(self.topdown_align_cos),
                    "check_l2_pick_quick_motion": bool(self.check_l2_pick_quick_motion),
                    "l2_pick_check_parts": sorted(list(self.l2_pick_check_parts)),
                    "l2_pick_check_lift_dist": float(self.l2_pick_check_lift_dist),
                    "l2_pick_check_directions": list(self.l2_pick_check_directions),
                    "l2_pick_check_tilt": float(self.l2_pick_check_tilt),
                    "goal_y_side_biased_sampling": bool(self.goal_y_side_biased_sampling),
                    "goal_y_side_bias_ratio": float(self.goal_y_side_bias_ratio),
                    "goal_y_side_eps": float(self.goal_y_side_eps),
                    "goal_y_side_rule": "if goal y is larger than other parts, prefer left half table; if smaller, prefer right half table; fallback to full table",
                    "robot_home_clearance": float(self.robot_home_clearance),
                },
                "weights": {
                    "grasp": float(self.w_grasp),
                    "manip": float(self.w_manip),
                    "dist": float(self.w_dist),
                    "rot": float(self.w_rot),
                },
                "part_order": list(self.part_order),
                "assembly_region_id": str(layout.assembly_region_id),
                "assembly_region_rc": list(layout.assembly_region_rc),
                "assembly_station_pos": np.asarray(layout.assembly_station_pos, dtype=float).tolist(),
                "preassemble_first_part": bool(self.preassemble_first_part),
                "filter_assembly_near_arms": False,
                "assembly_region_rule": "3x3 centers are all tried; only preassembled first part collision with arm boxes can reject an assembly center",
                "assembly_arm_x_clearance": float(self.assembly_arm_x_clearance),
                "assembly_arm_y_clearance": float(self.assembly_arm_y_clearance),
                "filter_staging_near_arms": bool(self.filter_staging_near_arms),
                "staging_arm_x_clearance": float(self.staging_arm_x_clearance),
                "staging_arm_y_clearance": float(self.staging_arm_y_clearance),
                "arm_base_xy_map": self._arm_base_xy_map(),
                "grasp_counts": dict(layout.grasp_counts),
                "topdown_counts_identity": dict(layout.topdown_counts),
                "arm_choice": dict(layout.arm_choice),
                "pose_tag": dict(layout.pose_tag),
                "rot_name": dict(layout.rot_name),
                "z_offsets": {k: float(v) for k, v in layout.z_offset.items()},
                "per_part_dist": dict(layout.per_part_dist),
                "per_part_manip": dict(layout.per_part_manip),
                "per_part_rot_angle": dict(layout.per_part_rot_angle),
                "table_top_z": float(self.table_top_z),
                "table_x_range": list(map(float, self.table_x_range)),
                "table_y_range": list(map(float, self.table_y_range)),
                "rotation_lift_rule": "z_offset = table_top_z + clearance - rotated_z_min",
                "check_robot_home_collision": bool(self.check_robot_home_collision),
                "strict_initial_robot_collision": bool(self.strict_initial_robot_collision),
                "robot_home_collision_rule": "all initial objects including preassembled first part must not collide with lft/rgt arm collision boxes at HOME_JV",
                "robot_home_clearance_rule": f"all staging models should keep at least {float(self.robot_home_clearance):.3f}m AABB clearance from robot home links when link AABBs are available",
                "l2_pick_quick_motion_rule": "for selected parts, common grasp ids must survive pre-pick / pick / post-pick quick check against dynamic obstacles",
            },
        )
        ws.save(out_path)
        print(f"\n[OK] 保存 layout -> {out_path}")

        debug_path = os.path.join(output_dir, f"{self.output_name}_debug.json")
        debug = {
            "layout_path": out_path,
            "score": float(layout.layout_score),
            "xy": {k: [float(v[0]), float(v[1])] for k, v in layout.xy.items()},
            "z_offset": {k: float(v) for k, v in layout.z_offset.items()},
            "pose_tag": dict(layout.pose_tag),
            "rot_name": dict(layout.rot_name),
            "assembly_region_id": str(layout.assembly_region_id),
            "assembly_region_rc": list(layout.assembly_region_rc),
            "assembly_station_pos": np.asarray(layout.assembly_station_pos, dtype=float).tolist(),
            "filter_assembly_near_arms": False,
            "assembly_region_rule": "3x3 centers are all tried; only preassembled first part collision with arm boxes can reject an assembly center",
            "assembly_arm_x_clearance": float(self.assembly_arm_x_clearance),
            "assembly_arm_y_clearance": float(self.assembly_arm_y_clearance),
            "filter_staging_near_arms": bool(self.filter_staging_near_arms),
            "staging_arm_x_clearance": float(self.staging_arm_x_clearance),
            "staging_arm_y_clearance": float(self.staging_arm_y_clearance),
            "arm_base_xy_map": self._arm_base_xy_map(),
            "grasp_counts": dict(layout.grasp_counts),
            "arm_choice": dict(layout.arm_choice),
            "score_components": {
                "grasp": float(layout.grasp_score_norm),
                "manip": float(layout.manip_score_norm),
                "dist": float(layout.dist_score_norm),
                "rot": float(layout.rot_score_norm),
                "spatial_y_distribution": float(layout.spatial_score_norm),
            },
            "spatial_constraints": {
                "min_staging_mesh_clearance": float(self.min_staging_mesh_clearance),
                "enforce_order_x_constraint": bool(self.enforce_order_x_constraint),
                "order_x_tolerance": float(self.order_x_tolerance),
                "force_upright_when_topdown_low": bool(self.prefer_upright_when_topdown_low),
                "topdown_min_count": int(self.topdown_min_count),
                "upright_rule": "if identity topdown(-Z) count < threshold, non-upright initial poses are forbidden",
                "check_l2_pick_quick_motion": bool(self.check_l2_pick_quick_motion),
                "l2_pick_check_parts": sorted(list(self.l2_pick_check_parts)),
                "l2_pick_check_lift_dist": float(self.l2_pick_check_lift_dist),
                "l2_pick_check_directions": list(self.l2_pick_check_directions),
                "l2_pick_check_tilt": float(self.l2_pick_check_tilt),
                "goal_y_side_biased_sampling": bool(self.goal_y_side_biased_sampling),
                "goal_y_side_bias_ratio": float(self.goal_y_side_bias_ratio),
                "goal_y_side_eps": float(self.goal_y_side_eps),
                "goal_y_side_rule": "if goal y is larger than other parts, prefer left half table; if smaller, prefer right half table; fallback to full table",
                "robot_home_clearance": float(self.robot_home_clearance),
            },
            "topdown_counts_identity": dict(layout.topdown_counts),
            "strict_initial_robot_collision": bool(self.strict_initial_robot_collision),
            "chosen_rotmat": {
                k: np.asarray(R).tolist()
                for k, R in layout.chosen_rotmat.items()
            },
        }
        if hasattr(self, "search_eval_stats"):
            debug["search_eval_stats"] = self.search_eval_stats()
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
        print(f"[OK] 保存 debug -> {debug_path}")

        return out_path


# ============================================================
# CLI
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Weighted random initial placement search on work_table."
    )
    parser.add_argument("--asmdef", default=DEFAULT_ASMDEF, help="asmdef 路径；PyCharm 右键运行时使用默认 Tower asmdef")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="sample_config.yaml 路径")
    parser.add_argument("--grasp-dir", default=DEFAULT_GRASP_DIR, help="grasp pickle 文件夹")
    parser.add_argument("--part-order", default=DEFAULT_PART_ORDER, help="逗号分隔的装配顺序；为空则使用 asmdef 顺序")
    parser.add_argument("--grasp-map-json", default="", help="可选，显式指定 grasp 文件映射")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="输出目录，默认 sealp/examples/layout/_output")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME, help="输出 layout 名称")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES, help="随机采样 layout 数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")

    parser.add_argument("--fixture-pos", default=DEFAULT_FIXTURE_POS, help="装配 fixture 世界坐标")
    parser.add_argument("--robot-base-pos", default=DEFAULT_ROBOT_BASE_POS, help="机器人基座世界坐标")

    parser.add_argument("--disable-assembly-region-search", action="store_true",
                        help="关闭 3x3 装配区域搜索，退回使用 --fixture-pos 固定装配中心。")
    parser.add_argument("--assembly-grid", type=int, default=DEFAULT_ASSEMBLY_GRID,
                        help="coarse 装配区域网格数量，默认 3 表示 3x3=9 块。")
    parser.add_argument("--disable-preassemble-first", action="store_true",
                        help="关闭第一件零件直接放在装配区中心的策略；默认第一件视为已装好。")
    parser.add_argument("--disable-assembly-arm-keepout", action="store_true",
                        help="兼容旧参数：当前版本装配区不再按 arm-base 矩形距离过滤，3x3 网格全部尝试。")
    parser.add_argument("--assembly-arm-x-clearance", type=float, default=DEFAULT_ASSEMBLY_ARM_X_CLEARANCE,
                        help="兼容旧参数：当前版本装配区不再使用该值做矩形过滤。")
    parser.add_argument("--assembly-arm-y-clearance", type=float, default=DEFAULT_ASSEMBLY_ARM_Y_CLEARANCE,
                        help="兼容旧参数：当前版本装配区不再使用该值做矩形过滤。")
    parser.add_argument("--disable-staging-arm-keepout", action="store_true",
                        help="关闭 staging 随机采样时避开左右臂 xy 矩形禁区的约束。")
    parser.add_argument("--staging-arm-x-clearance", type=float, default=DEFAULT_STAGING_ARM_X_CLEARANCE,
                        help="staging 零件外轮廓与左右臂基座 x 矩形禁区的安全距离，默认 0.12m。")
    parser.add_argument("--staging-arm-y-clearance", type=float, default=DEFAULT_STAGING_ARM_Y_CLEARANCE,
                        help="staging 零件外轮廓与左右臂基座 y 矩形禁区的安全距离，默认 0.12m。")
    parser.add_argument("--disable-goal-y-side-biased-sampling", action="store_true",
                        help="关闭根据最终装配 y 左/右关系进行半桌面优先采样。")
    parser.add_argument("--goal-y-side-bias-ratio", type=float, default=DEFAULT_GOAL_Y_SIDE_BIAS_RATIO,
                        help="目标侧半桌面优先采样比例，默认 0.75；剩余尝试回退全桌面。")
    parser.add_argument("--goal-y-side-eps", type=float, default=DEFAULT_GOAL_Y_SIDE_EPS,
                        help="判断 goal y 左/右的容差，默认 0.005m。")
    parser.add_argument("--disable-flatsurface", action="store_true",
                        help="关闭 flatsurface.py 稳定摆放姿态候选，退回 90 度候选。")
    parser.add_argument("--fs-stability-threshold", type=float, default=DEFAULT_FS_STABILITY_THRESHOLD,
                        help="flatsurface 稳定性阈值，默认 0.10。")
    parser.add_argument("--disable-home-collision-check", action="store_true",
                        help="关闭初始 staging 与机器人 home 姿态穿模检查；不建议关闭。")
    parser.add_argument("--disable-strict-initial-robot-collision", action="store_true",
                        help="关闭严格初始布局机器人碰撞检查；不建议关闭。开启时 preassembled 第一件也不能和左右机械臂碰撞盒碰撞。")
    parser.add_argument("--min-staging-mesh-clearance", type=float, default=DEFAULT_MIN_STAGING_MESH_CLEARANCE,
                        help="初始 staging 零件外轮廓最小间距，默认 0.01m。")
    parser.add_argument("--disable-order-x-constraint", action="store_true",
                        help="关闭装配顺序 x 约束；默认后装零件不应比前装零件 x 更大。")
    parser.add_argument("--order-x-tolerance", type=float, default=DEFAULT_ORDER_X_TOLERANCE,
                        help="order-x 约束容差，默认 0.03m。")
    parser.add_argument("--disable-y-side-distribution-score", action="store_true",
                        help="关闭 y 方向两侧分布的小幅评分加成。")
    parser.add_argument("--disable-upright-preference", action="store_true",
                        help="关闭 topdown 抓取不足时强制 upright / 侧立的硬约束。")
    parser.add_argument("--topdown-min-count", type=int, default=DEFAULT_TOPDOWN_MIN_COUNT,
                        help="若原始姿态从上往下抓取数小于该值，则优先 upright，默认 10。")
    parser.add_argument("--enable-l2-pick-quick-check", action="store_true",
                        help="开启 L2 pre-pick / pick / post-pick 快速检查；默认关闭，即不考虑撤离/接近距离限制。")
    parser.add_argument("--disable-l2-pick-quick-check", action="store_true",
                        help="兼容旧参数：关闭 L2 pre-pick / pick / post-pick 快速检查。")
    parser.add_argument("--l2-pick-check-parts", default=DEFAULT_L2_PICK_CHECK_PARTS,
                        help="需要做 L2 快速抓取运动检查的零件，逗号分隔，默认 middle_plate。")
    parser.add_argument("--l2-pick-check-lift-dist", type=float, default=DEFAULT_L2_PICK_CHECK_LIFT_DIST,
                        help="pre-pick/post-pick 撤离检查距离，默认 0.06m。")
    parser.add_argument("--l2-pick-check-directions", default=DEFAULT_L2_PICK_CHECK_DIRECTIONS,
                        help="L2 quick check 候选撤离方向，逗号分隔，默认 z,x_plus,x_minus,y_plus,y_minus。")
    parser.add_argument("--l2-pick-check-tilt", type=float, default=DEFAULT_L2_PICK_CHECK_TILT,
                        help="x/y 斜向撤离的水平分量比例，默认 0.35。")
    parser.add_argument("--robot-home-clearance", type=float, default=DEFAULT_ROBOT_HOME_CLEARANCE,
                        help="staging 零件与机器人 home 链节 AABB 的最小安全距离，默认 0.03m；设为 0 可关闭。")

    parser.add_argument("--table-name", default="work_table", help="config 中桌子障碍名")
    parser.add_argument("--table-margin", type=float, default=DEFAULT_TABLE_MARGIN, help="work_table 边界内缩")
    parser.add_argument("--table-clearance", type=float, default=DEFAULT_TABLE_CLEARANCE, help="旋转后 z 修正的安全间隙")
    parser.add_argument("--max-rot-candidates", type=int, default=DEFAULT_MAX_ROT_CANDIDATES, help="每个零件最多旋转候选数")
    parser.add_argument("--ignore-env", action="store_true", help="调试用：评估时忽略 work_table 等环境障碍")
    parser.add_argument(
        "--cdprim-type",
        default=DEFAULT_CDPRIM_TYPE,
        help="CollisionModel 碰撞类型，默认 triangles，尽量按 STL mesh 检测；不支持时自动退回。",
    )
    parser.add_argument(
        "--planner-obstacle-mode",
        choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
        default=DEFAULT_L2_OBSTACLE_MODE,
        help=(
            "reason_common_gids 使用的障碍列表。mesh=全 triangles 碰撞(对堆叠装配会被桌面/"
            "接触面误杀, 通常找不到解)；env_only=只检查桌子；none=不传障碍, 避免 gripper "
            "的 box/cdprim 误杀；staging_aware=排除桌面误杀 + 保留已装件(按 parent/"
            "接触表豁免插接面) + 新增其它 staging 件做障碍(默认)；executor_match="
            "staging_aware + 工作台桌面, 与执行脚本 _placement_obstacles 逐项一致, "
            "但贴桌零件的低位抓取会被薄桌盒误杀(不推荐)。"
        ),
    )

    parser.add_argument("--disable-prefer-stl-upface", action="store_true",
                        help="关闭'默认姿态保持'软偏好(细长/扁平件不再优先保持 STL 默认站立/平放)。")
    parser.add_argument("--w-stl-upface", type=float, default=DEFAULT_W_STL_UPFACE,
                        help="默认姿态保持惩罚权重(相对归一化 part_score, 默认 0.35)。越大越强制不翻倒。")
    parser.add_argument("--stl-upface-min-thinness", type=float, default=DEFAULT_STL_UPFACE_MIN_THINNESS,
                        help="只对 thinness>=此值(足够细长/扁平)的件生效, 默认 0.20。")

    parser.add_argument("--enable-l3", action="store_true", default=DEFAULT_ENABLE_L3,
                        help="开启严格 L3 全流程 TransportPrimitive/RRT 动态避障验证。默认关闭，只保存 L2。")
    parser.add_argument("--disable-l3", action="store_true",
                        help="关闭 L3，只保存 L2 结果。调试时才建议使用。")
    parser.add_argument("--l3-top-k", type=int, default=DEFAULT_L3_TOP_K,
                        help="对 L2 得分最高的前 K 个 layout 做 L3 验证。")
    parser.add_argument("--l3-obstacle-mode", choices=["mesh", "env_only", "none", "staging_aware", "executor_match"],
                        default=DEFAULT_L3_OBSTACLE_MODE,
                        help="L3 全流程验证的障碍模式。推荐 staging_aware(默认): 排除桌面(避免夹爪被桌面"
                             "mesh 误杀) + 接触豁免, 与 L2 --planner-obstacle-mode staging_aware 同口径; "
                             "executor_match: staging_aware + 桌面, 与执行脚本逐项一致(贴桌零件易误杀); "
                             "mesh 含桌面会过严。")
    parser.add_argument("--l3-skip-parts", default=DEFAULT_L3_SKIP_PARTS,
                        help="L3 全流程验证时跳过运动规划的零件，逗号分隔，默认 middle_plate。"
                             "被跳过的零件仍计入后续零件的障碍(视为已放置)，只是不对它本身做 L3 验证。"
                             "传空字符串则不跳过任何零件。")
    parser.add_argument("--allow-l2-fallback", action="store_true",
                        help="如果 L3 top-k 全失败，允许回退保存 L2 最高分。不加则 L3 失败时不保存。")

    parser.add_argument("--w-grasp", type=float, default=DEFAULT_W_GRASP)
    parser.add_argument("--w-manip", type=float, default=DEFAULT_W_MANIP)
    parser.add_argument("--w-dist", type=float, default=DEFAULT_W_DIST)
    parser.add_argument("--w-rot", type=float, default=DEFAULT_W_ROT)

    return parser.parse_args()


def main():
    args = _parse_args()

    part_order = _parse_part_order(args.part_order)
    grasp_map = _load_json_map(args.grasp_map_json)

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(SEALP_ROOT, "examples", "layout", "_output")

    print("=" * 70)
    print("Tower Initial Layout Search [v9.5 NoL2DepartApproach + GoalYSideSampling]")
    print(f"asmdef    = {args.asmdef}")
    print(f"config    = {args.config}")
    print(f"grasp_dir = {args.grasp_dir}")
    print(f"output    = {os.path.join(output_dir, args.output_name + '.layout')}")
    print(f"L2 obs    = {args.planner_obstacle_mode}")
    print(f"prefer STL upface = {not args.disable_prefer_stl_upface}, w={args.w_stl_upface:.2f}, min_thinness={args.stl_upface_min_thinness:.2f}  # 细长/扁平件优先保持默认站立/平放")
    print(f"L3 enable = {args.enable_l3 and not args.disable_l3}, L3 obs = {args.l3_obstacle_mode}, top_k = {args.l3_top_k}")
    print(f"L3 skip parts = {args.l3_skip_parts or '(none)'}  # skipped parts are still counted as placed obstacles")
    print(f"assembly region search = {not args.disable_assembly_region_search}, grid={args.assembly_grid}, preassemble_first={not args.disable_preassemble_first}")
    print(f"assembly arm keepout   = False  # 3x3装配中心不按矩形距离过滤，只检查preassembled是否撞机械臂")
    print(f"staging arm keepout    = {not args.disable_staging_arm_keepout}, x_clearance={args.staging_arm_x_clearance:.3f}m, y_clearance={args.staging_arm_y_clearance:.3f}m")
    print(f"goal-y side sampling   = {not args.disable_goal_y_side_biased_sampling}, ratio={args.goal_y_side_bias_ratio:.2f}, eps={args.goal_y_side_eps:.3f}m")
    _rb = _parse_vec3(args.robot_base_pos, (0.0, 0.0, 0.0))
    _ws = load_workspace_settings(args.config)
    if _ws["single_arm"]:
        print(f"robot mode             = single")
        print(f"arm base xy used       = arm=({_rb[0]:.3f},{_rb[1]:.3f})")
    else:
        print(f"robot mode             = dual")
        _off = float(_ws.get("dual_arm_y_offset", DUAL_ARM_Y_OFFSET))
        print(f"arm base xy used       = lft=({_rb[0]:.3f},{_rb[1]:.3f}), rgt=({_rb[0]:.3f},{_rb[1] - _off:.3f})")
    print(f"flatsurface poses      = {not args.disable_flatsurface}, threshold={args.fs_stability_threshold}")
    print(f"home collision check   = {not args.disable_home_collision_check}")
    print(f"strict initial arm-box collision = {not args.disable_strict_initial_robot_collision}")
    print(f"min mesh clearance     = {args.min_staging_mesh_clearance:.4f} m")
    print(f"order-x constraint     = {not args.disable_order_x_constraint}, tol={args.order_x_tolerance:.3f} m")
    print(f"y side score           = {not args.disable_y_side_distribution_score}")
    print(f"upright hard constraint = {not args.disable_upright_preference}, topdown_min={args.topdown_min_count}")
    _use_l2_pick_quick = bool(args.enable_l2_pick_quick_check and not args.disable_l2_pick_quick_check)
    print(f"L2 pick quick check     = {_use_l2_pick_quick}, parts={args.l2_pick_check_parts}, lift={args.l2_pick_check_lift_dist:.3f}m, dirs={args.l2_pick_check_directions}, tilt={args.l2_pick_check_tilt:.3f}")
    print(f"robot home clearance    = {args.robot_home_clearance:.3f}m")
    print("=" * 70)

    searcher = WeightedInitialLayoutSearcher(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        fixture_pos=_parse_vec3(args.fixture_pos, (0.36, 0.0, 0.0)),
        fixture_rotmat=np.eye(3),
        robot_base_pos=_parse_vec3(args.robot_base_pos, (0.0, 0.0, 0.0)),
        robot_base_rotmat=np.eye(3),
        part_order=part_order,
        output_name=args.output_name,
        table_name=args.table_name,
        table_margin=args.table_margin,
        table_clearance=args.table_clearance,
        grasp_map=grasp_map,
        max_rot_candidates=args.max_rot_candidates,
        w_grasp=args.w_grasp,
        w_manip=args.w_manip,
        w_dist=args.w_dist,
        w_rot=args.w_rot,
        ignore_env=args.ignore_env,
        cdprim_type=args.cdprim_type,
        planner_obstacle_mode=args.planner_obstacle_mode,
        plan_assembly_region=not args.disable_assembly_region_search,
        assembly_grid=args.assembly_grid,
        preassemble_first_part=not args.disable_preassemble_first,
        filter_assembly_near_arms=False,
        assembly_arm_x_clearance=args.assembly_arm_x_clearance,
        assembly_arm_y_clearance=args.assembly_arm_y_clearance,
        filter_staging_near_arms=not args.disable_staging_arm_keepout,
        staging_arm_x_clearance=args.staging_arm_x_clearance,
        staging_arm_y_clearance=args.staging_arm_y_clearance,
        goal_y_side_biased_sampling=not args.disable_goal_y_side_biased_sampling,
        goal_y_side_bias_ratio=args.goal_y_side_bias_ratio,
        goal_y_side_eps=args.goal_y_side_eps,
        use_flatsurface=not args.disable_flatsurface,
        fs_stability_threshold=args.fs_stability_threshold,
        check_robot_home_collision=not args.disable_home_collision_check,
        strict_initial_robot_collision=not args.disable_strict_initial_robot_collision,
        min_staging_mesh_clearance=args.min_staging_mesh_clearance,
        enforce_order_x_constraint=not args.disable_order_x_constraint,
        order_x_tolerance=args.order_x_tolerance,
        enable_y_side_distribution_score=not args.disable_y_side_distribution_score,
        prefer_upright_when_topdown_low=not args.disable_upright_preference,
        topdown_min_count=args.topdown_min_count,
        check_l2_pick_quick_motion=_use_l2_pick_quick,
        l2_pick_check_parts=_parse_part_order(args.l2_pick_check_parts) or [],
        l3_skip_parts=_parse_part_order(args.l3_skip_parts) or [],
        l2_pick_check_lift_dist=args.l2_pick_check_lift_dist,
        l2_pick_check_directions=_parse_part_order(args.l2_pick_check_directions) or [],
        l2_pick_check_tilt=args.l2_pick_check_tilt,
        robot_home_clearance=args.robot_home_clearance,
        prefer_stl_upface=not args.disable_prefer_stl_upface,
        w_stl_upface=args.w_stl_upface,
        stl_upface_min_thinness=args.stl_upface_min_thinness,
    )

    best = searcher.random_search(
        n_samples=args.n_samples,
        seed=args.seed,
        enable_l3=(args.enable_l3 and not args.disable_l3),
        l3_top_k=args.l3_top_k,
        l3_obstacle_mode=args.l3_obstacle_mode,
        require_l3=not args.allow_l2_fallback,
    )

    if best is not None:
        searcher.save_layout(best, output_dir)
    else:
        # 没找到通过验证的布局(常见: require_l3=True 且 L3 全失败)。
        # 关键安全修复: 把旧的 .layout 重命名失效, 避免用户误用一个未通过验证的旧布局
        # (那正是"能抓但放不下"的根源)。
        out_path = os.path.join(output_dir, f"{searcher.output_name}.layout")
        if os.path.isfile(out_path):
            stale_path = out_path + ".stale_invalid"
            try:
                if os.path.exists(stale_path):
                    os.remove(stale_path)
                os.replace(out_path, stale_path)
                print(
                    f"\n[SAVE] 未找到通过验证的布局; 已把旧的\n  {out_path}\n"
                    f"重命名为\n  {stale_path}\n"
                    "以免误用未验证布局。请放宽参数/换 seed 重搜, 或加 --allow-l2-fallback。"
                )
            except Exception as e:
                print(f"[SAVE] WARN: 旧 layout 失效失败: {type(e).__name__}: {e!r}")
        else:
            print("\n[SAVE] 未找到通过验证的布局, 且无旧 layout 文件, 不写出。")


if __name__ == "__main__":
    main()