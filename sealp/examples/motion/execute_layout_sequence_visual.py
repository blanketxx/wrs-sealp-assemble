#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execute + Visualize Assembly From .layout
================================================

这个脚本用于读取已经搜索得到的 WorkspaceLayout（.layout），然后按照 asmdef
顺序执行装配路径规划，并把能成功规划到的步骤动画显示出来。

特点：
1. 通用性：
   - 不写死 tower 的具体坐标；
   - 读取 .asmdef 决定零件模型和目标装配位姿；
   - 读取 .layout 决定 assembly_station 和 staging 初始摆放；
   - 读取 .layout.metadata["arm_choice"] 作为优先手臂；config 中 robot.mode=single 时只规划左臂。
2. 动态障碍物：
   - 已装好的零件：使用 goal pose，作为后续步骤障碍物；
   - 未装零件：使用 staging pose，作为后续步骤障碍物；
   - 当前搬运零件：从静态障碍列表中移除，由 TransportPrimitive 作为被抓物体处理。
3. 尽力执行：
   - 如果某一步失败，不会崩溃，也不会停止；
   - 会继续尝试后续零件；
   - 最后播放所有成功规划出来的步骤动画；
   - 同时显示初始布局、目标 ghost、已成功装配的零件。
4. middle_plate 走 **单臂**：先试强制 Cartesian 直线 approach/depart 的 pick-place，
   全部失败后改用 ``DirectTransportPrimitive``（同一只手臂、同一个 grasp，取放段用 RRT
   连接，不生成强制直线段，不换手、不中途放下）。双臂换手默认关闭，仅
   ``--middle-plate-handover`` 时才启用，届时需要预生成
   ``tower_handover/middle_plate_hopg.pickle``。
5. 默认使用 **box（AABB 包围盒）** 做避障碰撞，比 mesh/triangles 更快、更保守。
   - 需要更精细碰撞可传 ``--cdprim-type triangles``。
6. 运输障碍与落位接触豁免分离：
   - middle_plate 运输过程中会把四根柱子作为动态障碍，减少动画穿模；
   - 最终落位/插接附近仍保留必要的接触豁免。
7. 动画效果仿照 LRMate200id_ppp_animation.py：
   - 初始零件留在 layout 位置；
   - 抓住后由夹爪动画接管；
   - 放下后固定到目标位置；
   - 每帧补显示另一只手臂，避免“用到谁才出现谁”。
8. 全部步骤成功后保存运动路径 pkl，下次可直接缓存回放：
   - 默认缓存路径：sealp/examples/motion/_output/{layout名}_motions.pkl
   - 强制重算：--replan
   - 直接读缓存(不重规划)：不加 --replan；即使 fingerprint 变了也会回放已有 pkl
   - 若需旧行为(fingerprint 不匹配就拒绝缓存)：加 --strict-cache

推荐放置路径：
    sealp/examples/motion/execute_layout_sequence_visual.py

默认运行：
    python -m sealp.examples.motion.execute_layout_sequence_visual

指定 tower layout：
    python -m sealp.examples.motion.execute_layout_sequence_visual ^
      --layout D:/Project/wrs-sealp/sealp/examples/layout/_output/tower_optimal_initial.layout ^
      --asmdef D:/Project/wrs-sealp/sealp/assembly_sequence/_demo_output/topdown_tower.asmdef ^
      --grasp-dir D:/Project/wrs-sealp/sealp/examples/grasp/tower_grasp

说明：
    默认会优先使用 .layout 里的 assembly_station_pos。
    如果 layout 里 base_plate 被标记为 preassembled，则 step=0 会跳过抓取，
    直接把 base_plate 作为已经装好的动态障碍物加入后续规划。
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh as trm

import wrs.basis.robot_math as rm
import wrs.manipulation.handover_regrasp as horeg
from wrs import wd, mgm, mcm
# 允许直接右键运行
_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # 假设脚本位于 sealp/examples/motion 下
            return os.path.abspath(os.path.join(start_dir, "..", ".."))
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sealp.assembly_sequence import AssemblyDef
from sealp.config import load_config
from sealp.config.workspace_settings import load_workspace_settings
from sealp.colliders import StaticEnvironment
from sealp.layout import WorkspaceLayout
from sealp.layout._viz_common import load_table_box
from sealp.layout.layout_robot_factory import (
    create_layout_robot,
    get_layout_arm,
    goto_home_joints,
    iter_layout_arms,
)
from sealp.primitives.transport import TransportPrimitive
from sealp.primitives.direct_transport import DirectTransportPrimitive
from wrs.grasping.grasp import GraspCollection


# ============================================================
# 默认路径
# ============================================================

DEFAULT_ASMDEF = os.path.join(
    SEALP_ROOT, "assembly_sequence", "_demo_output", "topdown_tower.asmdef"
)
DEFAULT_LAYOUT = os.path.join(
    SEALP_ROOT, "examples", "layout", "_output", "tower_optimal_initial.layout"
)
DEFAULT_CONFIG = os.path.join(SEALP_ROOT, "config", "sample_config.yaml")
DEFAULT_GRASP_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_grasp")
DEFAULT_HANDOVER_DIR = os.path.join(SEALP_ROOT, "examples", "grasp", "tower_handover")
DEFAULT_MOTION_CACHE_DIR = os.path.join(SEALP_ROOT, "examples", "motion", "_output")
MOTION_CACHE_FORMAT_VERSION = "2.4"

# middle_plate 直接走换手，不走单臂 pick-place（默认关，见 --middle-plate-handover）
HANDOVER_PART_IDS = frozenset({"middle_plate"})

# 允许走"无强制 Cartesian 直线段"单臂规划的零件。这些件被周围已装件(如四根立柱)包围,
# 标准 pick-place 强制生成的直线 approach/depart 要求沿线每个插值位姿都有 IK, middle_plate
# 的所有共同抓取都死在 `IK not solvable in gen_linear_motion` 上。把距离缩到 0 治不了根:
# distance=0 时 start_tcp_pos == goal_tcp_pos, gen_linear_motion 仍会被调用并重解 IK。
# DirectTransportPrimitive 以认证过的 q_pick/q_place 为锚点用 RRT 连接, 全程单臂同一个
# grasp, 不换手也不中途放下 —— 与之前换手路径里 toggle_start_approach/end_depart=False
# 实际走出来的那条单臂轨迹同口径。
NOLINEAR_PART_IDS = frozenset({"middle_plate"})
LINEAR_FREE_TAG = "linear_free"


# ============================================================
# 规划参数：默认 box 避障碰撞（AABB 包围盒）
# ============================================================

DUAL_ARM_Y_OFFSET = 0.6
HOME_JV = np.zeros(6)

APPROACH_DIST = 0.0
PICK_DEPART_DIST = 0.03
PLACE_APPROACH_DIST = 0.03
PLACE_DEPART_DIST = 0.03
LINEAR_GRANULARITY = 0.04

# 多方向候选的水平倾斜量
MOTION_TILT = 0.35

# RRT 稀疏化，不降低障碍物要求，只降低搜索开销
RRT_EXT_DIST = 0.30
RRT_SMOOTHING_N_ITER = 150
RRT_MAX_TIME = 10.0

# 最后一件放置完成后的收尾策略：
# 1) 识别夹爪由闭合变为张开的 release 帧；
# 2) 只保留极短的原始脱离前缀，删除 TransportPrimitive 自动追加的长撤离尾巴；
# 3) 从第一个无碰撞的脱离帧重新用 RRT-Connect 规划到 HOME。
FINAL_RELEASE_HOLD_FRAMES = 3
FINAL_HOME_MAX_CLEARANCE_FRAMES = 8
FINAL_HOME_RRT_MAX_TIME = 10.0

# 默认使用 box（AABB）；triangles 更精细但更慢
DEFAULT_CDPRIM_TYPE = "box"

# 最终可视化桌面：按当前 work_table 尺寸生成真实贯穿孔洞洞板。
# 这里只替换显示模型；规划碰撞仍使用 config 中原来的实心 work_table，
# 因此不会改变已有运动规划结果或缓存指纹。
PERFORATED_TABLE_LONG_HOLES = 46
PERFORATED_TABLE_SHORT_HOLES = 23
PERFORATED_TABLE_HOLE_SEGMENTS = 16
PERFORATED_TABLE_HOLE_DIAMETER = None  # None: 自动取较小孔距的 42%



# ============================================================
# 最终显示用洞洞板桌面（只改视觉，不改规划碰撞）
# ============================================================

def _square_ring_points(
    half_x: float,
    half_y: float,
    radius: float,
    segments: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成一圈方形单元边界和对应圆孔边界，二者顶点数一致。"""
    segments = max(8, int(math.ceil(segments / 4.0)) * 4)
    per_edge = segments // 4
    outer: List[Tuple[float, float]] = []

    for i in range(per_edge):
        t = i / per_edge
        outer.append((-half_x + 2.0 * half_x * t, -half_y))
    for i in range(per_edge):
        t = i / per_edge
        outer.append((half_x, -half_y + 2.0 * half_y * t))
    for i in range(per_edge):
        t = i / per_edge
        outer.append((half_x - 2.0 * half_x * t, half_y))
    for i in range(per_edge):
        t = i / per_edge
        outer.append((-half_x, half_y - 2.0 * half_y * t))

    outer_arr = np.asarray(outer, dtype=float)
    norms = np.linalg.norm(outer_arr, axis=1, keepdims=True)
    inner_arr = radius * outer_arr / np.maximum(norms, 1e-12)
    return outer_arr, inner_arr


def _append_perforated_cell(
    vertices: List[List[float]],
    faces: List[List[int]],
    cx: float,
    cy: float,
    z_bottom: float,
    z_top: float,
    pitch_x: float,
    pitch_y: float,
    radius: float,
    segments: int,
    close_bottom: bool,
    close_right: bool,
    close_top: bool,
    close_left: bool,
) -> None:
    """添加一个带真实贯穿圆孔的矩形单元。"""
    outer_xy, inner_xy = _square_ring_points(
        half_x=pitch_x / 2.0,
        half_y=pitch_y / 2.0,
        radius=radius,
        segments=segments,
    )
    n = len(outer_xy)
    start = len(vertices)

    # 顶面外圈、顶面孔圈、底面外圈、底面孔圈。
    for xy in outer_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_top])
    for xy in inner_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_top])
    for xy in outer_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_bottom])
    for xy in inner_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_bottom])

    outer_top = start
    inner_top = start + n
    outer_bottom = start + 2 * n
    inner_bottom = start + 3 * n

    for i in range(n):
        j = (i + 1) % n

        # 顶面环形区域。
        faces.append([outer_top + i, outer_top + j, inner_top + j])
        faces.append([outer_top + i, inner_top + j, inner_top + i])

        # 底面环形区域。
        faces.append([outer_bottom + i, inner_bottom + j, outer_bottom + j])
        faces.append([outer_bottom + i, inner_bottom + i, inner_bottom + j])

        # 圆孔内壁。
        faces.append([inner_top + i, inner_bottom + j, inner_bottom + i])
        faces.append([inner_top + i, inner_top + j, inner_bottom + j])

    # 仅在桌面最外边界封侧壁，单元之间不生成重复内壁。
    per_edge = n // 4
    edge_ranges = []
    if close_bottom:
        edge_ranges.append(range(0, per_edge))
    if close_right:
        edge_ranges.append(range(per_edge, 2 * per_edge))
    if close_top:
        edge_ranges.append(range(2 * per_edge, 3 * per_edge))
    if close_left:
        edge_ranges.append(range(3 * per_edge, 4 * per_edge))

    for edge_range in edge_ranges:
        for i in edge_range:
            j = (i + 1) % n
            faces.append([outer_top + i, outer_bottom + i, outer_bottom + j])
            faces.append([outer_top + i, outer_bottom + j, outer_top + j])


def _build_perforated_table_mesh(
    extent: Sequence[float],
    pos: Sequence[float],
    long_count: int = PERFORATED_TABLE_LONG_HOLES,
    short_count: int = PERFORATED_TABLE_SHORT_HOLES,
    hole_diameter: Optional[float] = PERFORATED_TABLE_HOLE_DIAMETER,
    hole_segments: int = PERFORATED_TABLE_HOLE_SEGMENTS,
) -> Tuple[trm.Trimesh, dict]:
    """按 work_table 当前尺寸生成封闭的真实贯穿孔网格。"""
    extent = np.asarray(extent, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float).reshape(-1)
    if extent.size < 3 or pos.size < 3 or np.any(extent[:3] <= 0):
        raise ValueError(f"Invalid work_table geometry: extent={extent}, pos={pos}")

    size_x, size_y, thickness = map(float, extent[:3])
    if size_x >= size_y:
        nx, ny = int(long_count), int(short_count)
    else:
        nx, ny = int(short_count), int(long_count)

    pitch_x = size_x / nx
    pitch_y = size_y / ny
    min_pitch = min(pitch_x, pitch_y)

    if hole_diameter is None:
        hole_diameter = 0.42 * min_pitch
    hole_diameter = float(hole_diameter)
    max_diameter = 0.90 * min_pitch
    if not (0.0 < hole_diameter < max_diameter):
        raise ValueError(
            f"hole_diameter={hole_diameter:.6f} must be in (0, {max_diameter:.6f})"
        )

    radius = hole_diameter / 2.0
    x_min = float(pos[0] - size_x / 2.0)
    y_min = float(pos[1] - size_y / 2.0)
    z_bottom = float(pos[2] - thickness / 2.0)
    z_top = float(pos[2] + thickness / 2.0)

    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    for ix in range(nx):
        cx = x_min + (ix + 0.5) * pitch_x
        for iy in range(ny):
            cy = y_min + (iy + 0.5) * pitch_y
            _append_perforated_cell(
                vertices=vertices,
                faces=faces,
                cx=cx,
                cy=cy,
                z_bottom=z_bottom,
                z_top=z_top,
                pitch_x=pitch_x,
                pitch_y=pitch_y,
                radius=radius,
                segments=hole_segments,
                close_bottom=(iy == 0),
                close_right=(ix == nx - 1),
                close_top=(iy == ny - 1),
                close_left=(ix == 0),
            )

    mesh = trm.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    if not mesh.is_watertight:
        raise RuntimeError("Generated perforated work_table mesh is not watertight")

    info = {
        "nx": nx,
        "ny": ny,
        "hole_count": nx * ny,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "hole_diameter": hole_diameter,
        "is_watertight": bool(mesh.is_watertight),
    }
    return mesh, info


def _attach_perforated_table_visual(
    base,
    extent: Sequence[float],
    pos: Sequence[float],
    rgba: Sequence[float],
):
    """将洞洞板作为最终场景显示模型附加到 Panda3D。"""
    mesh, info = _build_perforated_table_mesh(extent=extent, pos=pos)

    # 先导出 STL 再由 WRS 加载。大网格直接传入 CollisionModel 在部分版本中
    # 可能只生成碰撞对象而不生成可见几何。
    extent_arr = np.asarray(extent, dtype=float)
    cache_dir = Path(_THIS_DIR) / "_generated_meshes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = (
        f"work_table_perforated_{info['nx']}x{info['ny']}_"
        f"{extent_arr[0] * 1000.0:.1f}x{extent_arr[1] * 1000.0:.1f}x"
        f"{extent_arr[2] * 1000.0:.1f}mm_"
        f"d{info['hole_diameter'] * 1000.0:.3f}mm_"
        f"seg{PERFORATED_TABLE_HOLE_SEGMENTS}.stl"
    )
    mesh_path = cache_dir / mesh_name
    mesh.export(str(mesh_path))

    board = mcm.CollisionModel(str(mesh_path))
    rgba_arr = np.asarray(rgba, dtype=float).reshape(-1)
    if rgba_arr.size >= 4:
        board.rgba = rgba_arr[:4]
    elif rgba_arr.size >= 3:
        board.rgb = rgba_arr[:3]
        board.alpha = 1.0
    else:
        board.rgba = np.array([0.60, 0.60, 0.60, 1.0])
    board._sealp_name = "work_table_perforated_visual"
    board._sealp_role = "environment_visual_only"
    board.attach_to(base)

    print(
        "[环境/视觉] work_table 已替换为洞洞板："
        f"{info['nx']}x{info['ny']}={info['hole_count']} 个真实贯穿孔，"
        f"孔径={info['hole_diameter'] * 1000.0:.2f} mm，"
        f"watertight={info['is_watertight']}。"
    )
    return board


def _find_work_table_obstacle_index(
    obs_list: List,
    table_extent: Sequence[float],
    table_pos: Sequence[float],
) -> Optional[int]:
    """从 StaticEnvironment 中稳健识别原实心 ``work_table``。

    识别顺序：
    1. 名称中直接包含 ``work_table`` / ``table``；
    2. 环境中只有一个障碍物时，直接把它视为桌面；
    3. 最后再按位置与尺寸进行几何匹配。

    之前只使用 ``get_extents()`` 做严格几何匹配。部分 WRS 版本返回的是
    变换后的包围盒或不同格式，因此即使环境里只有一张桌子也可能识别失败。
    """
    if not obs_list:
        return None

    # 先尝试通过对象名称识别。不同 WRS 版本使用的名称字段可能不同。
    for idx, obs in enumerate(obs_list):
        names = []
        for attr in ("name", "_name", "_sealp_name", "model_name"):
            try:
                value = getattr(obs, attr, None)
                if value is not None:
                    names.append(str(value).lower())
            except Exception:
                pass
        joined = " ".join(names)
        if "work_table" in joined or "worktable" in joined:
            print(f"[环境/视觉] 通过名称识别 work_table: obstacle[{idx}]")
            return idx

    # 当前 sample_config.yaml 只有一项静态环境障碍，它就是 work_table。
    # 这是最可靠的兜底，避免 get_extents() 在不同 WRS 版本中的差异。
    if len(obs_list) == 1:
        print(
            "[环境/视觉] 静态环境仅有 1 个障碍物，"
            "将 obstacle[0] 视为 work_table 并替换其显示模型。"
        )
        return 0

    target_extent = np.asarray(table_extent, dtype=float).reshape(3)
    target_pos = np.asarray(table_pos, dtype=float).reshape(3)
    best_idx = None
    best_score = float("inf")

    for idx, obs in enumerate(obs_list):
        try:
            obs_pos = np.asarray(obs.pos, dtype=float).reshape(3)
        except Exception:
            continue

        # 尺寸读取失败时，仍可仅按中心位置给出候选。
        obs_extent = None
        try:
            raw_extent = np.asarray(obs.get_extents(), dtype=float).reshape(-1)
            if raw_extent.size >= 3:
                obs_extent = np.abs(raw_extent[:3])
        except Exception:
            pass

        pos_scale = max(float(np.linalg.norm(target_extent)), 1e-6)
        pos_score = float(np.linalg.norm(obs_pos - target_pos)) / pos_scale

        if obs_extent is not None:
            extent_scale = np.maximum(np.abs(target_extent), 1e-6)
            extent_score = float(
                np.linalg.norm((obs_extent - target_extent) / extent_scale)
            )
        else:
            extent_score = 0.5

        score = pos_score + extent_score
        if score < best_score:
            best_score = score
            best_idx = idx

    # 多障碍环境下放宽阈值，但仍避免明显误识别。
    if best_idx is None or best_score > 1.25:
        print(
            f"[WARN] work_table 几何匹配失败: best_idx={best_idx}, "
            f"best_score={best_score:.3f}"
        )
        return None

    print(
        f"[环境/视觉] 通过几何匹配识别 work_table: "
        f"obstacle[{best_idx}], score={best_score:.3f}"
    )
    return best_idx

# ============================================================
# 数据结构
# ============================================================

@dataclass
class StepMotion:
    step_id: int
    part_id: str
    arm_tag: str
    motion_tag: str
    mot_data: object
    # 用于保存/重建轨迹缓存：单臂通常是 [res.mot_data]，换手是 motion_list。
    motion_segments: Optional[List] = None
    # 每个 segment 对应的逐帧物体位姿 [(pos, rotmat) | None]。
    # 在 _add_other_arm_to_motion_* 把对侧手臂塞进帧 cm_list 之前抢先抽出来，
    # 这样后续 cache 重放时不再用 staging→goal 直线插值假装物体。
    obj_pose_per_segment: Optional[List[List[Optional[Tuple[np.ndarray, np.ndarray]]]]] = None


@dataclass
class FailedStep:
    step_id: int
    part_id: str
    reason: str


@dataclass
class ExecutionSummary:
    success_steps: List[StepMotion]
    failed_steps: List[FailedStep]
    failed_step_id: Optional[int] = None
    failed_part_id: Optional[str] = None
    failed_reason: str = ""


@dataclass
class _AnimMotionData:
    """动画载体。mesh_list 可以是真 list，也可以是惰性序列。"""

    mesh_list: object


class _LazyCachedMeshList:
    """按需生成缓存回放帧，避免启动时一次性构建全部 mesh。

    旧逻辑会在读取 pkl 后立即把所有关节帧都转成 Panda3D mesh，
    对 100+ 帧的装配动画很容易造成卡顿、黑屏或显存压力。
    这个序列只保存轻量的关节角/物体位姿，在用户按 SPACE 到某一帧时
    才生成该帧 mesh，并且只保留很少的最近帧。
    """

    def __init__(self, runner, part_id: str, frame_states: List[dict], max_keep: int = 1):
        self.runner = runner
        self.part_id = part_id
        self.frame_states = list(frame_states)
        self.max_keep = max(0, int(max_keep))
        self._cache: Dict[int, object] = {}
        self._order: List[int] = []

    def __len__(self):
        return len(self.frame_states)

    def __bool__(self):
        return len(self.frame_states) > 0

    def __iter__(self):
        for i in range(len(self.frame_states)):
            yield self[i]

    def _remember(self, idx: int, frame) -> None:
        if self.max_keep <= 0:
            return
        self._cache[idx] = frame
        self._order.append(idx)
        while len(self._order) > self.max_keep:
            old_idx = self._order.pop(0)
            old_frame = self._cache.pop(old_idx, None)
            if old_frame is not frame:
                try:
                    old_frame.detach()
                except Exception:
                    pass

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self.frame_states)
        if idx < 0 or idx >= len(self.frame_states):
            raise IndexError(idx)
        if idx in self._cache:
            return self._cache[idx]

        st = self.frame_states[idx]
        frame = _make_cached_dual_frame(
            self.runner,
            st["lft_jv"],
            st["rgt_jv"],
            lft_ee=st.get("lft_ee"),
            rgt_ee=st.get("rgt_ee"),
            active_sides=st.get("active_sides"),
        )
        obj_pose = st.get("obj_pose")
        if frame is not None and obj_pose is not None:
            obj_pos, obj_rot = obj_pose
            _attach_object_at_pose_to_frame(
                frame,
                self.runner.asm,
                self.part_id,
                pos=obj_pos,
                rotmat=obj_rot,
                cdprim_type=self.runner.cdprim_type,
            )
        self._remember(idx, frame)
        return frame


class _QuietHandoverPlanner(horeg.HandoverPlanner):
    def show_graph(self):
        return


def _duplicate_grasp_collection(gc: GraspCollection) -> GraspCollection:
    try:
        return gc.copy()
    except Exception:
        return copy.deepcopy(gc)


def _merge_motion_mesh_list(motion_list: List) -> _AnimMotionData:
    mesh_list = []
    for md in motion_list:
        mesh_list.extend(getattr(md, "mesh_list", []) or [])
    return _AnimMotionData(mesh_list=mesh_list)


def _slice_motion_data_inplace(md, start: int = 0, stop: Optional[int] = None) -> None:
    """原地裁剪 MotionData 的所有逐帧字段，保持 jv/ev/mesh/物体位姿严格对齐。"""
    if md is None:
        return
    fields = ("_jv_list", "_ev_list", "_mesh_list", "_oiee_gl_pose_list")
    for attr in fields:
        seq = getattr(md, attr, None)
        if seq is None:
            continue
        try:
            setattr(md, attr, list(seq[start:stop]))
        except Exception:
            pass


def _ev_to_scalar(ev) -> Optional[float]:
    """把 jaw width/ee value 尽量转成一个可比较标量。"""
    if ev is None:
        return None
    try:
        if np.isscalar(ev):
            return float(ev)
    except Exception:
        pass
    try:
        arr = np.asarray(ev, dtype=float).reshape(-1)
        if arr.size:
            return float(arr[0])
    except Exception:
        pass
    return None


def _find_release_frame(md) -> Optional[int]:
    """寻找“夹爪闭合后重新张开”的第一帧，用于删除最后一件的多余撤离尾巴。"""
    jv_list = list(getattr(md, "jv_list", None) or [])
    ev_list = _extract_ev_list(md)
    if not jv_list or not ev_list:
        return None

    values = [_ev_to_scalar(ev) for ev in ev_list]
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None

    v_min, v_max = min(valid), max(valid)
    # 正常夹爪应存在明显的开合差；差值太小则不做误判。
    if v_max - v_min < 1e-4:
        return None
    threshold = v_min + 0.55 * (v_max - v_min)

    seen_closed = False
    for i, v in enumerate(values):
        if v is None:
            continue
        if v < threshold:
            seen_closed = True
        elif seen_closed and v >= threshold:
            return i
    return None


def _apply_motion_end_states(motion_list: List) -> None:
    for md in motion_list:
        robot = getattr(md, "robot", None)
        jv_list = getattr(md, "jv_list", None)
        if robot is None or not jv_list:
            continue
        try:
            robot.goto_given_conf(jv_list[-1])
        except Exception:
            pass


def _robot_arm_side(robot) -> str:
    """尽量判断 MotionData 对应的是左臂还是右臂。"""
    name = str(getattr(robot, "name", "") or "").lower()
    if "rgt" in name or "right" in name:
        return "rgt"
    if "lft" in name or "left" in name:
        return "lft"

    try:
        pos_y = float(np.asarray(getattr(robot, "pos", np.zeros(3)), dtype=float)[1])
        return "rgt" if pos_y < 0 else "lft"
    except Exception:
        return "lft"


def _get_arm_jv(arm) -> np.ndarray:
    try:
        return np.asarray(arm.get_jnt_values(), dtype=float)
    except Exception:
        return np.asarray(HOME_JV, dtype=float)


def _get_arm_ee(arm):
    try:
        return arm.get_ee_values()
    except Exception:
        return None


def _goto_arm(arm, jv, ee=None) -> None:
    try:
        if ee is not None:
            arm.goto_given_conf(np.asarray(jv, dtype=float), ee_values=ee)
        else:
            arm.goto_given_conf(np.asarray(jv, dtype=float))
    except TypeError:
        try:
            arm.goto_given_conf(np.asarray(jv, dtype=float))
        except Exception:
            pass
    except Exception:
        pass


def _gen_arm_mesh(arm, alpha: float = 1.0):
    """生成单臂 mesh。只作为补充显示，不参与碰撞。"""
    try:
        return arm.gen_meshmodel(alpha=alpha)
    except TypeError:
        try:
            return arm.gen_meshmodel()
        except Exception:
            return None
    except Exception:
        return None


def _extract_obj_pose_from_mesh(mesh) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """从一帧 ModelCollection 里抓出代表 “被持物体” 的 cm 的世界位姿。

    必须在 ``_add_other_arm_to_motion_*`` 改写 cm_list 之前调用。规则：
    1. 任何 cm.``_sealp_part_id`` 非空者（我们自己塞过的）优先；
    2. 否则取 ``cm_list[-1]`` —— 单臂 ``gen_meshmodel`` / ``obj_cmodel_copy.attach_to``
       都把 obj 追加在末尾。
    """
    if mesh is None:
        return None
    cm_list = getattr(mesh, "cm_list", None) or []
    if not cm_list:
        return None

    target = None
    for cm in reversed(cm_list):
        if getattr(cm, "_sealp_part_id", None):
            target = cm
            break
    if target is None:
        target = cm_list[-1]

    try:
        return (
            np.asarray(target.pos, dtype=float).copy(),
            np.asarray(target.rotmat, dtype=float).copy(),
        )
    except Exception:
        return None


def _extract_obj_pose_per_frame(mesh_list: List) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
    """逐帧抽取 obj (pos, rotmat)；None 表示该帧无法识别。"""
    out: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    last_seen: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for mesh in mesh_list or []:
        pose = _extract_obj_pose_from_mesh(mesh)
        if pose is not None:
            last_seen = pose
            out.append(pose)
        else:
            # 抽不出就沿用上一帧；动画不会因为缺失帧而出现物体瞬移。
            out.append(last_seen)
    return out


def _add_other_arm_to_motion_mesh_list(
    mesh_list: List,
    dual_robot,
    active_arm_tag: str,
    alpha: float = 1.0,
) -> None:
    """给单臂 MotionData 的每一帧补上另一只手臂。

    原来的动画帧通常只包含“正在运动的那只手臂”，看起来会像另一只手臂突然消失。
    这里不改变原帧里的主动手臂和被抓物体，只额外把另一只手臂以当前关节角画出来。
    """
    if not mesh_list:
        return

    other = dual_robot.rgt_arm if active_arm_tag == "lft" else dual_robot.lft_arm
    other_jv = _get_arm_jv(other)
    other_ee = _get_arm_ee(other)

    try:
        other.backup_state()
    except Exception:
        pass

    try:
        for frame in mesh_list:
            if frame is None:
                continue
            _goto_arm(other, other_jv, other_ee)
            other_mesh = _gen_arm_mesh(other, alpha=alpha)
            if other_mesh is not None:
                try:
                    other_mesh.attach_to(frame)
                except Exception:
                    pass
    finally:
        try:
            other.restore_state()
        except Exception:
            pass


def _add_other_arm_to_motion_segments(motion_list: List, dual_robot, alpha: float = 1.0) -> None:
    """给换手 motion_list 的每一段补上另一只手臂，使动画始终双臂可见。"""
    if not motion_list:
        return

    lft_jv = _get_arm_jv(dual_robot.lft_arm)
    rgt_jv = _get_arm_jv(dual_robot.rgt_arm)
    lft_ee = _get_arm_ee(dual_robot.lft_arm)
    rgt_ee = _get_arm_ee(dual_robot.rgt_arm)

    for md in motion_list:
        side = _robot_arm_side(getattr(md, "robot", None))
        mesh_list = getattr(md, "mesh_list", None) or []

        if side == "rgt":
            other = dual_robot.lft_arm
            other_jv, other_ee = lft_jv, lft_ee
        else:
            other = dual_robot.rgt_arm
            other_jv, other_ee = rgt_jv, rgt_ee

        try:
            other.backup_state()
        except Exception:
            pass
        try:
            for frame in mesh_list:
                if frame is None:
                    continue
                _goto_arm(other, other_jv, other_ee)
                other_mesh = _gen_arm_mesh(other, alpha=alpha)
                if other_mesh is not None:
                    try:
                        other_mesh.attach_to(frame)
                    except Exception:
                        pass
        finally:
            try:
                other.restore_state()
            except Exception:
                pass

        # 更新主动臂末状态，给后续段作为“另一只手臂”的静态状态使用。
        jv_list = getattr(md, "jv_list", None) or []
        ev_list = getattr(md, "ev_list", None) or []
        if jv_list:
            if side == "rgt":
                rgt_jv = np.asarray(jv_list[-1], dtype=float)
                if ev_list:
                    rgt_ee = ev_list[-1]
            else:
                lft_jv = np.asarray(jv_list[-1], dtype=float)
                if ev_list:
                    lft_ee = ev_list[-1]


def _extract_ev_list(md) -> List:
    ev_list = getattr(md, "ev_list", None)
    if ev_list is None:
        ev_list = getattr(md, "ee_values_list", None)
    return list(ev_list or [])


def _serialize_ev(ev):
    if ev is None:
        return None
    try:
        if np.isscalar(ev):
            return float(ev)
    except Exception:
        pass
    try:
        return np.asarray(ev, dtype=float).tolist()
    except Exception:
        return None


def _deserialize_ev(ev):
    if ev is None:
        return None
    try:
        if isinstance(ev, (int, float)):
            return float(ev)
        return np.asarray(ev, dtype=float)
    except Exception:
        return None


def _interp_rotmat(r0, r1, t: float) -> np.ndarray:
    """两个旋转矩阵之间做近似插值，只用于缓存回放时补显示物体。"""
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    t = float(np.clip(t, 0.0, 1.0))
    m = (1.0 - t) * r0 + t * r1
    try:
        u, _, vt = np.linalg.svd(m)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        return r
    except Exception:
        return r0 if t < 0.5 else r1


def _attach_object_at_pose_to_frame(
    frame,
    asm: AssemblyDef,
    part_id: str,
    pos: np.ndarray,
    rotmat: np.ndarray,
    cdprim_type: str,
) -> None:
    """缓存回放时按规划时记录的真实位姿，把物体贴到一帧上。"""
    if frame is None:
        return
    mesh_path = asm.model_path(part_id)
    if not mesh_path or not os.path.isfile(mesh_path):
        # 不静默失败：缺 STL 时零件不会跟夹爪走，表现为“只有臂和桌子”。
        print(
            f"[WARN] 无法贴零件到动画帧: part={part_id!r} "
            f"mesh 不存在: {mesh_path!r}"
        )
        return

    try:
        obj = make_collision_model(mesh_path, cdprim_type=cdprim_type)
        obj.pos = np.asarray(pos, dtype=float)
        obj.rotmat = np.asarray(rotmat, dtype=float)
        obj._sealp_part_id = part_id
        obj._sealp_role = "cached_moving_object_visual"
        obj.attach_to(frame)
    except Exception as e:
        print(
            f"[WARN] 贴零件到动画帧失败: part={part_id!r} "
            f"{type(e).__name__}: {e}"
        )


def _attach_interpolated_object_to_frame(
    frame,
    asm: AssemblyDef,
    part_id: str,
    start_pose: Tuple[np.ndarray, np.ndarray],
    goal_pose: Tuple[np.ndarray, np.ndarray],
    t: float,
    cdprim_type: str,
) -> None:
    """缓存里没有该帧的精确 obj 位姿时（兼容旧 cache）才会用到的兜底。"""
    if frame is None:
        return
    sp, sr = start_pose
    gp, gr = goal_pose
    pos = (1.0 - t) * np.asarray(sp, dtype=float) + t * np.asarray(gp, dtype=float)
    rot = _interp_rotmat(sr, gr, t)
    _attach_object_at_pose_to_frame(
        frame, asm, part_id, pos=pos, rotmat=rot, cdprim_type=cdprim_type,
    )


# ============================================================
# 基础工具
# ============================================================

def _patch_rrt():
    """让 RRT 稍微稀疏一点，避免太慢。"""
    try:
        from wrs.motion.probabilistic.rrt_connect import RRTConnect
    except Exception:
        return

    if getattr(RRTConnect.plan, "_layout_visual_patched", False):
        return

    _orig_plan = RRTConnect.plan

    def _patched_plan(self, *args, **kwargs):
        kwargs["ext_dist"] = RRT_EXT_DIST
        kwargs["smoothing_n_iter"] = RRT_SMOOTHING_N_ITER
        kwargs["max_time"] = RRT_MAX_TIME
        return _orig_plan(self, *args, **kwargs)

    _patched_plan._layout_visual_patched = True
    RRTConnect.plan = _patched_plan
    print(
        f"[RRT patch] ext_dist={RRT_EXT_DIST}, "
        f"smoothing={RRT_SMOOTHING_N_ITER}, max_time={RRT_MAX_TIME}s"
    )


def _unit_vec(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def make_collision_model(mesh_path: str, cdprim_type: str = DEFAULT_CDPRIM_TYPE):
    """创建 CollisionModel。

    cdprim_type="box"/"aabb"：AABB 包围盒碰撞（规划默认，快且保守）；
    cdprim_type="triangles"：三角 mesh 碰撞（更精细但慢）。
    """
    requested = str(cdprim_type or DEFAULT_CDPRIM_TYPE).lower()
    if requested in ("bbox", "bounding_box", "bounding-box", "aabb_box"):
        requested = "box"

    trials = []
    if requested in ("box", "aabb", "obb"):
        for t in ("box", "aabb", "obb"):
            trials.append({"initor": mesh_path, "cdprim_type": t})
        trials.append({"initor": mesh_path})
        trials.append({"initor": mesh_path, "cdprim_type": "convex_hull"})
    elif requested in ("default", "none", ""):
        trials.append({"initor": mesh_path})
        trials.append({"initor": mesh_path, "cdprim_type": "convex_hull"})
    else:
        trials.append({"initor": mesh_path, "cdprim_type": requested})
        if requested != "convex_hull":
            trials.append({"initor": mesh_path, "cdprim_type": "convex_hull"})
        trials.append({"initor": mesh_path})

    last_err = None
    for kw in trials:
        try:
            cm = mcm.CollisionModel(**kw)
            cm._sealp_cdprim_type = kw.get("cdprim_type", "default")
            return cm
        except TypeError as e:
            last_err = e
            try:
                if "cdprim_type" in kw:
                    cm = mcm.CollisionModel(mesh_path, cdprim_type=kw["cdprim_type"])
                    cm._sealp_cdprim_type = kw["cdprim_type"]
                    return cm
                cm = mcm.CollisionModel(mesh_path)
                cm._sealp_cdprim_type = "default"
                return cm
            except Exception as ee:
                last_err = ee
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"无法创建 CollisionModel: {mesh_path}, cdprim_type={cdprim_type}, "
        f"last_err={last_err!r}"
    )


def load_env_obstacles(config_path: str, base=None) -> List:
    """加载静态环境障碍物，并仅在视觉上把实心 work_table 换成洞洞板。

    返回的 ``obs_list`` 完全保留原始实心桌面，因此运动规划、碰撞检测和缓存
    逻辑不变；Panda3D 场景中跳过原桌面显示，改为 46×23 个真实贯穿孔的网格。
    """
    if not config_path or not os.path.isfile(config_path):
        print(f"[WARN] config 不存在，不加载环境障碍物: {config_path}")
        return []

    cfg = load_config(config_path)
    env = StaticEnvironment(obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
    obs_list = list(env.obstacle_list)

    table_extent = table_pos = table_rgba = None
    table_obs_idx = None
    try:
        table_extent, table_pos, table_rgba = load_table_box(config_path, "work_table")
        table_obs_idx = _find_work_table_obstacle_index(
            obs_list, table_extent=table_extent, table_pos=table_pos,
        )
    except Exception as e:
        print(
            f"[WARN] 无法读取/识别 work_table，保留原实心桌面显示: "
            f"{type(e).__name__}: {e}"
        )

    for idx, obs in enumerate(obs_list):
        obs._sealp_role = "environment_obstacle"
        if base is not None and idx != table_obs_idx:
            obs.attach_to(base)

    if base is not None and table_obs_idx is not None:
        try:
            _attach_perforated_table_visual(
                base=base,
                extent=table_extent,
                pos=table_pos,
                rgba=table_rgba,
            )
        except Exception as e:
            # 洞洞板生成失败时回退显示原实心桌面，规划障碍物始终不受影响。
            print(
                f"[WARN] 洞洞板显示失败，回退原实心 work_table: "
                f"{type(e).__name__}: {e}"
            )
            try:
                obs_list[table_obs_idx].attach_to(base)
            except Exception:
                pass
    elif base is not None and table_obs_idx is None:
        print("[WARN] 未识别出 work_table；所有环境障碍物保持原显示。")

    print(
        f"[环境] 已加载 {len(obs_list)} 个静态障碍物。"
        "碰撞仍使用原实心 work_table，显示使用洞洞板。"
    )
    return obs_list


def _model_id_for_part(asm: AssemblyDef, part_id: str) -> Optional[str]:
    """从 AssemblyDef 中尽量取出 part 对应的 model id。"""
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

    return None


def _find_grasp_pickle(
    asm: AssemblyDef,
    part_id: str,
    grasp_dir: str,
    explicit_map: Optional[Dict[str, str]] = None,
) -> str:
    """为 part_id 找 grasp pickle。

    兼容 tower:
        post_bl/post_fl/post_br/post_fr 共用 tower_post_grasps.pickle
    """
    explicit_map = explicit_map or {}

    mesh_path = asm.model_path(part_id)
    mesh_base = os.path.splitext(os.path.basename(mesh_path))[0]
    model_id = _model_id_for_part(asm, part_id)

    keys = [part_id, mesh_base]
    if model_id:
        keys.append(model_id)
    if part_id.startswith("post_"):
        keys.append("post")

    # 显式映射优先
    for k in keys:
        if k in explicit_map:
            p = explicit_map[k]
            if not os.path.isabs(p):
                p = os.path.join(grasp_dir, p)
            if os.path.isfile(p):
                return p

    names = []
    for k in keys:
        names += [
            f"{k}_grasps.pickle",
            f"{k}_grasps_topdown.pickle",
            f"{k}.pickle",
            f"tower_{k}_grasps.pickle",
            f"tower_{k}_grasps_topdown.pickle",
        ]

    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = os.path.join(grasp_dir, name)
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        f"找不到 {part_id} 的 grasp pickle。尝试 keys={keys}, grasp_dir={grasp_dir}"
    )


def load_grasp_cache(
    asm: AssemblyDef,
    part_ids: List[str],
    grasp_dir: str,
    explicit_map: Optional[Dict[str, str]] = None,
) -> Dict[str, GraspCollection]:
    out = {}
    file_cache = {}

    print("\n========== Grasp 文件加载 ==========")
    for pid in part_ids:
        pkl = _find_grasp_pickle(asm, pid, grasp_dir, explicit_map)
        if pkl not in file_cache:
            file_cache[pkl] = GraspCollection.load_from_disk(file_name=pkl)
        out[pid] = file_cache[pkl]
        print(f"{pid:16s}: {pkl}  n={len(out[pid])}")

    return out


def _load_json_map(text: str) -> Dict[str, str]:
    if not text:
        return {}
    return dict(json.loads(text))


def _motion_candidate_kwargs(pid: str, arm_tag: str, *, skip_place_depart: bool = False):
    """多套 pick/place 接近撤离方向候选。

    先试纯 Z，再试带一点水平偏置的方向。
    skip_place_depart: 最后一件(如 top_cross)放置后不再做 place_depart 撤离段,
    避免 pkl/动画末尾出现一段多余的"类似 middle_plate 附近"的关节角。
    """
    t = float(MOTION_TILT)

    specs = [
        ("z",       [0, 0, 1],      [0, 0, -1],      [0, 0, 1]),
        ("x_plus",  [t, 0, 1],      [-t, 0, -1],     [t, 0, 1]),
        ("x_minus", [-t, 0, 1],     [t, 0, -1],      [-t, 0, 1]),
        ("y_plus",  [0, t, 1],      [0, -t, -1],     [0, t, 1]),
        ("y_minus", [0, -t, 1],     [0, t, -1],      [0, -t, 1]),
    ]

    if arm_tag == "rgt":
        order = ["z", "y_minus", "x_plus", "x_minus", "y_plus"]
    else:
        order = ["z", "y_plus", "x_minus", "x_plus", "y_minus"]

    spec_map = {name: (pd, pa, pld) for name, pd, pa, pld in specs}

    # 直线段距离缩到 0 并不会跳过直线段: gen_rel_linear_motion 在 distance=0 时算出
    # start_tcp_pos == goal_tcp_pos, 然后照样调 gen_linear_motion 并以 seed=None 重解 IK,
    # 所以既没省掉 IK, 还可能落到与已认证抓取不同的 IK 分支上。真正不生成这段的做法见
    # LINEAR_FREE_TAG / DirectTransportPrimitive; 这里的 0 距离只是"尽量短", 保留作为
    # 第一批候选(能过就仍然用更严格的强制直线解)。
    no_linear = pid in NOLINEAR_PART_IDS
    pick_dep_d = 0.0 if no_linear else PICK_DEPART_DIST
    place_app_d = 0.0 if no_linear else PLACE_APPROACH_DIST
    place_dep_d = 0.0 if (no_linear or skip_place_depart) else PLACE_DEPART_DIST

    out = []
    for name in order:
        pd, pa, pld = spec_map[name]
        out.append((
            name,
            dict(
                pick_depart_direction=_unit_vec(pd),
                pick_depart_distance=pick_dep_d,
                place_approach_direction_list=[_unit_vec(pa)],
                place_approach_distance_list=[place_app_d],
                place_depart_direction_list=[_unit_vec(pld)],
                place_depart_distance_list=[place_dep_d],
            )
        ))
    return out


def _linear_free_kwargs(pid: str, arm_tag: str) -> Dict:
    """落位方向传给无直线段规划器: 它只用这个方向摆放"接触前的让位构型"。"""
    _, kwargs = _motion_candidate_kwargs(pid, arm_tag)[0]
    return {"place_depart_direction_list": kwargs["place_depart_direction_list"]}


def _plan_candidates(pid: str, arm_tag: str, *, skip_place_depart: bool = False):
    """本步要依次尝试的规划方案。

    先把所有强制 Cartesian 直线段的方向候选试完 —— 它约束更强, 能过就用它。全部失败后,
    对 ``NOLINEAR_PART_IDS`` 里的零件再试一次无强制直线段的单臂规划(仍是同一只手臂、
    同一个 grasp, 不换手也不中途放下)。
    """
    out = _motion_candidate_kwargs(pid, arm_tag, skip_place_depart=skip_place_depart)
    if pid in NOLINEAR_PART_IDS:
        out = list(out) + [(LINEAR_FREE_TAG, {})]
    return out


def _last_assembly_part_id(part_order: List[str], layout: WorkspaceLayout) -> Optional[str]:
    """装配顺序里最后一个需要规划的零件(跳过 preassembled 如 base_plate)。"""
    for pid in reversed(part_order):
        if _is_preassembled(pid, layout):
            continue
        return pid
    return None


def _goto_home_sim(runner, base=None) -> None:
    """仿真里把机械臂回到 home 并可选显示 home 姿态 mesh。"""
    if runner is None:
        return
    single_arm = bool(getattr(runner, "single_arm_mode", False))
    goto_home_joints(runner.robot, HOME_JV, single_arm=single_arm)
    if base is not None:
        try:
            prev = getattr(runner, "_home_pose_mesh", None)
            if prev is not None:
                try:
                    prev.detach()
                except Exception:
                    pass
            mesh = runner.robot.gen_meshmodel(alpha=1.0)
            mesh.attach_to(base)
            runner._home_pose_mesh = mesh
        except Exception as e:
            print(f"[WARN] home 姿态 mesh 显示失败: {type(e).__name__}: {e}")
    print("[HOME] 全部零件放置完成，机械臂回到 home。")


def _arm_try_order(
    pid: str,
    preferred: Optional[str],
    *,
    single_arm: bool = False,
) -> List[str]:
    """优先使用 layout 里给出的 arm_choice；单臂模式下只规划左臂。"""
    if single_arm:
        return ["lft"]

    if preferred in ("lft", "rgt"):
        other = "rgt" if preferred == "lft" else "lft"
        return [preferred, other]

    # 没有 preferred 时按零件名给个直觉顺序，不是强制
    if pid.endswith("_r") or pid.endswith("br") or pid.endswith("fr"):
        return ["rgt", "lft"]
    return ["lft", "rgt"]


def _is_preassembled(pid: str, layout: WorkspaceLayout) -> bool:
    meta = getattr(layout, "metadata", {}) or {}
    arm_choice = meta.get("arm_choice", {})
    pose_tag = meta.get("pose_tag", {})

    if arm_choice.get(pid) == "preassembled":
        return True
    if str(pose_tag.get(pid, "")).startswith("preassembled"):
        return True

    # 兼容 metadata 里的 preassemble_first_part
    part_order = meta.get("part_order", [])
    if meta.get("preassemble_first_part", False) and part_order and pid == part_order[0]:
        return True

    return False


def _step_for_part(asm: AssemblyDef, part_id: str):
    """根据 part_id 找对应 StepDef。"""
    for s in asm.steps:
        if s.part_id == part_id:
            return s
    return None


def _default_contact_exclusion_map(asm: AssemblyDef) -> Dict[str, List[str]]:
    """默认接触/插接豁免表。

    通用规则：
        当前零件的 direct parent 会自动在 _contact_exclusion_set 中加入；
        这里主要放一些 asmdef parent 无法表达但几何上明显插接/承托的关系。

    对当前 tower：
        - top_cross 竖着插入 middle_plate 顶面方孔，所以规划 top_cross 时要临时排除 middle_plate；
        - middle_plate 放到四根 post 顶部，最终接触面附近可能被 mesh 判交，所以规划 middle_plate 时可临时排除四根 post。
    """
    part_ids = set(getattr(asm, "part_ids", []))
    out: Dict[str, List[str]] = {}

    if "top_cross" in part_ids and "middle_plate" in part_ids:
        out.setdefault("top_cross", []).append("middle_plate")

    post_ids = [p for p in ("post_bl", "post_fl", "post_br", "post_fr") if p in part_ids]
    if "middle_plate" in part_ids and post_ids:
        out.setdefault("middle_plate", []).extend(post_ids)

    return out


def _part_order_from_asm_or_layout(asm: AssemblyDef, layout: WorkspaceLayout) -> List[str]:
    meta = getattr(layout, "metadata", {}) or {}
    order = meta.get("part_order")
    if order:
        return [p for p in order if p in asm.part_ids]

    return [
        s.part_id for s in asm.steps
        if s.part_id in asm.part_ids
    ]


# ============================================================
# 可视化
# ============================================================

def attach_goal_ghosts(base, asm, world_poses, cdprim_type: str):
    print("\n========== 目标 ghost ==========")
    for pid, (gp, gr) in world_poses.items():
        if pid not in asm.part_ids:
            continue
        mp = asm.model_path(pid)
        if not os.path.isfile(mp):
            continue

        ghost = make_collision_model(mp, cdprim_type=cdprim_type)
        ghost.pos = np.asarray(gp, dtype=float)
        ghost.rotmat = np.asarray(gr, dtype=float)
        ghost.rgba = np.array([0.7, 0.7, 0.7, 0.18])
        ghost.attach_to(base)
        print(f"ghost {pid:14s}: pos={np.round(gp, 4).tolist()}")


def attach_staging_visuals(base, asm, layout, part_order, cdprim_type: str):
    colors = [
        np.array([0.90, 0.45, 0.35, 0.85]),
        np.array([0.25, 0.60, 0.95, 0.85]),
        np.array([0.25, 0.80, 0.45, 0.85]),
        np.array([0.85, 0.55, 0.20, 0.85]),
        np.array([0.75, 0.35, 0.85, 0.85]),
        np.array([0.35, 0.85, 0.85, 0.85]),
        np.array([0.85, 0.85, 0.35, 0.85]),
    ]

    vis = {}
    print("\n========== 初始 staging 彩色显示 ==========")
    missing = []
    for i, pid in enumerate(part_order):
        st = layout.staging_positions.get(pid)
        if st is None:
            continue
        pos, rot = st
        mp = asm.model_path(pid)
        if not os.path.isfile(mp):
            missing.append((pid, mp))
            continue

        cm = make_collision_model(mp, cdprim_type=cdprim_type)
        cm.pos = np.asarray(pos, dtype=float)
        cm.rotmat = np.asarray(rot, dtype=float)
        cm.rgba = colors[i % len(colors)]
        cm.attach_to(base)
        mgm.gen_frame(pos=pos, rotmat=rot, ax_length=0.035).attach_to(base)
        vis[pid] = cm
        print(f"{pid:14s}: pos={np.round(pos, 4).tolist()}")

    if missing:
        print(f"[WARN] {len(missing)} 个零件 STL 找不到，staging/跟随夹爪都会缺失:")
        for pid, mp in missing:
            print(f"  {pid:14s}: {mp}")
    elif not vis:
        print("[WARN] 没有可显示的 staging 零件（layout staging 为空或 part_order 不匹配）。")

    return vis


def _safe_cm_pos(cm) -> Optional[np.ndarray]:
    if cm is None:
        return None
    try:
        return np.asarray(cm.pos, dtype=float).reshape(3)
    except Exception:
        return None


def _frame_has_cm_at(mesh, target_pos: np.ndarray, eps: float = 5e-3) -> bool:
    """判断一帧里是否有物体仍停在 staging 位置。"""
    if mesh is None:
        return False
    cm_list = getattr(mesh, "cm_list", None) or []
    for cm in cm_list:
        p = _safe_cm_pos(cm)
        if p is None:
            continue
        if float(np.linalg.norm(p - target_pos)) <= eps:
            return True
    return False


def _detect_carry_start_by_pose(mesh_list: List, staging_pos: Optional[np.ndarray], eps: float = 5e-3) -> Optional[int]:
    """仿照 LRMate 动画：从“物体不再贴在 staging”开始视为被夹爪接管。"""
    if staging_pos is None:
        return None
    staging_pos = np.asarray(staging_pos, dtype=float).reshape(3)
    for i, mesh in enumerate(mesh_list):
        if mesh is None:
            continue
        if not _frame_has_cm_at(mesh, staging_pos, eps):
            return i
    return None


_ANIME_RELEASE_COUNTER = 0


def _hard_release_mesh(mesh) -> None:
    """释放上一帧，并周期性触发 GC，降低长动画回放时的内存压力。"""
    global _ANIME_RELEASE_COUNTER
    if mesh is None:
        return
    try:
        mesh.detach()
    except Exception:
        pass
    _ANIME_RELEASE_COUNTER += 1
    if _ANIME_RELEASE_COUNTER % 40 == 0:
        try:
            gc.collect()
        except Exception:
            pass


def animate_success_steps(
    base,
    step_motions: List[StepMotion],
    runner=None,
    interval: float = 0.03,  # 保留参数仅为向后兼容，已不再使用 task tick。
    auto_play: bool = False,  # 保留参数仅为向后兼容，自动播放已彻底关闭。
    frame_stride: int = 1,    # >1 时每按一次 SPACE 跳 N 帧，减少重型 mesh 上传次数。
):
    """SPACE 驱动播放：按一次推进一帧；全部播放完后再按 SPACE 重新开始。

    动画状态机：
        PRE   ：零件保持在 layout 的 staging 初始位置；
        CARRY ：进入持物帧后，外部 staging 模型 detach，由夹爪 mesh 内的被持物体接管显示；
        DONE  ：整段动作结束后，把零件固定到目标装配位姿。

    循环播放：所有 step 播完后，恢复 staging 颜色块、清掉 assembled 实心件，
    下一次按 SPACE 重新从头播。
    """
    stride = max(1, int(frame_stride))
    if not step_motions:
        print("\n无成功轨迹可播放：仅显示初始布局、目标 ghost 和机械臂。")
        return

    motion_items = []
    for sm in step_motions:
        md = sm.mot_data
        mesh_list = getattr(md, "mesh_list", None)
        if mesh_list is None or len(mesh_list) == 0:
            continue
        motion_items.append(sm)

    if not motion_items:
        print("\n成功步骤没有 mesh_list，无法播放动画。")
        return

    # 收掉规划阶段静态机器人，避免“静态机器人 + 动画机器人”重影。
    if runner is not None:
        static_robot_mesh = getattr(runner, "static_robot_mesh", None)
        if static_robot_mesh is not None:
            try:
                static_robot_mesh.detach()
            except Exception:
                pass
            runner.static_robot_mesh = None

    PRE, CARRY, DONE = 0, 1, 2
    phases = []

    print("\n========== 动画控制 ==========")
    print(f"  按 SPACE 推进 {stride} 帧；全部播放完后再按 SPACE 重新开始（循环）。")
    print(f"  可播放成功步骤数：{len(motion_items)}")

    for sm in motion_items:
        mesh_list = getattr(sm.mot_data, "mesh_list", []) or []
        staging_pos = None
        if runner is not None:
            st_model = getattr(runner, "staging_models", {}).get(sm.part_id)
            staging_pos = _safe_cm_pos(st_model)
            if staging_pos is None:
                st = runner.layout.staging_positions.get(sm.part_id)
                if st is not None:
                    staging_pos = np.asarray(st[0], dtype=float).reshape(3)

        cached_cs = getattr(sm.mot_data, "cached_carry_start", None)
        if cached_cs is not None:
            cs = cached_cs
        else:
            cs = _detect_carry_start_by_pose(mesh_list, staging_pos)
            if cs is None and mesh_list:
                cs = 0

        phases.append({
            "state": PRE,
            "carry_start": cs,
            "done_solid": None,
        })
        print(
            f"  [anime] step={sm.step_id:2d} pid={sm.part_id:14s} "
            f"frames={len(mesh_list):4d} carry_start={cs} "
            f"staging_pos={None if staging_pos is None else np.round(staging_pos, 4).tolist()}"
        )

    def _detach_staging(pid: str) -> None:
        if runner is None:
            return
        vis = getattr(runner, "staging_visuals", {}).get(pid)
        if vis is not None:
            try:
                vis.detach()
            except Exception:
                pass

    def _reattach_staging(pid: str) -> None:
        if runner is None:
            return
        vis = getattr(runner, "staging_visuals", {}).get(pid)
        if vis is not None:
            try:
                vis.attach_to(base)
            except Exception:
                pass

    def _attach_done(pid: str):
        if runner is None:
            return None
        _detach_staging(pid)
        try:
            return runner._attach_assembled_solid(
                pid,
                rgba=np.array([0.25, 0.85, 0.35, 0.70]),
            )
        except Exception as e:
            print(f"[WARN] attach done visual failed for {pid}: {type(e).__name__}: {e}")
            return None

    def _detach_done_solid(solid) -> None:
        if solid is None:
            return
        try:
            solid.detach()
        except Exception:
            pass

    class _AnimeState:
        __slots__ = ("step_idx", "frame_idx", "last_attached", "finished", "loop_count", "home_done")

        def __init__(self):
            self.step_idx = 0
            self.frame_idx = 0
            self.last_attached = None
            self.finished = False
            self.loop_count = 0
            self.home_done = False

    st = _AnimeState()

    def _finalize_step(idx: int) -> None:
        if idx < 0 or idx >= len(motion_items):
            return
        sm = motion_items[idx]
        ph = phases[idx]
        if ph["state"] != DONE:
            ph["done_solid"] = _attach_done(sm.part_id)
            ph["state"] = DONE
        print(f"[动画] step={sm.step_id} {sm.part_id} 播放结束，零件固定到目标位。")

    def _advance_phase(idx: int, frame_idx: int) -> None:
        sm = motion_items[idx]
        ph = phases[idx]
        cs = ph["carry_start"]

        if cs is not None and ph["state"] == PRE and frame_idx >= int(cs):
            _detach_staging(sm.part_id)
            ph["state"] = CARRY
            print(
                f"[动画] step={sm.step_id} {sm.part_id} 进入 CARRY: "
                f"frame={frame_idx}, staging 模型 detach。"
            )

    def _reset_for_loop() -> None:
        """循环重置：恢复 staging 颜色块、清掉 assembled 实心件。"""
        _hard_release_mesh(st.last_attached)
        st.last_attached = None
        for idx, ph in enumerate(phases):
            sm = motion_items[idx]
            _detach_done_solid(ph.get("done_solid"))
            ph["done_solid"] = None
            ph["state"] = PRE
            _reattach_staging(sm.part_id)
        st.step_idx = 0
        st.frame_idx = 0
        st.finished = False
        st.home_done = False
        st.loop_count += 1
        print(f"\n[动画] 开始第 {st.loop_count + 1} 轮播放。")

    def _show_current_frame() -> None:
        """显示当前 (step_idx, frame_idx) 帧；跳过空 mesh，越界则推进到下一 step。"""
        while st.step_idx < len(motion_items):
            sm = motion_items[st.step_idx]
            mesh_list = getattr(sm.mot_data, "mesh_list", []) or []

            if st.frame_idx >= len(mesh_list):
                _hard_release_mesh(st.last_attached)
                st.last_attached = None
                _finalize_step(st.step_idx)
                st.step_idx += 1
                st.frame_idx = 0
                continue

            mesh = mesh_list[st.frame_idx]
            if mesh is None:
                st.frame_idx += 1
                continue

            _advance_phase(st.step_idx, st.frame_idx)

            _hard_release_mesh(st.last_attached)
            st.last_attached = None

            mesh.attach_to(base)
            st.last_attached = mesh
            return

        _hard_release_mesh(st.last_attached)
        st.last_attached = None
        st.finished = True
        if not st.home_done:
            _goto_home_sim(runner, base)
            st.home_done = True
        print(
            f"[动画] 第 {st.loop_count + 1} 轮播放完毕。再按 SPACE 重新开始。"
        )

    def _on_space():
        # 已经全部播完 → 下一次 SPACE 重置并播第一帧。
        if st.finished:
            _reset_for_loop()
            _show_current_frame()
            return

        # 首次按下 → 直接渲染第 0 帧；之后每按一次 frame_idx += stride。
        if st.last_attached is None:
            _show_current_frame()
        else:
            st.frame_idx += stride
            _show_current_frame()

    base.accept("space", _on_space)


# ============================================================
# 动态执行器
# ============================================================

class LayoutSequenceVisualizer:
    def __init__(
        self,
        asm: AssemblyDef,
        layout: WorkspaceLayout,
        config_path: str,
        grasp_dir: str,
        base,
        cdprim_type: str = DEFAULT_CDPRIM_TYPE,
        grasp_map: Optional[Dict[str, str]] = None,
        contact_exclusion_map: Optional[Dict[str, List[str]]] = None,
        enable_middle_plate_regrasp: bool = True,
        handover_dir: str = DEFAULT_HANDOVER_DIR,
        dual_arm_overlay: bool = False,
    ):
        self.asm = asm
        self.layout = layout
        self.config_path = config_path
        self.grasp_dir = grasp_dir
        self.base = base
        self.cdprim_type = cdprim_type
        self.grasp_map = grasp_map or {}
        # 动画减压：默认不再给每一帧叠加"另一只手臂"的完整 mesh。
        # 该叠加会让每帧三角面几何近乎翻倍(全程 100+ 帧)，是卡顿/黑屏崩溃的主因，
        # 且正在运动的那只手臂本来就会画出来。需要双臂常驻可用 --dual-arm-overlay 打开。
        self.dual_arm_overlay = bool(dual_arm_overlay)
        self.enable_middle_plate_regrasp = bool(enable_middle_plate_regrasp)
        self._middle_plate_hopg = os.path.join(
            handover_dir or DEFAULT_HANDOVER_DIR, "middle_plate_hopg.pickle"
        )

        # 接触/插接豁免表：每个 step 规划时，从动态障碍物中临时排除这些已装件。
        # direct parent 会自动加入；这里叠加默认 tower 规则和用户传入规则。
        self.contact_exclusion_map = _default_contact_exclusion_map(self.asm)
        if contact_exclusion_map:
            for k, v in contact_exclusion_map.items():
                self.contact_exclusion_map.setdefault(k, [])
                for item in v:
                    if item not in self.contact_exclusion_map[k]:
                        self.contact_exclusion_map[k].append(item)

        self.part_order = _part_order_from_asm_or_layout(self.asm, self.layout)
        self._last_assembly_pid = _last_assembly_part_id(self.part_order, self.layout)

        self.fixture_pos = np.asarray(self.layout.assembly_station_pos, dtype=float)
        self.fixture_rotmat = np.asarray(self.layout.assembly_station_rotmat, dtype=float)

        self.world_poses = self.asm.compute_world_poses(
            fixture_pos=self.fixture_pos,
            fixture_rotmat=self.fixture_rotmat,
        )

        self.robot_base_pos = np.asarray(
            getattr(self.layout, "robot_base_pos", np.zeros(3)), dtype=float
        )
        self.robot_base_rotmat = np.asarray(
            getattr(self.layout, "robot_base_rotmat", np.eye(3)), dtype=float
        )

        ws = load_workspace_settings(config_path)
        self.single_arm_mode = bool(ws.get("single_arm", False))
        robot_bundle = create_layout_robot(
            single_arm=self.single_arm_mode,
            robot_base_pos=self.robot_base_pos,
            robot_base_rotmat=self.robot_base_rotmat,
            dual_arm_y_offset=float(ws.get("dual_arm_y_offset", DUAL_ARM_Y_OFFSET)),
            enable_cc=True,
        )
        self.robot = robot_bundle.robot
        goto_home_joints(self.robot, HOME_JV, single_arm=self.single_arm_mode)

        self.env_obstacles = load_env_obstacles(config_path, base)
        self.grasps = load_grasp_cache(self.asm, self.part_order, grasp_dir, self.grasp_map)

        self.staging_models: Dict[str, object] = {}
        self.goal_models: Dict[str, object] = {}
        self.staging_visuals: Dict[str, object] = {}
        self.static_robot_mesh = None

        self._build_collision_models()

    def _build_collision_models(self):
        print("\n========== 构建 staging / goal 动态障碍模型 ==========")

        for pid in self.part_order:
            if pid not in self.asm.part_ids:
                continue

            mp = self.asm.model_path(pid)

            # staging
            st = self.layout.staging_positions.get(pid)
            if st is not None and os.path.isfile(mp):
                pos, rot = st
                cm = make_collision_model(mp, cdprim_type=self.cdprim_type)
                cm.pos = np.asarray(pos, dtype=float)
                cm.rotmat = np.asarray(rot, dtype=float)
                cm._sealp_part_id = pid
                cm._sealp_role = "staging_on_table"
                self.staging_models[pid] = cm

            # goal
            if pid in self.world_poses and os.path.isfile(mp):
                gp, gr = self.world_poses[pid]
                gm = make_collision_model(mp, cdprim_type=self.cdprim_type)
                gm.pos = np.asarray(gp, dtype=float)
                gm.rotmat = np.asarray(gr, dtype=float)
                gm._sealp_part_id = pid
                gm._sealp_role = "assembled_at_goal"
                self.goal_models[pid] = gm

        print(f"staging_models = {list(self.staging_models.keys())}")
        print(f"goal_models    = {list(self.goal_models.keys())}")

    def _contact_exclusion_set(self, current_pid: str, placed: set) -> set:
        """当前 step 的接触/插接豁免集合。

        这些零件不会加入 obstacle_list：
        1. 当前零件的 direct parent；
        2. contact_exclusion_map 中声明的接触件。

        典型例子：
            post_bl 插入 base_plate 的孔时，base_plate 是父件，不能作为普通障碍；
            top_cross 插入 middle_plate 的方孔时，middle_plate 也应临时排除。
        """
        excl = set()

        step = _step_for_part(self.asm, current_pid)
        parent_id = getattr(step, "parent_id", None) if step is not None else None
        if parent_id and parent_id != "fixture":
            excl.add(parent_id)

        for p in self.contact_exclusion_map.get(current_pid, []):
            excl.add(p)

        # 只排除已经装好的接触件；还没装的零件仍然应该作为 staging 障碍。
        return {p for p in excl if p in placed}

    def _transit_obstacles(self, current_pid: str, placed: set) -> List:
        """运输/RRT 阶段使用的完整动态障碍，不做接触豁免。

        关键修复：
            middle_plate 在移动过程中必须把四根 post 当作真实障碍物，
            否则虽然最终落位需要接触豁免，但运输路径可能直接穿过柱子。
        """
        obs = list(self.env_obstacles)

        for pid in placed:
            if pid in self.goal_models:
                obs.append(self.goal_models[pid])

        for pid in self.part_order:
            if pid == current_pid or pid in placed:
                continue
            if pid in self.staging_models:
                obs.append(self.staging_models[pid])

        return obs

    def _placement_obstacles(self, current_pid: str, placed: set) -> List:
        """抓取/落位校验用的动态障碍，允许父件/支撑件接触豁免。"""
        obs = list(self.env_obstacles)
        excluded = self._contact_exclusion_set(current_pid, placed)

        if excluded:
            print(f"    [接触豁免/落位] 当前 {current_pid!r} 规划时临时排除已装件: {sorted(excluded)}")

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

    def _current_obstacles(self, current_pid: str, placed: set) -> List:
        """兼容旧调用：默认返回落位校验障碍。"""
        return self._placement_obstacles(current_pid, placed)

    def _maybe_debug_obstacles(self, current_pid: str, placed: set,
                               transit_obs: List, placement_obs: List) -> None:
        """打印当前 step 真正参与碰撞检测的障碍物。

        注意：屏幕上彩色 staging 可视化模型可能只是动画显示残留；
        真正参与规划碰撞的是 transit_obs / placement_obs 这两个列表。
        """
        if not bool(getattr(self, "debug_obstacles", False)):
            return

        dbg_part = str(getattr(self, "debug_part", "") or "").strip()
        if dbg_part and dbg_part != current_pid:
            return

        printed = getattr(self, "_debug_obstacles_printed", set())
        key = (current_pid, tuple(sorted(list(placed))))
        if key in printed:
            return
        printed.add(key)
        self._debug_obstacles_printed = printed

        excluded = sorted(list(self._contact_exclusion_set(current_pid, placed)))

        def _obs_name(obs) -> str:
            pid = getattr(obs, "_sealp_part_id", None)
            role = getattr(obs, "_sealp_role", None)
            cdprim = getattr(obs, "_sealp_cdprim_type", None)
            try:
                pos = np.round(np.asarray(obs.pos, dtype=float), 4).tolist()
            except Exception:
                pos = "?"
            try:
                cls = obs.__class__.__name__
            except Exception:
                cls = type(obs).__name__
            pid_s = str(pid) if pid is not None else "-"
            role_s = str(role) if role is not None else "-"
            cdprim_s = str(cdprim) if cdprim is not None else "-"
            return (
                f"pid={pid_s:14s} "
                f"role={role_s:24s} "
                f"cdprim={cdprim_s:8s} "
                f"pos={pos} cls={cls}"
            )

        print("\n========== DEBUG OBSTACLES ==========")
        print(f"current_pid = {current_pid}")
        print(f"placed      = {sorted(list(placed))}")
        print(f"excluded    = {excluded}")
        print("说明: transit_obs 用于搬运/RRT；placement_obs 用于抓取/落位/common grasp。")
        print("      屏幕上的彩色 staging 模型不一定参与碰撞，请以下面列表为准。")

        print(f"\n[transit_obs] n={len(transit_obs)}")
        for i, obs in enumerate(transit_obs):
            print(f"  T{i:02d}: {_obs_name(obs)}")

        print(f"\n[placement_obs] n={len(placement_obs)}")
        for i, obs in enumerate(placement_obs):
            print(f"  P{i:02d}: {_obs_name(obs)}")
        print("=====================================\n")

    def _uses_handover_direct(self, pid: str) -> bool:
        return self.enable_middle_plate_regrasp and pid in HANDOVER_PART_IDS

    def _handover_arm_pairs(self, pid: str):
        if self.single_arm_mode:
            return []
        meta = getattr(self.layout, "metadata", {}) or {}
        preferred = (meta.get("arm_choice", {}) or {}).get(pid)
        pairs = [
            ("lft", "rgt", self.robot.lft_arm, self.robot.rgt_arm),
            ("rgt", "lft", self.robot.rgt_arm, self.robot.lft_arm),
        ]
        if preferred == "rgt":
            pairs.reverse()
        elif preferred == "lft":
            pass
        return pairs

    def _try_middle_plate_handover(
        self,
        pid: str,
        start_pose: Tuple[np.ndarray, np.ndarray],
        goal_pose: Tuple[np.ndarray, np.ndarray],
        gc: GraspCollection,
        obstacle_list: List,
    ) -> Tuple[Optional[StepMotion], str]:
        if not os.path.isfile(self._middle_plate_hopg):
            return None, f"{pid}: hopg missing: {self._middle_plate_hopg}"

        obj_cm = make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
        obj_cm.pose = start_pose
        last_err = ""

        for sender_tag, receiver_tag, sender_arm, receiver_arm in self._handover_arm_pairs(pid):
            planner = _QuietHandoverPlanner(
                obj_cmodel=obj_cm,
                sender_robot=sender_arm,
                receiver_robot=receiver_arm,
                sender_reference_gc=gc,
                receiver_reference_gc=_duplicate_grasp_collection(gc),
            )
            try:
                planner.add_hopg_collection_from_disk(self._middle_plate_hopg)
            except Exception as e:
                last_err = f"{pid} load hopg failed: {e!r}"
                print(f"  [NO] {last_err}")
                continue

            try:
                motion_list = planner.plan_by_obj_poses(
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    obstacle_list=obstacle_list,
                    toggle_dbg=False,
                )
            except Exception as e:
                last_err = (
                    f"{pid} handover {sender_tag}->{receiver_tag}: "
                    f"exception {type(e).__name__}: {e!r}"
                )
                print(f"  [NO] {last_err}")
                continue

            if motion_list is None:
                last_err = f"{pid} handover {sender_tag}->{receiver_tag}: no path"
                print(f"  [NO] {last_err}")
                continue

            # 在 _add_other_arm_to_motion_segments 改写 cm_list 之前，
            # 先抓出每个 segment 的逐帧 obj 真位姿，供 cache 重放精确复现。
            obj_pose_per_segment: List[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = []
            for md in motion_list:
                obj_pose_per_segment.append(
                    _extract_obj_pose_per_frame(getattr(md, "mesh_list", []) or [])
                )

            # 动画显示：可选地给每一帧额外补上另一只手臂(默认关，减压)。
            if self.dual_arm_overlay:
                _add_other_arm_to_motion_segments(motion_list, self.robot)
            _apply_motion_end_states(motion_list)
            anim_md = _merge_motion_mesh_list(motion_list)
            print(
                f"  [OK/handover] pid={pid:14s} {sender_tag}->{receiver_tag} "
                f"frames={len(anim_md.mesh_list)}"
            )
            return StepMotion(
                step_id=-1,
                part_id=pid,
                arm_tag=f"{sender_tag}+{receiver_tag}",
                motion_tag=f"handover_{sender_tag}_to_{receiver_tag}",
                mot_data=anim_md,
                motion_segments=list(motion_list),
                obj_pose_per_segment=obj_pose_per_segment,
            ), ""

        return None, last_err or f"{pid}: handover failed"

    def _try_middle_plate_handover_step(
        self,
        pid: str,
        placed: set,
    ) -> Tuple[Optional[StepMotion], str]:
        if pid not in self.staging_models or pid not in self.world_poses:
            return None, f"{pid}: missing staging/goal for handover"

        gc = self.grasps.get(pid)
        if gc is None or len(gc) == 0:
            return None, f"{pid}: grasp collection missing for handover"

        st = self.staging_models[pid]
        start_pose = (np.asarray(st.pos, dtype=float), np.asarray(st.rotmat, dtype=float))
        gp, gr = self.world_poses[pid]
        goal_pose = (np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))
        # middle_plate 运输路径必须使用完整障碍物，不能把四根柱子排除；
        # 只有最终落位/接触校验才允许豁免支撑件。
        transit_obs = self._transit_obstacles(pid, placed)
        placement_obs = self._placement_obstacles(pid, placed)

        meta = getattr(self.layout, "metadata", {}) or {}
        preferred_arm = (meta.get("arm_choice", {}) or {}).get(pid)
        print(f"\n[HANDOVER] pid={pid} 直接换手（跳过单臂） preferred_sender={preferred_arm}")
        print(f"  hopg = {self._middle_plate_hopg}")
        print(f"  transit_obs={len(transit_obs)}, placement_obs={len(placement_obs)}")

        return self._try_middle_plate_handover(
            pid, start_pose, goal_pose, gc, transit_obs
        )

    def _home_return_obstacles(self, placed_after: set) -> List:
        """最后一件释放后回 HOME 使用的完整障碍物集合。"""
        obs = list(self.env_obstacles)
        for placed_pid in placed_after:
            gm = self.goal_models.get(placed_pid)
            if gm is not None:
                obs.append(gm)
        for other_pid in self.part_order:
            if other_pid in placed_after:
                continue
            sm = self.staging_models.get(other_pid)
            if sm is not None:
                obs.append(sm)
        return obs

    def _build_last_part_home_return(
        self,
        pid: str,
        arm_tag: str,
        arm,
        transport_md,
        placed: set,
    ) -> Tuple[Optional[List], str]:
        """删除最后一件 release 后的长撤离尾巴，并重新规划到 HOME。

        TransportPrimitive 即使 place_depart_distance=0，在部分 WRS 版本里仍可能把
        一段 end/depart 轨迹追加到 MotionData。这里不再信任那段尾巴，而是：
        1. 根据夹爪重新张开的帧识别 release；
        2. 仅保留极短的原撤离前缀，逐帧寻找一个不碰撞的 RRT 起点；
        3. 从该起点用 RRT-Connect 规划到 HOME_JV。

        只有 HOME 路径成功时，最后一件才算规划成功，避免缓存里再次留下“放完后
        去了别处但没有回 home”的轨迹。
        """
        if transport_md is None:
            return None, f"{pid}: final motion data missing"

        jv_list = list(getattr(transport_md, "jv_list", None) or [])
        ev_list = _extract_ev_list(transport_md)
        if not jv_list:
            return None, f"{pid}: final motion has no joint frames"

        release_idx = _find_release_frame(transport_md)
        if release_idx is None:
            return None, f"{pid}: cannot detect gripper release frame"

        release_ee = ev_list[release_idx] if release_idx < len(ev_list) else None
        hold_end = min(
            len(jv_list) - 1,
            release_idx + max(1, int(FINAL_RELEASE_HOLD_FRAMES)) - 1,
        )
        search_end = min(
            len(jv_list) - 1,
            hold_end + max(0, int(FINAL_HOME_MAX_CLEARANCE_FRAMES)),
        )

        placed_after = set(placed)
        placed_after.add(pid)
        home_obstacles = self._home_return_obstacles(placed_after)

        try:
            from wrs.motion.probabilistic.rrt_connect import RRTConnect
        except Exception as e:
            return None, f"{pid}: cannot import RRTConnect for home return: {e!r}"

        print(
            f"  [FINAL/HOME] release_frame={release_idx}, "
            f"hold_end={hold_end}, clearance_search={hold_end}..{search_end}, "
            f"home_obs={len(home_obstacles)}"
        )

        original_jv = _get_arm_jv(arm)
        original_ee = _get_arm_ee(arm)
        last_reason = ""
        for cut_idx in range(hold_end, search_end + 1):
            start_jv = np.asarray(jv_list[cut_idx], dtype=float)
            _goto_arm(arm, start_jv, release_ee)

            try:
                home_planner = RRTConnect(arm)
                home_md = home_planner.plan(
                    start_conf=start_jv,
                    goal_conf=np.asarray(HOME_JV, dtype=float),
                    obstacle_list=home_obstacles,
                    ext_dist=RRT_EXT_DIST,
                    smoothing_n_iter=RRT_SMOOTHING_N_ITER,
                    max_time=FINAL_HOME_RRT_MAX_TIME,
                    toggle_dbg=False,
                )
            except Exception as e:
                last_reason = (
                    f"cut_frame={cut_idx} home RRT exception "
                    f"{type(e).__name__}: {e!r}"
                )
                print(f"    [NO/HOME] {last_reason}")
                continue

            if home_md is None or not (getattr(home_md, "jv_list", None) or []):
                last_reason = f"cut_frame={cut_idx} home RRT returned no path"
                print(f"    [NO/HOME] {last_reason}")
                continue

            # RRT 路径首帧与 transport 保留段末帧相同，删掉重复首帧。
            if len(getattr(home_md, "jv_list", []) or []) > 1:
                _slice_motion_data_inplace(home_md, start=1, stop=None)

            # 现在才正式裁掉 TransportPrimitive 的长尾，只保留到可安全回 HOME 的帧。
            _slice_motion_data_inplace(transport_md, start=0, stop=cut_idx + 1)

            # HOME 段里零件已经释放，视觉上始终固定在最终装配位姿。
            gp, gr = self.world_poses[pid]
            for frame in getattr(home_md, "mesh_list", None) or []:
                _attach_object_at_pose_to_frame(
                    frame,
                    self.asm,
                    pid,
                    pos=np.asarray(gp, dtype=float),
                    rotmat=np.asarray(gr, dtype=float),
                    cdprim_type=self.cdprim_type,
                )

            print(
                f"    [OK/HOME] 保留原轨迹至 frame={cut_idx}，"
                f"删除其后多余尾巴；RRT HOME frames="
                f"{len(getattr(home_md, 'jv_list', []) or [])}"
            )
            return [transport_md, home_md], ""

        # 所有 HOME 尝试失败时恢复进入本函数前的手臂状态，避免污染下一个候选。
        _goto_arm(arm, original_jv, original_ee)
        return None, (
            f"{pid}: release 后无法在 {hold_end}..{search_end} 帧中找到可回 HOME 的起点; "
            f"last={last_reason}"
        )

    def _try_plan_step(self, pid: str, placed: set):
        if pid not in self.staging_models:
            return None, f"{pid}: staging model missing"

        if pid not in self.world_poses:
            return None, f"{pid}: goal world pose missing"

        gc = self.grasps.get(pid)
        if gc is None or len(gc) == 0:
            return None, f"{pid}: grasp collection missing or empty"

        if self._uses_handover_direct(pid):
            return self._try_middle_plate_handover_step(pid, placed)

        obj_cm = self.staging_models[pid]
        gp, gr = self.world_poses[pid]

        meta = getattr(self.layout, "metadata", {}) or {}
        preferred_arm = (meta.get("arm_choice", {}) or {}).get(pid)
        arm_order = _arm_try_order(pid, preferred_arm, single_arm=self.single_arm_mode)

        last_err = ""
        if pid == self._last_assembly_pid:
            print(f"[PLAN] pid={pid} 为最后一件: 放置后不做 place_depart, 结束即 home。")
        print(f"\n[PLAN] pid={pid}, preferred_arm={preferred_arm}, try_order={arm_order}")

        for arm_tag in arm_order:
            arm = get_layout_arm(self.robot, arm_tag, single_arm=self.single_arm_mode)
            transport = TransportPrimitive(arm)

            for motion_tag, motion_kwargs in _plan_candidates(
                    pid, arm_tag,
                    skip_place_depart=(pid == self._last_assembly_pid)):
                # 每次尝试复制一个 moving object，避免失败污染 staging model
                moving = make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
                moving.pos = obj_cm.pos.copy()
                moving.rotmat = obj_cm.rotmat.copy()
                moving._sealp_part_id = pid
                moving._sealp_role = "moving_object"

                # 运输阶段使用完整动态障碍；落位/插接校验才使用接触豁免障碍。
                transit_obs = self._transit_obstacles(pid, placed)
                placement_obs = self._placement_obstacles(pid, placed)

                debug_free_part = str(getattr(self, "debug_free_part", "") or "").strip()
                if debug_free_part and (debug_free_part == pid or debug_free_part.lower() == "all"):
                    print(
                        f"    [DEBUG/FREE-PART] {pid}: 强制清空 transit_obs 和 placement_obs。"
                        "这只用于诊断 No common grasp 是否由障碍物造成，不建议保存缓存。"
                    )
                    transit_obs = []
                    placement_obs = []

                self._maybe_debug_obstacles(pid, placed, transit_obs, placement_obs)

                try:
                    if motion_tag == LINEAR_FREE_TAG:
                        # 无强制 Cartesian 直线段的单臂规划: 同一只手臂、同一个 grasp,
                        # 以认证过的 q_pick/q_place 为锚点用 RRT 连接。全程单臂, 不换手,
                        # 不中途放下; IK/关节限位/碰撞照旧严格检查。
                        res = DirectTransportPrimitive(arm).plan(
                            obj_cmodel=moving,
                            grasp_collection=gc,
                            goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                            obstacle_list=transit_obs,
                            grasp_obstacle_list=placement_obs,
                            **_linear_free_kwargs(pid, arm_tag),
                        )
                    else:
                        try:
                            res = transport.plan(
                                obj_cmodel=moving,
                                grasp_collection=gc,
                                goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                                obstacle_list=transit_obs,
                                grasp_obstacle_list=placement_obs,
                                approach_distance=APPROACH_DIST,
                                depart_distance=PICK_DEPART_DIST,
                                linear_granularity=LINEAR_GRANULARITY,
                                **motion_kwargs,
                            )
                        except TypeError:
                            # 兼容旧版 TransportPrimitive.plan：没有 grasp_obstacle_list 时仍能运行。
                            res = transport.plan(
                                obj_cmodel=moving,
                                grasp_collection=gc,
                                goal_pose_list=[(np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))],
                                obstacle_list=transit_obs,
                                approach_distance=APPROACH_DIST,
                                depart_distance=PICK_DEPART_DIST,
                                linear_granularity=LINEAR_GRANULARITY,
                                **motion_kwargs,
                            )
                except Exception as e:
                    last_err = (
                        f"{pid} {arm_tag} motion={motion_tag}: "
                        f"exception {type(e).__name__}: {e!r}"
                    )
                    print(f"  [NO] {last_err}")
                    continue

                if not bool(getattr(res, "success", False)):
                    msg = getattr(res, "error_msg", "") or "no valid plan"
                    last_err = f"{pid} {arm_tag} motion={motion_tag}: {msg}"
                    print(f"  [NO] {last_err}")
                    continue

                transport_md = getattr(res, "mot_data", None)
                if transport_md is None:
                    last_err = f"{pid} {arm_tag} motion={motion_tag}: success=True but mot_data missing"
                    print(f"  [NO] {last_err}")
                    continue

                motion_segments = [transport_md]
                end_jv = np.asarray(getattr(res, "end_jnt_values", HOME_JV), dtype=float)

                # 最后一件必须把 TransportPrimitive 的自动撤离尾巴替换为明确的 HOME 路径。
                if pid == self._last_assembly_pid:
                    motion_segments, home_err = self._build_last_part_home_return(
                        pid=pid,
                        arm_tag=arm_tag,
                        arm=arm,
                        transport_md=transport_md,
                        placed=placed,
                    )
                    if motion_segments is None:
                        last_err = f"{pid} {arm_tag} motion={motion_tag}: {home_err}"
                        print(f"  [NO/FINAL-HOME] {last_err}")
                        continue
                    end_jv = np.asarray(HOME_JV, dtype=float)

                # 必须在对侧手臂 overlay 改写 cm_list 前抽取每段物体真实位姿。
                obj_pose_per_segment: List[List[Optional[Tuple[np.ndarray, np.ndarray]]]] = []
                for md in motion_segments:
                    obj_pose_per_segment.append(
                        _extract_obj_pose_per_frame(getattr(md, "mesh_list", []) or [])
                    )

                # 动画显示：可选地给每一段补上另一只静止手臂。
                if self.dual_arm_overlay:
                    for md in motion_segments:
                        _add_other_arm_to_motion_mesh_list(
                            getattr(md, "mesh_list", []) or [],
                            self.robot,
                            active_arm_tag=arm_tag,
                        )

                # 多段（最后一件 transport + HOME）合并为一个连续动画。
                anim_md = (
                    motion_segments[0]
                    if len(motion_segments) == 1
                    else _merge_motion_mesh_list(motion_segments)
                )

                total_frames = sum(
                    len(getattr(md, "mesh_list", []) or []) for md in motion_segments
                )
                print(
                    f"  [OK] pid={pid:14s} arm={arm_tag} motion={motion_tag} "
                    f"transit_obs={len(transit_obs)} placement_obs={len(placement_obs)} "
                    f"segments={len(motion_segments)} frames={total_frames} "
                    f"end={'HOME' if pid == self._last_assembly_pid else 'place/depart'}"
                )

                # 更新该臂末端状态；最后一件现在明确结束在 HOME_JV。
                _goto_arm(arm, end_jv)

                sm = StepMotion(
                    step_id=-1,
                    part_id=pid,
                    arm_tag=arm_tag,
                    motion_tag=(
                        f"{motion_tag}_then_home"
                        if pid == self._last_assembly_pid
                        else motion_tag
                    ),
                    mot_data=anim_md,
                    motion_segments=motion_segments,
                    obj_pose_per_segment=obj_pose_per_segment,
                )
                return sm, ""

        return None, last_err or f"{pid}: all arm/motion candidates failed"

    def _backup_robot_state(self):
        """备份机械臂状态。失败后恢复，避免失败尝试污染后续步骤。"""
        for arm in iter_layout_arms(self.robot, single_arm=self.single_arm_mode):
            try:
                arm.backup_state()
            except Exception:
                pass

    def _restore_robot_state(self):
        """恢复机械臂状态。"""
        for arm in iter_layout_arms(self.robot, single_arm=self.single_arm_mode):
            try:
                arm.restore_state()
            except Exception:
                pass

    def _attach_assembled_solid(self, pid: str, rgba=None):
        """把已经装好的零件以实心模型显示在 goal pose。

        用于 preassembled 件，例如 base_plate。
        """
        if pid not in self.goal_models:
            return None
        try:
            solid = self.goal_models[pid].copy()
        except Exception:
            solid = make_collision_model(self.asm.model_path(pid), cdprim_type=self.cdprim_type)
            gp, gr = self.world_poses[pid]
            solid.pos = np.asarray(gp, dtype=float)
            solid.rotmat = np.asarray(gr, dtype=float)

        try:
            solid.rgba = np.asarray(rgba if rgba is not None else [0.35, 0.75, 0.45, 0.88], dtype=float)
        except Exception:
            pass

        solid._sealp_part_id = pid
        solid._sealp_role = "assembled_at_goal_visual"
        solid.attach_to(self.base)
        return solid

    def execute_until_failure(self) -> ExecutionSummary:
        """按 asmdef 顺序尽力规划所有步骤。

        与旧版不同：
            - 某一步失败后不会停止；
            - 失败零件不会加入 placed，也不会从 staging 可视化中移除；
            - 后续零件仍继续尝试规划；
            - 最终播放所有成功规划出来的步骤动画。

        动态障碍物规则仍然保持：
            已成功装配的零件 -> goal pose，作为障碍；
            未成功装配的零件 -> staging pose，作为障碍；
            当前正在尝试的零件 -> 不作为静态障碍。
        """
        print("========== Layout Sequence Execute All Possible Steps ==========")
        print(f"assembly_station = {np.round(self.fixture_pos, 4).tolist()}")
        print(f"part_order       = {self.part_order}")
        print(f"cdprim_type      = {self.cdprim_type}")
        print("continue_on_fail = True")
        print(f"single_arm_mode  = {self.single_arm_mode}  # config robot.mode=single 时只规划左臂")

        placed = set()
        success_steps: List[StepMotion] = []
        failed_steps: List[FailedStep] = []

        attach_goal_ghosts(self.base, self.asm, self.world_poses, self.cdprim_type)
        self.staging_visuals = attach_staging_visuals(
            self.base, self.asm, self.layout, self.part_order, self.cdprim_type
        )

        self.static_robot_mesh = self.robot.gen_meshmodel(alpha=1.0)
        self.static_robot_mesh.attach_to(self.base)

        step_id_lookup = {s.part_id: s.step_id for s in self.asm.steps}

        for step in self.asm.steps:
            pid = step.part_id
            if pid not in self.part_order:
                continue

            sid = getattr(step, "step_id", step_id_lookup.get(pid, len(success_steps)))

            if _is_preassembled(pid, self.layout):
                print(
                    f"[SKIP/PREASSEMBLED] step={sid} pid={pid}: "
                    f"layout 中标记为已装配，直接加入 goal 动态障碍，并以实心模型显示。"
                )
                if pid in self.staging_visuals:
                    try:
                        self.staging_visuals[pid].detach()
                    except Exception:
                        pass

                self._attach_assembled_solid(
                    pid,
                    rgba=np.array([0.45, 0.70, 0.45, 0.92]),
                )

                placed.add(pid)
                continue

            # 关键修改：该 step 失败后，恢复机械臂状态，然后继续尝试后续零件。
            self._backup_robot_state()
            sm, err = self._try_plan_step(pid, placed)

            if sm is None:
                self._restore_robot_state()

                print("" + "=" * 70)
                print(f"[FAIL/CONTINUE] step={sid} pid={pid} 规划失败，但继续尝试后续零件。")
                print(f"reason: {err}")
                print("说明：该零件仍保留在 staging 位置，不加入 placed；后续规划会继续把它当作未装零件障碍。")
                print("=" * 70)

                failed_steps.append(FailedStep(
                    step_id=sid,
                    part_id=pid,
                    reason=str(err),
                ))
                continue

            sm.step_id = sid
            success_steps.append(sm)
            print(f"[SUCCESS] step={sid} pid={pid} 规划成功，加入已装配集合。")

            # 注意：这里不再立刻 detach staging / attach goal。
            # 仿照 LRMate200id_ppp_animation.py：动画开始时零件仍在真实 layout 初始位置；
            # 进入 CARRY 帧时由动画状态机 detach，动作结束后再固定到 goal pose。
            placed.add(pid)

        print("" + "=" * 70)
        print("[DONE] 已遍历所有步骤。")
        print(f"成功步骤数: {len(success_steps)}")
        print(f"失败步骤数: {len(failed_steps)}")
        print("=" * 70)

        if failed_steps:
            last = failed_steps[-1]
            return ExecutionSummary(
                success_steps=success_steps,
                failed_steps=failed_steps,
                failed_step_id=last.step_id,
                failed_part_id=last.part_id,
                failed_reason=last.reason,
            )

        # 不在规划结束时提前“瞬移”到 HOME；最后一件的轨迹本身已经包含
        # collision-free HOME 段，动画播放到末尾后再由状态机确认 HOME 姿态。
        return ExecutionSummary(
            success_steps=success_steps,
            failed_steps=[],
        )



# ============================================================
# 运动轨迹缓存：全部成功后保存，下次直接重建动画，不重新规划
# ============================================================

def _file_md5(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()
    except Exception:
        return ""


def _default_motion_cache_path(layout_path: str) -> str:
    name = os.path.splitext(os.path.basename(layout_path))[0]
    return os.path.join(DEFAULT_MOTION_CACHE_DIR, f"{name}_motions.pkl")


def _motion_cache_fingerprint(
    asmdef_path: str,
    layout_path: str,
    grasp_dir: str,
    cdprim_type: str,
    enable_middle_plate_regrasp: bool,
    handover_dir: str,
    grasp_map: Optional[Dict[str, str]] = None,
    contact_exclusion_map: Optional[Dict[str, List[str]]] = None,
    *,
    single_arm_mode: bool = False,
) -> str:
    """用关键输入生成缓存指纹，避免 layout/asmdef 改了还误用旧路径。"""
    meta = {
        "format_version": MOTION_CACHE_FORMAT_VERSION,
        "asmdef_md5": _file_md5(asmdef_path),
        "layout_md5": _file_md5(layout_path),
        "grasp_dir": os.path.abspath(grasp_dir),
        "cdprim_type": str(cdprim_type),
        "middle_plate_handover": bool(enable_middle_plate_regrasp),
        "handover_dir": os.path.abspath(handover_dir),
        "grasp_map": grasp_map or {},
        "contact_exclusion_map": contact_exclusion_map or {},
        "single_arm_mode": bool(single_arm_mode),
        "pick_depart_dist": float(PICK_DEPART_DIST),
        "place_depart_dist": float(PLACE_DEPART_DIST),
        "skip_place_depart_last": True,
        "trim_final_depart_tail": True,
        "final_release_hold_frames": int(FINAL_RELEASE_HOLD_FRAMES),
        "final_home_max_clearance_frames": int(FINAL_HOME_MAX_CLEARANCE_FRAMES),
        "final_home_rrt_max_time": float(FINAL_HOME_RRT_MAX_TIME),
    }
    return hashlib.md5(
        json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _serialize_obj_pose_entry(entry) -> Optional[List]:
    """把一帧的 (pos, rotmat) 变成可 pickle 的 list；None 保持 None。"""
    if entry is None:
        return None
    try:
        pos, rot = entry
        return [
            np.asarray(pos, dtype=float).tolist(),
            np.asarray(rot, dtype=float).tolist(),
        ]
    except Exception:
        return None


def _deserialize_obj_pose_entry(entry) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if entry is None:
        return None
    try:
        pos, rot = entry
        return (
            np.asarray(pos, dtype=float),
            np.asarray(rot, dtype=float),
        )
    except Exception:
        return None


def _serialize_motion_data(md, obj_pose_list: Optional[List] = None) -> dict:
    jv_list = getattr(md, "jv_list", None) or []
    ev_list = _extract_ev_list(md)
    if len(ev_list) < len(jv_list):
        ev_list = ev_list + [None] * (len(jv_list) - len(ev_list))

    # 对齐 obj_pose_list 长度。少了用 None 填，超了截掉，避免后期 zip 越界。
    obj_pose_list = list(obj_pose_list or [])
    if len(obj_pose_list) < len(jv_list):
        obj_pose_list = obj_pose_list + [None] * (len(jv_list) - len(obj_pose_list))
    elif len(obj_pose_list) > len(jv_list):
        obj_pose_list = obj_pose_list[: len(jv_list)]

    return {
        "arm_side": _robot_arm_side(getattr(md, "robot", None)),
        "jv_list": [np.asarray(jv, dtype=float).tolist() for jv in jv_list],
        "ev_list": [_serialize_ev(ev) for ev in ev_list],
        "obj_pose_list": [_serialize_obj_pose_entry(p) for p in obj_pose_list],
    }


def _infer_carry_start_for_step(sm: StepMotion, runner) -> Optional[int]:
    """保存缓存时记录 CARRY 起点，便于缓存回放仍接近 LRMate 动画效果。"""
    try:
        mesh_list = getattr(sm.mot_data, "mesh_list", []) or []
        st_model = getattr(runner, "staging_models", {}).get(sm.part_id)
        staging_pos = _safe_cm_pos(st_model)
        if staging_pos is None:
            st = runner.layout.staging_positions.get(sm.part_id)
            if st is not None:
                staging_pos = np.asarray(st[0], dtype=float).reshape(3)
        cs = _detect_carry_start_by_pose(mesh_list, staging_pos)
        return 0 if cs is None and mesh_list else cs
    except Exception:
        return None


def _serialize_step_motion(sm: StepMotion, runner=None) -> dict:
    segments = []
    src = sm.motion_segments if sm.motion_segments else [sm.mot_data]
    obj_pose_per_seg = list(sm.obj_pose_per_segment or [])
    for idx, md in enumerate(src):
        if md is None:
            continue
        if getattr(md, "jv_list", None):
            obj_pose_seg = obj_pose_per_seg[idx] if idx < len(obj_pose_per_seg) else None
            segments.append(_serialize_motion_data(md, obj_pose_list=obj_pose_seg))

    carry_start = _infer_carry_start_for_step(sm, runner) if runner is not None else None
    # 真机判定: 每段轨迹绑定 arm_side(lft/rgt)。若一步里出现 >=2 个不同手臂,
    # 即真机需要"换手"(sender 夹住->receiver 夹住->sender 松开); 否则真机单臂即可。
    arm_sides = [s.get("arm_side") for s in segments if s.get("arm_side") is not None]
    distinct_arms = [a for i, a in enumerate(arm_sides) if a not in arm_sides[:i]]
    is_handover = len(set(arm_sides)) > 1
    return {
        "step_id": int(sm.step_id),
        "part_id": sm.part_id,
        "arm_tag": sm.arm_tag,
        "motion_tag": sm.motion_tag,
        "segments": segments,
        "carry_start": carry_start,
        "arm_sides": arm_sides,
        "handover": bool(is_handover),
        "arm_sequence": distinct_arms,
    }


def _is_full_success(asm: AssemblyDef, layout: WorkspaceLayout, part_order: List[str],
                     summary: ExecutionSummary) -> bool:
    if summary.failed_steps:
        return False
    required = set()
    for step in asm.steps:
        pid = step.part_id
        if pid not in part_order:
            continue
        if _is_preassembled(pid, layout):
            continue
        required.add(pid)
    got = {sm.part_id for sm in summary.success_steps}
    return required.issubset(got)


def save_motion_cache(
    cache_path: str,
    fingerprint: str,
    asmdef_path: str,
    layout_path: str,
    summary: ExecutionSummary,
    runner=None,
    extra_meta: Optional[dict] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    payload = {
        "format_version": MOTION_CACHE_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "asmdef_path": os.path.abspath(asmdef_path),
        "layout_path": os.path.abspath(layout_path),
        "meta": extra_meta or {},
        "steps": [_serialize_step_motion(sm, runner=runner) for sm in summary.success_steps],
    }
    # 真机执行说明: 标注每步用哪只手, 以及哪些步是"真换手"(真机需双臂交接)。
    any_handover = False
    for st in payload["steps"]:
        if st.get("handover"):
            any_handover = True
    payload["meta"]["any_handover"] = bool(any_handover)

    with open(cache_path, "wb") as fh:
        pickle.dump(payload, fh)

    print(f"[CACHE/SAVE] 全部可规划步骤成功，已保存运动路径 -> {cache_path}")
    print(f"[CACHE/SAVE] steps={len(payload['steps'])}")
    print("[CACHE/SAVE] ===== 真机执行说明(每段轨迹绑定 arm_side) =====")
    for st in payload["steps"]:
        seq = st.get("arm_sequence") or st.get("arm_sides") or []
        if st.get("handover"):
            print(f"  step={st['step_id']:>2} pid={st['part_id']:14s} "
                  f"*** 真换手: 真机需双臂交接 {('->'.join(map(str, seq)))} "
                  f"(receiver 先夹住再让 sender 松开, 当前缓存未编码握手同步, 真机慎用)")
        else:
            arm = (seq[0] if seq else "?")
            print(f"  step={st['step_id']:>2} pid={st['part_id']:14s} "
                  f"单臂={arm}  (真机单臂执行, 不换手)")
    if any_handover:
        print("[CACHE/SAVE] 注意: 存在真换手步骤, 真机直接重放需补充双臂握手同步逻辑。")
    else:
        print("[CACHE/SAVE] 全部为单臂步骤: 真机按 arm_side 逐段重放即可, 不会换手。")
    return cache_path


def load_motion_cache(
    cache_path: str,
    expected_fingerprint: str,
    *,
    strict: bool = False,
) -> Optional[dict]:
    """Load motion pkl for playback.

    By default (``strict=False``) any readable non-empty cache is accepted even
    when fingerprint/format_version differ — only ``--replan`` forces a full
    re-search. Pass ``strict=True`` (CLI: ``--strict-cache``) to reject stale
    caches like the old behaviour.
    """
    if not cache_path or not os.path.isfile(cache_path):
        return None

    try:
        with open(cache_path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as e:
        print(f"[CACHE/LOAD] 读取缓存失败，忽略: {type(e).__name__}: {e}")
        return None

    if not payload.get("steps"):
        print(f"[CACHE/LOAD] 缓存为空，忽略: {cache_path}")
        return None

    cached_fp = payload.get("fingerprint")
    cached_ver = payload.get("format_version")
    fp_ok = cached_fp == expected_fingerprint
    ver_ok = cached_ver == MOTION_CACHE_FORMAT_VERSION

    if not ver_ok:
        msg = (
            f"[CACHE/LOAD] 缓存 format_version 不匹配: cached={cached_ver!r} "
            f"expected={MOTION_CACHE_FORMAT_VERSION!r}"
        )
        if strict:
            print(msg + "，忽略")
            return None
        print(msg + "；仍尝试回放")

    if not fp_ok:
        msg = (
            f"[CACHE/LOAD] 缓存 fingerprint 不匹配: {cache_path}\n"
            f"  expected={expected_fingerprint}\n"
            f"  cached  ={cached_fp}"
        )
        if strict:
            print(msg + "\n  -> 忽略( strict-cache )")
            return None
        print(msg + "\n  -> 仍使用该 pkl 回放；若要重算请加 --replan")

    print(f"[CACHE/LOAD] 命中缓存 -> {cache_path}  steps={len(payload['steps'])}")
    return payload


def _make_cached_dual_frame(
    runner,
    lft_jv: np.ndarray,
    rgt_jv: np.ndarray,
    lft_ee=None,
    rgt_ee=None,
    active_sides=None,
):
    """缓存回放时用左右臂关节角 + 夹爪宽度重建一帧动画。

    早期版本只设 jnt_values，gripper 始终维持默认 jaw width，回放里看不到
    夹爪开合。这里把 ev (jaw width) 也灌进去，pick 之后会合上、place 之后会松开。

    性能优化(默认开启): 单臂步骤只渲染正在动的那只手臂(``active_sides``)，
    另一只手臂由常驻的半透明 ghost 机器人代为显示, 从而把每帧几何量减半,
    显著缓解卡顿。需要每帧都画整台双臂时设 ``runner.anime_full_robot=True``
    (CLI: ``--anime-full-robot``)。
    """
    # 两臂关节角始终更新(保证 ghost/真换手渲染时位姿正确)。
    _goto_arm(runner.robot.lft_arm, lft_jv, lft_ee)
    if runner.robot.rgt_arm is not None:
        _goto_arm(runner.robot.rgt_arm, rgt_jv, rgt_ee)

    full_robot = bool(getattr(runner, "anime_full_robot", False))
    sides = set(active_sides) if active_sides else set()
    # 只有一只手臂参与这步 -> 只画那只手臂(轻量)。
    if not full_robot and len(sides) == 1:
        arm_tag = "rgt" if "rgt" in sides else "lft"
        arm = get_layout_arm(
            runner.robot,
            arm_tag,
            single_arm=bool(getattr(runner, "single_arm_mode", False)),
        )
        mesh = _gen_arm_mesh(arm, alpha=1.0)
        if mesh is not None:
            return mesh
        # 退回整机渲染。

    try:
        return runner.robot.gen_meshmodel(alpha=1.0)
    except Exception:
        return None


def _rebuild_mot_data_from_segments(
    step_payload: dict,
    runner,
) -> _AnimMotionData:
    """从 pkl 中重建动画数据。

    关键优化：不再一次性生成所有 mesh。这里只把缓存里的 jv/ev/obj_pose
    展开成轻量 frame_states，真正的 mesh 在动画播放到该帧时才生成。
    """
    part_id = step_payload["part_id"]
    segments = step_payload.get("segments") or []

    st_model = runner.staging_models.get(part_id)
    if st_model is not None:
        start_pose = (np.asarray(st_model.pos, dtype=float), np.asarray(st_model.rotmat, dtype=float))
    else:
        st = runner.layout.staging_positions.get(part_id)
        if st is None:
            # 常见原因: bash 用了反引号导致 --layout 没传进去，默认 layout 与 pkl 零件不一致。
            layout_pids = sorted(runner.layout.staging_positions.keys())
            raise ValueError(
                f"缓存步骤 part_id={part_id!r} 在当前 layout 的 staging 中不存在。"
                f" layout_staging={layout_pids}。"
                f" 请确认 --layout 指向与该 pkl 匹配的 .layout"
                f"（bash 换行请用 \\，不要用反引号 `）。"
            )
        start_pose = (np.asarray(st[0], dtype=float), np.asarray(st[1], dtype=float))

    if part_id not in runner.world_poses:
        raise ValueError(
            f"缓存步骤 part_id={part_id!r} 在当前 asmdef/layout 的 goal poses 中不存在。"
            f" 请确认 --asmdef/--layout 与 motion cache 匹配。"
        )
    gp, gr = runner.world_poses[part_id]
    goal_pose = (np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))

    # 让不同 step 之间的缓存回放保持关节 + 夹爪宽度连续。
    lft_jv = np.asarray(getattr(runner, "_cache_replay_lft_jv", np.asarray(HOME_JV, dtype=float)), dtype=float)
    rgt_jv = np.asarray(getattr(runner, "_cache_replay_rgt_jv", np.asarray(HOME_JV, dtype=float)), dtype=float)
    lft_ee = getattr(runner, "_cache_replay_lft_ee", None)
    rgt_ee = getattr(runner, "_cache_replay_rgt_ee", None)

    total_frames = sum(len(seg.get("jv_list") or []) for seg in segments)
    carry_start = step_payload.get("carry_start")
    if carry_start is None:
        carry_start = 0 if total_frames > 0 else None

    # 这步实际用到的手臂集合: 单臂步骤 -> 每帧只渲染该臂(减半几何, 缓解卡顿)。
    step_sides = {
        seg.get("arm_side", "lft")
        for seg in segments
        if seg.get("jv_list")
    }

    frame_states: List[dict] = []
    global_i = 0

    for seg in segments:
        side = seg.get("arm_side", "lft")
        jv_list = seg.get("jv_list") or []
        ev_list = seg.get("ev_list") or []
        obj_pose_list = seg.get("obj_pose_list") or []

        for local_i, jv in enumerate(jv_list):
            jv_arr = np.asarray(jv, dtype=float)
            ev = _deserialize_ev(ev_list[local_i]) if local_i < len(ev_list) else None
            if side == "rgt":
                rgt_jv = jv_arr
                if ev is not None:
                    rgt_ee = ev
            else:
                lft_jv = jv_arr
                if ev is not None:
                    lft_ee = ev

            # 物体可视化：优先用规划时记录的逐帧真实位姿；
            # 实在缺失才退回 staging→goal 直线插值兜底。
            obj_pose = None
            if local_i < len(obj_pose_list):
                obj_pose = _deserialize_obj_pose_entry(obj_pose_list[local_i])

            if obj_pose is None and carry_start is not None and global_i >= int(carry_start):
                denom = max(1, total_frames - int(carry_start) - 1)
                t = (global_i - int(carry_start)) / float(denom)
                sp, sr = start_pose
                gp_, gr_ = goal_pose
                obj_pose = (
                    (1.0 - t) * np.asarray(sp, dtype=float) + t * np.asarray(gp_, dtype=float),
                    _interp_rotmat(sr, gr_, t),
                )

            frame_states.append({
                "lft_jv": np.asarray(lft_jv, dtype=float).copy(),
                "rgt_jv": np.asarray(rgt_jv, dtype=float).copy(),
                "lft_ee": copy.deepcopy(lft_ee),
                "rgt_ee": copy.deepcopy(rgt_ee),
                "active_sides": set(step_sides),
                "obj_pose": obj_pose,
            })
            global_i += 1

    runner._cache_replay_lft_jv = np.asarray(lft_jv, dtype=float)
    runner._cache_replay_rgt_jv = np.asarray(rgt_jv, dtype=float)
    runner._cache_replay_lft_ee = lft_ee
    runner._cache_replay_rgt_ee = rgt_ee

    mesh_list = _LazyCachedMeshList(
        runner=runner,
        part_id=part_id,
        frame_states=frame_states,
        max_keep=int(getattr(runner, "lazy_cache_keep_frames", 1)),
    )
    md = _AnimMotionData(mesh_list=mesh_list)
    md.cached_carry_start = carry_start
    md.lazy_cache_playback = True
    return md


def rebuild_summary_from_cache(payload: dict, runner) -> ExecutionSummary:
    print("[CACHE/LAZY] 使用惰性回放：只保存关节帧，按 SPACE 到达时再生成 mesh。")
    runner._cache_replay_lft_jv = np.asarray(HOME_JV, dtype=float)
    runner._cache_replay_rgt_jv = np.asarray(HOME_JV, dtype=float)
    runner._cache_replay_lft_ee = None
    runner._cache_replay_rgt_ee = None

    success_steps: List[StepMotion] = []
    for step in payload.get("steps", []):
        mot_data = _rebuild_mot_data_from_segments(step, runner)
        success_steps.append(StepMotion(
            step_id=int(step.get("step_id", -1)),
            part_id=step["part_id"],
            arm_tag=step.get("arm_tag", ""),
            motion_tag=step.get("motion_tag", ""),
            mot_data=mot_data,
            motion_segments=None,
        ))

    return ExecutionSummary(success_steps=success_steps, failed_steps=[])


def setup_scene_for_cached_playback(runner) -> None:
    """缓存回放时搭建与正常规划一致的初始场景。"""
    attach_goal_ghosts(runner.base, runner.asm, runner.world_poses, runner.cdprim_type)
    runner.staging_visuals = attach_staging_visuals(
        runner.base, runner.asm, runner.layout, runner.part_order, runner.cdprim_type
    )

    runner.static_robot_mesh = runner.robot.gen_meshmodel(alpha=1.0)
    runner.static_robot_mesh.attach_to(runner.base)

    for pid in runner.part_order:
        if _is_preassembled(pid, runner.layout):
            if pid in runner.staging_visuals:
                try:
                    runner.staging_visuals[pid].detach()
                except Exception:
                    pass
            runner._attach_assembled_solid(
                pid,
                rgba=np.array([0.45, 0.70, 0.45, 0.92]),
            )


# ============================================================
# CLI
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Read .layout, execute assembly sequence with dynamic obstacles, and visualize successful motions."
    )
    parser.add_argument("--asmdef", default=DEFAULT_ASMDEF, help="asmdef 文件路径")
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, help=".layout 文件路径")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="sample_config.yaml")
    parser.add_argument("--grasp-dir", default=DEFAULT_GRASP_DIR, help="grasp pickle 文件夹")
    parser.add_argument(
        "--grasp-map-json",
        default="",
        help="可选，显式 grasp 映射，例如 '{\"post\":\"tower_post_grasps.pickle\"}'",
    )
    parser.add_argument(
        "--cdprim-type",
        default=DEFAULT_CDPRIM_TYPE,
        help="碰撞模型类型，默认 box(AABB 包围盒)。更精细可填 triangles / convex_hull。",
    )
    parser.add_argument(
        "--contact-exclusion-json",
        default="",
        help=(
            "可选，额外接触豁免表。规划某零件时临时排除这些已装件，"
            "例如 '{\"top_cross\":[\"middle_plate\"]}'。direct parent 会自动排除。"
        ),
    )
    parser.add_argument(
        "--no-middle-plate-regrasp",
        action="store_true",
        help="[已默认] middle_plate 用单臂 pick-place。换手现已默认取消, 本参数保留向后兼容。",
    )
    parser.add_argument(
        "--middle-plate-handover",
        action="store_true",
        help="恢复 middle_plate 双臂换手(默认取消; 之前换手其实退化成单臂且较慢)。",
    )
    parser.add_argument(
        "--handover-dir",
        default=DEFAULT_HANDOVER_DIR,
        help="middle_plate 换手 hopg 目录(仅 --middle-plate-handover 时使用)",
    )
    parser.add_argument(
        "--no-auto-play",
        action="store_true",
        help="[已废弃] 当前播放固定为 SPACE 逐帧驱动，本参数保留仅为向后兼容，不再生效。",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.03,
        help="[已废弃] 当前不再使用 task tick，本参数保留仅为向后兼容，不再生效。",
    )
    parser.add_argument(
        "--motion-cache",
        default="",
        help="运动轨迹缓存 pkl 路径；默认 sealp/examples/motion/_output/{layout名}_motions.pkl",
    )
    parser.add_argument(
        "--replan",
        action="store_true",
        help="忽略已有缓存，重新规划全部步骤",
    )
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help="fingerprint/format_version 不匹配时拒绝使用已有 pkl(旧默认行为)。"
             "默认只要 pkl 存在且可读就直接回放，只有 --replan 才重算。",
    )
    parser.add_argument(
        "--no-save-cache",
        action="store_true",
        help="即使全部成功也不写入缓存",
    )
    parser.add_argument(
        "--dual-arm-overlay",
        action="store_true",
        help="给每一帧叠加另一只静止手臂的完整 mesh(默认关)。"
             "开启会让每帧几何近乎翻倍，容易卡顿/黑屏崩溃；"
             "正在运动的那只手臂本来就会画出，通常无需开启。",
    )
    parser.add_argument(
        "--anime-stride",
        type=int,
        default=1,
        help="动画每按一次 SPACE 推进的帧数(>1 可显著减少重型 mesh 上传次数、缓解卡顿)。",
    )
    parser.add_argument(
        "--lazy-cache-keep",
        type=int,
        default=1,
        help="缓存回放时最多保留最近多少帧 mesh。默认 1，显著降低显存/内存占用；调大可减少重复生成。",
    )
    parser.add_argument(
        "--anime-full-robot",
        action="store_true",
        help="缓存回放时每帧都渲染整台双臂(默认关)。默认单臂步骤只画运动臂以缓解卡顿，"
             "另一只手臂由常驻 ghost 机器人显示; 开启此项恢复整机渲染(更重、更卡)。",
    )
    parser.add_argument(
        "--debug-obstacles",
        action="store_true",
        help="调试用：打印当前 step 真正传入规划器的 transit_obs / placement_obs 障碍物列表。",
    )
    parser.add_argument(
        "--debug-part",
        default="",
        help="只打印指定零件的障碍物，例如 top_cross；为空则所有零件都打印。",
    )
    parser.add_argument(
        "--debug-free-part",
        default="",
        help=(
            "诊断用：对指定零件强制清空 transit_obs 和 placement_obs，"
            "用于判断 No common grasp 是否由障碍物导致。只建议配合 --no-save-cache 使用。"
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # 换手默认取消(单臂)。仅当显式 --middle-plate-handover 且未 --no-middle-plate-regrasp 时启用。
    mp_handover = bool(getattr(args, "middle_plate_handover", False)) and not args.no_middle_plate_regrasp

    _patch_rrt()

    asmdef_path = os.path.abspath(args.asmdef)
    layout_path = os.path.abspath(args.layout)
    config_path = os.path.abspath(args.config)
    grasp_dir = os.path.abspath(args.grasp_dir)

    if not os.path.isfile(asmdef_path):
        raise FileNotFoundError(f"asmdef 不存在: {asmdef_path}")
    if not os.path.isfile(layout_path):
        raise FileNotFoundError(f"layout 不存在: {layout_path}")

    print("=" * 78)
    print("Execute Layout Sequence Visualizer [v5 middle_plate handover]")
    print(f"asmdef     = {asmdef_path}")
    print(f"layout     = {layout_path}")
    print(f"config     = {config_path}")
    print(f"grasp_dir  = {grasp_dir}")
    print(f"cdprim     = {args.cdprim_type}  # 默认 box(AABB)")
    ws = load_workspace_settings(config_path)
    single_arm_mode = bool(ws.get("single_arm", False))
    print(f"robot_mode         = {'single(lft only)' if single_arm_mode else 'dual'}  # from config robot.mode")
    print(f"middle_plate_handover = {mp_handover}  # 默认 False(单臂); 用 --middle-plate-handover 恢复")
    if mp_handover:
        print(f"handover_dir = {os.path.abspath(args.handover_dir)}")
    print(f"dual_arm_overlay = {args.dual_arm_overlay}  # 默认 False(减压); --replan 时才会写入新缓存")
    print(f"anime_stride = {max(1, int(args.anime_stride))}  # 每按一次 SPACE 推进的帧数")
    print(f"anime_full_robot = {bool(getattr(args, 'anime_full_robot', False))}  "
          f"# 默认 False(单臂步骤只画运动臂, 缓存回放更顺); True 恢复整机渲染")
    print("=" * 78)

    asm = AssemblyDef.load(asmdef_path)
    layout = WorkspaceLayout.load(layout_path)

    print(f"Loaded asm    : {asm.name}, n_parts={asm.n_parts}, n_steps={asm.n_steps}")
    print(f"Loaded layout : {layout.name}")
    print(f"assembly pos  : {np.round(layout.assembly_station_pos, 4).tolist()}")
    print(f"metadata keys : {list((layout.metadata or {}).keys())}")

    base = wd.World(
        cam_pos=[1.05, -1.25, 0.85],
        lookat_pos=np.asarray(layout.assembly_station_pos, dtype=float) + np.array([0, 0, 0.10]),
    )
    # 与 LRMate200id_ppp_animation.py 保持一致：让 RRT/PPP 内部能拿到 ShowBase，
    # 从而生成完整 mesh_list，避免动画段缺帧或直接跳过。
    sys.modules['__main__'].base = base
    mgm.gen_frame(pos=np.asarray(layout.assembly_station_pos, dtype=float), ax_length=0.10).attach_to(base)

    runner = LayoutSequenceVisualizer(
        asm=asm,
        layout=layout,
        config_path=config_path,
        grasp_dir=grasp_dir,
        base=base,
        cdprim_type=args.cdprim_type,
        grasp_map=_load_json_map(args.grasp_map_json),
        contact_exclusion_map=_load_json_map(args.contact_exclusion_json),
        enable_middle_plate_regrasp=mp_handover,
        handover_dir=os.path.abspath(args.handover_dir),
        dual_arm_overlay=args.dual_arm_overlay,
    )
    runner.debug_obstacles = bool(getattr(args, "debug_obstacles", False))
    runner.debug_part = str(getattr(args, "debug_part", "") or "").strip()
    runner.debug_free_part = str(getattr(args, "debug_free_part", "") or "").strip()
    runner._debug_obstacles_printed = set()
    if runner.debug_obstacles:
        print(f"debug_obstacles = True  debug_part = {runner.debug_part or '<all>'}")
    if runner.debug_free_part:
        print(f"debug_free_part = {runner.debug_free_part}  # 诊断模式：该零件规划时强制清空所有障碍物")

    # 缓存回放渲染策略: 默认单臂步骤只画运动臂(减压); --anime-full-robot 恢复整机。
    runner.anime_full_robot = bool(getattr(args, "anime_full_robot", False))
    runner.lazy_cache_keep_frames = max(0, int(getattr(args, "lazy_cache_keep", 1)))
    print(f"lazy_cache_playback = True  # pkl 回放按需生成 mesh，避免一次性构建全部帧")
    print(f"lazy_cache_keep     = {runner.lazy_cache_keep_frames}  # 最近保留帧数")

    grasp_map = _load_json_map(args.grasp_map_json)
    contact_exclusion_map = _load_json_map(args.contact_exclusion_json)
    cache_path = os.path.abspath(args.motion_cache) if args.motion_cache else _default_motion_cache_path(layout_path)
    fingerprint = _motion_cache_fingerprint(
        asmdef_path=asmdef_path,
        layout_path=layout_path,
        grasp_dir=grasp_dir,
        cdprim_type=args.cdprim_type,
        enable_middle_plate_regrasp=mp_handover,
        handover_dir=os.path.abspath(args.handover_dir),
        grasp_map=grasp_map,
        contact_exclusion_map=contact_exclusion_map,
        single_arm_mode=single_arm_mode,
    )
    print(f"motion_cache = {cache_path}")
    print(f"cache_fp     = {fingerprint}")
    if args.replan:
        print("[CACHE] --replan: 强制重新规划，忽略已有 pkl")
    elif os.path.isfile(cache_path):
        print("[CACHE] 未指定 --replan: 优先使用已有 pkl( fingerprint 不匹配也会回放 )")
    else:
        print("[CACHE] pkl 不存在，将执行规划")

    summary: ExecutionSummary
    loaded_from_cache = False
    if not args.replan:
        cached = load_motion_cache(
            cache_path,
            fingerprint,
            strict=bool(getattr(args, "strict_cache", False)),
        )
        if cached is not None:
            summary = rebuild_summary_from_cache(cached, runner)
            setup_scene_for_cached_playback(runner)
            loaded_from_cache = True
            print("[CACHE/LOAD] 跳过规划，直接使用缓存关节路径重建动画。")

    if not loaded_from_cache:
        if not args.replan and os.path.isfile(cache_path):
            print("[CACHE] 无法加载已有 pkl，回退到重新规划")
        summary = runner.execute_until_failure()
        if (
            not args.no_save_cache
            and _is_full_success(asm, layout, runner.part_order, summary)
        ):
            save_motion_cache(
                cache_path=cache_path,
                fingerprint=fingerprint,
                asmdef_path=asmdef_path,
                layout_path=layout_path,
                summary=summary,
                runner=runner,
                extra_meta={
                    "cdprim_type": args.cdprim_type,
                    "middle_plate_handover": mp_handover,
                },
            )
        elif summary.failed_steps:
            print("[CACHE/SAVE] 存在失败步骤，不写入缓存。")
        elif args.no_save_cache:
            print("[CACHE/SAVE] 已跳过保存（--no-save-cache）。")

    print("\n========== 执行总结 ==========")
    print(f"成功步骤数: {len(summary.success_steps)}")
    for sm in summary.success_steps:
        tag = sm.motion_tag or ""
        if tag.startswith("handover"):
            via = "handover"
        elif tag.startswith("regrasp"):
            via = "regrasp"
        else:
            via = "single-arm"
        print(
            f"  [OK]   step={sm.step_id:2d} pid={sm.part_id:14s} "
            f"arm={sm.arm_tag} motion={sm.motion_tag} via={via}"
        )

    print(f"失败步骤数: {len(summary.failed_steps)}")
    for fs in summary.failed_steps:
        print(
            f"  [FAIL] step={fs.step_id:2d} pid={fs.part_id:14s} "
            f"reason={fs.reason}"
        )

    if not summary.failed_steps:
        print("\n全部步骤成功。")
    else:
        print("\n注意：失败步骤不会播放动画；成功步骤会继续播放。")

    animate_success_steps(
        base,
        summary.success_steps,
        runner=runner,
        frame_stride=args.anime_stride,
    )

    base.run()


if __name__ == "__main__":
    main()