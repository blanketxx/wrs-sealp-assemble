#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""View Initial Layout Only
===========================

只用于查看已经生成的 .layout 初始布局，不做抓取规划、不做 RRT、不播放装配动画。

功能：
1. 读取 .layout 文件；
2. 读取 .asmdef 文件；
3. 显示 46×23 真实贯穿孔洞洞板 work_table，桌面透明度固定为 1；
4. 只显示左机械臂 home 姿态，不显示右机械臂；
5. 显示每个零件的初始 staging 位置；
6. 如果某个零件是 preassembled，例如 base_plate，则以实心模型显示在装配区；
7. 可选显示最终目标装配位姿 ghost。

推荐放置路径：
    D:/Project/wrs-sealp/sealp/examples/layout/show_initial_layout_from_file.py

运行：
    python -m sealp.examples.layout.show_initial_layout_from_file

指定 layout：
    python -m sealp.examples.layout.show_initial_layout_from_file ^
      --layout D:/Project/wrs-sealp/sealp/examples/layout/_output/tower_optimal_initial.layout ^
      --asmdef D:/Project/wrs-sealp/sealp/assembly_sequence/_demo_output/topdown_tower.asmdef
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh as trm
import yaml

from wrs import wd, mgm, mcm

_THIS_FILE = os.path.abspath(__file__)
_THIS_DIR = os.path.dirname(_THIS_FILE)


def _find_sealp_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.basename(cur) == "sealp":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(os.path.join(start_dir, "..", ".."))
        cur = parent


SEALP_ROOT = _find_sealp_root(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SEALP_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sealp.assembly_sequence import AssemblyDef
from sealp.layout import WorkspaceLayout
from sealp.config import load_config
from sealp.colliders import StaticEnvironment
from sealp.layout._viz_common import load_table_box


# 兼容旧版/搜索脚本生成的 .layout：
# 有些 .layout 文件里会把 tuple 保存成 YAML 的 !!python/tuple 标签。
# WorkspaceLayout.load 内部使用 yaml.safe_load，默认不认识这个标签，
# 所以这里只给 SafeLoader 补一个 tuple 构造器，不改成 unsafe_load。
def _enable_yaml_tuple_safe_load():
    tag = "tag:yaml.org,2002:python/tuple"

    def _construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor(tag, _construct_python_tuple)


_enable_yaml_tuple_safe_load()

try:
    import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda
except Exception:
    pda = None

DEFAULT_ASMDEF = os.path.join(
    SEALP_ROOT, "assembly_sequence", "_demo_output", "topdown_tower.asmdef"
)
DEFAULT_LAYOUT = os.path.join(
    SEALP_ROOT, "examples", "layout", "_output", "tower_optimal_initial.layout"
)
DEFAULT_CONFIG = os.path.join(SEALP_ROOT, "config", "sample_config.yaml")

DUAL_ARM_Y_OFFSET = 0.62
HOME_JV = np.zeros(6)

# 洞洞板参数：长边 46 孔、短边 23 孔。
PERFORATED_TABLE_LONG_HOLES = 46
PERFORATED_TABLE_SHORT_HOLES = 23
PERFORATED_TABLE_HOLE_SEGMENTS = 16
PERFORATED_TABLE_HOLE_DIAMETER = None  # None: 自动取较小孔距的 42%


def make_model(mesh_path: str, rgba=None):
    cm = mcm.CollisionModel(mesh_path)
    if rgba is not None:
        cm.rgba = np.asarray(rgba, dtype=float)
    return cm


def _square_ring_points(
        half_x: float,
        half_y: float,
        radius: float,
        segments: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成一个方形单元外圈和圆孔内圈，二者顶点数一致。"""
    segments = max(8, int(math.ceil(segments / 4.0)) * 4)
    per_edge = segments // 4
    outer = []

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
        vertices,
        faces,
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
    """向网格中添加一个带真实贯穿圆孔的矩形单元。"""
    outer_xy, inner_xy = _square_ring_points(
        pitch_x / 2.0,
        pitch_y / 2.0,
        radius,
        segments,
    )
    n = len(outer_xy)
    start = len(vertices)

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

        # 顶面。
        faces.append([outer_top + i, outer_top + j, inner_top + j])
        faces.append([outer_top + i, inner_top + j, inner_top + i])

        # 底面。
        faces.append([outer_bottom + i, inner_bottom + j, outer_bottom + j])
        faces.append([outer_bottom + i, inner_bottom + i, inner_bottom + j])

        # 圆孔内壁。
        faces.append([inner_top + i, inner_bottom + j, inner_bottom + i])
        faces.append([inner_top + i, inner_top + j, inner_bottom + j])

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
) -> Tuple[trm.Trimesh, dict]:
    """按 work_table 当前尺寸生成 46×23 真实贯穿孔洞洞板。"""
    extent = np.asarray(extent, dtype=float).reshape(3)
    pos = np.asarray(pos, dtype=float).reshape(3)

    size_x, size_y, thickness = map(float, extent)

    if size_x >= size_y:
        nx = PERFORATED_TABLE_LONG_HOLES
        ny = PERFORATED_TABLE_SHORT_HOLES
    else:
        nx = PERFORATED_TABLE_SHORT_HOLES
        ny = PERFORATED_TABLE_LONG_HOLES

    pitch_x = size_x / nx
    pitch_y = size_y / ny
    min_pitch = min(pitch_x, pitch_y)

    hole_diameter = (
        0.42 * min_pitch
        if PERFORATED_TABLE_HOLE_DIAMETER is None
        else float(PERFORATED_TABLE_HOLE_DIAMETER)
    )
    radius = hole_diameter / 2.0

    x_min = pos[0] - size_x / 2.0
    y_min = pos[1] - size_y / 2.0
    z_bottom = pos[2] - thickness / 2.0
    z_top = pos[2] + thickness / 2.0

    vertices = []
    faces = []

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
                segments=PERFORATED_TABLE_HOLE_SEGMENTS,
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
        raise RuntimeError("生成的洞洞板网格不是 watertight")

    return mesh, {
        "nx": nx,
        "ny": ny,
        "hole_count": nx * ny,
        "hole_diameter": hole_diameter,
        "watertight": bool(mesh.is_watertight),
    }


def _attach_perforated_table(
        config_path: str,
        base,
):
    """显示完全不透明的洞洞板 work_table。"""
    extent, pos, rgba = load_table_box(config_path, "work_table")
    mesh, info = _build_perforated_table_mesh(extent, pos)

    cache_dir = Path(_THIS_DIR) / "_generated_meshes"
    cache_dir.mkdir(parents=True, exist_ok=True)

    extent_arr = np.asarray(extent, dtype=float)
    mesh_path = cache_dir / (
        f"work_table_perforated_{info['nx']}x{info['ny']}_"
        f"{extent_arr[0] * 1000.0:.1f}x"
        f"{extent_arr[1] * 1000.0:.1f}x"
        f"{extent_arr[2] * 1000.0:.1f}mm_"
        f"d{info['hole_diameter'] * 1000.0:.3f}mm_"
        f"seg{PERFORATED_TABLE_HOLE_SEGMENTS}.stl"
    )

    if not mesh_path.is_file():
        mesh.export(str(mesh_path))

    table = mcm.CollisionModel(str(mesh_path))

    rgba_arr = np.asarray(rgba, dtype=float).reshape(-1)
    if rgba_arr.size >= 3:
        table.rgba = np.array(
            [rgba_arr[0], rgba_arr[1], rgba_arr[2], 1.0],
            dtype=float,
        )
    else:
        table.rgba = np.array([0.55, 0.55, 0.55, 1.0])

    table.attach_to(base)

    print(
        "[TABLE] perforated work_table: "
        f"{info['nx']}x{info['ny']}={info['hole_count']} holes, "
        f"diameter={info['hole_diameter'] * 1000.0:.2f} mm, "
        f"alpha=1.0, watertight={info['watertight']}"
    )
    return table


def _load_env_obstacles(config_path: str, base) -> List:
    """加载环境，并把原实心 work_table 的显示替换成不透明洞洞板。"""
    if not config_path or not os.path.isfile(config_path):
        print(f"[WARN] config 不存在，跳过环境障碍物: {config_path}")
        return []

    cfg = load_config(config_path)
    env = StaticEnvironment(
        obstacle_defs=cfg.obstacle_defs,
        base_dir=cfg.config_dir,
    )
    obs_list = list(env.obstacle_list)

    table_extent, table_pos, _table_rgba = load_table_box(
        config_path,
        "work_table",
    )

    # 当前 sample_config 通常只有一个环境障碍物，即 work_table。
    table_idx = 0 if len(obs_list) == 1 else None

    # 多障碍物时，按中心位置寻找最接近 work_table 的障碍物。
    if table_idx is None:
        target_pos = np.asarray(table_pos, dtype=float).reshape(3)
        best_dist = float("inf")
        for idx, obs in enumerate(obs_list):
            try:
                obs_pos = np.asarray(obs.pos, dtype=float).reshape(3)
            except Exception:
                continue
            dist = float(np.linalg.norm(obs_pos - target_pos))
            if dist < best_dist:
                best_dist = dist
                table_idx = idx

    # 除桌面之外的环境障碍物保持原样显示。
    for idx, obs in enumerate(obs_list):
        if idx == table_idx:
            continue
        try:
            obs.rgba = np.array([0.55, 0.55, 0.55, 1.0])
        except Exception:
            pass
        obs.attach_to(base)

    try:
        _attach_perforated_table(config_path, base)
    except Exception as e:
        print(
            f"[WARN] 洞洞板生成失败，回退到原实心桌面: "
            f"{type(e).__name__}: {e}"
        )
        if table_idx is not None:
            try:
                obs_list[table_idx].rgba = np.array([0.55, 0.55, 0.55, 1.0])
            except Exception:
                pass
            obs_list[table_idx].attach_to(base)

    print(f"[ENV] loaded obstacles: {len(obs_list)}")
    return obs_list


def _part_order_from_layout_or_asm(asm: AssemblyDef, layout: WorkspaceLayout) -> List[str]:
    meta = getattr(layout, "metadata", {}) or {}
    order = meta.get("part_order")
    if order:
        return [p for p in order if p in asm.part_ids]

    return [
        step.part_id
        for step in asm.steps
        if step.part_id in asm.part_ids
    ]


def _is_preassembled(pid: str, layout: WorkspaceLayout, part_order: List[str]) -> bool:
    meta = getattr(layout, "metadata", {}) or {}
    arm_choice = meta.get("arm_choice", {}) or {}
    pose_tag = meta.get("pose_tag", {}) or {}

    if arm_choice.get(pid) == "preassembled":
        return True

    if str(pose_tag.get(pid, "")).startswith("preassembled"):
        return True

    if meta.get("preassemble_first_part", False) and part_order and pid == part_order[0]:
        return True

    return False


def _metadata_map(layout: WorkspaceLayout, key: str) -> Dict:
    meta = getattr(layout, "metadata", {}) or {}
    v = meta.get(key, {})
    return v if isinstance(v, dict) else {}


def attach_robot_home(base, layout: WorkspaceLayout, show_robot: bool = True):
    """只显示左机械臂，不显示右机械臂/双臂整体模型。"""
    if not show_robot:
        return None

    if pda is None:
        print("[WARN] 无法导入 DualPantheraHTNoBody，跳过机器人显示。")
        return None

    robot_base_pos = np.asarray(
        getattr(layout, "robot_base_pos", np.zeros(3)), dtype=float
    )
    robot_base_rotmat = np.asarray(
        getattr(layout, "robot_base_rotmat", np.eye(3)), dtype=float
    )

    # 仍构造双臂机器人对象，以保证模型/坐标配置与执行程序一致；
    # 但可视化时只 attach lft_arm 的 mesh，右臂完全不显示。
    robot = pda.DualPantheraHTNoBody(
        pos=robot_base_pos,
        rotmat=robot_base_rotmat,
        arm_y_offset=DUAL_ARM_Y_OFFSET,
        enable_cc=True,
    )

    try:
        robot.lft_arm.goto_given_conf(HOME_JV)
    except Exception:
        pass

    try:
        robot.lft_arm.gen_meshmodel(alpha=0.32).attach_to(base)
        print(
            f"[ROBOT] LEFT arm only, home at "
            f"{np.round(robot_base_pos, 4).tolist()}"
        )
    except Exception as e:
        print(
            f"[WARN] 左机械臂 mesh 显示失败: "
            f"{type(e).__name__}: {e}"
        )

    return robot


def attach_goal_ghosts(base, asm: AssemblyDef, world_poses: Dict, show_goal_ghosts: bool):
    if not show_goal_ghosts:
        return

    print("\n========== Goal Ghosts ==========")
    for pid, pose in world_poses.items():
        if pid not in asm.part_ids:
            continue

        mesh_path = asm.model_path(pid)
        if not os.path.isfile(mesh_path):
            print(f"[WARN] missing mesh for goal ghost: {pid}, {mesh_path}")
            continue

        gp, gr = pose
        cm = make_model(mesh_path, rgba=[0.70, 0.70, 0.70, 0.18])
        cm.pos = np.asarray(gp, dtype=float)
        cm.rotmat = np.asarray(gr, dtype=float)
        cm.attach_to(base)

        print(f"ghost {pid:14s}: pos={np.round(gp, 4).tolist()}")


def attach_initial_layout(base, asm: AssemblyDef, layout: WorkspaceLayout, part_order: List[str],
                          show_frames: bool = True):
    pose_tag = _metadata_map(layout, "pose_tag")
    rot_name = _metadata_map(layout, "rot_name")
    arm_choice = _metadata_map(layout, "arm_choice")
    grasp_counts = _metadata_map(layout, "grasp_counts")
    topdown_counts = _metadata_map(layout, "topdown_counts_identity")

    colors = [
        np.array([0.90, 0.45, 0.35, 0.88]),
        np.array([0.20, 0.60, 0.95, 0.88]),
        np.array([0.25, 0.80, 0.45, 0.88]),
        np.array([0.95, 0.60, 0.20, 0.88]),
        np.array([0.75, 0.35, 0.85, 0.88]),
        np.array([0.20, 0.85, 0.85, 0.88]),
        np.array([0.85, 0.85, 0.30, 0.88]),
        np.array([0.65, 0.65, 0.95, 0.88]),
    ]

    print("\n========== Initial Layout / Staging ==========")
    shown = []

    for i, pid in enumerate(part_order):
        if pid not in asm.part_ids:
            continue

        st = layout.staging_positions.get(pid)
        if st is None:
            print(f"[WARN] no staging pose for {pid}")
            continue

        mesh_path = asm.model_path(pid)
        if not os.path.isfile(mesh_path):
            print(f"[WARN] missing mesh for {pid}: {mesh_path}")
            continue

        pos, rot = st
        pos = np.asarray(pos, dtype=float)
        rot = np.asarray(rot, dtype=float)

        preassembled = _is_preassembled(pid, layout, part_order)

        if preassembled:
            rgba = np.array([0.40, 0.78, 0.42, 0.95])
        else:
            rgba = colors[i % len(colors)]

        cm = make_model(mesh_path, rgba=rgba)
        cm.pos = pos
        cm.rotmat = rot
        cm.attach_to(base)

        if show_frames:
            try:
                mgm.gen_frame(pos=pos, rotmat=rot, ax_length=0.04).attach_to(base)
            except Exception:
                pass

        shown.append(pid)

        print(
            f"{pid:14s}: "
            f"pos={np.round(pos, 4).tolist()} "
            f"preassembled={preassembled} "
            f"arm={arm_choice.get(pid, '-')} "
            f"pose={pose_tag.get(pid, '-')} "
            f"rot={rot_name.get(pid, '-')} "
            f"grasp={grasp_counts.get(pid, '-')} "
            f"topdown={topdown_counts.get(pid, '-')}"
        )

    print(f"\n[OK] shown parts: {shown}")


def main():
    parser = argparse.ArgumentParser(
        description="Only view initial staging layout from a .layout file."
    )
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, help=".layout 文件路径")
    parser.add_argument("--asmdef", default=DEFAULT_ASMDEF, help=".asmdef 文件路径")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="sample_config.yaml 路径")
    parser.add_argument("--show-goal-ghosts", action="store_true", help="显示最终目标 ghost；默认不显示")
    parser.add_argument("--hide-robot", action="store_true", help="不显示左机械臂 home 姿态")
    parser.add_argument("--hide-env", action="store_true", help="不显示洞洞板 work_table 等环境")
    parser.add_argument("--show-frames", action="store_true", help="显示装配中心和零件坐标系；默认不显示")
    parser.add_argument(
        "--cam-pos",
        default="1.05,-1.25,0.85",
        help="相机位置，格式 x,y,z",
    )
    args = parser.parse_args()

    layout_path = os.path.abspath(args.layout)
    asmdef_path = os.path.abspath(args.asmdef)
    config_path = os.path.abspath(args.config)

    if not os.path.isfile(layout_path):
        raise FileNotFoundError(f"layout 不存在: {layout_path}")
    if not os.path.isfile(asmdef_path):
        raise FileNotFoundError(f"asmdef 不存在: {asmdef_path}")

    asm = AssemblyDef.load(asmdef_path)
    layout = WorkspaceLayout.load(layout_path)

    part_order = _part_order_from_layout_or_asm(asm, layout)

    assembly_pos = np.asarray(layout.assembly_station_pos, dtype=float)
    assembly_rot = np.asarray(layout.assembly_station_rotmat, dtype=float)

    cam_pos = [float(x.strip()) for x in args.cam_pos.split(",")]

    print("=" * 78)
    print("View Initial Layout Only")
    print(f"asmdef      = {asmdef_path}")
    print(f"layout      = {layout_path}")
    print(f"config      = {config_path}")
    print(f"assembly    = {np.round(assembly_pos, 4).tolist()}")
    print(f"part_order  = {part_order}")
    print(f"metadata    = {list((layout.metadata or {}).keys())}")
    print("=" * 78)

    base = wd.World(
        cam_pos=cam_pos,
        lookat_pos=assembly_pos + np.array([0.0, 0.0, 0.10]),
    )

    if args.show_frames:
        mgm.gen_frame(
            pos=assembly_pos,
            rotmat=assembly_rot,
            ax_length=0.12,
        ).attach_to(base)

    if not args.hide_env:
        _load_env_obstacles(config_path, base)

    attach_robot_home(base, layout, show_robot=not args.hide_robot)

    world_poses = asm.compute_world_poses(
        fixture_pos=assembly_pos,
        fixture_rotmat=assembly_rot,
    )

    attach_goal_ghosts(
        base,
        asm,
        world_poses,
        show_goal_ghosts=args.show_goal_ghosts,
    )

    attach_initial_layout(base, asm, layout, part_order, show_frames=args.show_frames)

    print("\n提示：")
    print("  绿色实心模型通常表示 preassembled，例如 base_plate。")
    print("  彩色模型表示 layout 中的初始 staging 零件。")
    print("  默认只显示：左机械臂 + 洞洞板 + 初始布局零件。")
    print("  --show-goal-ghosts 可额外显示最终目标 ghost。")
    print("  --show-frames 可额外显示坐标系。")

    base.run()


if __name__ == "__main__":
    main()
