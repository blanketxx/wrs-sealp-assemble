#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize Dual Arms on a Perforated Work Table
=================================================

显示 ``sample_config.yaml`` 中的 work_table 与 Panthera 双臂 home 姿态。
默认把原来的实心桌面替换为真正带通孔的洞洞板：

- 长边 46 个孔；
- 短边 23 个孔；
- 孔不是圆柱贴图，而是桌面三角网格中的真实贯穿孔。

运行::

    python -m sealp.layout.show_dual_arms_perforated

指定孔径（单位 m）::

    python -m sealp.layout.show_dual_arms_perforated --hole-diameter 0.010

恢复原来的实心桌面::

    python -m sealp.layout.show_dual_arms_perforated --solid-table
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trimesh as trm

from wrs import mcm, mgm, wd

from sealp.config import load_config
from sealp.layout._viz_common import (
    DEFAULT_CONFIG,
    DEFAULT_TABLE_FLOOR_Z,
    DUAL_ARM_Y_OFFSET,
    attach_env_obstacles,
    hide_rot_center_marker,
    load_table_box,
    resolve_config_path,
)

try:
    import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda
except Exception:
    pda = None

HOME_JV = np.zeros(6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize dual Panthera arms and a truly perforated work table."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument(
        "--arm-y-offset",
        type=float,
        default=DUAL_ARM_Y_OFFSET,
        help="Right arm y offset from left arm base (m)",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="Draw 3x3 assembly-region boundary lines above the perforated table",
    )
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--robot-alpha", type=float, default=1.0, help="Robot mesh alpha")
    parser.add_argument("--hide-env", action="store_true", help="Hide the work table")
    parser.add_argument(
        "--solid-table",
        action="store_true",
        help="Use the original solid work_table instead of the perforated mesh",
    )
    parser.add_argument(
        "--no-legs",
        action="store_true",
        help="Draw tabletop only, without four corner legs",
    )
    parser.add_argument(
        "--table-floor-z",
        type=float,
        default=DEFAULT_TABLE_FLOOR_Z,
        help="Floor z for table legs (default: -0.75 m)",
    )
    parser.add_argument(
        "--hole-long-count",
        type=int,
        default=46,
        help="Number of through holes along the longer tabletop side",
    )
    parser.add_argument(
        "--hole-short-count",
        type=int,
        default=23,
        help="Number of through holes along the shorter tabletop side",
    )
    parser.add_argument(
        "--hole-diameter",
        type=float,
        default=None,
        help=(
            "Through-hole diameter in meters. Default: 42%% of the smaller hole pitch "
            "computed from the current tabletop size"
        ),
    )
    parser.add_argument(
        "--hole-segments",
        type=int,
        default=16,
        help="Polygon segments for each round hole; larger is smoother but slower",
    )
    parser.add_argument(
        "--table-alpha",
        type=float,
        default=1.0,
        help="Perforated tabletop alpha",
    )
    parser.add_argument(
        "--cam-pos",
        default="1.05,-1.25,0.85",
        help="Camera position x,y,z",
    )
    return parser.parse_args()


def _square_ring_points(
    half_x: float,
    half_y: float,
    radius: float,
    segments: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return matching CCW outer-square and inner-circle rings.

    The outer ring follows the complete square perimeter, including all four corners.
    The inner point paired with each outer point lies on the same radial direction.
    This creates a non-overlapping annular patch for one perforation cell.
    """
    segments = max(8, int(math.ceil(segments / 4.0)) * 4)
    per_edge = segments // 4

    outer: List[Tuple[float, float]] = []

    # Bottom edge: left -> right.
    for i in range(per_edge):
        t = i / per_edge
        outer.append((-half_x + 2.0 * half_x * t, -half_y))

    # Right edge: bottom -> top.
    for i in range(per_edge):
        t = i / per_edge
        outer.append((half_x, -half_y + 2.0 * half_y * t))

    # Top edge: right -> left.
    for i in range(per_edge):
        t = i / per_edge
        outer.append((half_x - 2.0 * half_x * t, half_y))

    # Left edge: top -> bottom.
    for i in range(per_edge):
        t = i / per_edge
        outer.append((-half_x, half_y - 2.0 * half_y * t))

    outer_arr = np.asarray(outer, dtype=float)
    norms = np.linalg.norm(outer_arr, axis=1, keepdims=True)
    inner_arr = radius * outer_arr / np.maximum(norms, 1e-12)
    return outer_arr, inner_arr


def _append_cell_with_round_hole(
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
    close_bottom: bool = False,
    close_right: bool = False,
    close_top: bool = False,
    close_left: bool = False,
) -> None:
    """Append one rectangular cell with a real cylindrical through hole.

    The four ``close_*`` flags add the external side wall only when this cell
    lies on the corresponding tabletop boundary.  This keeps the whole mesh
    watertight while avoiding duplicate internal walls between adjacent cells.
    """
    outer_xy, inner_xy = _square_ring_points(
        half_x=pitch_x / 2.0,
        half_y=pitch_y / 2.0,
        radius=radius,
        segments=segments,
    )
    n = len(outer_xy)
    start = len(vertices)

    # Vertex blocks: outer top, inner top, outer bottom, inner bottom.
    for xy in outer_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_top])
    for xy in inner_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_top])
    for xy in outer_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_bottom])
    for xy in inner_xy:
        vertices.append([cx + xy[0], cy + xy[1], z_bottom])

    ot = start
    it = start + n
    ob = start + 2 * n
    ib = start + 3 * n

    for i in range(n):
        j = (i + 1) % n

        # Top annulus, normal approximately +Z.
        faces.append([ot + i, ot + j, it + j])
        faces.append([ot + i, it + j, it + i])

        # Bottom annulus, reverse winding for -Z.
        faces.append([ob + i, ib + j, ob + j])
        faces.append([ob + i, ib + i, ib + j])

        # Inner cylindrical wall, normal points into the empty hole.
        faces.append([it + i, ib + j, ib + i])
        faces.append([it + i, it + j, ib + j])

    # Close only the external tabletop boundary.  The outer ring is ordered as
    # bottom, right, top, left; each edge contains ``n // 4`` segments.
    per_edge = n // 4
    boundary_ranges = []
    if close_bottom:
        boundary_ranges.append(range(0, per_edge))
    if close_right:
        boundary_ranges.append(range(per_edge, 2 * per_edge))
    if close_top:
        boundary_ranges.append(range(2 * per_edge, 3 * per_edge))
    if close_left:
        boundary_ranges.append(range(3 * per_edge, 4 * per_edge))

    for edge_range in boundary_ranges:
        for i in edge_range:
            j = (i + 1) % n
            faces.append([ot + i, ob + i, ob + j])
            faces.append([ot + i, ob + j, ot + j])


def _append_outer_walls(
    vertices: List[List[float]],
    faces: List[List[int]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_bottom: float,
    z_top: float,
) -> None:
    """Close the four external side walls of the tabletop."""
    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    start = len(vertices)
    for x, y in corners:
        vertices.append([x, y, z_top])
    for x, y in corners:
        vertices.append([x, y, z_bottom])

    top = start
    bottom = start + 4
    for i in range(4):
        j = (i + 1) % 4
        faces.append([top + i, bottom + i, bottom + j])
        faces.append([top + i, bottom + j, top + j])


def _build_perforated_table_mesh(
    extent: Sequence[float],
    pos: Sequence[float],
    long_count: int = 46,
    short_count: int = 23,
    hole_diameter: Optional[float] = None,
    hole_segments: int = 16,
) -> Tuple[trm.Trimesh, dict]:
    """Build one watertight tabletop mesh containing actual through holes."""
    extent = np.asarray(extent, dtype=float)
    pos = np.asarray(pos, dtype=float)

    if extent.shape[0] < 3 or np.any(extent[:3] <= 0):
        raise ValueError(f"Invalid table extent: {extent}")
    if long_count <= 0 or short_count <= 0:
        raise ValueError("Hole counts must be positive")

    size_x, size_y, thickness = map(float, extent[:3])

    # Automatically map 46 holes to the physical long side and 23 to the short side.
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
            f"hole_diameter={hole_diameter:.6f} m is invalid; "
            f"it must be in (0, {max_diameter:.6f}) for the current pitch"
        )

    radius = hole_diameter / 2.0
    x_min = float(pos[0] - size_x / 2.0)
    x_max = float(pos[0] + size_x / 2.0)
    y_min = float(pos[1] - size_y / 2.0)
    y_max = float(pos[1] + size_y / 2.0)
    z_bottom = float(pos[2] - thickness / 2.0)
    z_top = float(pos[2] + thickness / 2.0)

    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    for ix in range(nx):
        cx = x_min + (ix + 0.5) * pitch_x
        for iy in range(ny):
            cy = y_min + (iy + 0.5) * pitch_y
            _append_cell_with_round_hole(
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
    try:
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
    except Exception:
        pass

    if not mesh.is_watertight:
        raise RuntimeError(
            "Generated perforated tabletop is not watertight. "
            "Try reducing --hole-segments or checking the table dimensions."
        )

    info = {
        "nx": nx,
        "ny": ny,
        "pitch_x": pitch_x,
        "pitch_y": pitch_y,
        "hole_diameter": hole_diameter,
        "hole_count": nx * ny,
        "is_watertight": bool(mesh.is_watertight),
    }
    return mesh, info


def _attach_table_legs(
    base,
    extent: Sequence[float],
    pos: Sequence[float],
    floor_z: float,
    rgb: Sequence[float],
    alpha: float,
) -> None:
    extent = np.asarray(extent, dtype=float)
    pos = np.asarray(pos, dtype=float)
    size_x, size_y, thickness = map(float, extent[:3])

    table_bottom_z = float(pos[2] - thickness / 2.0)
    leg_height = table_bottom_z - float(floor_z)
    if leg_height <= 0.0:
        print(
            f"[WARN] table_floor_z={floor_z:.4f} is not below tabletop bottom "
            f"z={table_bottom_z:.4f}; skip table legs."
        )
        return

    leg_side = min(0.055, 0.10 * min(size_x, size_y))
    inset = max(0.035, 0.8 * leg_side)
    x_off = size_x / 2.0 - inset
    y_off = size_y / 2.0 - inset
    leg_z = float(floor_z + leg_height / 2.0)

    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            mgm.gen_box(
                xyz_lengths=[leg_side, leg_side, leg_height],
                pos=np.array(
                    [pos[0] + sx * x_off, pos[1] + sy * y_off, leg_z],
                    dtype=float,
                ),
                rgb=np.asarray(rgb, dtype=float),
                alpha=float(alpha),
            ).attach_to(base)


def _attach_perforated_table(
    base,
    extent: Sequence[float],
    pos: Sequence[float],
    rgba: Sequence[float],
    long_count: int,
    short_count: int,
    hole_diameter: Optional[float],
    hole_segments: int,
    table_alpha: float,
    with_legs: bool,
    floor_z: float,
):
    mesh, info = _build_perforated_table_mesh(
        extent=extent,
        pos=pos,
        long_count=long_count,
        short_count=short_count,
        hole_diameter=hole_diameter,
        hole_segments=hole_segments,
    )

    # WRS/Panda3D is more reliable when a large generated mesh is loaded from
    # an STL file instead of being passed as a large in-memory trimesh object.
    # The old in-memory path could silently create a collision object whose
    # render geometry was missing, which is why only the four legs appeared.
    cache_dir = Path(__file__).resolve().parent / "_generated_meshes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = (
        f"work_table_perforated_{info['nx']}x{info['ny']}_"
        f"d{info['hole_diameter'] * 1000.0:.3f}mm_"
        f"seg{int(hole_segments)}.stl"
    )
    mesh_path = cache_dir / mesh_name
    mesh.export(str(mesh_path))

    board = mcm.CollisionModel(str(mesh_path))
    rgba = np.asarray(rgba, dtype=float).reshape(-1)
    rgb = rgba[:3] if rgba.size >= 3 else np.array([0.65, 0.65, 0.65])
    board.rgb = np.asarray(rgb, dtype=float)
    board.alpha = float(table_alpha)
    board.attach_to(base)

    # Metadata is useful if this model is later reused as a planner obstacle.
    board._sealp_role = "environment_obstacle"
    board._sealp_name = "work_table_perforated"

    if with_legs:
        _attach_table_legs(
            base=base,
            extent=extent,
            pos=pos,
            floor_z=floor_z,
            rgb=rgb,
            alpha=table_alpha,
        )

    print(
        "[table] perforated tabletop: "
        f"nx={info['nx']}, ny={info['ny']}, total={info['hole_count']} holes, "
        f"pitch=({info['pitch_x'] * 1000.0:.2f}, "
        f"{info['pitch_y'] * 1000.0:.2f}) mm, "
        f"diameter={info['hole_diameter'] * 1000.0:.2f} mm, "
        f"watertight={info['is_watertight']}, "
        f"mesh={mesh_path}"
    )
    return board


def _attach_grid_lines(
    base,
    extent: Sequence[float],
    pos: Sequence[float],
    rows: int,
    cols: int,
) -> None:
    """Draw only region boundary lines, so the holes remain visible."""
    extent = np.asarray(extent, dtype=float)
    pos = np.asarray(pos, dtype=float)
    size_x, size_y, thickness = map(float, extent[:3])
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    z = float(pos[2] + thickness / 2.0 + 0.0015)
    line_width = 0.003
    line_height = 0.0015
    line_rgb = np.array([0.18, 0.18, 0.18], dtype=float)

    for c in range(1, cols):
        x = float(pos[0] - size_x / 2.0 + c * size_x / cols)
        mgm.gen_box(
            xyz_lengths=[line_width, size_y, line_height],
            pos=np.array([x, pos[1], z]),
            rgb=line_rgb,
            alpha=0.9,
        ).attach_to(base)

    for r in range(1, rows):
        y = float(pos[1] - size_y / 2.0 + r * size_y / rows)
        mgm.gen_box(
            xyz_lengths=[size_x, line_width, line_height],
            pos=np.array([pos[0], y, z]),
            rgb=line_rgb,
            alpha=0.9,
        ).attach_to(base)


def _attach_dual_arms(base, robot_pos, robot_rot, arm_y_offset: float, alpha: float):
    if pda is None:
        print("[WARN] Cannot import DualPantheraHTNoBody; skip robot display.")
        return None

    robot = pda.DualPantheraHTNoBody(
        pos=np.asarray(robot_pos, dtype=float),
        rotmat=np.asarray(robot_rot, dtype=float),
        arm_y_offset=float(arm_y_offset),
        enable_cc=True,
    )
    try:
        robot.lft_arm.goto_given_conf(HOME_JV)
        robot.rgt_arm.goto_given_conf(HOME_JV)
    except Exception:
        pass

    robot.gen_meshmodel(alpha=float(alpha)).attach_to(base)

    rb = np.asarray(robot_pos, dtype=float)
    mgm.gen_frame(
        pos=rb,
        rotmat=np.asarray(robot_rot, dtype=float),
        ax_length=0.12,
    ).attach_to(base)

    lft_base = rb + np.array([0.0, 0.0, 0.0])
    rgt_base = rb + np.array([0.0, -float(arm_y_offset), 0.0])
    mgm.gen_frame(pos=lft_base, ax_length=0.06).attach_to(base)
    mgm.gen_frame(pos=rgt_base, ax_length=0.06).attach_to(base)

    print(f"[arms] robot_base_pos = {np.round(rb, 4).tolist()}")
    print(f"[arms] lft_arm_base    = {np.round(lft_base, 4).tolist()}")
    print(f"[arms] rgt_arm_base    = {np.round(rgt_base, 4).tolist()}")
    return robot


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)

    robot_pos = np.asarray(cfg.robot.pos, dtype=float)
    robot_rot = np.asarray(cfg.robot.rotmat, dtype=float)
    extent, pos, rgba = load_table_box(config_path, "work_table")

    if args.show_grid:
        look = np.array([pos[0], pos[1], pos[2] + 0.1], dtype=float)
    else:
        look = robot_pos + np.array([0.23, -0.35, 0.10])

    cam_pos = np.array([float(x.strip()) for x in args.cam_pos.split(",")], dtype=float)
    base = wd.World(cam_pos=cam_pos, lookat_pos=look)
    hide_rot_center_marker(base)
    mgm.gen_frame(ax_length=0.1).attach_to(base)

    print(f"[arms] config = {config_path}")
    if not args.hide_env:
        if args.solid_table:
            attach_env_obstacles(
                config_path,
                base,
                alpha=float(args.table_alpha),
                table_with_legs=not args.no_legs,
                table_floor_z=args.table_floor_z,
            )
        else:
            _attach_perforated_table(
                base=base,
                extent=extent,
                pos=pos,
                rgba=rgba,
                long_count=args.hole_long_count,
                short_count=args.hole_short_count,
                hole_diameter=args.hole_diameter,
                hole_segments=args.hole_segments,
                table_alpha=args.table_alpha,
                with_legs=not args.no_legs,
                floor_z=args.table_floor_z,
            )

    if args.show_grid and not args.hide_env:
        _attach_grid_lines(
            base=base,
            extent=extent,
            pos=pos,
            rows=args.grid_rows,
            cols=args.grid_cols,
        )

    _attach_dual_arms(
        base,
        robot_pos=robot_pos,
        robot_rot=robot_rot,
        arm_y_offset=args.arm_y_offset,
        alpha=args.robot_alpha,
    )

    print("[arms] Close window to exit.")
    base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
