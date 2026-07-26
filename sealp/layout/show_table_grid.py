#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize work_table Grid Regions
===================================

把 ``work_table`` 顶面均分为网格区域，用不同颜色显示 3x3 装配区候选。

运行::

    python -m sealp.layout.show_table_grid

指定配置::

    python -m sealp.layout.show_table_grid --config sealp/config/sample_config.yaml
"""

from __future__ import annotations

import argparse

import numpy as np

from wrs import mgm, wd

from sealp.layout._viz_common import (
    DEFAULT_CONFIG,
    DEFAULT_TABLE_FLOOR_Z,
    DEFAULT_TABLE_TOP_THICKNESS,
    TABLE_VIS_RGB,
    attach_env_obstacles,
    attach_pegboard_holes,
    attach_work_table_with_legs,
    build_table_grid_tiles,
    hide_rot_center_marker,
    load_table_box,
    resolve_config_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize work_table top surface as a colored grid."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--table-name", default="work_table", help="Table obstacle name")
    parser.add_argument("--rows", type=int, default=3, help="Rows along x")
    parser.add_argument("--cols", type=int, default=3, help="Columns along y")
    parser.add_argument("--gap", type=float, default=0.004, help="Gap between tiles (m)")
    parser.add_argument("--tile-thickness", type=float, default=0.004, help="Tile thickness (m)")
    parser.add_argument("--z-lift", type=float, default=0.001, help="Lift above table top (m)")
    parser.add_argument("--alpha", type=float, default=0.85, help="Tile alpha")
    parser.add_argument("--no-table", action="store_true", help="Hide table body, show grid only")
    parser.add_argument("--no-legs", action="store_true", help="Draw flat tabletop only, no corner legs")
    parser.add_argument(
        "--table-floor-z",
        type=float,
        default=DEFAULT_TABLE_FLOOR_Z,
        help="Floor z for table legs (default: -0.75 m)",
    )
    parser.add_argument(
        "--use-env-loader",
        action="store_true",
        help="Load full environment via StaticEnvironment instead of drawing one box.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    extent, pos, rgba = load_table_box(config_path, args.table_name)

    print(f"[table] config = {config_path}")
    print(f"[table] {args.table_name} extent={extent.tolist()} pos={pos.tolist()}")
    print(f"[table] grid = {args.rows} x {args.cols}")

    look = np.array([pos[0], pos[1], pos[2] + 0.1], dtype=float)
    base = wd.World(cam_pos=look + np.array([0.9, -1.0, 0.9]), lookat_pos=look)
    hide_rot_center_marker(base)
    mgm.gen_frame(ax_length=0.1).attach_to(base)

    if args.use_env_loader:
        attach_env_obstacles(
            config_path,
            base,
            alpha=float(rgba[3]) if len(rgba) > 3 else 0.55,
            table_with_legs=not args.no_legs,
            table_floor_z=args.table_floor_z,
        )
    elif not args.no_table:
        if args.no_legs:
            top_surface_z = float(pos[2]) + float(extent[2]) / 2.0
            visual_ez = max(float(extent[2]), DEFAULT_TABLE_TOP_THICKNESS)
            visual_pos = np.array([pos[0], pos[1], top_surface_z - visual_ez / 2.0], dtype=float)
            mgm.gen_box(
                xyz_lengths=np.array([extent[0], extent[1], visual_ez], dtype=float),
                pos=visual_pos,
                rgb=TABLE_VIS_RGB,
                alpha=float(rgba[3]) if len(rgba) > 3 else 0.62,
            ).attach_to(base)
            attach_pegboard_holes(
                base,
                extent,
                pos,
                top_surface_z=top_surface_z,
                top_thickness=visual_ez,
            )
        else:
            attach_work_table_with_legs(
                base,
                extent,
                pos,
                rgba,
                floor_z=args.table_floor_z,
                top_alpha=float(rgba[3]) if len(rgba) > 3 else 0.55,
            )

    tiles = build_table_grid_tiles(
        extent,
        pos,
        args.rows,
        args.cols,
        gap=args.gap,
        tile_thickness=args.tile_thickness,
        z_lift=args.z_lift,
    )
    for (r, c), center, tile_extent, rgb in tiles:
        mgm.gen_box(
            xyz_lengths=tile_extent,
            pos=center,
            rgb=np.array(rgb),
            alpha=float(args.alpha),
        ).attach_to(base)
        mgm.gen_frame(pos=center, ax_length=0.03).attach_to(base)
        print(
            f"  r{r}c{c}: center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) "
            f"size=({tile_extent[0]:.3f} x {tile_extent[1]:.3f})"
        )

    print("[table] Press close window to exit.")
    base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
