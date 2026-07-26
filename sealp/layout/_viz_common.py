"""Shared helpers for sealp.layout visualization scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import yaml

from wrs import mgm

_SEALP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SEALP_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_CONFIG = os.path.join(_SEALP_ROOT, "config", "sample_config.yaml")
DUAL_ARM_Y_OFFSET = 0.62
DEFAULT_TABLE_FLOOR_Z = -0.75
DEFAULT_TABLE_TOP_THICKNESS = 0.06
TABLE_VIS_RGB = np.array([0.68, 0.68, 0.70])
TABLE_VIS_ALPHA = 0.72
PEGBOARD_NX = 23
PEGBOARD_NY = 46
PEGBOARD_HOLE_RGB = np.array([0.30, 0.30, 0.32])
PEGBOARD_HOLE_RADIUS_RATIO = 0.20
PEGBOARD_HOLE_DEPTH_RATIO = 0.85

TABLE_GRID_PALETTE: Sequence[Tuple[float, float, float]] = (
    (0.90, 0.30, 0.30),
    (0.95, 0.65, 0.20),
    (0.95, 0.90, 0.25),
    (0.40, 0.80, 0.35),
    (0.25, 0.75, 0.75),
    (0.30, 0.55, 0.90),
    (0.55, 0.40, 0.85),
    (0.95, 0.55, 0.80),
    (0.55, 0.55, 0.55),
)


def hide_rot_center_marker(base) -> None:
    rot_center = getattr(getattr(base, "inputmgr", None), "rot_center", None)
    if rot_center is not None:
        rot_center.detach()


def resolve_config_path(config: str) -> str:
    path = Path(config)
    if not path.is_absolute():
        for base in (_PROJECT_ROOT, _SEALP_ROOT, Path.cwd()):
            candidate = (base / path).resolve()
            if candidate.is_file():
                return str(candidate)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    return str(path)


def load_table_box(config_path: str, name: str = "work_table"):
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    obstacles = (((cfg or {}).get("environment") or {}).get("obstacles")) or []
    for obs in obstacles:
        if obs.get("name") == name and obs.get("type") == "box":
            extent = np.asarray(obs["extent"], dtype=float)
            pos = np.asarray(obs["pos"], dtype=float)
            rgba = obs.get("rgba", [0.55, 0.45, 0.35, 0.8])
            return extent, pos, rgba
    raise ValueError(
        f"Box obstacle {name!r} not found in {config_path} environment.obstacles."
    )


def _table_corner_leg_centers(
    extent,
    pos,
    *,
    leg_width: float,
) -> List[np.ndarray]:
    ex, ey = float(extent[0]), float(extent[1])
    px, py = float(pos[0]), float(pos[1])
    x0, x1 = px - ex / 2.0, px + ex / 2.0
    y0, y1 = py - ey / 2.0, py + ey / 2.0
    half = float(leg_width) / 2.0
    return [
        np.array([x0 + half, y0 + half]),
        np.array([x0 + half, y1 - half]),
        np.array([x1 - half, y0 + half]),
        np.array([x1 - half, y1 - half]),
    ]


def build_pegboard_grid_centers(
    extent,
    pos,
    *,
    nx: int = PEGBOARD_NX,
    ny: int = PEGBOARD_NY,
    hole_model: str = "center",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Return hole centers on x/y axes and pitches (center model, same as layout_to_pegboard)."""
    ex, ey = float(extent[0]), float(extent[1])
    px, py = float(pos[0]), float(pos[1])
    x_min, x_max = px - ex / 2.0, px + ex / 2.0
    y_min, y_max = py - ey / 2.0, py + ey / 2.0

    if hole_model == "center":
        pitch_x = (x_max - x_min) / float(nx)
        pitch_y = (y_max - y_min) / float(ny)
        x_centers = np.array([x_min + (i + 0.5) * pitch_x for i in range(nx)], dtype=float)
        y_centers = np.array([y_min + (j + 0.5) * pitch_y for j in range(ny)], dtype=float)
    elif hole_model == "edge":
        pitch_x = (x_max - x_min) / float(max(nx - 1, 1))
        pitch_y = (y_max - y_min) / float(max(ny - 1, 1))
        x_centers = np.array([x_min + i * pitch_x for i in range(nx)], dtype=float)
        y_centers = np.array([y_min + j * pitch_y for j in range(ny)], dtype=float)
    else:
        raise ValueError(f"Unknown hole_model: {hole_model!r}")

    return x_centers, y_centers, float(pitch_x), float(pitch_y)


def _build_pegboard_holes_mesh(
    top_surface_z: float,
    top_thickness: float,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    pitch_x: float,
    pitch_y: float,
):
    import wrs.basis.trimesh.creation as trm_creation
    import wrs.basis.trimesh.util as trm_util

    pitch = min(float(pitch_x), float(pitch_y))
    hole_radius = max(pitch * PEGBOARD_HOLE_RADIUS_RATIO, 0.003)
    hole_depth = max(float(top_thickness) * PEGBOARD_HOLE_DEPTH_RATIO, 0.008)
    hole_center_z = float(top_surface_z) - hole_depth / 2.0

    template = trm_creation.cylinder(height=hole_depth, radius=hole_radius, n_sec=12)
    parts = []
    for x in x_centers:
        for y in y_centers:
            hole = template.copy()
            hole.apply_translation(np.array([float(x), float(y), hole_center_z], dtype=float))
            parts.append(hole)

    if not parts:
        return None

    merged = parts[0]
    for part in parts[1:]:
        merged = trm_util.concatenate(merged, part)
    return merged


def attach_pegboard_holes(
    base,
    extent,
    pos,
    top_surface_z: float,
    top_thickness: float,
    *,
    nx: int = PEGBOARD_NX,
    ny: int = PEGBOARD_NY,
    hole_model: str = "center",
    show_pegboard: bool = True,
) -> Optional[object]:
    if not show_pegboard:
        return None

    x_centers, y_centers, pitch_x, pitch_y = build_pegboard_grid_centers(
        extent, pos, nx=nx, ny=ny, hole_model=hole_model
    )
    holes_mesh = _build_pegboard_holes_mesh(
        top_surface_z=top_surface_z,
        top_thickness=top_thickness,
        x_centers=x_centers,
        y_centers=y_centers,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
    )
    if holes_mesh is None:
        return None

    holes = mgm.StaticGeometricModel(
        initor=holes_mesh,
        rgb=PEGBOARD_HOLE_RGB,
        alpha=0.95,
    )
    holes.attach_to(base)
    return holes


def attach_work_table_with_legs(
    base,
    extent,
    pos,
    rgba,
    *,
    floor_z: float = DEFAULT_TABLE_FLOOR_Z,
    leg_width: Optional[float] = None,
    leg_width_ratio: float = 5.5,
    top_thickness: float = DEFAULT_TABLE_TOP_THICKNESS,
    top_alpha: Optional[float] = None,
    leg_alpha: Optional[float] = None,
    pegboard_nx: int = PEGBOARD_NX,
    pegboard_ny: int = PEGBOARD_NY,
    show_pegboard: bool = True,
) -> List:
    """Draw tabletop, four corner legs, and pegboard holes (visual only)."""
    extent = np.asarray(extent, dtype=float)
    pos = np.asarray(pos, dtype=float)
    rgba = np.asarray(rgba, dtype=float)
    alpha = float(
        top_alpha
        if top_alpha is not None
        else (leg_alpha if leg_alpha is not None else (rgba[3] if len(rgba) > 3 else TABLE_VIS_ALPHA))
    )

    ex, ey, ez = float(extent[0]), float(extent[1]), float(extent[2])
    if leg_width is None:
        leg_width = min(ex, ey) / float(leg_width_ratio)
    leg_width = max(float(leg_width), 0.03)

    top_surface_z = float(pos[2]) + ez / 2.0
    visual_ez = max(float(ez), float(top_thickness))
    visual_pos = np.array([pos[0], pos[1], top_surface_z - visual_ez / 2.0], dtype=float)
    visual_extent = np.array([ex, ey, visual_ez], dtype=float)

    tabletop_bottom_z = float(visual_pos[2]) - visual_ez / 2.0
    leg_height = max(tabletop_bottom_z - float(floor_z), 0.05)
    leg_center_z = tabletop_bottom_z - leg_height / 2.0
    leg_extent = np.array([leg_width, leg_width, leg_height], dtype=float)

    attached = []
    top = mgm.gen_box(
        xyz_lengths=visual_extent,
        pos=visual_pos,
        rgb=TABLE_VIS_RGB,
        alpha=alpha,
    )
    top.attach_to(base)
    attached.append(top)

    for corner_xy in _table_corner_leg_centers(visual_extent, visual_pos, leg_width=leg_width):
        leg_pos = np.array([corner_xy[0], corner_xy[1], leg_center_z], dtype=float)
        leg = mgm.gen_box(
            xyz_lengths=leg_extent,
            pos=leg_pos,
            rgb=TABLE_VIS_RGB,
            alpha=alpha,
        )
        leg.attach_to(base)
        attached.append(leg)

    holes = attach_pegboard_holes(
        base,
        extent,
        pos,
        top_surface_z=top_surface_z,
        top_thickness=visual_ez,
        nx=pegboard_nx,
        ny=pegboard_ny,
        show_pegboard=show_pegboard,
    )
    if holes is not None:
        attached.append(holes)

    return attached


def attach_env_obstacles(
    config_path: str,
    base,
    *,
    alpha: float = 0.55,
    table_name: str = "work_table",
    table_with_legs: bool = True,
    table_floor_z: float = DEFAULT_TABLE_FLOOR_Z,
) -> List:
    from sealp.config import load_config
    from sealp.colliders import StaticEnvironment

    cfg = load_config(config_path)
    env = StaticEnvironment(obstacle_defs=cfg.obstacle_defs, base_dir=cfg.config_dir)
    obs_list = []
    table_drawn = False

    if table_with_legs:
        try:
            extent, pos, rgba = load_table_box(config_path, table_name)
            attach_work_table_with_legs(
                base,
                extent,
                pos,
                rgba,
                floor_z=table_floor_z,
                top_alpha=float(rgba[3]) if len(rgba) > 3 else alpha,
            )
            table_drawn = True
        except ValueError:
            table_drawn = False

    for name in env.names():
        if table_drawn and name == table_name:
            continue
        obs = env.get(name)
        try:
            obs.rgba = np.array([0.55, 0.55, 0.55, float(alpha)])
        except Exception:
            pass
        obs.attach_to(base)
        obs_list.append(obs)

    return obs_list


def build_table_grid_tiles(
    extent,
    pos,
    rows: int,
    cols: int,
    *,
    gap: float,
    tile_thickness: float,
    z_lift: float,
):
    ex, ey, ez = float(extent[0]), float(extent[1]), float(extent[2])
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    top_z = pz + ez / 2.0 + z_lift + tile_thickness / 2.0
    cell_x = ex / rows
    cell_y = ey / cols
    x0 = px - ex / 2.0
    y0 = py - ey / 2.0

    tiles = []
    for r in range(rows):
        for c in range(cols):
            cx = x0 + (r + 0.5) * cell_x
            cy = y0 + (c + 0.5) * cell_y
            tile_extent = np.array([
                max(1e-3, cell_x - gap),
                max(1e-3, cell_y - gap),
                tile_thickness,
            ])
            center = np.array([cx, cy, top_z])
            rgb = TABLE_GRID_PALETTE[(r * cols + c) % len(TABLE_GRID_PALETTE)]
            tiles.append(((r, c), center, tile_extent, rgb))
    return tiles
