#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate YuanChair mesh assets (round-stool kit).

    yuanchair-part1.stl : seat slab (square) with 4 leg sockets
    yuanchair-part2.stl : leg post (square cross-section)

Seat is resized to 20 cm (was 27 cm mesh / 38 cm rotated footprint). The four
leg sockets are moved inward to ``HOLE_OFFSET`` so the legs stay inside the
smaller seat; the assembly rel_pos in ``yuanchair.asmdef`` must match
``HOLE_OFFSET`` (see the constant printed at the end of this script).

All meshes have their bottom face at local z = 0.

Run::

    python sealp/assets/models/yuanchair/gen_yuanchair_meshes.py
"""

from __future__ import annotations

import os

import numpy as np
import trimesh
import trimesh.creation as tcreation

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEALP_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROJ_ROOT = os.path.dirname(_SEALP_ROOT)
_OUT_DIR = _HERE  # STL files live directly in the yuanchair/ folder

# ── seat (part1) ─────────────────────────────────────────────────
SEAT_XY = 0.20          # was 0.27 mesh -> now 20 cm
SEAT_H = 0.04
SOCKET_DEPTH = 0.02     # pocket cut from the top face down (leg embeds 0.02)

# ── leg (part2) ──────────────────────────────────────────────────
LEG_XY = 0.039
LEG_H = 0.16
SOCKET_XY = LEG_XY + 0.001   # 0.5 mm clearance per side

# ── leg socket / assembly offset (MUST equal asmdef rel_pos xy) ──
HOLE_OFFSET = 0.07      # was 0.11; moved inward for the 20 cm seat
LEG_REL_Z = 0.02        # leg bottom sits 0.02 above the seat bottom


def _box(lx: float, ly: float, lz: float,
         cx: float = 0.0, cy: float = 0.0, cz: float = 0.0) -> trimesh.Trimesh:
    mesh = tcreation.box(extents=np.array([lx, ly, lz], dtype=float))
    mesh.vertices += np.array([cx, cy, cz], dtype=float)
    return mesh


def _box_bottom_at_z0(lx: float, ly: float, lz: float,
                      cx: float = 0.0, cy: float = 0.0) -> trimesh.Trimesh:
    return _box(lx, ly, lz, cx=cx, cy=cy, cz=lz / 2.0)


def _make_rect_layer_with_holes(
    outer_x: float,
    outer_y: float,
    z_min: float,
    z_max: float,
    holes: list[tuple[float, float, float, float]],
) -> trimesh.Trimesh:
    """A rectangular slab layer [z_min, z_max] with axis-aligned rectangular holes.

    ``holes`` = list of (center_x, center_y, half_x, half_y).
    """
    x_min, x_max = -outer_x / 2.0, outer_x / 2.0
    y_min, y_max = -outer_y / 2.0, outer_y / 2.0
    xs = [x_min, x_max]
    ys = [y_min, y_max]
    for cx, cy, hx, hy in holes:
        xs.extend([cx - hx, cx + hx])
        ys.extend([cy - hy, cy + hy])
    xs = sorted(set(round(v, 10) for v in xs))
    ys = sorted(set(round(v, 10) for v in ys))

    lz = z_max - z_min
    cz = (z_min + z_max) / 2.0
    parts: list[trimesh.Trimesh] = []
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        if xb <= xa:
            continue
        for j in range(len(ys) - 1):
            ya, yb = ys[j], ys[j + 1]
            if yb <= ya:
                continue
            cell_cx = (xa + xb) / 2.0
            cell_cy = (ya + yb) / 2.0
            inside = False
            for hcx, hcy, hhx, hhy in holes:
                if hcx - hhx <= cell_cx <= hcx + hhx and hcy - hhy <= cell_cy <= hcy + hhy:
                    inside = True
                    break
            if inside:
                continue
            parts.append(_box(xb - xa, yb - ya, lz, cell_cx, cell_cy, cz))
    if not parts:
        raise RuntimeError("rect layer with holes produced no solids")
    return trimesh.util.concatenate(parts)


def _socket_holes() -> list[tuple[float, float, float, float]]:
    hx = hy = SOCKET_XY / 2.0
    return [
        (HOLE_OFFSET, HOLE_OFFSET, hx, hy),
        (-HOLE_OFFSET, HOLE_OFFSET, hx, hy),
        (HOLE_OFFSET, -HOLE_OFFSET, hx, hy),
        (-HOLE_OFFSET, -HOLE_OFFSET, hx, hy),
    ]


def gen_seat() -> trimesh.Trimesh:
    """Square seat slab with a solid lower layer + top layer with 4 sockets."""
    holes = _socket_holes()
    bottom_h = SEAT_H - SOCKET_DEPTH
    deck = _box_bottom_at_z0(SEAT_XY, SEAT_XY, bottom_h)
    top_layer = _make_rect_layer_with_holes(
        SEAT_XY, SEAT_XY, SEAT_H - SOCKET_DEPTH, SEAT_H, holes)
    return trimesh.util.concatenate([deck, top_layer])


def gen_leg() -> trimesh.Trimesh:
    """Square-section leg post, bottom at z=0."""
    return _box_bottom_at_z0(LEG_XY, LEG_XY, LEG_H)


def _export(mesh: trimesh.Trimesh, name: str) -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, name)
    mesh.export(out_path)
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(
        f"[OK] {name:20s} -> {os.path.relpath(out_path, _PROJ_ROOT)}  "
        f"n_verts={len(mesh.vertices)} n_faces={len(mesh.faces)}  "
        f"bbox(m)=[{ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}]"
    )


def main() -> None:
    print("========== YuanChair STL generation (seat=20cm) ==========")
    print(f"output dir  = {os.path.relpath(_OUT_DIR, _PROJ_ROOT)}")
    print(f"SEAT_XY     = {SEAT_XY:.3f} m  SEAT_H = {SEAT_H:.3f} m")
    print(f"HOLE_OFFSET = {HOLE_OFFSET:.3f} m  (asmdef leg rel_pos xy must match)")
    print(f"LEG         = {LEG_XY:.3f} x {LEG_XY:.3f} x {LEG_H:.3f} m")
    print(f"LEG_REL_Z   = {LEG_REL_Z:.3f} m  (asmdef leg rel_pos z)\n")

    _export(gen_seat(), "yuanchair-part1.stl")
    _export(gen_leg(), "yuanchair-part2.stl")

    print("\nYuanChair STL assets generated (overwritten).")
    print(f"  -> set asmdef leg rel_pos to (+/-{HOLE_OFFSET:.2f}, +/-{HOLE_OFFSET:.2f}, {LEG_REL_Z:.2f})")
    print("  -> if geometry_cache/hint_cache/grasp for the SEAT exist, they are stale")
    print("     (seat is the preassembled first part -> not used by the GA leg pools).")


if __name__ == "__main__":
    main()
