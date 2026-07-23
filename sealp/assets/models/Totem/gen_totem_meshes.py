#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate Totem Mesh Assets - Furniture Side-Table Kit
========================================================

Totem 家具版：一套 **边桌 / 矮柜** 零件，不再是塔或小车。

    base_plate   : 踢脚线底座（四角腿孔 + 底部围板）
    post (x4)    : 锥形家具腿（上粗下细 + 脚垫 + 顶部箍）
    middle_plate : 中层固定搁板（板面 + 前挡裙板）
    top_cross    : 宽台面（薄板 + 四边封边，横向主体）

装配拓扑、part_id、rel_pos、孔位尺寸仍与 TopDownTower 对齐。
所有网格底面在 local z=0。

运行::

    python sealp/assets/models/Totem/gen_totem_meshes.py
"""

from __future__ import annotations

import os

import numpy as np
import trimesh
import trimesh.creation as tcreation

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEALP_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROJ_ROOT = os.path.dirname(_SEALP_ROOT)
_OUT_DIR = os.path.join(_HERE, "model")

# ── 差异化尺寸: 仅放大 XY 外形, 高度/孔位/rel_pos 与 tower 保持一致 (m) ──
# base/middle/top_cross 外框放大, 让 extent/footprint 明显区别于 tower/pavilion,
# 迫使布局模型依赖输入几何 (对抗单任务坍缩)。POST_OFFSET / socket / 高度不变。
BASE_X = 0.230
BASE_Y = 0.175
BASE_H = 0.0243
BASE_POCKET_DEPTH = 0.01215

POST_OFFSET_X = 0.06375
POST_OFFSET_Y = 0.04875
POST_BODY = 0.0195
POST_FOOT = 0.022
POST_FOOT_H = 0.01215
POST_H = 0.135
BASE_SOCKET_XY = 0.02535

MIDDLE_X = 0.195
MIDDLE_Y = 0.140
MIDDLE_H = 0.0216
MIDDLE_SOCKET_DEPTH = 0.0108
MIDDLE_SOCKET_XY = 0.020

FINIAL_TOTAL_H = 0.052
FINIAL_PEG_XY = 0.018
FINIAL_PEG_DEPTH = 0.018
FINIAL_PEG_H = 0.020
# 台面 XY 略小于 middle_plate，避免 footprint 过大挤占 staging 空间
TOP_CROSS_X = 0.150
TOP_CROSS_Y = 0.100


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


def _export(mesh: trimesh.Trimesh, name: str) -> None:
    os.makedirs(_OUT_DIR, exist_ok=True)
    out_path = os.path.join(_OUT_DIR, name)
    mesh.export(out_path)
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(
        f"[OK] {name:16s} -> {os.path.relpath(out_path, _PROJ_ROOT)}  "
        f"n_verts={len(mesh.vertices)} n_faces={len(mesh.faces)}  "
        f"bbox(m)=[{ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}]"
    )


def _corner_pocket_holes() -> list[tuple[float, float, float, float]]:
    hx = BASE_SOCKET_XY / 2.0
    hy = BASE_SOCKET_XY / 2.0
    return [
        (POST_OFFSET_X, POST_OFFSET_Y, hx, hy),
        (-POST_OFFSET_X, POST_OFFSET_Y, hx, hy),
        (POST_OFFSET_X, -POST_OFFSET_Y, hx, hy),
        (-POST_OFFSET_X, -POST_OFFSET_Y, hx, hy),
    ]


def gen_base_plate() -> trimesh.Trimesh:
    """踢脚线底座：板面四角腿孔 + 四边下垂围板（家具底座）。"""
    holes = _corner_pocket_holes()
    bottom_h = BASE_H - BASE_POCKET_DEPTH
    deck = _box_bottom_at_z0(BASE_X, BASE_Y, bottom_h)
    top_layer = _make_rect_layer_with_holes(
        BASE_X, BASE_Y, BASE_H - BASE_POCKET_DEPTH, BASE_H, holes)

    skirt_h = 0.009
    skirt_drop = 0.007
    skirt_z = skirt_drop / 2.0
    inset = 0.010
    skirts = [
        _box(BASE_X - inset, 0.008, skirt_h, 0.0, BASE_Y / 2.0 - 0.006, skirt_z),
        _box(BASE_X - inset, 0.008, skirt_h, 0.0, -BASE_Y / 2.0 + 0.006, skirt_z),
        _box(0.008, BASE_Y - inset, skirt_h, BASE_X / 2.0 - 0.006, 0.0, skirt_z),
        _box(0.008, BASE_Y - inset, skirt_h, -BASE_X / 2.0 + 0.006, 0.0, skirt_z),
    ]
    return trimesh.util.concatenate([deck, top_layer, *skirts])


def gen_post() -> trimesh.Trimesh:
    """锥形家具腿：脚垫 + 下细上粗两段 + 顶部箍圈。"""
    foot = _box_bottom_at_z0(POST_FOOT, POST_FOOT, POST_FOOT_H)

    lower_h = 0.048
    upper_h = POST_H - POST_FOOT_H - lower_h - 0.008
    lower = _box(0.015, 0.015, lower_h, 0.0, 0.0, POST_FOOT_H + lower_h / 2.0)
    upper = _box(POST_BODY, POST_BODY, upper_h, 0.0, 0.0, POST_FOOT_H + lower_h + upper_h / 2.0)

    collar_h = 0.008
    collar = _box(0.023, 0.023, collar_h, 0.0, 0.0, POST_H - collar_h / 2.0)

    return trimesh.util.concatenate([foot, lower, upper, collar])


def gen_middle_plate() -> trimesh.Trimesh:
    """中层搁板：板面中心孔 + 前侧挡裙板（抽屉柜层板感）。"""
    socket_hx = MIDDLE_SOCKET_XY / 2.0
    socket_hy = MIDDLE_SOCKET_XY / 2.0
    holes = [(0.0, 0.0, socket_hx, socket_hy)]

    bottom_h = MIDDLE_H - MIDDLE_SOCKET_DEPTH
    shelf = _box_bottom_at_z0(MIDDLE_X, MIDDLE_Y, bottom_h)
    top_layer = _make_rect_layer_with_holes(
        MIDDLE_X, MIDDLE_Y,
        MIDDLE_H - MIDDLE_SOCKET_DEPTH, MIDDLE_H,
        holes,
    )

    apron_h = 0.012
    apron_z = MIDDLE_H - apron_h / 2.0
    apron = _box(MIDDLE_X * 0.92, 0.010, apron_h, 0.0, -MIDDLE_Y / 2.0 + 0.006, apron_z)

    return trimesh.util.concatenate([shelf, top_layer, apron])


def gen_top_cross() -> trimesh.Trimesh:
    """宽台面：中心插脚 + 薄板面 + 四边封边条（边桌桌面）。"""
    peg = _box_bottom_at_z0(FINIAL_PEG_XY, FINIAL_PEG_DEPTH, FINIAL_PEG_H)

    top_xy = (TOP_CROSS_X, TOP_CROSS_Y)
    slab_h = 0.012
    slab_z = FINIAL_PEG_H + slab_h / 2.0
    slab = _box(top_xy[0], top_xy[1], slab_h, 0.0, 0.0, slab_z)

    band_h = FINIAL_TOTAL_H - FINIAL_PEG_H - slab_h
    band_z = FINIAL_PEG_H + slab_h + band_h / 2.0
    bands = [
        _box(top_xy[0], 0.008, band_h, 0.0, top_xy[1] / 2.0 - 0.004, band_z),
        _box(top_xy[0], 0.008, band_h, 0.0, -top_xy[1] / 2.0 + 0.004, band_z),
        _box(0.008, top_xy[1], band_h, top_xy[0] / 2.0 - 0.004, 0.0, band_z),
        _box(0.008, top_xy[1], band_h, -top_xy[0] / 2.0 + 0.004, 0.0, band_z),
    ]
    return trimesh.util.concatenate([peg, slab, *bands])


def main() -> None:
    print("========== Totem STL 生成 (家具边桌 v4) ==========")
    print(f"输出目录 = {os.path.relpath(_OUT_DIR, _PROJ_ROOT)}")
    print("造型：踢脚底座 + 锥形桌腿 + 中层搁板 + 宽台面\n")

    _export(gen_base_plate(), "base_plate.stl")
    _export(gen_post(), "post.stl")
    _export(gen_middle_plate(), "middle_plate.stl")
    _export(gen_top_cross(), "top_cross.stl")

    print("\nTotem furniture STL assets generated (overwritten).")
    print("  python sealp/assets/models/Totem/_preview_totem_assembled.py")
    print("  python sealp/assets/models/Totem/show_totem.py")


if __name__ == "__main__":
    main()
                                                        