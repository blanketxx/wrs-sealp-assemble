#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate TopDownPavilionV1 mesh assets.

四柱微型观景亭 —— topdown_tower 的受控几何替换。
仅改变外观，连接区/孔位/包围盒与 tower 对齐（±5%）。

输出 7 个独立 STL（四根立柱各一份）::

    sealp/assets/models/TopDownPavilionV1/model/
        base_plate.stl
        post_bl.stl  post_fl.stl  post_br.stl  post_fr.stl
        middle_plate.stl
        top_cross.stl

运行::

    python sealp/assets/models/TopDownPavilionV1/gen_pavilion_meshes.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import trimesh
import trimesh.creation as tcreation

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEALP_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_PROJ_ROOT = os.path.dirname(_SEALP_ROOT)
_OUT_DIR = os.path.join(_HERE, "model")

# ── 目标包围盒 (m): 差异化 XY 外形, 高度/孔位/rel_pos 与 tower 一致 ───────
# 紧凑化 base/middle/top_cross 外框, 让 extent/footprint 明显区别于
# tower/totem, 迫使布局模型依赖输入几何 (对抗单任务坍缩)。
REF = {
    "base_plate": np.array([0.1700, 0.1700, 0.0243]),
    "post": np.array([0.0220, 0.0220, 0.1350]),
    "middle_plate": np.array([0.1450, 0.1450, 0.0216]),
    "top_cross": np.array([0.1000, 0.1000, 0.0675]),
}

# ── 连接区尺寸（不可改: 高度/孔位/插接几何, 保证装配与 motion 兼容）─────
BASE_H = 0.0243
BASE_POCKET_DEPTH = 0.01215
POST_OFFSET_X, POST_OFFSET_Y = 0.06375, 0.04875
BASE_SOCKET_XY = 0.02535
POST_FOOT_CLEARANCE = 0.002

POST_BODY = 0.0195
POST_FOOT = 0.022
POST_FOOT_H = 0.01215
POST_H = 0.135

MIDDLE_H = 0.0216
MIDDLE_SOCKET_DEPTH = 0.0108
MIDDLE_SOCKET_XY = 0.020

TOP_H, TOP_D = 0.0675, 0.018
PEG_XY, PEG_D, PEG_H = 0.018, 0.018, 0.020

# ── 差异化 XY 外形 (仅外框, 不影响插接) ──────────────────────────────────
BASE_X, BASE_Y = 0.170, 0.170
MIDDLE_X, MIDDLE_Y = 0.145, 0.145
TOP_W = 0.100

MIN_THICKNESS = 0.004
BBOX_TOL = 0.05


def _box(lx: float, ly: float, lz: float,
         cx: float = 0.0, cy: float = 0.0, cz: float = 0.0) -> trimesh.Trimesh:
    m = tcreation.box(extents=np.array([lx, ly, lz], dtype=float))
    m.vertices += np.array([cx, cy, cz], dtype=float)
    return m


def _box_bottom_at_z0(lx: float, ly: float, lz: float,
                      cx: float = 0.0, cy: float = 0.0) -> trimesh.Trimesh:
    return _box(lx, ly, lz, cx, cy, lz / 2.0)


def _concat(parts: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.util.concatenate(parts)


def _octagonal_slab(ox: float, oy: float, lz: float, chamfer: float,
                    cz: float) -> trimesh.Trimesh:
    """圆角八边形：中心十字 + 四边，无布尔。"""
    cx_bar = ox - 2.0 * chamfer
    cy_bar = oy - 2.0 * chamfer
    parts = [
        _box(cx_bar, oy, lz, 0.0, 0.0, cz),
        _box(ox, cy_bar, lz, 0.0, 0.0, cz),
    ]
    return _concat(parts)


def _make_rect_layer_with_holes(
    outer_x: float, outer_y: float,
    z_min: float, z_max: float,
    holes: List[Tuple[float, float, float, float]],
) -> trimesh.Trimesh:
    x_min, x_max = -outer_x / 2.0, outer_x / 2.0
    y_min, y_max = -outer_y / 2.0, outer_y / 2.0
    xs = [x_min, x_max]
    ys = [y_min, y_max]
    for hx_c, hy_c, hx, hy in holes:
        xs.extend([hx_c - hx, hx_c + hx])
        ys.extend([hy_c - hy, hy_c + hy])
    xs = sorted(set(round(v, 10) for v in xs))
    ys = sorted(set(round(v, 10) for v in ys))
    lz = z_max - z_min
    cz = (z_min + z_max) / 2.0
    parts: List[trimesh.Trimesh] = []
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
            if any(
                hcx - hhx <= cell_cx <= hcx + hhx and hcy - hhy <= cell_cy <= hcy + hhy
                for hcx, hcy, hhx, hhy in holes
            ):
                continue
            parts.append(_box(xb - xa, yb - ya, lz, cell_cx, cell_cy, cz))
    if not parts:
        raise RuntimeError("hole layer empty")
    return _concat(parts)


def _corner_pocket_holes() -> List[Tuple[float, float, float, float]]:
    hx = BASE_SOCKET_XY / 2.0
    hy = BASE_SOCKET_XY / 2.0
    return [
        (POST_OFFSET_X, POST_OFFSET_Y, hx, hy),
        (-POST_OFFSET_X, POST_OFFSET_Y, hx, hy),
        (POST_OFFSET_X, -POST_OFFSET_Y, hx, hy),
        (-POST_OFFSET_X, -POST_OFFSET_Y, hx, hy),
    ]


def gen_base_plate() -> trimesh.Trimesh:
    """圆角八边形底座 + 四角立柱孔 + 浅凸筋 + 中央装饰台。"""
    chamfer = 0.022
    bottom_h = BASE_H - BASE_POCKET_DEPTH
    deck = _octagonal_slab(BASE_X, BASE_Y, bottom_h, chamfer, bottom_h / 2.0)
    top_layer = _make_rect_layer_with_holes(
        BASE_X, BASE_Y, BASE_H - BASE_POCKET_DEPTH, BASE_H, _corner_pocket_holes())

    rib_h = 0.003
    rib_z = BASE_H - rib_h / 2.0
    ribs = [
        _box(BASE_X * 0.55, 0.006, rib_h, 0.0, 0.0, rib_z),
        _box(0.006, BASE_Y * 0.55, rib_h, 0.0, 0.0, rib_z),
    ]

    plinth_h = 0.002
    plinth = _box(0.046, 0.034, plinth_h, 0.0, 0.0, BASE_H - plinth_h / 2.0)

    step_h = 0.002
    step = _octagonal_slab(BASE_X * 0.94, BASE_Y * 0.94, step_h, chamfer * 0.85,
                           BASE_H - BASE_POCKET_DEPTH - step_h / 2.0)
    return _concat([deck, top_layer, step, *ribs, plinth])


def _post_collars(body_z0: float, body_h: float) -> List[trimesh.Trimesh]:
    """上下套环（方框套）。"""
    ring_h = 0.005
    outer = 0.021
    inner = 0.015
    wall = (outer - inner) / 2.0
    rings = []
    for zc in (body_z0 + 0.010, body_z0 + body_h - 0.010):
        rings.extend([
            _box(outer, wall, ring_h, 0.0, 0.0, zc),
            _box(wall, outer, ring_h, 0.0, 0.0, zc),
        ])
    return rings


def _post_grooved_body(body_z0: float, body_h: float) -> trimesh.Trimesh:
    """带纵槽的矩形立柱：中心芯 + 四面浅肋（装饰不超出脚垫包络）。"""
    core = 0.013
    rib_t = 0.003
    parts = [_box(core, core, body_h, 0.0, 0.0, body_z0 + body_h / 2.0)]
    half = core / 2.0 + rib_t / 2.0
    parts.extend([
        _box(core + 2 * rib_t, rib_t, body_h, 0.0, half, body_z0 + body_h / 2.0),
        _box(core + 2 * rib_t, rib_t, body_h, 0.0, -half, body_z0 + body_h / 2.0),
        _box(rib_t, core + 2 * rib_t, body_h, half, 0.0, body_z0 + body_h / 2.0),
        _box(rib_t, core + 2 * rib_t, body_h, -half, 0.0, body_z0 + body_h / 2.0),
    ])
    return _concat(parts)


def gen_post() -> trimesh.Trimesh:
    """立柱：脚垫 + 下细上粗两段 + 纵槽 + 套环。"""
    foot = _box_bottom_at_z0(POST_FOOT, POST_FOOT, POST_FOOT_H)
    lower_h = 0.050
    upper_h = POST_H - POST_FOOT_H - lower_h
    lower = _box(0.017, 0.017, lower_h, 0.0, 0.0, POST_FOOT_H + lower_h / 2.0)
    upper = _post_grooved_body(POST_FOOT_H + lower_h, upper_h)
    body_h = POST_H - POST_FOOT_H
    return _concat([foot, lower, upper, *_post_collars(POST_FOOT_H, body_h)])


def gen_middle_plate() -> trimesh.Trimesh:
    """上层平台：八边形外廓 + 中心功能孔 + 大窗装饰框 + 浅边框。"""
    chamfer = 0.018
    socket_hx = MIDDLE_SOCKET_XY / 2.0
    socket_hy = MIDDLE_SOCKET_XY / 2.0

    bottom_h = MIDDLE_H - MIDDLE_SOCKET_DEPTH
    deck = _octagonal_slab(MIDDLE_X, MIDDLE_Y, bottom_h, chamfer, bottom_h / 2.0)

    holes = [(0.0, 0.0, socket_hx, socket_hy)]
    top_layer = _make_rect_layer_with_holes(
        MIDDLE_X, MIDDLE_Y,
        MIDDLE_H - MIDDLE_SOCKET_DEPTH, MIDDLE_H,
        holes,
    )

    # 较大矩形窗口：仅装饰框，不穿透主体
    wx, wy = 0.032, 0.024
    frame_t = 0.006
    frame_h = 0.004
    frame_z = MIDDLE_H - MIDDLE_SOCKET_DEPTH / 2.0
    frame = [
        _box(wx, frame_t, frame_h, 0.0, wy / 2.0 - frame_t / 2.0, frame_z),
        _box(wx, frame_t, frame_h, 0.0, -wy / 2.0 + frame_t / 2.0, frame_z),
        _box(frame_t, wy, frame_h, wx / 2.0 - frame_t / 2.0, 0.0, frame_z),
        _box(frame_t, wy, frame_h, -wx / 2.0 + frame_t / 2.0, 0.0, frame_z),
    ]

    rim_h = 0.003
    rim_z = MIDDLE_H - rim_h / 2.0
    rim = _octagonal_slab(MIDDLE_X, MIDDLE_Y, rim_h, chamfer * 0.9, rim_z)
    return _concat([deck, top_layer, *frame, rim])


def _arch_beam_stepped() -> trimesh.Trimesh:
    """浅拱顶梁：十字肩 + box 阶梯拱（无布尔），包围盒对齐 tower top_cross。"""
    z0 = PEG_H
    h_above = TOP_H - PEG_H
    cross_z = z0 + h_above * 0.52
    cross_h = 0.022
    bar_x = _box(TOP_W, TOP_D, cross_h, 0.0, 0.0, cross_z)
    bar_y = _box(TOP_D, TOP_W, cross_h, 0.0, 0.0, cross_z)

    step_fracs = [
        (0.52, 0.008), (0.68, 0.010), (0.84, 0.010), (0.68, 0.008), (0.50, 0.006),
    ]
    arch_parts = []
    for frac, dz in step_fracs:
        zc = z0 + h_above * frac
        if zc + dz / 2.0 > TOP_H:
            dz = max(MIN_THICKNESS, 2.0 * (TOP_H - zc))
        arch_parts.append(_box(TOP_W * frac, TOP_D, dz, 0.0, 0.0, zc))

    crest_z = min(TOP_H - 0.003, z0 + h_above * 0.92)
    crest = _box(0.014, TOP_D, 0.006, 0.0, 0.0, crest_z)
    shoulder_h = cross_h
    shoulder_z = z0 + h_above * 0.36
    shoulder_y = TOP_W * 0.34
    shoulders = [
        _box(0.016, TOP_D, shoulder_h, -shoulder_y, 0.0, shoulder_z),
        _box(0.016, TOP_D, shoulder_h, shoulder_y, 0.0, shoulder_z),
    ]
    return _concat([bar_x, bar_y, *arch_parts, crest, *shoulders])


def gen_top_cross() -> trimesh.Trimesh:
    """拱形顶梁：底部插脚接口与 tower 一致。"""
    peg = _box_bottom_at_z0(PEG_XY, PEG_D, PEG_H)
    beam = _arch_beam_stepped()
    return _concat([peg, beam])


POST_FACETS = {
    "post_bl": "post_bl",
    "post_fl": "post_fl",
    "post_br": "post_br",
    "post_fr": "post_fr",
}


def _validate_mesh(mesh: trimesh.Trimesh, name: str, ref_bbox: np.ndarray) -> Dict:
    ext = mesh.bounds[1] - mesh.bounds[0]
    ratio = ext / ref_bbox
    bbox_ok = bool(np.all(ratio >= 1.0 - BBOX_TOL) and np.all(ratio <= 1.0 + BBOX_TOL))
    wt = bool(mesh.is_watertight)
    winding = bool(mesh.is_winding_consistent) if hasattr(mesh, "is_winding_consistent") else True
    vol = float(mesh.volume) if wt else 0.0
    return {
        "name": name,
        "bbox_m": [float(x) for x in ext],
        "ref_bbox_m": [float(x) for x in ref_bbox],
        "bbox_within_5pct": bbox_ok,
        "watertight": wt,
        "winding_consistent": winding,
        "volume_m3": vol,
        "n_verts": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
    }


def _export(mesh: trimesh.Trimesh, filename: str, ref_key: str) -> Dict:
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, filename)
    mesh.export(path)
    report = _validate_mesh(mesh, filename, REF[ref_key])
    status = "OK" if report["bbox_within_5pct"] and report["watertight"] else "WARN"
    print(f"[{status}] {filename:<18s} bbox={report['bbox_m']} watertight={report['watertight']}")
    return report


def main() -> None:
    print("========== TopDownPavilionV1 STL 生成 ==========")
    print(f"输出目录 = {os.path.relpath(_OUT_DIR, _PROJ_ROOT)}")
    print(f"立柱脚间隙 = {POST_FOOT_CLEARANCE*1000:.1f} mm (相对孔口)\n")

    reports: List[Dict] = []
    reports.append(_export(gen_base_plate(), "base_plate.stl", "base_plate"))
    for pid in POST_FACETS:
        reports.append(_export(gen_post(), f"{pid}.stl", "post"))
    reports.append(_export(gen_middle_plate(), "middle_plate.stl", "middle_plate"))
    reports.append(_export(gen_top_cross(), "top_cross.stl", "top_cross"))

    manifest = os.path.join(_HERE, "generation_report.json")
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({"parts": reports, "clearance_mm": POST_FOOT_CLEARANCE * 1000.0},
                  fh, indent=2)
    print(f"\n报告 -> {os.path.relpath(manifest, _PROJ_ROOT)}")
    print("完成。请配合 topdown_pavilion_v1.asmdef 使用。")


if __name__ == "__main__":
    main()
