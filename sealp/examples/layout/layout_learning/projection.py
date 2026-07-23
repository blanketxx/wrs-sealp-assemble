"""Deterministic constraint projection for generator proposals."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import features as F


def _finite_xy(xy: np.ndarray) -> np.ndarray:
    out = np.asarray(xy, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def _clamp_with_footprint(
    xy: np.ndarray,
    footprint: np.ndarray,
    bounds: Tuple[float, float, float, float],
) -> np.ndarray:
    xlo, xhi, ylo, yhi = bounds
    hx, hy = float(footprint[0]) / 2.0, float(footprint[1]) / 2.0
    x = float(np.clip(xy[0], xlo + hx, xhi - hx))
    y = float(np.clip(xy[1], ylo + hy, yhi - hy))
    return np.array([x, y], dtype=np.float32)


def _rect_overlap(
    ca: np.ndarray,
    fa: np.ndarray,
    cb: np.ndarray,
    fb: np.ndarray,
) -> Tuple[float, float]:
    """Axis-aligned footprint overlap along x / y (meters)."""
    ox = max(0.0, (float(fa[0]) + float(fb[0])) / 2.0 - abs(float(ca[0]) - float(cb[0])))
    oy = max(0.0, (float(fa[1]) + float(fb[1])) / 2.0 - abs(float(ca[1]) - float(cb[1])))
    return ox, oy


def _total_overlap(
    projected: Dict[str, np.ndarray],
    footprints: Dict[str, np.ndarray],
    preassembled: Optional[str] = None,
) -> float:
    pids = [pid for pid in projected if pid != preassembled]
    total = 0.0
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            ox, oy = _rect_overlap(
                projected[a], footprints.get(a, np.array([0.05, 0.05])),
                projected[b], footprints.get(b, np.array([0.05, 0.05])),
            )
            total += ox * oy
    return total


def _keepout_violation(
    xy: np.ndarray,
    footprint: np.ndarray,
    keepout_centers: np.ndarray,
    keepout_radius: float,
) -> bool:
    if keepout_centers is None or len(keepout_centers) == 0:
        return False
    fp_r = 0.5 * float(np.linalg.norm(footprint[:2]))
    for c in keepout_centers:
        if float(np.linalg.norm(xy - c[:2])) < (keepout_radius + fp_r):
            return True
    return False


def _push_keepout(
    xy: np.ndarray,
    footprint: np.ndarray,
    keepout_centers: np.ndarray,
    keepout_radius: float,
) -> np.ndarray:
    out = xy.copy()
    fp_r = 0.5 * float(np.linalg.norm(footprint[:2]))
    for c in keepout_centers:
        delta = out[:2] - np.asarray(c[:2], dtype=np.float32)
        dist = float(np.linalg.norm(delta))
        need = keepout_radius + fp_r + 0.01
        if dist < need:
            if dist < 1e-6:
                # 默认沿 +x 推开，避免落到左臂基座 (0,0) 附近
                delta = np.array([need, 0.0], dtype=np.float32)
            else:
                delta = delta / dist * need
            out[:2] = np.asarray(c[:2], dtype=np.float32) + delta
    return out


def _arm_rect_violation(
    xy: np.ndarray,
    footprint: np.ndarray,
    arm_xy: np.ndarray,
    clear_x: float,
    clear_y: float,
) -> bool:
    fp_x = float(footprint[0])
    fp_y = float(footprint[1])
    req_x = float(clear_x) + fp_x / 2.0
    req_y = float(clear_y) + fp_y / 2.0
    dx = abs(float(xy[0]) - float(arm_xy[0]))
    dy = abs(float(xy[1]) - float(arm_xy[1]))
    return dx < req_x and dy < req_y


def _push_arm_rect(
    xy: np.ndarray,
    footprint: np.ndarray,
    arm_xy: np.ndarray,
    clear_x: float,
    clear_y: float,
) -> np.ndarray:
    """Push part center out of arm-base rectangular keepout (minimum displacement)."""
    out = xy.copy()
    fp_x = float(footprint[0])
    fp_y = float(footprint[1])
    req_x = float(clear_x) + fp_x / 2.0
    req_y = float(clear_y) + fp_y / 2.0
    ax, ay = float(arm_xy[0]), float(arm_xy[1])
    x, y = float(out[0]), float(out[1])
    dx = abs(x - ax)
    dy = abs(y - ay)
    if dx >= req_x or dy >= req_y:
        return out
    push_x = req_x - dx
    push_y = req_y - dy
    if push_x <= push_y:
        out[0] = ax + (req_x if x >= ax else -req_x)
    else:
        out[1] = ay + (req_y if y >= ay else -req_y)
    return out


def _push_arm_keepouts(
    xy: np.ndarray,
    footprint: np.ndarray,
    arm_keepouts: List[Tuple[np.ndarray, float, float]],
) -> np.ndarray:
    out = xy.copy()
    for arm_xy, clear_x, clear_y in arm_keepouts:
        out = _push_arm_rect(out, footprint, arm_xy, clear_x, clear_y)
    return out


def _resolve_rect_overlap(
    pa: np.ndarray,
    fa: np.ndarray,
    pb: np.ndarray,
    fb: np.ndarray,
    min_spacing: float,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Separate one overlapping pair using footprint rectangles."""
    a = pa.copy()
    b = pb.copy()
    moved = False
    ox, oy = _rect_overlap(a, fa, b, fb)
    if ox <= 1e-9 and oy <= 1e-9:
        return a, b, False

    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    if ox > 1e-9:
        sign_x = 1.0 if dx >= 0 else -1.0
        if abs(dx) < 1e-6:
            sign_x = 1.0
        shift = (ox + min_spacing) * 0.5
        a[0] -= sign_x * shift
        b[0] += sign_x * shift
        moved = True
    if oy > 1e-9:
        sign_y = 1.0 if dy >= 0 else -1.0
        if abs(dy) < 1e-6:
            sign_y = 1.0
        shift = (oy + min_spacing) * 0.5
        a[1] -= sign_y * shift
        b[1] += sign_y * shift
        moved = True
    return a, b, moved


def _spread_clustered_parts(
    projected: Dict[str, np.ndarray],
    footprints: Dict[str, np.ndarray],
    bounds: Tuple[float, float, float, float],
    preassembled: Optional[str],
    min_spacing: float,
    anchor_xy: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """If free parts sit on top of each other, fan them out on a loose grid."""
    free = [pid for pid in projected if pid != preassembled]
    if len(free) < 2:
        return projected

    centers = np.stack([projected[pid] for pid in free], axis=0)
    spread = float(np.max(np.std(centers, axis=0)))
    if spread > 0.04:
        return projected

    xlo, xhi, ylo, yhi = bounds
    if anchor_xy is not None:
        cx, cy = float(anchor_xy[0]), float(anchor_xy[1])
    else:
        cx = 0.5 * (xlo + xhi)
        cy = 0.5 * (ylo + yhi)
    n = len(free)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_w = max(0.12, (xhi - xlo) / max(cols + 1, 1))
    cell_h = max(0.12, (yhi - ylo) / max(rows + 1, 1))
    out = {k: v.copy() for k, v in projected.items()}
    for idx, pid in enumerate(free):
        r, c = divmod(idx, cols)
        fp = footprints.get(pid, np.array([0.05, 0.05], dtype=np.float32))
        x = cx + (c - (cols - 1) / 2.0) * cell_w
        y = cy + (r - (rows - 1) / 2.0) * cell_h
        out[pid] = _clamp_with_footprint(
            np.array([x, y], dtype=np.float32), fp, bounds)
    return out


def project_proposal(
    xy_by_part: Dict[str, np.ndarray],
    footprints: Dict[str, np.ndarray],
    bounds: Tuple[float, float, float, float],
    preassembled: Optional[str] = None,
    goal_xy: Optional[Dict[str, np.ndarray]] = None,
    keepout_centers: Optional[np.ndarray] = None,
    keepout_radius: float = 0.12,
    arm_keepouts: Optional[List[Tuple[np.ndarray, float, float]]] = None,
    min_spacing: float = 0.02,
    max_iters: int = 24,
    spread_if_clustered: bool = True,
) -> Dict:
    """Repair invalid proposals: boundary clamp, keepout, rectangle non-overlap."""
    raw = {k: _finite_xy(v) for k, v in xy_by_part.items()}
    projected = {k: v.copy() for k, v in raw.items()}
    had_overlap = False
    n_iters = 0

    if preassembled and goal_xy and preassembled in goal_xy:
        projected[preassembled] = _finite_xy(goal_xy[preassembled])

    anchor = None
    if keepout_centers is not None and len(keepout_centers) > 0:
        anchor = np.asarray(keepout_centers[0], dtype=np.float32)[:2]
    if spread_if_clustered:
        projected = _spread_clustered_parts(
            projected, footprints, bounds, preassembled, min_spacing, anchor_xy=anchor)

    pids = list(projected.keys())
    arm_keepouts = list(arm_keepouts or [])
    for _ in range(max_iters):
        n_iters += 1
        moved = False
        for pid in pids:
            if pid == preassembled:
                continue
            fp = footprints.get(pid, np.array([0.05, 0.05], dtype=np.float32))
            before = projected[pid].copy()
            projected[pid] = _clamp_with_footprint(projected[pid], fp, bounds)
            if keepout_centers is not None:
                projected[pid] = _push_keepout(
                    projected[pid], fp, keepout_centers, keepout_radius)
            if arm_keepouts:
                projected[pid] = _push_arm_keepouts(projected[pid], fp, arm_keepouts)
            projected[pid] = _clamp_with_footprint(projected[pid], fp, bounds)
            if float(np.linalg.norm(projected[pid] - before)) > 1e-6:
                moved = True

        for i, a in enumerate(pids):
            if a == preassembled:
                continue
            for b in pids[i + 1:]:
                if b == preassembled:
                    continue
                pa, pb = projected[a], projected[b]
                fa = footprints.get(a, np.array([0.05, 0.05]))
                fb = footprints.get(b, np.array([0.05, 0.05]))
                ox, oy = _rect_overlap(pa, fa, pb, fb)
                if ox > 1e-9 or oy > 1e-9:
                    had_overlap = True
                    na, nb, pair_moved = _resolve_rect_overlap(
                        pa, fa, pb, fb, min_spacing)
                    if pair_moved:
                        projected[a] = _clamp_with_footprint(na, fa, bounds)
                        projected[b] = _clamp_with_footprint(nb, fb, bounds)
                        moved = True
        if not moved:
            break

    residual_overlap = _total_overlap(projected, footprints, preassembled)
    failure_reason = None
    if had_overlap and residual_overlap > 1e-9:
        failure_reason = "overlap_push"
    elif keepout_centers is not None:
        for pid in pids:
            if pid == preassembled:
                continue
            fp = footprints.get(pid, np.array([0.05, 0.05], dtype=np.float32))
            if _keepout_violation(projected[pid], fp, keepout_centers, keepout_radius):
                failure_reason = "keepout_push"
                break
    if failure_reason is None and arm_keepouts:
        for pid in pids:
            if pid == preassembled:
                continue
            fp = footprints.get(pid, np.array([0.05, 0.05], dtype=np.float32))
            for arm_xy, clear_x, clear_y in arm_keepouts:
                if _arm_rect_violation(projected[pid], fp, arm_xy, clear_x, clear_y):
                    failure_reason = "arm_keepout_push"
                    break
            if failure_reason:
                break

    displacement = {
        pid: float(np.linalg.norm(projected[pid] - raw.get(pid, projected[pid])))
        for pid in projected
    }
    return {
        "raw_xy": raw,
        "projected_xy": projected,
        "displacement": displacement,
        "repair_iterations": n_iters,
        "residual_overlap": float(residual_overlap),
        "failure_reason": failure_reason,
    }


def proposal_from_structured(
    structured: Dict,
    cond: Dict,
    part_ids: List[str],
) -> Dict:
    """Convert structured generator output to table-absolute xy dict."""
    bounds = F._table_bounds(cond)
    station_norm = np.asarray(structured["station_xy_norm"], dtype=np.float32)
    station_xy = F.denormalize_xy(station_norm, bounds)
    xy: Dict[str, np.ndarray] = {}
    parts = F._ordered_parts(cond)
    for i, pid in enumerate(part_ids):
        if i >= len(parts):
            break
        if bool(parts[i].get("is_first", False)):
            goal = np.asarray(parts[i].get("goal_pos", [0, 0, 0]), dtype=np.float32)[:2]
            xy[pid] = goal
            continue
        off_norm = np.asarray(structured["parts"][i]["offset_xy_norm"], dtype=np.float32)
        xy[pid] = station_xy + F.denormalize_offset(off_norm, bounds)
    return {
        "assembly_station_xy": station_xy,
        "xy": xy,
        "pose_choice": {
            pid: structured["parts"][i].get("pose_index", 0)
            for i, pid in enumerate(part_ids) if i < len(structured["parts"])
        },
        "rotation_choice": {
            pid: structured["parts"][i].get("rotation_index", 0)
            for i, pid in enumerate(part_ids) if i < len(structured["parts"])
        },
        "proposal_logprob": float(structured.get("proposal_logprob", 0.0)),
    }
