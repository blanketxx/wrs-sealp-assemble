"""Required-motion collision masks ``W_req`` / ``F_{k,j}`` (paper §IV-C).

For a candidate initial placement ``x_k``, assembly site ``a`` and grasp ``g``,
``W_req(a, x_k, g)`` is the swept volume of the three *prescribed* local motions:

  1. post-grasp retreat from the initial pose (pick depart, world +Z)
  2. prescribed insertion (place approach along the mating axis)
  3. post-release retreat from the partial assembly (place depart)

A later part ``j > k`` belongs to the forbidden set ``F_{k,j}`` when its initial
geometry intersects ``W_req``. Backward search already has those later parts
assigned, so an intersection rejects the current placement / grasp (paper:
remove ``g`` from ``G_k^{loc}``).

Soundness
---------
* Motions (1)–(2) move the *attached* object. Sampling the part collision model
  along those translations is a sound necessary condition: a hit proves the
  prescribed local motion collides with the later staging part.
* Motion (3) is an empty-hand retreat. Without a gripper/robot model we use a
  thin under-approximating capsule along the depart axis (never larger than the
  true EE swept volume we can justify). A capsule hit is treated as a hard
  prune; a miss does not prove clearance — L1/L2/L3 still run.

The legacy XY transfer-corridor prune (``swept_segment_mask``) is *not* part of
``W_req``; it remains optional and off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Keep distances / directions aligned with the L3 transport defaults in
# find_optimal_initial_layout_tower_strict_pycharm.py (duplicated here to avoid
# importing that heavy module — and Panda3D — at mask-construction time).
PICK_DEPART_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
PICK_DEPART_DIST = 0.05
PLACE_APPROACH_DIR = np.array([0.0, 0.0, -1.0], dtype=float)
PLACE_APPROACH_DIST = 0.05
PLACE_DEPART_DIR = np.array([0.0, 0.0, 1.0], dtype=float)
PLACE_DEPART_DIST = 0.05

DEFAULT_SAMPLES = 5
# Under-approx EE / jaw half-width for post-release capsule (m).
DEFAULT_RELEASE_RADIUS = 0.015


@dataclass(frozen=True)
class MotionSegment:
    """One prescribed local-motion segment contributing to ``W_req``."""
    kind: str                 # "pick_depart" | "insertion" | "post_release"
    p0: np.ndarray            # (3,) world start
    p1: np.ndarray            # (3,) world end
    rotmat: np.ndarray        # (3,3) object orientation during the segment
    mode: str                 # "attached" (mesh CD) | "release_capsule"


def _unit(v: Sequence[float], default: np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(-1)[:3]
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        d = np.asarray(default, dtype=float).reshape(3)
        dn = float(np.linalg.norm(d))
        return d / dn if dn >= 1e-9 else np.array([0.0, 0.0, 1.0])
    return a / n


def _mating_dirs(searcher, pid: str, gr: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """World approach / depart unit vectors, falling back to L3 defaults."""
    approach, depart = None, None
    fn = getattr(searcher, "_assembly_mating_dirs", None)
    if callable(fn):
        try:
            approach, depart = fn(pid, gr)
        except Exception:
            approach, depart = None, None
    if approach is None:
        approach = PLACE_APPROACH_DIR
    if depart is None:
        depart = PLACE_DEPART_DIR
    return (_unit(approach, PLACE_APPROACH_DIR),
            _unit(depart, PLACE_DEPART_DIR))


def prescribed_segments(searcher, pid: str,
                        sp: Sequence[float], sr: np.ndarray,
                        gp: Sequence[float], gr: np.ndarray,
                        *,
                        pick_depart_dist: float = PICK_DEPART_DIST,
                        place_approach_dist: float = PLACE_APPROACH_DIST,
                        place_depart_dist: float = PLACE_DEPART_DIST,
                        include_post_release: bool = True,
                        ) -> List[MotionSegment]:
    """Build the three (or two) prescribed local-motion segments for part ``pid``."""
    sp = np.asarray(sp, dtype=float).reshape(3)
    gp = np.asarray(gp, dtype=float).reshape(3)
    sr = np.asarray(sr, dtype=float).reshape(3, 3)
    gr = np.asarray(gr, dtype=float).reshape(3, 3)
    lift = _unit(PICK_DEPART_DIR, PICK_DEPART_DIR)
    approach, depart = _mating_dirs(searcher, pid, gr)

    segs = [
        MotionSegment(
            kind="pick_depart",
            p0=sp.copy(),
            p1=sp + lift * float(pick_depart_dist),
            rotmat=sr.copy(),
            mode="attached",
        ),
        MotionSegment(
            kind="insertion",
            # pre-insert -> seated (place approach travels *along* approach into gp)
            p0=gp - approach * float(place_approach_dist),
            p1=gp.copy(),
            rotmat=gr.copy(),
            mode="attached",
        ),
    ]
    if include_post_release:
        segs.append(MotionSegment(
            kind="post_release",
            p0=gp.copy(),
            p1=gp + depart * float(place_depart_dist),
            rotmat=gr.copy(),
            mode="release_capsule",
        ))
    return segs


def later_staging_pids(searcher, pid: str,
                       staged_pids: Sequence[str]) -> List[str]:
    """Assembly-later parts among ``staged_pids`` (paper ``j > k``)."""
    order = list(getattr(searcher, "_active_pick_part_order", lambda: [])() or [])
    if not order:
        order = list(getattr(searcher, "part_order", []) or [])
    try:
        k = order.index(pid)
        later = set(order[k + 1:])
    except ValueError:
        later = set(staged_pids) - {pid}
    out = []
    for q in staged_pids:
        if q == pid or q not in later:
            continue
        if q not in getattr(searcher, "staging_models", {}):
            continue
        out.append(q)
    return out


def _segment_samples(p0: np.ndarray, p1: np.ndarray, n: int) -> List[np.ndarray]:
    n = max(int(n), 2)
    return [(1.0 - t) * p0 + t * p1 for t in np.linspace(0.0, 1.0, n)]


def _attached_sweep_hit(searcher, pid: str, seg: MotionSegment,
                        later: Sequence[str], n_samples: int
                        ) -> Optional[str]:
    """Translate ``pid``'s collision model along ``seg``; return first hit pid."""
    cm = searcher.staging_models.get(pid)
    if cm is None:
        return None
    old_pos = np.asarray(cm.pos, dtype=float).copy()
    old_rot = np.asarray(cm.rotmat, dtype=float).copy()
    try:
        cm.rotmat = np.asarray(seg.rotmat, dtype=float)
        for pos in _segment_samples(seg.p0, seg.p1, n_samples):
            cm.pos = np.asarray(pos, dtype=float)
            for q in later:
                other = searcher.staging_models.get(q)
                if other is None:
                    continue
                try:
                    if cm.is_mcdwith(other):
                        return q
                except Exception:
                    continue
        return None
    finally:
        cm.pos = old_pos
        cm.rotmat = old_rot


def part_xy_radius(searcher, pid: str) -> float:
    """Half-diagonal of the largest staging footprint for ``pid`` (m)."""
    best = 0.02
    for c in (getattr(searcher, "rot_cands", {}) or {}).get(pid, []) or []:
        fp = np.asarray(getattr(c, "footprint", [0.04, 0.04]), dtype=float)[:2]
        best = max(best, 0.5 * float(np.linalg.norm(fp)))
    return best


# back-compat alias used inside this module
_part_xy_radius = part_xy_radius


def _point_segment_dist_xy(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    a2, b2, p2 = a[:2], b[:2], p[:2]
    ab = b2 - a2
    denom = float(ab @ ab) or 1.0
    t = float(np.clip(((p2 - a2) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p2 - (a2 + t * ab)))


def _release_capsule_hit(searcher, seg: MotionSegment, later: Sequence[str],
                         radius: float) -> Optional[str]:
    """Under-approx post-release EE corridor vs later-part staging origins.

    Uses the staging origin only (no outer footprint inflation) so a hit is a
    conservative *under*-approximation of a true EE / later-part collision and
    remains a sound hard-prune. Misses are left to L1/L2/L3.
    """
    for q in later:
        cm = searcher.staging_models.get(q)
        if cm is None:
            continue
        cxy = np.asarray(cm.pos, dtype=float)
        d = _point_segment_dist_xy(cxy, seg.p0, seg.p1)
        if d <= float(radius):
            return q
    return None


def forbidden_later_parts(searcher, pid: str,
                          sp: Sequence[float], sr: np.ndarray,
                          gp: Sequence[float], gr: np.ndarray,
                          staged_pids: Sequence[str],
                          *,
                          n_samples: int = DEFAULT_SAMPLES,
                          include_post_release: bool = True,
                          release_radius: float = DEFAULT_RELEASE_RADIUS,
                          ) -> List[Tuple[str, str]]:
    """Return ``[(later_pid, segment_kind), ...]`` for every ``F_{k,j}`` hit.

    Empty list means no later staging part intersects the constructed ``W_req``.
    """
    later = later_staging_pids(searcher, pid, staged_pids)
    if not later:
        return []
    hits: List[Tuple[str, str]] = []
    seen = set()
    for seg in prescribed_segments(
            searcher, pid, sp, sr, gp, gr,
            include_post_release=include_post_release):
        if seg.mode == "attached":
            q = _attached_sweep_hit(searcher, pid, seg, later, n_samples)
        else:
            q = _release_capsule_hit(searcher, seg, later, release_radius)
        if q is not None and (q, seg.kind) not in seen:
            seen.add((q, seg.kind))
            hits.append((q, seg.kind))
    return hits


def w_req_cell_mask(grid, searcher, pid: str,
                    sp: Sequence[float], sr: np.ndarray,
                    gp: Sequence[float], gr: np.ndarray,
                    *,
                    n_samples: int = DEFAULT_SAMPLES,
                    include_post_release: bool = True,
                    release_radius: float = DEFAULT_RELEASE_RADIUS,
                    inflate: float = 0.0) -> int:
    """Bitmap of staging cells whose centres lie inside an XY inflation of ``W_req``.

    Intended for *forward* domain filtering of undecided later parts (paper
    ``F_{k,j}`` shrinking). Not used to block earlier parts in backward search
    (those are absent during step ``k`` by suffix preservation).
    """
    mask = 0
    segs = prescribed_segments(
        searcher, pid, sp, sr, gp, gr,
        include_post_release=include_post_release)
    # Approximate object XY radius for attached sweeps.
    obj_r = _part_xy_radius(searcher, pid) + float(inflate)
    for seg in segs:
        r = release_radius + float(inflate) if seg.mode == "release_capsule" else obj_r
        for j, c in enumerate(grid.cell_xy):
            if _point_segment_dist_xy(c, seg.p0, seg.p1) <= r:
                mask |= (1 << j)
    return mask


def filter_xy_by_w_req(cand_xy: Sequence[np.ndarray],
                       searcher, pid: str,
                       sp: Sequence[float], sr: np.ndarray,
                       gp: Sequence[float], gr: np.ndarray,
                       *,
                       include_post_release: bool = True,
                       release_radius: float = DEFAULT_RELEASE_RADIUS,
                       inflate: float = 0.0) -> List[np.ndarray]:
    """Drop continuous ``(x,y)`` candidates that fall inside ``F`` for this step."""
    segs = prescribed_segments(
        searcher, pid, sp, sr, gp, gr,
        include_post_release=include_post_release)
    obj_r = _part_xy_radius(searcher, pid) + float(inflate)
    kept = []
    for xy in cand_xy:
        p = np.asarray(xy, dtype=float)
        blocked = False
        for seg in segs:
            r = (release_radius + float(inflate)
                 if seg.mode == "release_capsule" else obj_r)
            # candidate is for a *later* part; use that part's own radius when
            # known via inflate; here inflate carries the later footprint margin.
            if _point_segment_dist_xy(p, seg.p0, seg.p1) <= r:
                blocked = True
                break
        if not blocked:
            kept.append(xy)
    return kept
