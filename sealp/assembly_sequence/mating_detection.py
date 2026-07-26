"""Automatic mating/contact part detection from assembly relations + goal geometry.

Replaces per-assembly hardcoded tables (``middle_plate`` vs ``post_*``, ``top_cross`` vs
``middle_plate``, ...) with two assembly-agnostic sources:

1. the asmdef ``parent_id`` of the current step, and
2. geometric proximity **at the final goal pose** between the current part and the parts that
   are already assembled.

Only the goal configuration is inspected.  Two parts that merely pass close to each other during
transport are never treated as mating: the query is "once this part is seated, which assembled
parts does it touch", which is exactly the set whose surface contact is legitimate.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# Surfaces closer than this at the goal pose count as an assembly contact.  Sized above the
# surface-sampling resolution used below and well under the smallest real clearance in the
# benchmark assemblies (parts that do not mate are centimetres apart).
CONTACT_DISTANCE_THRESHOLD = 0.005

# Surface samples per mesh for the distance estimate.  Only evaluated for pairs whose world AABBs
# are already nearly touching, so this stays cheap.
_SURFACE_SAMPLES = 3000


def _world_vertices(cmodel) -> Optional[np.ndarray]:
    trm = getattr(cmodel, "trm_mesh", None)
    if trm is None:
        return None
    verts = np.asarray(getattr(trm, "vertices", None), dtype=float)
    if verts.ndim != 2 or verts.size == 0:
        return None
    rotmat = np.asarray(cmodel.rotmat, dtype=float).reshape(3, 3)
    pos = np.asarray(cmodel.pos, dtype=float).reshape(3)
    return verts @ rotmat.T + pos


def _world_surface_points(cmodel, n_samples: int = _SURFACE_SAMPLES) -> Optional[np.ndarray]:
    """Vertices plus surface samples, in world coordinates.

    Vertices alone are not enough: a post standing under the middle of a large plate has its top
    face far from every plate vertex, so a vertex-only distance would wrongly report centimetres.
    """
    trm = getattr(cmodel, "trm_mesh", None)
    if trm is None:
        return None
    chunks: List[np.ndarray] = []
    verts = _world_vertices(cmodel)
    if verts is not None:
        chunks.append(verts)
    try:
        samples = trm.sample_surface(int(n_samples))
        if isinstance(samples, tuple):
            samples = samples[0]
        samples = np.asarray(samples, dtype=float)
        if samples.ndim == 2 and samples.size:
            rotmat = np.asarray(cmodel.rotmat, dtype=float).reshape(3, 3)
            pos = np.asarray(cmodel.pos, dtype=float).reshape(3)
            chunks.append(samples @ rotmat.T + pos)
    except Exception:
        pass
    if not chunks:
        return None
    return np.vstack(chunks)


def _world_aabb(cmodel) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    verts = _world_vertices(cmodel)
    if verts is None:
        return None
    return verts.min(axis=0), verts.max(axis=0)


def aabb_gap(cm_a, cm_b) -> Optional[float]:
    """Axis-aligned box gap, a lower bound on the true surface distance (0 when boxes overlap)."""
    box_a = _world_aabb(cm_a)
    box_b = _world_aabb(cm_b)
    if box_a is None or box_b is None:
        return None
    lo = np.maximum(box_a[0], box_b[0])
    hi = np.minimum(box_a[1], box_b[1])
    gap = np.maximum(lo - hi, 0.0)
    return float(np.linalg.norm(gap))


def surface_distance(cm_a, cm_b, threshold: float = CONTACT_DISTANCE_THRESHOLD) -> float:
    """Approximate minimum surface distance between two posed collision models.

    The AABB gap is a valid lower bound, so pairs that are clearly apart are rejected without any
    sampling.  Otherwise both surfaces are sampled and the nearest-neighbour distance is taken in
    both directions.
    """
    gap = aabb_gap(cm_a, cm_b)
    if gap is None:
        return float("inf")
    if gap > threshold:
        return gap
    pts_a = _world_surface_points(cm_a)
    pts_b = _world_surface_points(cm_b)
    if pts_a is None or pts_b is None:
        return gap
    try:
        from scipy.spatial import cKDTree
    except Exception:
        d = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)
        return float(d.min())
    d_ab = cKDTree(pts_b).query(pts_a, k=1)[0].min()
    d_ba = cKDTree(pts_a).query(pts_b, k=1)[0].min()
    return float(min(d_ab, d_ba))


def auto_detect_mating_parts(current_pid: str,
                             placed: Iterable[str],
                             *,
                             goal_model_of: Callable[[str], object],
                             parent_of: Optional[Callable[[str], Optional[str]]] = None,
                             threshold: float = CONTACT_DISTANCE_THRESHOLD,
                             verbose: bool = True) -> Set[str]:
    """Parts already assembled that the current part touches once it reaches its goal pose.

    :param goal_model_of: ``pid -> CollisionModel`` posed at the part's final assembled pose.
    :param parent_of: ``pid -> parent pid`` from the asmdef; ``"fixture"`` and ``None`` are ignored.
    """
    placed_ids = [p for p in placed if p != current_pid]
    mating: Set[str] = set()
    parent = parent_of(current_pid) if parent_of is not None else None
    if parent and parent != "fixture" and parent in placed_ids:
        mating.add(parent)

    current_goal = goal_model_of(current_pid)
    measured: List[Tuple[str, float]] = []
    if current_goal is not None:
        for other_pid in placed_ids:
            other_goal = goal_model_of(other_pid)
            if other_goal is None:
                continue
            dist = surface_distance(current_goal, other_goal, threshold=threshold)
            measured.append((other_pid, dist))
            if dist <= threshold:
                mating.add(other_pid)

    if verbose:
        _print_report(current_pid, parent, measured, mating, threshold)
    return mating


def _print_report(current_pid: str,
                  parent: Optional[str],
                  measured: Sequence[Tuple[str, float]],
                  mating: Set[str],
                  threshold: float) -> None:
    dist_of = dict(measured)
    print("[AUTO CONTACT]")
    print(f"  current = {current_pid}")
    if parent and parent in mating and dist_of.get(parent, float("inf")) > threshold:
        print(f"  parent  = {parent} (kept on the asmdef relation; "
              f"goal distance {dist_of[parent] * 1000:.1f}mm)")
    else:
        print(f"  parent  = {parent if parent else '-'}")
    near = sorted((m for m in measured if m[1] <= threshold), key=lambda kv: kv[1])
    far = sorted((m for m in measured if m[1] > threshold), key=lambda kv: kv[1])
    print(f"  goal-near parts (<= {threshold * 1000:.1f}mm) =")
    if near:
        for pid, dist in near:
            print(f"      {pid:14s} distance={dist * 1000:.2f}mm")
    else:
        print("      -")
    if far:
        preview = ", ".join(f"{pid}={dist * 1000:.0f}mm" for pid, dist in far[:6])
        print(f"  other placed parts = {preview}")
    print(f"  detected mating parts = {sorted(mating) if mating else '[]'}")


class MatingCache:
    """Per-step memo so the geometry is measured once per ``(part, placed set)``."""

    def __init__(self,
                 goal_model_of: Callable[[str], object],
                 parent_of: Optional[Callable[[str], Optional[str]]] = None,
                 threshold: float = CONTACT_DISTANCE_THRESHOLD,
                 verbose: bool = True):
        self._goal_model_of = goal_model_of
        self._parent_of = parent_of
        self._threshold = float(threshold)
        self._verbose = bool(verbose)
        self._cache: Dict[Tuple[str, frozenset], Set[str]] = {}

    def get(self, current_pid: str, placed: Iterable[str]) -> Set[str]:
        key = (current_pid, frozenset(placed))
        hit = self._cache.get(key)
        if hit is not None:
            return set(hit)
        found = auto_detect_mating_parts(
            current_pid, key[1],
            goal_model_of=self._goal_model_of,
            parent_of=self._parent_of,
            threshold=self._threshold,
            verbose=self._verbose,
        )
        self._cache[key] = set(found)
        return set(found)
