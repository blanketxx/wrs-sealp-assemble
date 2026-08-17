"""RRT-free cost model for BSFS.

Design decision (locked with the user):
  * The primary cost c_k of assembly step k is the END-EFFECTOR CARTESIAN
    POLYLINE LENGTH through the mandatory keyposes of the pick-transfer-insert
    motion, NOT the motion *time*.  It is computed purely from the staging pose,
    the goal pose and fixed approach/lift/retreat clearances -- no RRT, no chosen
    grasp id, so it is cheap and deterministic.

        home -> pre-pick(lift) -> pick -> lift -> transfer -> pre-insert(lift)
             -> insert -> retreat(lift) -> home

    We use the transported OBJECT origin as the end-effector proxy (grasp
    independent). The empty-arm approach/return segments use a fixed home TCP
    when available.

  * The ADMISSIBLE LOWER BOUND used by the exact A*/branch-and-bound search is
    the straight-line Cartesian chord ||p_pick - p_place||.  Because the object
    must travel from staging to goal, any real path length >= this chord
    (triangle inequality on the sp -> sp_up -> gp_up -> gp polyline, plus the
    non-negative empty-arm segments). Hence h never overestimates -> A* optimal.

  * Two SECONDARY metrics -- minimum clearance margin and manipulability -- are
    HARD feasibility thresholds (tau_clear, tau_manip). They do NOT enter the
    scalar objective (which would break the admissibility proof); instead they
    gate feasibility and provide a lexicographic tiebreak among equal-length
    layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# fixed motion clearances (m) -- keypose geometry only, no planner involved
DEFAULT_LIFT = 0.10          # vertical lift above pick / above insert
DEFAULT_TAU_CLEAR = 0.005    # min mesh clearance margin gate (m)
DEFAULT_TAU_MANIP = 1.0e-3   # min endpoint manipulability gate


@dataclass
class CostParams:
    lift: float = DEFAULT_LIFT
    tau_clear: float = DEFAULT_TAU_CLEAR
    tau_manip: float = DEFAULT_TAU_MANIP
    include_empty_arm: bool = True
    home_tcp: Optional[np.ndarray] = None   # (3,) world xyz of the arm home TCP
    # Required-motion collision masks (paper W_req / F_{k,j}); on by default.
    use_req_motion_masks: bool = True
    req_motion_samples: int = 5
    req_motion_post_release: bool = True
    req_motion_release_radius: float = 0.015


def _v3(p: Sequence[float]) -> np.ndarray:
    a = np.asarray(p, dtype=float).reshape(-1)
    if a.shape[0] < 3:
        a = np.concatenate([a, np.zeros(3 - a.shape[0])])
    return a[:3]


def step_cost(sp: Sequence[float], gp: Sequence[float],
              params: CostParams) -> float:
    """End-effector polyline length for one pick-transfer-insert step (no RRT).

    ``sp`` = staging (pick) position, ``gp`` = goal (insert) position.
    """
    sp = _v3(sp)
    gp = _v3(gp)
    up = np.array([0.0, 0.0, float(params.lift)])
    sp_up = sp + up
    gp_up = gp + up
    length = 0.0
    if params.include_empty_arm and params.home_tcp is not None:
        home = _v3(params.home_tcp)
        length += float(np.linalg.norm(home - sp_up))     # approach (empty)
    length += float(np.linalg.norm(sp_up - sp))           # lower to pick
    length += float(np.linalg.norm(gp_up - sp_up))        # transfer (carrying)
    length += float(np.linalg.norm(gp - gp_up))           # insert
    if params.include_empty_arm and params.home_tcp is not None:
        home = _v3(params.home_tcp)
        length += float(np.linalg.norm(gp_up - home))     # retreat + return (empty)
    return length


def step_lb(sp: Sequence[float], gp: Sequence[float]) -> float:
    """Admissible lower bound on ``step_cost``: straight-line pick->place chord."""
    return float(np.linalg.norm(_v3(sp) - _v3(gp)))


def cost_params_from_args(args, *, mode: Optional[str] = None) -> CostParams:
    """Build ``CostParams`` from a CLI / experiment args namespace."""
    params = CostParams(
        lift=float(getattr(args, "lift", DEFAULT_LIFT)),
        tau_clear=float(getattr(args, "tau_clear", DEFAULT_TAU_CLEAR)),
        tau_manip=float(getattr(args, "tau_manip", DEFAULT_TAU_MANIP)),
        use_req_motion_masks=bool(getattr(args, "req_motion_masks", True)),
        req_motion_samples=int(getattr(args, "req_motion_samples", 5)),
        req_motion_post_release=bool(
            getattr(args, "req_motion_post_release", True)),
        req_motion_release_radius=float(
            getattr(args, "req_motion_release_radius", 0.015)),
    )
    # Exact A*: hard-prune only on sound attached-object sweeps.
    if mode == "exact":
        params.req_motion_post_release = False
    return params


def passes_thresholds(clearance: float, manip: float, params: CostParams) -> bool:
    """Hard secondary gates: min clearance and manipulability must exceed tau."""
    if clearance is not None and clearance < params.tau_clear:
        return False
    if manip is not None and manip < params.tau_manip:
        return False
    return True


def lex_key(length: float, min_clearance: float, min_manip: float):
    """Ascending sort key: minimise length, then MAXIMISE clearance, then manip.

    Used as the beam ordering and as the tiebreak among equal-cost exact optima.
    """
    return (float(length), -float(min_clearance), -float(min_manip))
