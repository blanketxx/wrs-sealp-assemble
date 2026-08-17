"""Layered per-step feasibility oracle phi_k wrapping WeightedInitialLayoutSearcher.

phi_k(a, x_k, x_{k+1..n}) is evaluated with the SAME real WRS machinery the
forward ``evaluate_layout`` uses, driven one step at a time. Fidelity levels:

    L0 : bounds / staging overlap / mesh clearance / arm keepout  (geometry)
    L1 : IK reachability + common-grasp reasoning + home collision (kinematics)
    L2 : quick pick-motion check (prescribed pre-pick/pick/post-pick)
    L3 : full transport / RRT (final witness only)

Soundness of pruning (assumption A7): a DETERMINISTIC L0/L1 failure (bounds,
overlap, missing grasp, no common grasp, home collision) is a proof of
infeasibility and may be cached / hard-pruned. An L3 RRT *timeout* is NOT a
proof of infeasibility; it returns the ``UNPROVEN`` sentinel and is never used
to hard-prune.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from find_optimal_initial_layout_tower_strict_pycharm import PickPlacePlanner
from sealp.layout.layout_robot_factory import get_layout_arm

from .cost import CostParams, step_cost, step_lb, passes_thresholds
from .req_motion_masks import forbidden_later_parts

# deterministic (cacheable / hard-prunable) failure reasons
HARD_FAIL = frozenset({
    "arm_keepout", "upright_constraint", "pair_collision", "mesh_clearance",
    "home_collision", "home_clearance", "no_common_gids", "no_grasp_collection",
    "req_motion_mask",   # later staging part intersects W_req (paper F_{k,j})
})
UNPROVEN = "unproven"    # L3 RRT timeout etc. -- never hard-prune (A7)


class StepOracle:
    """Stateful per-step certifier bound to one searcher."""

    def __init__(self, searcher, params: CostParams):
        self.s = searcher
        self.params = params
        if params.home_tcp is None:
            params.home_tcp = self._probe_home_tcp()

    def _probe_home_tcp(self) -> Optional[np.ndarray]:
        """Best-effort world TCP of the arm at home (for empty-arm cost term)."""
        try:
            arm = get_layout_arm(self.s.robot, "lft", single_arm=self.s.single_arm_mode)
            tcp = arm.gl_tcp_pos if hasattr(arm, "gl_tcp_pos") else None
            if tcp is None and hasattr(arm, "get_gl_tcp"):
                tcp = arm.get_gl_tcp()[0]
            if tcp is not None:
                return np.asarray(tcp, dtype=float).reshape(-1)[:3]
        except Exception:
            pass
        return None

    # ---- helpers -------------------------------------------------
    def cand_by_rotname(self, pid: str, rot_name: str):
        for c in self.s.rot_cands.get(pid, []) or []:
            if str(getattr(c, "rot_name", "")) == str(rot_name):
                return c
        return None

    def apply_suffix(self, assign: Dict[str, Dict], preassembled_pid: Optional[str]) -> None:
        """Pin the preassembled base at goal + every decided suffix part at staging."""
        self.s._apply_first_part_as_assembled()
        for q, rec in assign.items():
            cand = self.cand_by_rotname(q, rec["rot_name"])
            if cand is not None:
                self.s._apply_staging_pose(q, np.asarray(rec["xy"], dtype=float), cand)

    def _clearance_proxy(self, pid: str, xy: np.ndarray, staged_pids: List[str]) -> float:
        """Cheap min horizontal separation to other staged parts (tiebreak metric)."""
        p = np.asarray(xy, dtype=float)[:2]
        best = float("inf")
        for q in staged_pids:
            if q == pid:
                continue
            m = self.s.staging_models.get(q)
            if m is None:
                continue
            d = float(np.linalg.norm(np.asarray(m.pos, dtype=float)[:2] - p))
            best = min(best, d)
        return best if np.isfinite(best) else 1.0

    # ---- the oracle ----------------------------------------------
    def certify(self, pid: str, xy: np.ndarray, cand, placed: set,
                staged_pids: List[str], level: int = 2
                ) -> Tuple[Optional[Dict], str]:
        """Certify assembly step for ``pid`` at fidelity ``level`` (1, 2, or 3).

        Returns (record, "") on success, else (None, reason). ``reason`` in
        ``HARD_FAIL`` may be cached; ``UNPROVEN`` (L3 timeout) may not.
        """
        s = self.s
        # ---- L0 geometry (same per-part gates as forward evaluate_layout) ----
        if s._staging_arm_keepout_reason(pid, xy, cand):
            return None, "arm_keepout"
        if s._upright_hard_constraint_reason(pid, cand):
            return None, "upright_constraint"
        s._apply_staging_pose(pid, xy, cand)
        active = [pid] + [p for p in staged_pids if p != pid]
        if s._pairwise_collision(active_pids=active):
            return None, "pair_collision"
        if s._mesh_clearance_reason(active_pids=active):
            return None, "mesh_clearance"
        if s._robot_home_collision_reason(active_pids=[pid]):
            return None, "home_collision"
        if s._robot_home_clearance_reason(active_pids=[pid]):
            return None, "home_clearance"

        sp = s.staging_models[pid].pos.copy()
        sr = s.staging_models[pid].rotmat.copy()
        gp, gr = s.world_poses[pid]
        gp = np.asarray(gp, dtype=float)
        gr = np.asarray(gr, dtype=float)

        # ---- Required-motion collision masks W_req / F_{k,j} (paper) ----
        # Cheap sound necessary check against already-assigned later staging
        # parts before grasp reasoning / L2. Disabled via CostParams.
        if getattr(self.params, "use_req_motion_masks", True):
            hits = forbidden_later_parts(
                s, pid, sp, sr, gp, gr, staged_pids,
                n_samples=int(getattr(self.params, "req_motion_samples", 5)),
                include_post_release=bool(
                    getattr(self.params, "req_motion_post_release", True)),
                release_radius=float(
                    getattr(self.params, "req_motion_release_radius", 0.015)),
            )
            if hits:
                return None, "req_motion_mask"

        # ---- L1 kinematics / common grasp ------------------------
        gc = s._grasp_collection(pid)
        if gc is None or len(gc) == 0:
            return None, "no_grasp_collection"

        obs = s._step_obstacles(pid, placed)
        planner_obs = s._planner_obstacles(obs, current_pid=pid, placed=placed)

        best_gc, best_arm, best_gids = 0, None, None
        for arm_tag in s._arm_order(pid):
            arm = get_layout_arm(s.robot, arm_tag, single_arm=s.single_arm_mode)
            planner = PickPlacePlanner(robot=arm)
            try:
                gids = planner.reason_common_gids(
                    grasp_collection=gc,
                    goal_pose_list=[(sp, sr), (gp, gr)],
                    obstacle_list=planner_obs,
                )
            except Exception:
                gids = None
            n = len(gids) if gids else 0
            if n > best_gc:
                best_gc, best_arm, best_gids = n, arm_tag, list(gids)
        if best_gc < 1:
            return None, "no_common_gids"
        sel_gids = list(best_gids)

        arm = get_layout_arm(s.robot, best_arm, single_arm=s.single_arm_mode)

        # ---- L1.5 continuous pick-depart motion gate (ALWAYS, correctness) ----
        # Runs at every fidelity level (including the beam's default level 1) so
        # the search itself never selects a placement whose only common grasps
        # cannot be lifted along the prescribed +Z L3 segment. Endpoint IK
        # feasibility does NOT imply this local motion is feasible; the helper
        # pins HOME_JV per grasp for deterministic IK seeds.
        try:
            gids_dep, _ = s._l2_pick_depart_motion_gids(
                arm=arm, gc=gc, sp=sp, sr=sr, gids=list(best_gids))
        except Exception:
            gids_dep = list(best_gids)
        if not gids_dep:
            return None, "l2_pick_depart_motion"        # motion-ish -> not cached
        best_gc = len(gids_dep)
        sel_gids = list(gids_dep)

        # ---- L2 quick pick motion --------------------------------
        if level >= 2:
            planner = PickPlacePlanner(robot=arm)
            try:
                gids2, _ = s._l2_pick_quick_check_gids(
                    pid=pid, planner=planner, gc=gc, gids=list(sel_gids),
                    sp=sp, sr=sr, obs=obs)
            except Exception:
                gids2 = sel_gids
            if not gids2:
                return None, "l2_pick_quick_check"      # motion-ish -> not cached
            best_gc = len(gids2)
            sel_gids = list(gids2)

        # ---- metrics ---------------------------------------------
        manip = float(s._endpoint_manip(arm, gc, sp, sr, gp, gr, planner_obs))
        clearance = self._clearance_proxy(pid, xy, active)
        # hard secondary gates (clearance + manip)
        if not passes_thresholds(clearance, manip, self.params):
            return None, "below_threshold"             # marginal -> not cached
        cost = step_cost(sp, gp, self.params)

        rec = {
            "pid": pid,
            "xy": np.asarray(xy, dtype=float)[:2],
            "rot_name": str(getattr(cand, "rot_name", "unknown")),
            "pose_tag": str(getattr(cand, "tag", "unknown")),
            "arm": str(best_arm),
            "gids": [int(g) for g in sel_gids],
            "footprint": np.asarray(getattr(cand, "footprint", [0.05, 0.05]),
                                    dtype=float)[:2],
            "common_grasp_count": int(best_gc),
            "manipulability": manip,
            "clearance": float(clearance),
            "dist_to_goal": float(np.linalg.norm(sp[:2] - gp[:2])),
            "cost": float(cost),
            "cost_lb": float(step_lb(sp, gp)),
        }
        return rec, ""
