"""BSFS end-to-end orchestration entry point.

    python -m sealp.examples.layout.bsfs.run \
        --asmdef sealp/assembly_sequence/_demo_output/yuanchair.asmdef \
        --grasp-dir sealp/examples/grasp/yuanchair_grasp \
        --center-search coarse-to-fine --coarse-grid-n 4 --coarse-top-k 4 \
        --refine-grid-n 3 --refine-spacing-factor 0.5 \
        --workers 4 --parallel-level auto --witness-retries 5 --seed 0 \
        --output-json bsfs_best_layout.json

This module ONLY orchestrates the already-validated BSFS components (search,
oracle, cost, domain, pruning); it never duplicates their logic. The nine
pipeline stages are:

    [1/9] load task/robot/CAD/grasps/stable-poses/cost
    [2/9] generate coarse assembly centers
    [3/9] cheap DETERMINISTIC center screening (no RRT / evaluate_layout / BSFS)
    [4/9] refine promising center regions
    [5/9] full BSFS over the selected centers (parallel)
    [6/9] global-best aggregation (deterministic)
    [7/9] final yaw refinement (parallel per-part, sequential across parts)
    [8/9] robust full WRS evaluate_layout witness (bounded, seeded retries)
    [9/9] save self-contained result + summary + FINAL verdict

Reproducibility: every stochastic evaluation (WRS IK random restarts) is seeded
from stable task identity, so ``--workers 1`` and ``--workers 4`` produce the
same logical results (process scheduling cannot change numerics).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Deterministic numerics (MUST run before numpy imports its BLAS backend).
# OpenBLAS/MKL use multi-threaded, non-associative float reductions whose
# summation order depends on the live thread count, so an unpinned main process
# (e.g. 24 BLAS threads) and a pinned worker (1 thread) disagree at the ULP
# level -- enough to flip a near-tie yaw-refinement score and make ``--workers 1``
# vs ``--workers 4`` pick a different final heading for the same layout. Pinning
# EVERY process (the main process AND every spawn-reimported worker) to a single
# BLAS/OpenMP thread fixes the reduction order and makes the whole pipeline
# bit-reproducible across worker counts. The affected ops are tiny 6-DOF
# Jacobian/SVD computations, so the single-thread cost is negligible.
# ---------------------------------------------------------------------------
for _thr_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thr_var] = "1"

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LAYOUT_DIR = os.path.dirname(_THIS_DIR)
for _d in (_LAYOUT_DIR, _THIS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
from find_optimal_initial_layout_tower_strict_pycharm import (
    LayoutCandidate,
    RotCandidate,
    _bounds_after_rotation,
)
from sealp.examples.layout.infer_assembly_ga import (
    _build_searcher,
    _feasible_center_bounds,
    _parse_vec3,
)
from sealp.examples.layout.bsfs.search import search_site, rec_jsonable
from sealp.examples.layout.bsfs.oracle import StepOracle
from sealp.examples.layout.bsfs.cost import CostParams
from sealp.examples.layout.bsfs.seeding import (
    center_id, seed_everything, task_seed,
)

# Fail reasons that PROVE geometric infeasibility (stop retrying immediately).
DET_FAIL_KEYS = frozenset({
    "pair_collision", "mesh_clearance", "robot_home_collision", "home_clearance",
    "upright_constraint", "staging_arm_keepout",
})
# Fail reasons that MAY be caused by stochastic IK/grasp search (retry-able).
STO_FAIL_KEYS = frozenset({"no_common_gids", "l2_pick_quick_check", "reason_exception"})

# Yaw refinement keeps layout_score as the criterion, but two headings whose
# scores are within this band are treated as a TIE and resolved deterministically
# by the smallest yaw angle (then stable candidate id). The band is sized to
# exceed the observed WRS IK warm-start score noise: the witness grasp-reasoner
# (wrs/grasping/reasoner.py) calls robot.ik() without a fixed seed, so it
# warm-starts from the arm's current joint config, which depends on process
# history and therefore differs between workers=1 and workers=4 (per-heading
# score jitter ~0.04-0.06). A 0.06 band absorbs that jitter so the good heading
# cluster resolves to the same smallest-angle heading regardless of worker count,
# while remaining far below the >0.15 gap to clearly-worse headings.
YAW_TIE_TOL = 0.06


# ------------------------------------------------------------------
# args
# ------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BSFS backward search (exact + beam)")
    p.add_argument("--config", default=fol.DEFAULT_CONFIG)
    p.add_argument("--asmdef", default=fol.DEFAULT_ASMDEF)
    p.add_argument("--grasp-dir", default=fol.DEFAULT_GRASP_DIR)
    p.add_argument("--part-order", default="")
    p.add_argument("--goal-pos", default="0.373,0.0,0.0")
    # mode
    p.add_argument("--mode", choices=["exact", "beam"], default="beam")
    # ---- assembly-center search ----
    p.add_argument("--center-search", choices=["single", "all", "coarse-to-fine"],
                   default="single",
                   help="single=one center at --goal-pos; all=exhaustive grid (paper "
                        "validation); coarse-to-fine=cheap screen a coarse grid then refine "
                        "the top-k (normal full-scale runs).")
    p.add_argument("--coarse-grid-n", type=int, default=4)
    p.add_argument("--coarse-top-k", type=int, default=4)
    p.add_argument("--refine-grid-n", type=int, default=3)
    p.add_argument("--refine-spacing-factor", type=float, default=0.5)
    # legacy --all-centers knobs (still honoured by center-search=all)
    p.add_argument("--center-spacing", type=float, default=0.05)
    p.add_argument("--center-cap", type=int, default=0)
    # search hyper-params
    p.add_argument("--beam-width", type=int, default=8)
    p.add_argument("--cand-per-part", type=int, default=12)
    p.add_argument("--poses-per-xy", type=int, default=1)
    p.add_argument("--grid-spacing", type=float, default=0.05)
    p.add_argument("--max-nodes", type=int, default=0)
    p.add_argument("--exact-cell-cap", type=int, default=0)
    p.add_argument("--discrete-domain", action="store_true")
    p.add_argument("--do-quick-check", action="store_true")
    p.add_argument("--swept-prune", action="store_true")
    p.add_argument("--enable-order-x", action="store_true")
    # final precise filtering: z-yaw sampling on the stable flatsurface poses
    p.add_argument("--yaw-step-deg", type=float, default=20.0,
                   help="Final precise filtering: sample yaw about world-z every this many "
                        "degrees on each part's chosen stable pose; keep best feasible heading "
                        "(cost is yaw-invariant). 0 disables.")
    # cost model
    p.add_argument("--lift", type=float, default=0.10)
    p.add_argument("--tau-clear", type=float, default=0.005)
    p.add_argument("--tau-manip", type=float, default=1.0e-3)
    # witness / parallelism / io
    p.add_argument("--witness-retries", type=int, default=5,
                   help="Bounded retries for the full WRS witness. WRS IK uses random "
                        "restarts, so a marginal but feasible layout can fail one unlucky "
                        "draw; deterministic geometric failures stop immediately.")
    p.add_argument("--l3-witness", action="store_true")
    p.add_argument("--workers", type=int, default=4,
                   help="Worker processes for independent WRS evaluations (1=serial).")
    p.add_argument("--parallel-level", choices=["auto", "centers", "candidates", "none"],
                   default="auto")
    p.add_argument("--seed", type=int, default=0,
                   help="Global base seed; all task seeds derive deterministically from it.")
    p.add_argument("--output-json", default="")
    return p.parse_args(argv)


def _cfg_tag(args) -> str:
    """Stable task-configuration fingerprint component."""
    return "|".join(str(x) for x in (
        args.mode, args.grid_spacing, args.cand_per_part, args.poses_per_xy,
        args.beam_width, args.lift, args.tau_clear, args.tau_manip,
        args.do_quick_check, args.enable_order_x, args.part_order))


# ==================================================================
# assembly-center generation / screening / refinement
# ==================================================================
def _grid_over_bounds(bounds: Tuple[float, float, float, float], nx: int, ny: int,
                      top_z: float) -> List[np.ndarray]:
    xlo, ylo, xhi, yhi = bounds
    xs = np.linspace(xlo, xhi, max(int(nx), 1)) if xhi > xlo else np.array([0.5 * (xlo + xhi)])
    ys = np.linspace(ylo, yhi, max(int(ny), 1)) if yhi > ylo else np.array([0.5 * (ylo + yhi)])
    return [np.array([float(x), float(y), top_z], dtype=float) for y in ys for x in xs]


def _screen_centers(searcher, centers: List[np.ndarray]):
    """Cheap DETERMINISTIC necessary-condition screening (no RRT / full witness /
    BSFS). Only ``_assembly_region_reject_reason`` (center bounds + preassembled
    seat arm-keepout + deterministic seat/env collision) may HARD-prune a center.
    Returns (survivors, rejected[list of (center, reason)])."""
    survivors, rejected = [], []
    for c in centers:
        reason = searcher._assembly_region_reject_reason(np.asarray(c, dtype=float))
        if reason is None:
            survivors.append(c)
        else:
            rejected.append((c, reason))
    return survivors, rejected


def _rank_centers(searcher, centers: List[np.ndarray]) -> List[np.ndarray]:
    """Rank survivors by a cheap STAGING-INDEPENDENT heuristic (centrality in the
    feasible region => more staging room). This only ORDERS centers; it is never
    an infeasibility proof. Deterministic tie-break by (x, y)."""
    xlo, ylo, xhi, yhi = _feasible_center_bounds(searcher)
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)

    def _key(c):
        c = np.asarray(c, dtype=float)
        d = float(np.hypot(c[0] - cx, c[1] - cy))
        return (d, round(float(c[0]), 4), round(float(c[1]), 4))

    return sorted(centers, key=_key)


def _refine_centers(retained: List[np.ndarray], coarse_spacing: Tuple[float, float],
                    args, bounds: Tuple[float, float, float, float],
                    top_z: float) -> List[np.ndarray]:
    """Local refine-grid around each retained center; clip to legal bounds + dedup."""
    xlo, ylo, xhi, yhi = bounds
    sx = float(coarse_spacing[0]) * float(args.refine_spacing_factor)
    sy = float(coarse_spacing[1]) * float(args.refine_spacing_factor)
    m = max(int(args.refine_grid_n), 1)
    half = (m - 1) / 2.0
    out, seen = [], set()
    for c in retained:
        c = np.asarray(c, dtype=float)
        for iy in range(m):
            for ix in range(m):
                x = float(np.clip(c[0] + (ix - half) * sx, xlo, xhi))
                y = float(np.clip(c[1] + (iy - half) * sy, ylo, yhi))
                key = (round(x, 4), round(y, 4))
                if key in seen:
                    continue
                seen.add(key)
                out.append(np.array([x, y, top_z], dtype=float))
    return out


def plan_centers(searcher, args) -> Dict:
    """Return the centers that will receive FULL BSFS plus stage bookkeeping."""
    top_z = float(searcher.table_top_z)
    bounds = _feasible_center_bounds(searcher)
    info = {"mode": args.center_search, "stage2_skipped": False,
            "stage3_skipped": False, "stage4_skipped": False,
            "coarse_centers_total": 0, "coarse_centers_hard_rejected": 0,
            "coarse_centers_survived": 0, "coarse_centers_retained": 0,
            "refined_centers_generated": 0, "refined_centers_unique": 0,
            "full_bsfs_centers_evaluated": 0}

    if args.center_search == "single":
        goal = _parse_vec3(args.goal_pos, (0.373, 0.0, 0.0))
        xlo, ylo, xhi, yhi = bounds
        c = np.array([float(np.clip(goal[0], xlo, xhi)),
                      float(np.clip(goal[1], ylo, yhi)), top_z], dtype=float)
        info.update(stage2_skipped=True, stage3_skipped=True, stage4_skipped=True)
        info["full_bsfs_centers_evaluated"] = 1
        info["centers"] = [c]
        return info

    if args.center_search == "all":
        xlo, ylo, xhi, yhi = bounds
        step = max(float(args.center_spacing), 0.02)
        nx = max(1, int(np.floor((xhi - xlo) / step)) + 1)
        ny = max(1, int(np.floor((yhi - ylo) / step)) + 1)
        coarse = _grid_over_bounds(bounds, nx, ny, top_z)
        info["coarse_centers_total"] = len(coarse)
        survivors, rejected = _screen_centers(searcher, coarse)
        info["coarse_centers_hard_rejected"] = len(rejected)
        info["coarse_centers_survived"] = len(survivors)
        survivors = _rank_centers(searcher, survivors)
        if int(args.center_cap) > 0:
            survivors = survivors[:int(args.center_cap)]
        info.update(stage4_skipped=True)
        info["full_bsfs_centers_evaluated"] = len(survivors)
        info["centers"] = survivors
        return info

    # ---- coarse-to-fine ----
    xlo, ylo, xhi, yhi = bounds
    ncoarse = max(int(args.coarse_grid_n), 1)
    coarse = _grid_over_bounds(bounds, ncoarse, ncoarse, top_z)
    info["coarse_centers_total"] = len(coarse)
    coarse_sx = (xhi - xlo) / max(ncoarse - 1, 1)
    coarse_sy = (yhi - ylo) / max(ncoarse - 1, 1)

    survivors, rejected = _screen_centers(searcher, coarse)
    info["coarse_centers_hard_rejected"] = len(rejected)
    info["coarse_centers_survived"] = len(survivors)

    ranked = _rank_centers(searcher, survivors)
    retained = ranked[:max(int(args.coarse_top_k), 1)]
    info["coarse_centers_retained"] = len(retained)

    refined = _refine_centers(retained, (coarse_sx, coarse_sy), args, bounds, top_z)
    info["refined_centers_generated"] = len(refined)
    refined_ok, _ = _screen_centers(searcher, refined)   # deterministic re-screen
    refined_ok = _rank_centers(searcher, refined_ok)
    info["refined_centers_unique"] = len(refined_ok)
    info["full_bsfs_centers_evaluated"] = len(refined_ok)
    info["centers"] = refined_ok
    return info


# ==================================================================
# layout <-> assign helpers  (reused by every stage)
# ==================================================================
def _layout_from_assign(searcher, assign: Dict[str, Dict],
                        preassembled_pid: Optional[str],
                        station: np.ndarray) -> LayoutCandidate:
    xy = {pid: np.asarray(rec["xy"], dtype=float) for pid, rec in assign.items()}
    if preassembled_pid is not None and preassembled_pid in searcher.world_poses:
        gp, _ = searcher.world_poses[preassembled_pid]
        xy[preassembled_pid] = np.asarray(gp[:2], dtype=float)
    cand = LayoutCandidate(xy=xy)
    cand.assembly_station_pos = np.asarray(station, dtype=float).copy()
    cand.forced_rot_name = {pid: rec["rot_name"] for pid, rec in assign.items()}
    return cand


def _compact_assign(assign: Dict[str, Dict]) -> Dict[str, Dict]:
    return {pid: {"xy": np.asarray(rec["xy"], dtype=float).tolist(),
                  "rot_name": rec["rot_name"], "cost": float(rec.get("cost", 0.0))}
            for pid, rec in assign.items()}


def _cand_by_rotname(searcher, pid: str, rot_name: str):
    for c in searcher.rot_cands.get(pid, []) or []:
        if str(getattr(c, "rot_name", "")) == str(rot_name):
            return c
    return None


def _register_yaw_cand(searcher, pid: str, base_rotmat: np.ndarray, deg: float) -> str:
    base_name = None
    for c in searcher.rot_cands.get(pid, []) or []:
        if np.allclose(np.asarray(c.rotmat), base_rotmat, atol=1e-6):
            base_name = str(c.rot_name)
            break
    tag_base = (base_name or "fs").split("_y")[0]
    name = f"{tag_base}_y{int(round(deg)) % 360:03d}"
    if _cand_by_rotname(searcher, pid, name) is not None:
        return name
    th = np.deg2rad(float(deg))
    c, s = np.cos(th), np.sin(th)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ np.asarray(base_rotmat, dtype=float)
    verts = searcher.mesh_vertices[pid]
    bmin, _, extent = _bounds_after_rotation(verts, R)
    z_off = float(searcher.table_top_z) + float(searcher.table_clearance) - float(bmin[2])
    searcher.rot_cands[pid].append(RotCandidate(
        rotmat=R, z_offset=z_off, tag="fs_yaw", extent=np.asarray(extent, dtype=float),
        footprint=np.asarray(extent[:2], dtype=float), rot_name=name, fs_pos=None))
    return name


def _parse_yaw_deg(rot_name: str) -> float:
    s = str(rot_name)
    if "_y" in s:
        try:
            return float(int(s.rsplit("_y", 1)[1]))
        except Exception:
            return 0.0
    return 0.0


def _ensure_yaw_cand(searcher, pid: str, rot_name: str) -> None:
    """Make sure a refined yaw candidate (``fs_XX_yDDD``) exists in rot_cands so
    forced_rot_name resolves in ANY process (main or worker)."""
    rot_name = str(rot_name)
    if "_y" not in rot_name:
        return
    if _cand_by_rotname(searcher, pid, rot_name) is not None:
        return
    base_name, deg_s = rot_name.rsplit("_y", 1)
    try:
        deg = int(deg_s)
    except Exception:
        return
    base = _cand_by_rotname(searcher, pid, base_name)
    if base is not None:
        _register_yaw_cand(searcher, pid, np.asarray(base.rotmat, dtype=float), deg)


def _ensure_all_yaw(searcher, assign: Dict[str, Dict]) -> None:
    for pid, rec in assign.items():
        _ensure_yaw_cand(searcher, pid, rec["rot_name"])


# ==================================================================
# robust full WRS witness (canonical for ALL complete verifications)
# ==================================================================
def _classify_fail(cand) -> Tuple[str, str]:
    part = str(getattr(cand, "fail_part", "") or "")
    detail = getattr(cand, "fail_detail", {}) or {}
    reason = str(getattr(cand, "fail_reason", "") or "")
    if part.startswith("final_") or part == "order_x_constraint":
        return "DETERMINISTIC", reason or part
    if sum(int(detail.get(k, 0)) for k in DET_FAIL_KEYS) > 0:
        return "DETERMINISTIC", reason
    if sum(int(detail.get(k, 0)) for k in STO_FAIL_KEYS) > 0:
        return "STOCHASTIC", reason
    # empty detail with a non-step fail_part => deterministic geometric/global
    if part and not detail:
        return "DETERMINISTIC", reason or part
    return "STOCHASTIC", reason or "unknown"


def _layout_fingerprint(assign: Dict[str, Dict], center, part_order, cfg_tag) -> str:
    c = np.asarray(center, dtype=float).reshape(-1)
    items = []
    for pid in part_order:
        if pid in assign:
            xy = np.asarray(assign[pid]["xy"], dtype=float)
            items.append(f"{pid}:{xy[0]:.4f},{xy[1]:.4f}:{assign[pid]['rot_name']}")
    return f"C{c[0]:.4f},{c[1]:.4f}|{'|'.join(items)}|{cfg_tag}"


def robust_evaluate_layout(searcher, assign, preassembled_pid, center, args,
                           part_order, stats) -> Tuple[str, object, Dict]:
    """Canonical bounded, deterministically-seeded full WRS witness.

    Returns (status, last_cand, info) with status in
    {SUCCESS, DETERMINISTIC_FAIL, STOCHASTIC_FAIL}. A deterministic geometric
    failure stops immediately (not hidden behind retries); stochastic IK/grasp
    failures are retried up to --witness-retries. Every retry uses a reproducible
    seed derived from the layout fingerprint + retry index.
    """
    fp = _layout_fingerprint(assign, center, part_order, _cfg_tag(args))
    retries = max(int(args.witness_retries), 1)
    base = int(args.seed)
    _ensure_all_yaw(searcher, assign)
    attempts, last_cand = [], None
    for r in range(retries):
        s = task_seed(base, fp, "witness", r)
        seed_everything(s)
        cand = _layout_from_assign(searcher, assign, preassembled_pid, center)
        stats["full_witness_calls"] = stats.get("full_witness_calls", 0) + 1
        stats["witness_attempts_total"] = stats.get("witness_attempts_total", 0) + 1
        last_cand = cand
        if bool(searcher.evaluate_layout(cand)):
            attempts.append({"retry": r, "seed": s, "result": "SUCCESS"})
            return "SUCCESS", cand, {
                "configured_retries": retries, "attempts": attempts,
                "successful_retry_index": r, "reused_from_cache": False,
                "fingerprint": fp, "seeds": [a["seed"] for a in attempts]}
        kind, reason = _classify_fail(cand)
        attempts.append({"retry": r, "seed": s, "result": "FAIL",
                         "reason": reason, "kind": kind})
        if kind == "DETERMINISTIC":
            return "DETERMINISTIC_FAIL", cand, {
                "configured_retries": retries, "attempts": attempts,
                "successful_retry_index": None, "reused_from_cache": False,
                "fingerprint": fp, "seeds": [a["seed"] for a in attempts]}
    return "STOCHASTIC_FAIL", last_cand, {
        "configured_retries": retries, "attempts": attempts,
        "successful_retry_index": None, "reused_from_cache": False,
        "fingerprint": fp, "seeds": [a["seed"] for a in attempts]}


# ==================================================================
# result extraction
# ==================================================================
def _final_certification(searcher, args, assign, center, preassembled_pid) -> Dict[str, Dict]:
    """Re-run StepOracle.certify along the pick order on the FINAL poses so the
    saved per-step record (grasp ids/arm/clearance/manip/cost) matches exactly."""
    params = CostParams(lift=float(args.lift), tau_clear=float(args.tau_clear),
                        tau_manip=float(args.tau_manip))
    oracle = StepOracle(searcher, params)
    searcher._set_assembly_station(np.asarray(center, dtype=float))
    _ensure_all_yaw(searcher, assign)
    pick_order = searcher._active_pick_part_order()
    n = len(pick_order)
    cid = center_id(center)
    recs: Dict[str, Dict] = {}
    for k in range(n, 0, -1):
        pid = pick_order[k - 1]
        if pid not in assign:
            continue
        placed = set(pick_order[:k - 1])
        if preassembled_pid is not None:
            placed.add(preassembled_pid)
        suffix = {q: assign[q] for q in pick_order[k:] if q in assign}
        oracle.apply_suffix(suffix, preassembled_pid)
        staged = ([preassembled_pid] if preassembled_pid else []) + list(pick_order[k:])
        cand = _cand_by_rotname(searcher, pid, assign[pid]["rot_name"])
        xy = np.asarray(assign[pid]["xy"], dtype=float)
        seed_everything(task_seed(int(args.seed), cid, "finalcert", pid))
        rec, reason = oracle.certify(pid, xy, cand, placed, staged, level=2)
        if rec is None:
            recs[pid] = {"pid": pid, "certified": False, "reason": reason,
                         "rot_name": assign[pid]["rot_name"],
                         "yaw_deg": _parse_yaw_deg(assign[pid]["rot_name"]),
                         "xy": [float(xy[0]), float(xy[1])]}
        else:
            j = rec_jsonable(rec)
            j["certified"] = True
            j["yaw_deg"] = _parse_yaw_deg(rec.get("rot_name", ""))
            recs[pid] = j
    return recs


def _extract_layout(searcher, layout, preassembled_pid, center, score, cost,
                    l3_pass, assign, fingerprint, witness_info) -> Dict:
    return {
        "assembly_center": np.asarray(center, dtype=float).tolist(),
        "layout_score": float(score),
        "total_cost": float(cost) if cost is not None else None,
        "l2_pass": True,
        "l3_pass": bool(l3_pass),
        "preassembled_part": preassembled_pid,
        "_assign": _compact_assign(assign),
        "_fingerprint": fingerprint,
        "_witness_info": witness_info,
        "picked_parts": [p for p in layout.xy if p != preassembled_pid],
        "best_layout": {
            pid: {
                "init_pos": [float(layout.xy[pid][0]), float(layout.xy[pid][1]),
                             float(layout.z_offset.get(pid, 0.0))],
                "init_rotmat": np.asarray(layout.chosen_rotmat.get(pid, np.eye(3)),
                                          dtype=float).reshape(-1).tolist(),
                "pose_tag": layout.pose_tag.get(pid),
                "arm_choice": layout.arm_choice.get(pid),
                "grasp_count": int(layout.grasp_counts.get(pid, 0)),
                "manipulability": float(layout.per_part_manip.get(pid, 0.0)),
                "dist_to_goal": float(layout.per_part_dist.get(pid, 0.0)),
            }
            for pid in layout.xy
        },
    }


def _better(a: Optional[Dict], b: Optional[Dict]) -> bool:
    """Fully deterministic global-best test: minimise objective cost, then break
    ties by the DETERMINISTIC layout fingerprint.

    ``layout_score`` is intentionally NOT used to choose between equal-cost
    layouts: it is derived from WRS IK (warm-start dependent, so not bit-stable
    across worker counts) and is retained only as a reported diagnostic. Ordering
    by cost -> fingerprint makes the selection independent of process scheduling
    (workers=1 and workers=4 pick the identical layout)."""
    if a is None:
        return False
    if b is None:
        return True
    ca, cb = a.get("total_cost"), b.get("total_cost")
    if ca is not None and cb is not None and abs(ca - cb) > 1e-9:
        return ca < cb
    return str(a.get("_fingerprint", "")) < str(b.get("_fingerprint", ""))


# ==================================================================
# one-site full BSFS (runs inside a worker or serially in main)
# ==================================================================
def _site_best(searcher, args, center, stats, cert_batch_fn=None) -> Optional[Dict]:
    reason = searcher._assembly_region_reject_reason(np.asarray(center, dtype=float))
    if reason is not None:
        return None
    stats["sites_tried"] += 1
    assigns = search_site(searcher, args, center, args.mode, stats, verbose=True,
                          cert_batch_fn=cert_batch_fn)
    if not assigns:
        return None
    stats["sites_feasible"] += 1
    first_pid = searcher._first_part_id()
    preassembled_pid = first_pid if searcher.preassemble_first_part else None
    searcher._set_assembly_station(np.asarray(center, dtype=float))
    part_order = list(searcher.part_order)
    best = None
    for assign in assigns:
        status, cand, info = robust_evaluate_layout(
            searcher, assign, preassembled_pid, center, args, part_order, stats)
        if status != "SUCCESS":
            fr = getattr(cand, "fail_part", None), getattr(cand, "fail_reason", "")
            print(f"    [witness-{status}] part={fr[0]} reason={fr[1]}")
            continue
        score = float(getattr(cand, "layout_score", 0.0))
        cost = float(sum(r.get("cost", 0.0) for r in assign.values()))
        l3_pass = False
        if args.l3_witness:
            try:
                l3_pass = bool(searcher.validate_full_sequence_l3(cand, verbose=False))
            except Exception:
                l3_pass = False
            if not l3_pass:
                continue
        fp = info["fingerprint"]
        res = _extract_layout(searcher, cand, preassembled_pid, center, score, cost,
                              l3_pass, assign, fp, info)
        if _better(res, best):
            best = res
    return best


# ==================================================================
# yaw refinement (parallel per-part, sequential across parts)
# ==================================================================
def _yaw_refine(searcher, args, assign, center, preassembled_pid, stats,
                witness_batch_fn=None) -> Dict[str, Dict]:
    step_deg = float(args.yaw_step_deg)
    if step_deg <= 0:
        return assign
    searcher._set_assembly_station(np.asarray(center, dtype=float))
    part_order = list(searcher.part_order)
    base_rot = {}
    for pid, rec in assign.items():
        c = _cand_by_rotname(searcher, pid, rec["rot_name"])
        base_rot[pid] = np.asarray(c.rotmat, dtype=float) if c is not None else np.eye(3)
    cur = {pid: dict(rec) for pid, rec in assign.items()}
    degs = list(range(0, 360, max(int(step_deg), 1)))

    def _score_of(trial) -> float:
        status, cand, _ = robust_evaluate_layout(
            searcher, trial, preassembled_pid, center, args, part_order, stats)
        return float(getattr(cand, "layout_score", 0.0)) if status == "SUCCESS" else -1.0

    for _ in range(2):
        improved = False
        for pid in list(cur.keys()):
            # build the 18 independent yaw trials for this part
            trials, names = [], []
            for d in degs:
                name = _register_yaw_cand(searcher, pid, base_rot[pid], d)
                t = {q: dict(r) for q, r in cur.items()}
                t[pid]["rot_name"] = name
                trials.append(t)
                names.append((d, name))
            if witness_batch_fn is not None:
                scores = witness_batch_fn([{"assign": t, "center": np.asarray(center).tolist()}
                                           for t in trials])
            else:
                scores = [_score_of(t) for t in trials]
            # criterion = MAXIMISE layout_score; but resolve exact/near ties
            # DETERMINISTICALLY by smallest yaw angle, then stable candidate id
            # (name). A heading whose score is within YAW_TIE_TOL of the best is
            # considered tied, so IK warm-start score noise cannot flip the pick.
            feas = [(int(d), str(name), float(sc))
                    for (d, name), sc in zip(names, scores) if float(sc) > -1.0]
            if not feas:
                continue
            best_sc = max(s for _, _, s in feas)
            band = sorted(((d, name) for (d, name, s) in feas
                           if s >= best_sc - YAW_TIE_TOL), key=lambda t: (t[0], t[1]))
            chosen = band[0][1]
            if chosen != cur[pid]["rot_name"]:
                cur[pid]["rot_name"] = chosen
                improved = True
        if not improved:
            break
    return cur


# ==================================================================
# spawn-safe worker pool
# ==================================================================
_W: Dict[str, object] = {}


def _init_worker(args_ns):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    _W["args"] = args_ns
    _W["searcher"] = _build_searcher(args_ns)
    _W["oracle"] = StepOracle(_W["searcher"], CostParams(
        lift=float(args_ns.lift), tau_clear=float(args_ns.tau_clear),
        tau_manip=float(args_ns.tau_manip)))


def _worker_center(center) -> Dict:
    searcher, args = _W["searcher"], _W["args"]
    stats = _new_stats()
    best = _site_best(searcher, args, np.asarray(center, dtype=float), stats)
    return {"best": best, "stats": stats}


def _worker_certify(task) -> Dict:
    """Reconstruct the scene for one child and call the SAME StepOracle.certify."""
    searcher, oracle = _W["searcher"], _W["oracle"]
    seed_everything(int(task["seed"]))
    center = np.asarray(task["center"], dtype=float)
    searcher._set_assembly_station(center)
    searcher._apply_first_part_as_assembled()
    for s in task["suffix"]:
        _ensure_yaw_cand(searcher, s["pid"], s["rot_name"])
        cand = _cand_by_rotname(searcher, s["pid"], s["rot_name"])
        if cand is not None:
            searcher._apply_staging_pose(s["pid"], np.asarray(s["xy"], dtype=float), cand)
    _ensure_yaw_cand(searcher, task["pid"], task["rot_name"])
    cand = _cand_by_rotname(searcher, task["pid"], task["rot_name"])
    rec, reason = oracle.certify(task["pid"], np.asarray(task["xy"], dtype=float),
                                 cand, set(task["placed"]), list(task["staged"]),
                                 level=int(task["level"]))
    return {"rec": rec_jsonable(rec), "reason": reason}


def _worker_yaw(task) -> float:
    searcher, args = _W["searcher"], _W["args"]
    stats = {}
    status, cand, _ = robust_evaluate_layout(
        searcher, task["assign"], searcher._first_part_id()
        if searcher.preassemble_first_part else None,
        np.asarray(task["center"], dtype=float), args, list(searcher.part_order), stats)
    return float(getattr(cand, "layout_score", 0.0)) if status == "SUCCESS" else -1.0


def _new_stats() -> Dict:
    return {"certifications": 0, "max_depth_certified": 0, "sites_tried": 0,
            "sites_feasible": 0, "hall_pruned": 0, "node_expansions": 0,
            "hard_prunes": 0, "propagation_prunes": 0, "full_witness_calls": 0,
            "witness_attempts_total": 0}


def _merge_stats(agg: Dict, other: Dict) -> None:
    for k, v in other.items():
        if k == "max_depth_certified":
            agg[k] = max(agg.get(k, 0), v)
        else:
            agg[k] = agg.get(k, 0) + v


# ==================================================================
# main orchestration
# ==================================================================
def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    seed_everything(int(args.seed))
    print("========== BSFS end-to-end pipeline ==========")

    # ---- [1/9] load ----
    tl = time.time()
    searcher = _build_searcher(args)
    time_load = time.time() - tl
    print(f"[1/9] load task/robot/CAD/grasps/stable-poses/cost : "
          f"{len(searcher.part_order)} parts ({time_load:.1f}s)")

    workers = max(1, int(args.workers))
    part_order = list(searcher.part_order)

    # ---- [2/9]-[4/9] center generation / screening / refinement ----
    tc = time.time()
    cinfo = plan_centers(searcher, args)
    centers = cinfo["centers"]
    time_center_gen = time.time() - tc
    if not centers:
        print("[2/9] centers : NONE feasible")
        print("FINAL: FAIL (no feasible assembly center)")
        raise SystemExit("No feasible assembly center found.")
    _p2 = "SKIPPED (single)" if cinfo["stage2_skipped"] else \
        f"{cinfo['coarse_centers_total']} coarse centers"
    _p3 = "SKIPPED" if cinfo["stage3_skipped"] else \
        (f"{cinfo['coarse_centers_survived']} survived / "
         f"{cinfo['coarse_centers_hard_rejected']} hard-rejected "
         f"-> retain {cinfo['coarse_centers_retained']}")
    _p4 = "SKIPPED" if cinfo["stage4_skipped"] else \
        f"{cinfo['refined_centers_unique']} refined centers"
    print(f"[2/9] generate coarse assembly centers   : {_p2}")
    print(f"[3/9] cheap deterministic center screen   : {_p3}")
    print(f"[4/9] refine promising center regions     : {_p4}")
    print(f"      -> full BSFS on {len(centers)} center(s) "
          f"[center-search={args.center_search}]")

    # ---- parallel-level decision ----
    plevel = args.parallel_level
    if plevel == "auto":
        plevel = "none" if workers == 1 else ("centers" if len(centers) > 1 else "candidates")
    if workers == 1:
        plevel = "none"
    print(f"[5/9] full BSFS ({args.mode}) | workers={workers} parallel-level={plevel}")

    agg = _new_stats()
    best = None
    t_first_feasible = None
    worker_tasks_submitted = 0
    worker_tasks_completed = 0
    pool = None
    tb = time.time()
    try:
        if plevel == "centers":
            ctx = __import__("multiprocessing").get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                       initializer=_init_worker, initargs=(args,))
            worker_tasks_submitted = len(centers)
            for out in pool.map(_worker_center, [c.tolist() for c in centers]):
                worker_tasks_completed += 1
                _merge_stats(agg, out["stats"])
                if out["best"] is not None and t_first_feasible is None:
                    t_first_feasible = time.time() - tb
                if _better(out["best"], best):
                    best = out["best"]
        elif plevel == "candidates":
            ctx = __import__("multiprocessing").get_context("spawn")
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                       initializer=_init_worker, initargs=(args,))
            center = centers[0]

            def _cert_batch(tasks):
                nonlocal worker_tasks_submitted, worker_tasks_completed
                for t in tasks:
                    t["center"] = np.asarray(center, dtype=float).tolist()
                worker_tasks_submitted += len(tasks)
                res = list(pool.map(_worker_certify, tasks))
                worker_tasks_completed += len(res)
                return res

            best = _site_best(searcher, args, center, agg, cert_batch_fn=_cert_batch)
            if best is not None:
                t_first_feasible = time.time() - tb
        else:  # serial
            for ci, center in enumerate(centers):
                print(f"  [site {ci}] {args.mode} @({center[0]:.3f},{center[1]:.3f})")
                res = _site_best(searcher, args, center, agg)
                if res is not None and t_first_feasible is None:
                    t_first_feasible = time.time() - tb
                if _better(res, best):
                    best = res
        time_bsfs = time.time() - tb

        # ---- [6/9] aggregate + global best ----
        print(f"[6/9] global-best aggregation : {agg['sites_feasible']}/"
              f"{agg['sites_tried']} feasible site(s)")
        if best is None:
            _print_summary(args, agg, None, workers, plevel, cinfo)
            print("FINAL: FAIL (no L2-passing layout found)")
            raise SystemExit("No L2-passing layout found.")

        first_pid = searcher._first_part_id()
        preassembled_pid = first_pid if searcher.preassemble_first_part else None
        center = np.asarray(best["assembly_center"], dtype=float)
        final_assign = dict(best["_assign"])

        # ---- [7/9] final yaw refinement ----
        ty = time.time()
        if float(args.yaw_step_deg) > 0:
            wbf = None
            if plevel != "none" and pool is not None:
                def wbf(tasks):
                    return list(pool.map(_worker_yaw, tasks))
            final_assign = _yaw_refine(searcher, args, final_assign, center,
                                       preassembled_pid, agg, witness_batch_fn=wbf)
            print(f"[7/9] final yaw refinement ({args.yaw_step_deg:.0f} deg) done")
        else:
            print("[7/9] final yaw refinement : SKIPPED (--yaw-step-deg 0)")
        time_yaw = time.time() - ty
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    # ---- [8/9] robust full WRS witness (with fingerprint cache reuse) ----
    tw = time.time()
    new_fp = _layout_fingerprint(final_assign, center, part_order, _cfg_tag(args))
    cached_reuses = 0
    if new_fp == best.get("_fingerprint") and best.get("_witness_info"):
        status, cand = "SUCCESS", None
        witness_info = dict(best["_witness_info"])
        witness_info["reused_from_cache"] = True
        cached_reuses = 1
        l2_pass, l3_pass = True, bool(best.get("l3_pass"))
        print("[8/9] robust full WRS witness : REUSED cached stage-5 witness "
              "(identical layout)")
    else:
        status, cand, witness_info = robust_evaluate_layout(
            searcher, final_assign, preassembled_pid, center, args, part_order, agg)
        l2_pass = (status == "SUCCESS")
        l3_pass = False
        if l2_pass and args.l3_witness:
            try:
                l3_pass = bool(searcher.validate_full_sequence_l3(cand, verbose=False))
            except Exception:
                l3_pass = False
        print(f"[8/9] robust full WRS witness : {status} "
              f"(retry_idx={witness_info.get('successful_retry_index')})")
    time_witness = time.time() - tw

    if not l2_pass:
        _print_summary(args, agg, None, workers, plevel, cinfo)
        print(f"FINAL: FAIL ({status}) after {witness_info.get('configured_retries')} retries")
        raise SystemExit("Final robust witness failed.")

    # rebuild the extracted layout for the FINAL (possibly refined) assign
    if cand is None:  # cache-reuse path: re-derive layout fields deterministically
        seed_everything(task_seed(int(args.seed), new_fp, "witness", 0))
        cand = _layout_from_assign(searcher, final_assign, preassembled_pid, center)
        searcher.evaluate_layout(cand)
    cost = float(sum(r.get("cost", 0.0) for r in final_assign.values()))
    best = _extract_layout(searcher, cand, preassembled_pid, center,
                           float(getattr(cand, "layout_score", 0.0)), cost, l3_pass,
                           final_assign, new_fp, witness_info)

    # per-step certification + grasp-robustness on the exact final poses
    per_step = _final_certification(searcher, args, final_assign, center, preassembled_pid)
    grasp_counts = {}
    for pid, entry in best["best_layout"].items():
        r = per_step.get(pid)
        if r is None:
            continue
        entry["rot_name"] = r.get("rot_name")
        entry["yaw_deg"] = r.get("yaw_deg", 0.0)
        if r.get("certified", True):
            entry["gids"] = r.get("gids")
            entry["arm_choice"] = r.get("arm", entry.get("arm_choice"))
            if r.get("common_grasp_count") is not None:
                grasp_counts[pid] = int(r["common_grasp_count"])
    min_common = min(grasp_counts.values()) if grasp_counts else None

    # ---- [9/9] assemble self-contained result + save ----
    total_runtime = round(time.time() - t0, 2)
    witness_status = "L3_PASS" if l3_pass else "L2_PASS"
    for k in ("_assign", "_fingerprint", "_witness_info"):
        best.pop(k, None)
    result = {
        "method": f"BSFS-{args.mode}",
        "asmdef": args.asmdef,
        "num_parts": len(searcher.part_order),
        "witness_status": witness_status,
        "objective_cost": best["total_cost"],
        "layout_fingerprint": new_fp,
        "execution": {
            "workers": workers, "parallel_level": plevel,
            "center_search_mode": args.center_search, "seed": int(args.seed),
            "worker_tasks_submitted": worker_tasks_submitted,
            "worker_tasks_completed": worker_tasks_completed,
        },
        "center_search": {k: v for k, v in cinfo.items() if k != "centers"},
        "timing": {
            "time_load": round(time_load, 2),
            "time_center_generation": round(time_center_gen, 2),
            "time_center_screening": 0.0, "time_center_refinement": 0.0,
            "time_bsfs": round(time_bsfs, 2),
            "time_to_first_feasible": None if t_first_feasible is None
            else round(t_first_feasible, 2),
            "time_yaw_refinement": round(time_yaw, 2),
            "time_final_witness": round(time_witness, 2),
            "total_runtime": total_runtime,
        },
        "bsfs_stats": {
            "oracle_certifications": int(agg.get("certifications", 0)),
            "node_expansions": int(agg.get("node_expansions", 0)),
            "hard_prunes": int(agg.get("hard_prunes", 0)),
            "hall_prunes": int(agg.get("hall_pruned", 0)),
            "propagation_prunes": int(agg.get("propagation_prunes", 0)),
            "beam_width": int(args.beam_width),
            "max_depth_certified": int(agg.get("max_depth_certified", 0)),
        },
        "witness": {
            "witness_retries_configured": int(args.witness_retries),
            "full_witness_calls": int(agg.get("full_witness_calls", 0)),
            "witness_attempts_total": int(agg.get("witness_attempts_total", 0)),
            "cached_witness_reuses": cached_reuses,
            **witness_info,
        },
        "grasp_robustness": {
            "per_step_common_grasp_count": grasp_counts,
            "min_common_grasp_count": min_common,
        },
        "per_step_certification": per_step,
        **best,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as stream:
            stream.write(text)
        print(f"[9/9] saved result : {args.output_json}")
    else:
        print("[9/9] result (no --output-json; printed below)")
        print(text)

    _print_summary(args, agg, best, workers, plevel, cinfo)
    cc = best["assembly_center"]
    print(f"FINAL: SUCCESS  BSFS-{args.mode} witness={witness_status} "
          f"cost={best['total_cost']:.4f} score={best['layout_score']:.4f} "
          f"@({cc[0]:.3f},{cc[1]:.3f}) min_grasp={min_common} runtime={total_runtime:.1f}s")
    return result


def _print_summary(args, agg, best, workers, plevel, cinfo) -> None:
    print("\n===== BSFS pipeline summary =====")
    print(f"  mode / center-search   : {args.mode} / {args.center_search}")
    print(f"  workers / parallel     : {workers} / {plevel}")
    print(f"  full-BSFS centers      : {cinfo.get('full_bsfs_centers_evaluated', 0)}")
    print(f"  sites tried / feasible : {agg.get('sites_tried', 0)} / "
          f"{agg.get('sites_feasible', 0)}")
    print(f"  oracle certifications  : {agg.get('certifications', 0)}")
    print(f"  node expansions        : {agg.get('node_expansions', 0)}")
    print(f"  hard / hall / prop prune: {agg.get('hard_prunes', 0)} / "
          f"{agg.get('hall_pruned', 0)} / {agg.get('propagation_prunes', 0)}")
    print(f"  full witness calls     : {agg.get('full_witness_calls', 0)} "
          f"(attempts {agg.get('witness_attempts_total', 0)})")
    if best is not None:
        print(f"  preassembled / picked  : {best.get('preassembled_part')} / "
              f"{best.get('picked_parts')}")
        print(f"  objective cost / score : {best.get('total_cost')} / "
              f"{best.get('layout_score')}")


if __name__ == "__main__":
    main()
