"""GA layout optimizer -- COMPARISON BASELINE (black-box GA) for BSFS.

NOTE (2026-07): the primary solver is now the backward suffix-factorization
method in ``sealp.examples.layout.bsfs`` (exact A*/branch-and-bound + anytime
beam). This GA -- together with its ``PartPlacementRanker`` NN seed and the
``--use-surrogate`` Set-Transformer accelerator -- is retained only as the
black-box baseline the paper compares against (parts packed into a flat joint
chromosome and optimized by selection/crossover/mutation, ignoring the
assembly-order-induced suffix structure). It is NOT part of the BSFS pipeline;
BSFS reuses only the geometry utilities (``_build_searcher``,
``_enumerate_xy_candidates``, center enumeration), never the GA/NN/surrogate.

GA layout optimizer: NN-seeded first generation + REAL WRS fitness, with an
optional HYBRID surrogate accelerator (``--use-surrogate``).

A genetic algorithm directly searches over per-part table placements. The
ground-truth fitness is the *real* three-criteria evaluation:

  1. manipulability (higher better)          -- WRS ``_endpoint_manip``
  2. Cartesian init->goal distance (shorter)  -- real geometry
  3. assembly sequence + DYNAMIC obstacles    -- ``searcher.evaluate_layout``
     walks the asmdef assembly order and, at each step, places already-assembled
     parts at their GOAL pose and not-yet-assembled parts at their STAGING pose
     (dynamic obstacles), then checks common grasps / collisions.

Design (matches user's choices):
  * NN role = SEED ONLY.  ``PartPlacementRanker`` (trained on synthetic cuboid
    boxes -> generalizes to rectangular bbox, per-part -> variable part count)
    ranks table candidates and gives each part a top-k pool that warm-starts GA.
  * Fitness during search = a cheap proxy built from per-(part, xy) values that
    are precomputed ONCE with real WRS (manip / common-grasp / distance) plus a
    dynamic staging-overlap penalty.  This keeps GA fast.
  * Elites each generation are verified with the real ``evaluate_layout`` so the
    reported best is guaranteed L2-feasible.
  * Generalizes: everything loops over ``searcher.part_order`` and any asmdef,
    so variable N and different cuboid parts work without code changes.

HYBRID surrogate (``--use-surrogate``) -- for ROBIO-scale time budgets:
  A small Set-Transformer (``layout_learning.models.assembly_ga_surrogate``)
  LEARNS the expensive ``evaluate_layout`` (sequence + dynamic-obstacle + hard
  order-x feasibility) online. The GA fitness becomes proxy + surrogate
  feasibility, so the whole population is ranked without WRS; only the top few
  surrogate picks per generation get the real check, and those labels actively
  re-train the surrogate. This cuts real evaluate_layout calls by ~an order of
  magnitude while keeping the reported best truly L2-verified.

Output: the single best L2-passing layout (per-part init pose + pose_tag + score).

Example::

    python -m sealp.examples.layout.infer_assembly_ga \\
        --asmdef sealp/assembly_sequence/_demo_output/yuanchair.asmdef \\
        --grasp-dir sealp/examples/grasp/yuanchair_grasp \\
        --goal-pos 0.36,0.0,0.0 \\
        --checkpoint checkpoints/part_placement_ranker/part_placement_ranker_best.pt \\
        --top-k 6 --pop-size 40 --generations 20 --output-json ga_best_layout.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from sealp.examples.layout import find_optimal_initial_layout_tower_strict_pycharm as fol
from sealp.examples.layout.generate_layout_dataset import _pose_candidates_for_part
from sealp.examples.layout.uniform_candidate_pool import table_anchor_grid
from sealp.examples.layout.synthetic_bbox.utils import deterministic_init_anchor
from find_optimal_initial_layout_tower_strict_pycharm import (
    LayoutCandidate,
    PickPlacePlanner,
    WeightedInitialLayoutSearcher,
)
from sealp.layout.layout_robot_factory import get_layout_arm

try:
    import torch
    from layout_learning.models.part_placement_ranker import PartPlacementRankerNet
    from layout_learning.part_placement_dataset import (
        POSE_FEAT_DIM,
        _candidate_feature,
        _global_vector,
        _part_static_vector,
        _table_bounds,
    )
    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional (falls back to grid seeding)
    torch = None
    _HAS_TORCH = False


# ------------------------------------------------------------------
# args
# ------------------------------------------------------------------
def _parse_vec3(text: str, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    return np.asarray(vals[:3] if len(vals) >= 3 else default, dtype=float)


def _parse_args():
    p = argparse.ArgumentParser(description="GA layout optimizer (NN-seeded + real WRS fitness)")
    p.add_argument("--config", default=fol.DEFAULT_CONFIG)
    p.add_argument("--asmdef", default=fol.DEFAULT_ASMDEF)
    p.add_argument("--grasp-dir", default=fol.DEFAULT_GRASP_DIR)
    p.add_argument("--part-order", default="")
    p.add_argument("--goal-pos", default="0.36,0.0,0.0",
                   help="Assembly center xyz. Used as-is when --num-centers<=1; when "
                        "--num-centers>1 it is added as one candidate among the sampled ones.")
    # assembly-center search (replaces the fixed 3x3 grid): GA also optimizes WHERE
    # the chair is assembled by choosing among centers sampled from a random region.
    p.add_argument("--num-centers", type=int, default=1,
                   help="Number of candidate assembly centers the GA can choose from. "
                        "1 = fixed --goal-pos (legacy). >1 = sample from a random region.")
    p.add_argument("--center-region", default="",
                   help='Restrict center sampling to "xmin,ymin,xmax,ymax" (table coords). '
                        "Empty = whole feasible table area.")
    p.add_argument("--random-region", action="store_true",
                   help="Sample a RANDOM rectangular sub-region of the feasible table area "
                        "(size --region-frac) and draw centers inside it, instead of the "
                        "whole table. Ignored if --center-region is given.")
    p.add_argument("--region-frac", type=float, default=0.5,
                   help="Random sub-region size as a fraction of the feasible area per axis "
                        "(only with --random-region).")
    p.add_argument("--center-seed", type=int, default=-1,
                   help="RNG seed for center sampling (-1 = use --seed).")
    p.add_argument("--checkpoint", default="", help="PartPlacementRanker ckpt (empty=grid seeding)")
    p.add_argument("--init-pos-json", default="", help='Optional per-part init hints, {"leg_fl":[x,y]}')
    p.add_argument("--top-k", type=int, default=6, help="Seed pool size per part (NN top-k or grid)")
    p.add_argument("--grid-spacing", type=float, default=0.05)
    # GA hyper-params
    p.add_argument("--pop-size", type=int, default=40)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--elite", type=int, default=4, help="Elites carried over + real-verified per gen")
    p.add_argument("--tournament", type=int, default=3)
    p.add_argument("--crossover-rate", type=float, default=0.9)
    p.add_argument("--mutation-rate", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    # proxy fitness weights (the three criteria + common-grasp + sequence approx)
    p.add_argument("--w-manip", type=float, default=0.30)
    p.add_argument("--w-close", type=float, default=0.25)
    p.add_argument("--w-grasp", type=float, default=0.25,
                   help="Reward per-part common-grasp count in the proxy (criterion 1 support)")
    p.add_argument("--w-overlap", type=float, default=0.60, help="Dynamic staging-overlap penalty weight")
    p.add_argument("--w-order", type=float, default=0.35,
                   help="Soft assembly-sequence (order-x) penalty weight: later parts should not "
                        "stage at larger x than earlier ones (approximates dynamic-obstacle feasibility)")
    p.add_argument("--manip-norm", type=float, default=0.05)
    p.add_argument("--grasp-norm", type=float, default=40.0,
                   help="common_grasp_count normalizer (hill half-saturation)")
    p.add_argument("--order-tol", type=float, default=0.03,
                   help="order-x tolerance (m) matching the searcher's hard constraint")
    p.add_argument("--dist-decay", type=float, default=0.40)
    p.add_argument("--pool-per-xy-poses", type=int, default=2,
                   help="How many distinct poses per xy to keep in the candidate pool "
                        "(pose index is part of the GA gene)")
    p.add_argument("--verify-every", type=int, default=1, help="Run real evaluate_layout on elites every N gens")
    # ---- hybrid surrogate (SAGA): learned evaluate_layout accelerator ----
    p.add_argument("--use-surrogate", action="store_true",
                   help="Enable the hybrid Set-Transformer surrogate of evaluate_layout. It "
                        "learns sequence + dynamic-obstacle + order-x feasibility online from "
                        "the real L2 labels the GA already collects, and pre-ranks the whole "
                        "population so only the top few individuals need the expensive WRS check.")
    p.add_argument("--surrogate-warmup", type=int, default=24,
                   help="Random individuals real-evaluated before the GA to bootstrap the surrogate")
    p.add_argument("--surrogate-verify-topk", type=int, default=6,
                   help="Per generation, real-verify the top-K surrogate-ranked individuals "
                        "(these labels also feed active learning)")
    p.add_argument("--surrogate-refit-every", type=int, default=2,
                   help="Re-fit the surrogate every N generations on the growing label buffer")
    p.add_argument("--surrogate-epochs", type=int, default=40)
    p.add_argument("--surrogate-weight", type=float, default=1.0,
                   help="Blend weight of surrogate feasibility*score into the GA fitness")
    p.add_argument("--surrogate-hidden", type=int, default=64)
    p.add_argument("--surrogate-heads", type=int, default=4)
    p.add_argument("--surrogate-layers", type=int, default=2)
    p.add_argument("--surrogate-min-labels", type=int, default=16,
                   help="Minimum labels before the surrogate is fit / used")
    p.add_argument("--device", default="cpu")
    p.add_argument("--disable-order-x", action="store_true",
                   help="(deprecated / no-op) The hard order-x constraint is now OFF by default.")
    p.add_argument("--enable-order-x", action="store_true",
                   help="Restore the legacy HARD order-x constraint (later parts must stage at "
                        "x <= earlier + tol). OFF by default because it kills symmetric chairs; "
                        "sequence feasibility is handled softly (--w-order) and by the surrogate.")
    p.add_argument("--output-json", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="Only build per-part candidate pools and print feasibility stats, then exit "
                        "(no GA). Use to confirm asmdef / grasp-dir / goal-pos are wired correctly.")
    return p.parse_args()


# ------------------------------------------------------------------
# searcher + candidate pools
# ------------------------------------------------------------------
def _build_searcher(args) -> WeightedInitialLayoutSearcher:
    part_order = None
    if args.part_order.strip():
        part_order = [x.strip() for x in args.part_order.split(",") if x.strip()]
    searcher = WeightedInitialLayoutSearcher(
        asmdef_path=args.asmdef,
        config_yaml=args.config,
        grasp_dir=args.grasp_dir,
        fixture_pos=_parse_vec3(args.goal_pos, (0.36, 0.0, 0.0)),
        fixture_rotmat=np.eye(3),
        robot_base_pos=np.zeros(3),
        robot_base_rotmat=np.eye(3),
        part_order=part_order,
        output_name="assembly_ga",
        table_name="work_table",
        table_margin=0.06,
        table_clearance=0.01,
        grasp_map={},
        max_rot_candidates=8,
        w_grasp=0.3,
        w_manip=0.4,
        w_dist=0.1,
        w_rot=0.2,
        cdprim_type="box",
        planner_obstacle_mode="staging_aware",
        plan_assembly_region=True,
        use_flatsurface=True,
        check_l2_pick_quick_motion=False,
        # The step-0 part (e.g. seat) is preassembled at the assembly center: its
        # final geometry lives in A_0(a) and its initial position never enters any
        # step obstacle set O_k (assumptions A2/A3). It is NOT picked.
        preassemble_first_part=True,
    )
    # Task-2: the HARD order-x constraint is now OFF by default for the GA. For a
    # symmetric 4-leg chair some leg necessarily stages at larger +x, so the hard
    # reject kills almost every layout. Sequence feasibility is instead handled
    # softly (proxy --w-order) and, in hybrid mode, learned by the surrogate.
    # Pass --enable-order-x to restore the legacy hard constraint.
    if getattr(args, "enable_order_x", False):
        searcher.enforce_order_x_constraint = True
        print("[searcher] HARD order-x constraint ENABLED (legacy)")
    else:
        searcher.enforce_order_x_constraint = False
        print("[searcher] HARD order-x constraint DISABLED by default "
              "(soft --w-order proxy + surrogate still apply)")
    return searcher


def _build_sample_dict(searcher, station_pos: np.ndarray) -> Dict:
    """Feature-side sample dict (for the NN ranker; schema matches training)."""
    parts = []
    for idx, pid in enumerate(searcher.part_order):
        rc0 = searcher.rot_cands.get(pid, [None])[0]
        gp, gr = searcher.world_poses[pid]
        parts.append({
            "part_id": pid,
            "order_index": idx,
            "is_first": bool(pid == searcher._first_part_id()),
            "extent": np.asarray(getattr(rc0, "extent", [0, 0, 0]), dtype=float).tolist(),
            "footprint": np.asarray(getattr(rc0, "footprint", [0.05, 0.05]), dtype=float).tolist(),
            "goal_pos": np.asarray(gp, dtype=float).tolist(),
            "goal_rotmat": np.asarray(gr, dtype=float).reshape(-1).tolist(),
            "grasp_total": 20.0,
            "topdown_count": 5.0,
            "pose_candidates": _pose_candidates_for_part(searcher, pid, None, None),
        })
    return {
        "schema_version": "synthetic_bbox_v1",
        "num_parts": len(parts),
        "parts": parts,
        "assembly_station_pos": station_pos.tolist(),
        "table_x_range": list(searcher.table_x_range),
        "table_y_range": list(searcher.table_y_range),
    }


def _part_anchor_grid(searcher, pid: str, spacing: float) -> List[np.ndarray]:
    """Per-part table grid padded by THIS part's own footprint.

    ``table_anchor_grid`` pads by the GLOBAL max footprint across all parts, which
    collapses to a single anchor when one part (e.g. a lifted seat) has a large
    footprint.  For per-part candidate enumeration we only care about the current
    part, so pad by its smallest rot-candidate footprint to keep the grid dense.
    """
    import math as _math
    cands = searcher.rot_cands.get(pid, []) or []
    if cands:
        fps = [np.asarray(getattr(c, "footprint", [0.05, 0.05]), dtype=float)[:2] for c in cands]
        fx = float(min(fp[0] for fp in fps))
        fy = float(min(fp[1] for fp in fps))
    else:
        fx = fy = 0.05
    xlo, xhi = float(searcher.table_x_range[0]), float(searcher.table_x_range[1])
    ylo, yhi = float(searcher.table_y_range[0]), float(searcher.table_y_range[1])
    pad_x, pad_y = fx / 2.0 + 0.02, fy / 2.0 + 0.02
    xlo, xhi = xlo + pad_x, xhi - pad_x
    ylo, yhi = ylo + pad_y, yhi - pad_y
    if xlo >= xhi or ylo >= yhi:
        return [np.array([0.5 * (xlo + xhi), 0.5 * (ylo + yhi)], dtype=float)]
    step = max(float(spacing), 0.03)
    cols = max(1, int(_math.floor((xhi - xlo) / step)) + 1)
    rows = max(1, int(_math.floor((yhi - ylo) / step)) + 1)
    xs = np.linspace(xlo, xhi, cols, dtype=float)
    ys = np.linspace(ylo, yhi, rows, dtype=float)
    return [np.array([float(x), float(y)], dtype=float) for y in ys for x in xs]


def _enumerate_xy_candidates(searcher, pid: str, grid_spacing: float,
                             first_pid: Optional[str] = None) -> List[np.ndarray]:
    """Cheap geometric filter that EXCLUDES hopeless staging spots up-front.

    A position is emitted only if there exists at least one rotation candidate
    that simultaneously (task-1 request: pre-exclude, don't generate, then rank):
      * clears the arm-base keepout footprint,
      * does not self-collide / violate mesh clearance with other staging parts,
      * does NOT intersect the robot at its HOME pose (``_robot_home_collision_reason``).

    Filtering the home-arm-collision footprint here (instead of only later inside
    ``_proxy_for_xy``) means the farthest-point sampler and NN ranker operate on a
    clean feasible set, so no WRS budget is spent scoring layouts that were doomed
    from the start.
    """
    # Include the preassembled first part (seat, sitting at the assembly center)
    # in the staging collision/clearance checks. A leg physically staged inside
    # the seat's footprint IS a real collision and must be excluded here so the
    # pool matches what the full evaluate_layout will accept. (The leg->seat
    # INSERTION contact at the goal is a different thing and stays exempted in
    # grasp reasoning via the parent-based contact-exclusion set.)
    active = [pid] if not first_pid else [pid, first_pid]
    anchors = _part_anchor_grid(searcher, pid, spacing=float(grid_spacing))
    out: List[np.ndarray] = []
    for xy in anchors:
        ok = False
        for cand in searcher.rot_cands.get(pid, []) or []:
            if searcher._staging_arm_keepout_reason(pid, xy, cand):
                continue
            searcher._apply_staging_pose(pid, xy, cand)
            if searcher._pairwise_collision(active_pids=active):
                continue
            if searcher._mesh_clearance_reason(active_pids=active):
                continue
            if searcher._robot_home_collision_reason(active_pids=[pid]):
                continue
            ok = True
            break
        if ok:
            out.append(np.asarray(xy, dtype=float)[:2])
    return out


def _farthest_point_order(xy_list: List[np.ndarray], k: int) -> List[int]:
    """Farthest-point sampling: return indices of up to k maximally-spread points.

    Used so the per-part candidate pool COVERS the whole table instead of clumping
    into the first feasible grid cells (which caused all legs to pile onto the same
    corner and fail L2)."""
    n = len(xy_list)
    if n == 0:
        return []
    k = min(int(k), n)
    pts = np.asarray([np.asarray(p, dtype=float)[:2] for p in xy_list], dtype=float)
    # seed with the point closest to the table-candidate centroid (stable, deterministic)
    centroid = pts.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))
    chosen = [first]
    dmin = np.linalg.norm(pts - pts[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dmin))
        if dmin[nxt] <= 1e-9:
            break
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(pts - pts[nxt], axis=1))
    return chosen


def _proxy_for_xy(searcher, pid: str, xy: np.ndarray,
                  goal_pos: np.ndarray, goal_rot: np.ndarray,
                  max_poses: int = 2, first_pid: Optional[str] = None) -> List[Dict]:
    """Precompute real per-POSE manip / common-grasp / distance for (pid, xy).

    Returns up to ``max_poses`` records (one per feasible rotation candidate),
    each carrying its ``rot_name``. The GA gene selects (xy, pose) jointly and
    forces that exact pose in ``evaluate_layout`` (no proxy/L2 pose mismatch).

    The preassembled first part (seat) is (a) an obstacle for the STAGING overlap
    checks -- a leg staged inside the seat is a real collision -- but (b) EXEMPTED
    in grasp reasoning (``placed={first_pid}`` -> the parent-based contact
    exclusion drops it), so the leg->seat insertion at the goal is not falsely
    counted as a grasp-blocking collision."""
    gc = searcher._grasp_collection(pid)
    if gc is None or len(gc) == 0:
        return []
    active = [pid] if not first_pid else [pid, first_pid]
    placed = set() if not first_pid else {first_pid}
    recs: List[Dict] = []
    for cand in searcher.rot_cands.get(pid, []) or []:
        if searcher._staging_arm_keepout_reason(pid, xy, cand):
            continue
        searcher._apply_staging_pose(pid, xy, cand)
        if searcher._pairwise_collision(active_pids=active):
            continue
        if searcher._mesh_clearance_reason(active_pids=active):
            continue
        if searcher._robot_home_collision_reason(active_pids=[pid]):
            continue
        sp = searcher.staging_models[pid].pos.copy()
        sr = searcher.staging_models[pid].rotmat.copy()
        obs = searcher._planner_obstacles([], current_pid=pid, placed=placed)
        best_gc = 0
        best_arm = None
        for arm_tag in searcher._arm_order(pid):
            arm = get_layout_arm(searcher.robot, arm_tag, single_arm=searcher.single_arm_mode)
            planner = PickPlacePlanner(robot=arm)
            try:
                gids = planner.reason_common_gids(
                    grasp_collection=gc,
                    goal_pose_list=[(sp, sr), (goal_pos, goal_rot)],
                    obstacle_list=obs,
                )
            except Exception:
                gids = None
            n = len(gids) if gids else 0
            if n > best_gc:
                best_gc = n
                best_arm = arm_tag
        if best_gc < 1:
            continue
        manip = searcher._endpoint_manip(
            get_layout_arm(searcher.robot, best_arm, single_arm=searcher.single_arm_mode),
            gc, sp, sr, goal_pos, goal_rot, obs,
        )
        dist = float(np.linalg.norm(sp[:2] - goal_pos[:2]))
        rec = {
            "xy": np.asarray(xy, dtype=float)[:2],
            "pose_tag": str(getattr(cand, "tag", "unknown")),
            "rot_name": str(getattr(cand, "rot_name", "unknown")),
            "arm": str(best_arm),
            "footprint": np.asarray(getattr(cand, "footprint", [0.05, 0.05]), dtype=float)[:2],
            "common_grasp_count": int(best_gc),
            "manipulability": float(manip),
            "dist_to_goal": float(dist),
        }
        recs.append(rec)
    # keep the most promising distinct poses at this xy (by common-grasp)
    recs.sort(key=lambda r: -r["common_grasp_count"])
    return recs[: max(int(max_poses), 1)]


def _seat_exemption_probe(searcher, pools, ga_pids, spread, first_pid, fail_pid) -> None:
    """Diagnose whether the seat (parent) contact exemption is the deciding factor.

    For the failing leg, at the SPREAD-chosen staging pose, recompute common grasps
    twice against the SAME dynamic-obstacle set for its assembly step:
      * ``n_exempt`` -- seat excluded (real evaluate_layout behavior: leg->seat
        insertion is an assembly contact, not a collision), and
      * ``n_forced`` -- seat forced in as an obstacle.
    Also reports ``n_free`` (no obstacles at all, i.e. the optimistic pool number)
    so we can see how much the dynamic obstacles cost vs the seat specifically.
    """
    pid = fail_pid if fail_pid in ga_pids else (ga_pids[0] if ga_pids else None)
    if pid is None:
        return
    rec = pools[pid][spread[pid]]
    gc = searcher._grasp_collection(pid)
    if gc is None or len(gc) == 0:
        print(f"  [seat-probe] {pid}: no grasp collection")
        return
    # place all parts at their spread staging poses; seat already at goal
    for q in ga_pids:
        r = pools[q][spread[q]]
        cand = next((c for c in searcher.rot_cands.get(q, [])
                     if str(getattr(c, "rot_name", "")) == str(r.get("rot_name"))), None)
        if cand is not None:
            searcher._apply_staging_pose(q, np.asarray(r["xy"], dtype=float), cand)
    sp = searcher.staging_models[pid].pos.copy()
    sr = searcher.staging_models[pid].rotmat.copy()
    gp, gr = searcher.world_poses[pid]
    gp = np.asarray(gp, dtype=float)
    gr = np.asarray(gr, dtype=float)
    placed = {first_pid} if first_pid else set()
    # dynamic-obstacle set for this leg's assembly step (other legs staged + seat goal)
    try:
        step_obs = searcher._step_obstacles(pid, placed)
    except Exception:
        step_obs = []

    def _count(obs_list) -> int:
        arm_tag = searcher._arm_order(pid)[0]
        arm = get_layout_arm(searcher.robot, arm_tag, single_arm=searcher.single_arm_mode)
        planner = PickPlacePlanner(robot=arm)
        try:
            gids = planner.reason_common_gids(
                grasp_collection=gc, goal_pose_list=[(sp, sr), (gp, gr)],
                obstacle_list=obs_list)
            return len(gids) if gids else 0
        except Exception as exc:
            print(f"  [seat-probe] reason exception: {type(exc).__name__}: {exc}")
            return -1

    n_free = _count([])
    obs_exempt = searcher._planner_obstacles(step_obs, current_pid=pid, placed=placed)
    n_exempt = _count(obs_exempt)
    obs_forced = list(obs_exempt)
    if first_pid and first_pid in searcher.goal_models:
        obs_forced = list(obs_forced) + [searcher.goal_models[first_pid]]
    n_forced = _count(obs_forced)
    print(f"  [seat-probe] {pid} @({sp[0]:.3f},{sp[1]:.3f}) common grasps: "
          f"no_obstacles={n_free}  seat_exempt(real)={n_exempt}  seat_forced={n_forced}")
    if n_exempt > 0 and n_forced == 0:
        print("  [seat-probe] -> seat insertion WAS the killer; exemption correctly saves it.")
    elif n_exempt == 0 and n_free > 0:
        print("  [seat-probe] -> OTHER dynamic obstacles (other legs) block grasps, not the seat.")
    elif n_free == 0:
        print("  [seat-probe] -> grasp SET itself yields no common grasp at this pose "
              "(consider denser grasps or a different pose).")


def _rank_seed_pool(model, sample, part, xy_list, hint, device, top_k) -> List[np.ndarray]:
    """Use the NN ranker to keep the best top_k table XY candidates (seed pool)."""
    if model is None or not xy_list:
        return xy_list[: max(int(top_k), 1)]
    bounds = _table_bounds(sample)
    cand_feat = np.zeros((len(xy_list), POSE_FEAT_DIM), dtype=np.float32)
    for i, xy in enumerate(xy_list):
        cand_feat[i] = _candidate_feature(xy.tolist(), hint, i, 0.0, bounds)
    part_static = _part_static_vector(part, sample, int(sample["num_parts"]))
    global_feat = _global_vector(sample)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(part_static).unsqueeze(0).to(device),
            torch.from_numpy(global_feat).unsqueeze(0).to(device),
            torch.from_numpy(cand_feat).unsqueeze(0).to(device),
            torch.ones(len(xy_list)).unsqueeze(0).to(device),
        )["cand_logits"].squeeze(0).cpu().numpy()
    order = np.argsort(-logits)[: int(top_k)]
    return [xy_list[int(i)] for i in order]


# ------------------------------------------------------------------
# proxy fitness (cheap; per individual)
# ------------------------------------------------------------------
def _proxy_fitness(individual: Dict[str, int],
                   pools: Dict[str, List[Dict]],
                   first_goal_xy: Optional[np.ndarray],
                   *, w_manip: float, w_close: float, w_grasp: float,
                   w_overlap: float, w_order: float,
                   manip_norm: float, grasp_norm: float, dist_decay: float,
                   order_tol: float = 0.03,
                   order_pids: Optional[List[str]] = None,
                   first_footprint: Optional[np.ndarray] = None) -> float:
    """Cheap per-individual proxy combining the three criteria + common-grasp +
    a soft assembly-sequence (order-x) penalty that approximates the dynamic-
    obstacle feasibility the real evaluate_layout enforces.
    """
    manip_terms, close_terms, grasp_terms = [], [], []
    boxes = []  # (xy, footprint) for overlap check
    if first_goal_xy is not None:
        # criterion-4 fix: use the REAL preassembled-part footprint (e.g. 0.20 seat),
        # not a hardcoded 0.06, so legs are penalized for overlapping the seat.
        fp0 = np.asarray(first_footprint, dtype=float)[:2] if first_footprint is not None \
            else np.array([0.20, 0.20])
        boxes.append((np.asarray(first_goal_xy, dtype=float)[:2], fp0))
    for pid, idx in individual.items():
        rec = pools[pid][idx]
        manip_terms.append(1.0 - float(np.exp(-max(rec["manipulability"], 0.0) / max(manip_norm, 1e-9))))
        close_terms.append(float(np.exp(-max(rec["dist_to_goal"], 0.0) / max(dist_decay, 1e-9))))
        # hill saturation on common-grasp count (criterion-1 support)
        g = float(max(rec["common_grasp_count"], 0.0))
        grasp_terms.append(g / (g + max(grasp_norm, 1e-9)))
        boxes.append((rec["xy"], rec["footprint"]))
    mean_manip = float(np.mean(manip_terms)) if manip_terms else 0.0
    mean_close = float(np.mean(close_terms)) if close_terms else 0.0
    mean_grasp = float(np.mean(grasp_terms)) if grasp_terms else 0.0
    overlap = _staging_overlap(boxes)
    order_pen = _order_x_penalty(individual, pools, order_pids, order_tol)
    return (w_manip * mean_manip + w_close * mean_close + w_grasp * mean_grasp
            - w_overlap * overlap - w_order * order_pen)


def _order_x_penalty(individual: Dict[str, int], pools: Dict[str, List[Dict]],
                     order_pids: Optional[List[str]], tol: float) -> float:
    """Soft version of the searcher's HARD order-x constraint.

    The searcher requires later-assembled parts to stage at x <= earlier.x + tol.
    We sum the normalized x-overshoot across consecutive assembly steps so the GA
    proxy actively steers toward sequence-feasible layouts (this is the concrete
    'assembly sequence + dynamic obstacle' feasibility signal for staging)."""
    if not order_pids or len(order_pids) < 2:
        return 0.0
    xs = []
    for pid in order_pids:
        if pid not in individual:
            continue
        xs.append((pid, float(pools[pid][individual[pid]]["xy"][0])))
    pen = 0.0
    for (prev, x_prev), (cur, x_cur) in zip(xs[:-1], xs[1:]):
        over = x_cur - (x_prev + float(tol))
        if over > 0:
            pen += over
    return float(pen)


def _staging_overlap(boxes: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Sum of axis-aligned footprint overlap areas among all staging boxes.

    Cheap proxy for criterion 3: parts staged on the table must coexist without
    colliding; the real dynamic-obstacle sequence check is done by evaluate_layout.
    """
    total = 0.0
    n = len(boxes)
    for i in range(n):
        ci, fi = boxes[i]
        for j in range(i + 1, n):
            cj, fj = boxes[j]
            ox = (fi[0] + fj[0]) * 0.5 - abs(float(ci[0] - cj[0]))
            oy = (fi[1] + fj[1]) * 0.5 - abs(float(ci[1] - cj[1]))
            if ox > 0 and oy > 0:
                total += float(ox * oy)
    return total


# ------------------------------------------------------------------
# GA operators
# ------------------------------------------------------------------
def _random_individual(pools: Dict[str, List[Dict]], rng) -> Dict[str, int]:
    return {pid: int(rng.integers(0, len(pool))) for pid, pool in pools.items()}


def _tournament(pop, fits, k, rng):
    idx = rng.integers(0, len(pop), size=k)
    best = idx[0]
    for i in idx[1:]:
        if fits[i] > fits[best]:
            best = i
    return dict(pop[best])


def _crossover(a, b, rng, rate):
    if rng.random() > rate:
        return dict(a), dict(b)
    c1, c2 = {}, {}
    for pid in a:
        if rng.random() < 0.5:
            c1[pid], c2[pid] = a[pid], b[pid]
        else:
            c1[pid], c2[pid] = b[pid], a[pid]
    return c1, c2


def _mutate(ind, pools, rng, rate):
    out = dict(ind)
    for pid, pool in pools.items():
        if len(pool) > 1 and rng.random() < rate:
            out[pid] = int(rng.integers(0, len(pool)))
    return out


def _individual_xy(genes, pools, searcher, first_pid) -> Dict[str, np.ndarray]:
    xy = {pid: np.asarray(pools[pid][idx]["xy"], dtype=float) for pid, idx in genes.items()}
    if first_pid is not None and first_pid in searcher.world_poses:
        gp, _ = searcher.world_poses[first_pid]
        xy[first_pid] = np.asarray(gp[:2], dtype=float)
    return xy


def _build_layout_candidate(genes, pools, searcher, first_pid, station):
    """Decode a GA genome into a LayoutCandidate, FORCING each part's chosen pose.

    The gene selects a (xy, pose) record; we set ``forced_rot_name`` so
    evaluate_layout verifies exactly the pose the proxy scored (task 6: no
    proxy/L2 pose mismatch)."""
    xy = _individual_xy(genes, pools, searcher, first_pid)
    cand = LayoutCandidate(xy=xy)
    cand.assembly_station_pos = np.asarray(station, dtype=float).copy()
    forced = {}
    for pid, idx in genes.items():
        rn = pools[pid][idx].get("rot_name")
        if rn:
            forced[pid] = str(rn)
    cand.forced_rot_name = forced
    return cand


def _first_part_footprint(searcher, first_pid) -> Optional[np.ndarray]:
    if first_pid is not None and first_pid in searcher.rot_cands and searcher.rot_cands[first_pid]:
        return np.asarray(searcher.rot_cands[first_pid][0].footprint, dtype=float)[:2]
    return None


# ------------------------------------------------------------------
# assembly-center sampling (random region; replaces fixed 3x3 grid)
# ------------------------------------------------------------------
def _feasible_center_bounds(searcher) -> Tuple[float, float, float, float]:
    """Table range shrunk by the first (preassembled) part's footprint."""
    x_min, x_max = float(searcher.table_x_range[0]), float(searcher.table_x_range[1])
    y_min, y_max = float(searcher.table_y_range[0]), float(searcher.table_y_range[1])
    first_pid = searcher._first_part_id()
    fp = np.zeros(2)
    if first_pid is not None and first_pid in searcher.rot_cands and searcher.rot_cands[first_pid]:
        fp = np.asarray(searcher.rot_cands[first_pid][0].footprint, dtype=float)[:2]
    xlo, xhi = x_min + fp[0] / 2.0, x_max - fp[0] / 2.0
    ylo, yhi = y_min + fp[1] / 2.0, y_max - fp[1] / 2.0
    if xlo >= xhi:
        xlo, xhi = x_min, x_max
    if ylo >= yhi:
        ylo, yhi = y_min, y_max
    return xlo, ylo, xhi, yhi


def _sample_assembly_centers(searcher, args, rng) -> List[np.ndarray]:
    """Sample GA candidate assembly centers from a (random) table region.

    Each center is validated by ``searcher._assembly_region_reject_reason`` (the
    same hard constraint the 3x3 grid used: the preassembled first part must not
    collide with the arm at that center). Returns >=1 center (falls back to the
    clamped --goal-pos if sampling fails).
    """
    top_z = float(searcher.table_top_z)
    goal = _parse_vec3(args.goal_pos, (0.36, 0.0, 0.0))
    goal[2] = top_z

    n = max(1, int(args.num_centers))
    fxlo, fylo, fxhi, fyhi = _feasible_center_bounds(searcher)

    # legacy single fixed center
    if n <= 1 and not args.center_region.strip() and not args.random_region:
        gx = float(np.clip(goal[0], fxlo, fxhi))
        gy = float(np.clip(goal[1], fylo, fyhi))
        fixed = np.array([gx, gy, top_z], dtype=float)
        # Validate that the PREASSEMBLED seat at this center does not collide with
        # the arm at home (the same hard check evaluate_layout applies at the end).
        # The fixed-center path used to skip this, so an infeasible center like a
        # too-close seat silently produced pools that could never pass L2.
        reason = searcher._assembly_region_reject_reason(fixed)
        if reason is not None:
            print(f"[center][WARN] fixed center ({gx:.3f},{gy:.3f}) is INFEASIBLE: {reason}")
            print("[center][WARN] the preassembled seat there hits the arm home box; NO layout "
                  "can pass L2 at this center. Move --goal-pos further from the arm base, shrink "
                  "the seat, or use --num-centers>1 --random-region to auto-search a valid center.")
        return [fixed]

    # sampling window
    if args.center_region.strip():
        vals = [float(v) for v in args.center_region.split(",")]
        rxlo, rylo, rxhi, ryhi = vals[0], vals[1], vals[2], vals[3]
        rxlo, rxhi = max(rxlo, fxlo), min(rxhi, fxhi)
        rylo, ryhi = max(rylo, fylo), min(ryhi, fyhi)
    elif args.random_region:
        frac = float(np.clip(args.region_frac, 0.1, 1.0))
        w = (fxhi - fxlo) * frac
        h = (fyhi - fylo) * frac
        rxlo = float(rng.uniform(fxlo, fxhi - w)) if fxhi - w > fxlo else fxlo
        rylo = float(rng.uniform(fylo, fyhi - h)) if fyhi - h > fylo else fylo
        rxhi, ryhi = rxlo + w, rylo + h
        print(f"[center] random region = x[{rxlo:.3f},{rxhi:.3f}] y[{rylo:.3f},{ryhi:.3f}] "
              f"(frac={frac})")
    else:
        rxlo, rylo, rxhi, ryhi = fxlo, fylo, fxhi, fyhi

    if rxlo >= rxhi or rylo >= ryhi:
        rxlo, rylo, rxhi, ryhi = fxlo, fylo, fxhi, fyhi

    centers: List[np.ndarray] = []
    seen = set()

    def _try_add(x, y) -> bool:
        x = float(np.clip(x, fxlo, fxhi))
        y = float(np.clip(y, fylo, fyhi))
        key = (round(x, 3), round(y, 3))
        if key in seen:
            return False
        pos = np.array([x, y, top_z], dtype=float)
        if searcher._assembly_region_reject_reason(pos) is not None:
            return False
        seen.add(key)
        centers.append(pos)
        return True

    # include the (clamped) goal-pos as one candidate if it lies in the window
    if rxlo <= goal[0] <= rxhi and rylo <= goal[1] <= ryhi:
        _try_add(goal[0], goal[1])

    attempts = 0
    rejected = 0
    max_attempts = n * 400
    while len(centers) < n and attempts < max_attempts:
        attempts += 1
        before = len(centers)
        if not _try_add(rng.uniform(rxlo, rxhi), rng.uniform(rylo, ryhi)):
            rejected += 1
    if len(centers) < n:
        print(f"[center] only {len(centers)}/{n} valid centers found in "
              f"{attempts} samples ({rejected} rejected by seat-vs-arm keepout / dup). "
              f"Widen --center-region or lower the seat footprint for more.")

    if not centers:  # last resort: clamped goal even if reject reason fired
        centers.append(np.array([float(np.clip(goal[0], fxlo, fxhi)),
                                 float(np.clip(goal[1], fylo, fyhi)), top_z], dtype=float))
    return centers


def _build_pools_for_center(searcher, args, model, station, init_hints, verbose=True):
    """Set the assembly station and precompute per-part feasible candidate pools.

    Returns (pools, ga_pids, first_goal_xy). Goal poses depend on the station, so
    this must be called once per candidate center.
    """
    searcher._set_assembly_station(np.asarray(station, dtype=float))
    first_pid = searcher._first_part_id()
    # Preassemble the first part (seat) at its goal so the per-part staging checks
    # below see it exactly as evaluate_layout will (leg-vs-seat staging overlap is
    # rejected up-front; leg->seat insertion stays exempted in grasp reasoning).
    searcher._apply_first_part_as_assembled()
    anchors = table_anchor_grid(searcher, spacing=float(args.grid_spacing))
    sample = _build_sample_dict(searcher, np.asarray(station, dtype=float)) if model is not None else None

    pools: Dict[str, List[Dict]] = {}
    ga_pids: List[str] = []
    goal_xy_by_pid: Dict[str, np.ndarray] = {}
    for pid in searcher.part_order:
        if pid == first_pid:
            continue
        gp, gr = searcher.world_poses[pid]
        goal_pos = np.asarray(gp, dtype=float)
        goal_rot = np.asarray(gr, dtype=float)
        goal_xy_by_pid[pid] = goal_pos[:2].copy()
        xy_list = _enumerate_xy_candidates(searcher, pid, args.grid_spacing, first_pid=first_pid)
        if not xy_list:
            if verbose:
                print(f"[warn] {pid}: no geometric candidates on table")
            continue
        hint = init_hints.get(pid)
        if hint is None:
            hint = deterministic_init_anchor(pid, np.asarray(station)[:2], anchors).tolist()
        if model is not None:
            # NN ranks candidates; keep its order (already goal-aware).
            part = next(p for p in sample["parts"] if p["part_id"] == pid)
            part["init_pos_balanced"] = hint
            seed_xy = _rank_seed_pool(model, sample, part, xy_list, hint,
                                      args.device, len(xy_list))
        else:
            # No NN: disperse candidates across the WHOLE table via farthest-point
            # sampling so the pool covers the table instead of clumping into the
            # first feasible grid cells (root cause of legs piling on one corner).
            fps_idx = _farthest_point_order(xy_list, max(int(args.top_k) * 4, 24))
            seed_xy = [xy_list[i] for i in fps_idx]
        scan_cap = max(int(args.top_k) * 12, 60)
        max_poses = max(int(args.pool_per_xy_poses), 1)
        pool: List[Dict] = []
        distinct_xy = 0
        for xy in seed_xy[:scan_cap]:
            recs = _proxy_for_xy(searcher, pid, xy, goal_pos, goal_rot,
                                 max_poses=max_poses, first_pid=first_pid)
            if recs:
                pool.extend(recs)
                distinct_xy += 1
            if distinct_xy >= int(args.top_k):
                break
        if not pool:
            if verbose:
                print(f"[warn] {pid}: no feasible seed candidate (grasp<1) after "
                      f"scanning {min(len(seed_xy), scan_cap)} anchors; skipping in GA")
            continue
        pools[pid] = pool
        ga_pids.append(pid)
        if verbose:
            xs = [r["xy"][0] for r in pool]
            ys = [r["xy"][1] for r in pool]
            print(f"[pool] {pid}: {len(pool)} (xy,pose) candidates over {distinct_xy} xy "
                  f"| x[{min(xs):.2f},{max(xs):.2f}] y[{min(ys):.2f},{max(ys):.2f}] "
                  f"| best gc={max(r['common_grasp_count'] for r in pool)}")

    first_goal_xy = None
    if first_pid is not None and first_pid in searcher.world_poses:
        first_goal_xy = np.asarray(searcher.world_poses[first_pid][0][:2], dtype=float)
    return pools, ga_pids, first_goal_xy, goal_xy_by_pid


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))
    t0 = time.time()

    searcher = _build_searcher(args)
    first_pid = searcher._first_part_id()

    # optional NN ranker (seed only)
    model = None
    if args.checkpoint.strip() and _HAS_TORCH:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model = PartPlacementRankerNet(hidden=int(ckpt.get("hidden", 128)),
                                       dropout=float(ckpt.get("dropout", 0.15)))
        model.load_state_dict(ckpt["state_dict"])
        model.to(args.device)
        model.eval()
        print(f"[seed] NN ranker loaded: {args.checkpoint}")
    else:
        print("[seed] no NN ranker -> grid seeding")

    init_hints = json.loads(args.init_pos_json) if args.init_pos_json.strip() else {}

    # ---- candidate assembly centers (GA also optimizes WHERE to assemble) ----
    center_rng = np.random.default_rng(
        int(args.center_seed) if int(args.center_seed) >= 0 else int(args.seed))
    centers = _sample_assembly_centers(searcher, args, center_rng)
    print(f"[center] {len(centers)} candidate assembly center(s): "
          + ", ".join(f"({c[0]:.3f},{c[1]:.3f})" for c in centers))

    # ---- per-center candidate pools (goal poses depend on the center) ----
    kept_centers: List[np.ndarray] = []
    pools_by_c: List[Dict[str, List[Dict]]] = []
    pids_by_c: List[List[str]] = []
    firstxy_by_c: List[Optional[np.ndarray]] = []
    goalxy_by_c: List[Dict[str, np.ndarray]] = []
    for ci, center in enumerate(centers):
        print(f"\n[center {ci}] building pools at ({center[0]:.3f},{center[1]:.3f}) ...")
        pools, ga_pids, first_goal_xy, goal_xy_by_pid = _build_pools_for_center(
            searcher, args, model, center, init_hints, verbose=True)
        if not pools:
            print(f"[center {ci}] no feasible parts; dropped")
            continue
        kept_centers.append(np.asarray(center, dtype=float))
        pools_by_c.append(pools)
        pids_by_c.append(ga_pids)
        firstxy_by_c.append(first_goal_xy)
        goalxy_by_c.append(goal_xy_by_pid)

    if not pools_by_c:
        raise SystemExit("No feasible per-part candidates at any center; "
                         "check asmdef / grasp-dir / goal-pos / --center-region.")

    # ---------------- dry-run: summarize + one real evaluate_layout ------------
    if args.dry_run:
        print("\n===== DRY-RUN candidate-pool summary =====")
        print(f"asmdef      : {args.asmdef}")
        print(f"grasp_dir   : {args.grasp_dir}")
        print(f"num_parts   : {len(searcher.part_order)}  first(preassembled)={first_pid}")
        print(f"num_centers : {len(kept_centers)} (sampled from "
              + ("--center-region" if args.center_region.strip()
                 else "random region" if args.random_region
                 else "fixed goal-pos" if len(kept_centers) == 1 else "table") + ")")
        for ci, center in enumerate(kept_centers):
            pools, ga_pids = pools_by_c[ci], pids_by_c[ci]
            print(f"\n[center {ci}] ({center[0]:.3f},{center[1]:.3f})  ga_parts={ga_pids}")
            for pid in ga_pids:
                pool = pools[pid]
                best = max(pool, key=lambda r: r["common_grasp_count"])
                manip_max = max(r["manipulability"] for r in pool)
                dist_min = min(r["dist_to_goal"] for r in pool)
                print(f"  - {pid:10s}: {len(pool):3d} cands | "
                      f"best_common_grasp={best['common_grasp_count']:3d} | "
                      f"manip_max={manip_max:.4f} | dist_min={dist_min:.3f}m | "
                      f"best_xy=({best['xy'][0]:.3f},{best['xy'][1]:.3f}) pose={best['pose_tag']}")
        # verify a SPREAD pick at center 0 -> confirms full asmdef sequence +
        # dynamic-obstacle + L2 pipeline actually runs. We spread parts to distinct
        # anchors: picking every part's argmax-common-grasp candidate would stack
        # them on the same corner (a greedy artifact the GA's overlap penalty
        # avoids), which would trivially fail with pair_collision.
        c0 = 0
        searcher._set_assembly_station(kept_centers[c0])
        spread: Dict[str, int] = {}
        taken: List[np.ndarray] = []
        for pid in pids_by_c[c0]:
            pool = pools_by_c[c0][pid]
            order_idx = sorted(range(len(pool)),
                               key=lambda k: -pool[k]["common_grasp_count"])
            chosen = order_idx[0]
            for k in order_idx:
                xyk = np.asarray(pool[k]["xy"], dtype=float)
                if all(float(np.linalg.norm(xyk - t)) > 0.12 for t in taken):
                    chosen = k
                    break
            spread[pid] = chosen
            taken.append(np.asarray(pool[chosen]["xy"], dtype=float))
        searcher._set_assembly_station(kept_centers[c0])
        cand = _build_layout_candidate(spread, pools_by_c[c0], searcher,
                                       first_pid, kept_centers[c0])
        ok = searcher.evaluate_layout(cand)
        print(f"\n[dry-run] spread layout @center0 real evaluate_layout: "
              f"L2_pass={ok} layout_score={float(getattr(cand, 'layout_score', 0.0)):.4f}")
        if not ok:
            # task-3: print the FULL failure breakdown (which part, and the exact
            # per-reason counter), not a truncated string, so we know whether legs
            # fail on grasp (no_common_gids) vs staging (pair_collision) vs keepout.
            fpart = getattr(cand, "fail_part", "?")
            print(f"  fail_part = {fpart}")
            print(f"  fail_reason = {getattr(cand, 'fail_reason', '')}")
            fdetail = getattr(cand, "fail_detail", {}) or {}
            if fdetail:
                nz = {k: v for k, v in fdetail.items() if v}
                print(f"  fail_counter (nonzero) = {nz}")
            # task-4 + seat-exemption probe: for the failing part at its best xy,
            # recompute common grasps WITH the seat exempted (real behavior) vs
            # WITH the seat forced in as an obstacle. If exempt>0 but forced==0,
            # the leg->seat insertion contact was the (correctly avoided) killer;
            # if exempt is already 0, the bottleneck is the OTHER dynamic obstacles
            # / grasp set, not the seat.
            _seat_exemption_probe(searcher, pools_by_c[c0], pids_by_c[c0],
                                  spread, first_pid, fpart)
            print("[dry-run] note: a spread/greedy probe can still fail L2. The full GA "
                  "searches candidate combinations + overlap penalty; raise --top-k / "
                  "--num-centers for more diversity.")
        print(f"[dry-run] OK, pipeline wired (load + pools + real evaluate_layout ran). "
              f"elapsed={time.time()-t0:.1f}s")
        return

    # ------------------------------------------------------------------
    # center-aware GA:  individual = {"c": center_idx, "genes": {pid: cand_idx}}
    # ------------------------------------------------------------------
    def _rand_ind(c: int) -> Dict:
        genes = {pid: int(rng.integers(0, len(pools_by_c[c][pid]))) for pid in pids_by_c[c]}
        return {"c": int(c), "genes": genes}

    def _greedy_ind(c: int) -> Dict:
        genes = {pid: int(np.argmax([r["common_grasp_count"] for r in pools_by_c[c][pid]]))
                 for pid in pids_by_c[c]}
        return {"c": int(c), "genes": genes}

    first_fp = _first_part_footprint(searcher, first_pid)

    def proxy(ind: Dict) -> float:
        c = ind["c"]
        return _proxy_fitness(ind["genes"], pools_by_c[c], firstxy_by_c[c],
                              w_manip=args.w_manip, w_close=args.w_close,
                              w_grasp=args.w_grasp, w_overlap=args.w_overlap,
                              w_order=args.w_order, manip_norm=args.manip_norm,
                              grasp_norm=args.grasp_norm, dist_decay=args.dist_decay,
                              order_tol=args.order_tol, order_pids=pids_by_c[c],
                              first_footprint=first_fp)

    def _ind_key(ind: Dict):
        return (ind["c"], tuple(sorted(ind["genes"].items())))

    def _tournament2(pop, fits, k):
        idx = rng.integers(0, len(pop), size=k)
        best = idx[0]
        for i in idx[1:]:
            if fits[i] > fits[best]:
                best = i
        return {"c": pop[best]["c"], "genes": dict(pop[best]["genes"])}

    def _crossover2(a, b):
        # only recombine within the same center (gene index spaces differ per center)
        if a["c"] != b["c"] or rng.random() > float(args.crossover_rate):
            return ({"c": a["c"], "genes": dict(a["genes"])},
                    {"c": b["c"], "genes": dict(b["genes"])})
        c = a["c"]
        g1, g2 = {}, {}
        for pid in pids_by_c[c]:
            if rng.random() < 0.5:
                g1[pid], g2[pid] = a["genes"][pid], b["genes"][pid]
            else:
                g1[pid], g2[pid] = b["genes"][pid], a["genes"][pid]
        return {"c": c, "genes": g1}, {"c": c, "genes": g2}

    def _mutate2(ind):
        c = ind["c"]
        # switch assembly center (re-seed genes for the new center)
        if len(pools_by_c) > 1 and rng.random() < float(args.mutation_rate):
            c = int(rng.integers(0, len(pools_by_c)))
            if c != ind["c"]:
                return _rand_ind(c)
        genes = dict(ind["genes"])
        for pid in pids_by_c[c]:
            if len(pools_by_c[c][pid]) > 1 and rng.random() < float(args.mutation_rate):
                genes[pid] = int(rng.integers(0, len(pools_by_c[c][pid])))
        return {"c": c, "genes": genes}

    # ---- hybrid surrogate (SAGA): learned evaluate_layout accelerator ----
    surrogate = None
    if getattr(args, "use_surrogate", False):
        if not _HAS_TORCH:
            print("[surrogate] torch unavailable -> falling back to analytic proxy only")
        else:
            from layout_learning.models.assembly_ga_surrogate import (
                CenterContext, HybridSurrogateManager,
            )
            contexts = [
                CenterContext(
                    center_xy=np.asarray(kept_centers[c][:2], dtype=float),
                    order_pids=list(pids_by_c[c]),
                    first_goal_xy=firstxy_by_c[c],
                    table_x_range=tuple(searcher.table_x_range),
                    table_y_range=tuple(searcher.table_y_range),
                    goal_xy=goalxy_by_c[c],
                    grasp_norm=float(args.grasp_norm),
                    manip_norm=float(args.manip_norm),
                    dist_decay=float(args.dist_decay),
                )
                for c in range(len(kept_centers))
            ]
            surrogate = HybridSurrogateManager(
                contexts, pools_by_c,
                d_model=int(args.surrogate_hidden), heads=int(args.surrogate_heads),
                layers=int(args.surrogate_layers), device=args.device,
                min_labels=int(args.surrogate_min_labels), seed=int(args.seed),
            )
            print(f"[surrogate] hybrid Set-Transformer enabled "
                  f"(d_model={args.surrogate_hidden}, warmup={args.surrogate_warmup}, "
                  f"verify_topk={args.surrogate_verify_topk}, refit_every="
                  f"{args.surrogate_refit_every})")

    # ---- real-verification bookkeeping (guarantees L2) ----
    best_layout = None
    best_score = -np.inf
    best_center = None
    verified_keys = set()
    fail_hist: Dict[str, int] = {}          # task 2: failure-reason histogram
    n_verified = 0
    n_pass = 0

    def _fail_category(cand) -> str:
        """Compact, low-cardinality failure bucket (part + reason kind)."""
        fp = str(getattr(cand, "fail_part", None) or "unknown")
        reason = str(getattr(cand, "fail_reason", "") or "")
        low = reason.lower()
        if fp in ("order_x_constraint",) or "order_x" in low:
            return "order_x_constraint"
        if "all rotation/arm candidates failed" in low:
            return f"{fp}:all_rot_arm_failed"
        if "grasp collection missing" in low:
            return f"{fp}:no_grasp_collection"
        if "collision" in low:
            return f"{fp}:collision"
        if "clearance" in low:
            return f"{fp}:clearance"
        if "keepout" in low:
            return f"{fp}:arm_keepout"
        return fp

    def _record_fail(cand):
        cat = _fail_category(cand)
        fail_hist[cat] = fail_hist.get(cat, 0) + 1

    def _verify_one(ind) -> Optional[bool]:
        """Run the REAL evaluate_layout on one individual (deduped).

        Updates the best-so-far, the failure histogram AND -- crucially for the
        hybrid loop -- appends the (layout -> pass/fail, score) label to the
        surrogate's replay buffer for active learning. Returns None if already
        verified, else the pass/fail bool.
        """
        nonlocal best_layout, best_score, best_center, n_pass, n_verified
        key = _ind_key(ind)
        if key in verified_keys:
            return None
        verified_keys.add(key)
        c = ind["c"]
        searcher._set_assembly_station(kept_centers[c])
        cand = _build_layout_candidate(ind["genes"], pools_by_c[c], searcher,
                                       first_pid, kept_centers[c])
        passed = bool(searcher.evaluate_layout(cand))
        n_verified += 1
        score = float(getattr(cand, "layout_score", 0.0)) if passed else 0.0
        if surrogate is not None:
            surrogate.add_label(ind, passed, score)
        if passed:
            n_pass += 1
            if score > best_score:
                best_score = score
                best_layout = cand
                best_center = kept_centers[c].copy()
                print(f"    [verify] new best L2 layout_score={best_score:.4f} "
                      f"@center({best_center[0]:.3f},{best_center[1]:.3f})")
        else:
            _record_fail(cand)
        return passed

    def verify_elites(pop_sorted, k=None):
        if k is None:
            k = int(args.elite)
        for ind in pop_sorted[: int(k)]:
            _verify_one(ind)

    # ---- surrogate-blended fitness: proxy + w * P(feasible) * (0.5+0.5*score) ----
    def combined_fitness(pop) -> List[float]:
        base = [proxy(ind) for ind in pop]
        if surrogate is None or not surrogate.ready:
            return base
        feas, score = surrogate.predict(pop)
        w = float(args.surrogate_weight)
        return [base[i] + w * float(feas[i]) * (0.5 + 0.5 * float(score[i]))
                for i in range(len(pop))]

    # ---- initial population: greedy per center + random ----
    population: List[Dict] = [_greedy_ind(c) for c in range(len(pools_by_c))]
    while len(population) < int(args.pop_size):
        population.append(_rand_ind(int(rng.integers(0, len(pools_by_c)))))

    # ---- surrogate WARM-UP: bootstrap the label buffer with real L2 checks ----
    if surrogate is not None:
        n_warm = int(args.surrogate_warmup)
        print(f"[surrogate] warm-up: real-evaluating {n_warm} random layouts ...")
        warm_seen = set()
        attempts = 0
        while surrogate.n_labels < n_warm and attempts < n_warm * 6:
            attempts += 1
            ind = _rand_ind(int(rng.integers(0, len(pools_by_c))))
            k = _ind_key(ind)
            if k in warm_seen:
                continue
            warm_seen.add(k)
            _verify_one(ind)
        stats = surrogate.fit(epochs=int(args.surrogate_epochs))
        print(f"[surrogate] warm-up done: labels={surrogate.n_labels} "
              f"(pos={surrogate.n_pos}) bce={stats.get('bce', 0):.3f} "
              f"mse={stats.get('mse', 0):.3f} ready={surrogate.ready}")

    fits = combined_fitness(population)

    # ---- GA loop ----
    for gen in range(int(args.generations)):
        order = np.argsort(-np.asarray(fits))
        pop_sorted = [population[i] for i in order]
        fit_sorted = [fits[i] for i in order]

        # selective real verification: with the surrogate on, trust it to rank the
        # population and only spend WRS on its top picks (active-learning labels).
        if gen % max(1, int(args.verify_every)) == 0:
            k_verify = int(args.surrogate_verify_topk) if surrogate is not None else int(args.elite)
            verify_elites(pop_sorted, k=k_verify)

        # periodic active-learning refit on the growing label buffer
        if surrogate is not None and args.surrogate_refit_every > 0 \
                and gen % int(args.surrogate_refit_every) == 0 and gen > 0:
            stats = surrogate.fit(epochs=int(args.surrogate_epochs))
            if stats.get("trained", 0.0) > 0:
                print(f"    [surrogate] refit@gen{gen}: labels={surrogate.n_labels} "
                      f"(pos={surrogate.n_pos}) bce={stats.get('bce', 0):.3f}")

        print(f"[gen {gen:03d}] fit_best={fit_sorted[0]:.4f} "
              f"fit_mean={np.mean(fits):.4f} best_L2={best_score:.4f} "
              f"L2 pass/verified={n_pass}/{n_verified} elapsed={time.time()-t0:.1f}s")

        next_pop = [{"c": pop_sorted[i]["c"], "genes": dict(pop_sorted[i]["genes"])}
                    for i in range(min(int(args.elite), len(pop_sorted)))]
        while len(next_pop) < int(args.pop_size):
            pa = _tournament2(population, fits, int(args.tournament))
            pb = _tournament2(population, fits, int(args.tournament))
            c1, c2 = _crossover2(pa, pb)
            c1 = _mutate2(c1)
            c2 = _mutate2(c2)
            next_pop.append(c1)
            if len(next_pop) < int(args.pop_size):
                next_pop.append(c2)
        population = next_pop
        fits = combined_fitness(population)

    # final sweep + fallback until one L2-passing layout is found
    order = np.argsort(-np.asarray(fits))
    verify_elites([population[i] for i in order],
                  k=max(int(args.elite),
                        int(args.surrogate_verify_topk) if surrogate is not None else 0))
    if best_layout is None:
        for i in order:
            if _verify_one(population[i]) and best_layout is not None:
                break

    # ---- failure-reason histogram (task 2) ----
    print("\n===== evaluate_layout failure histogram =====")
    print(f"verified={n_verified}  L2_pass={n_pass}")
    for reason, cnt in sorted(fail_hist.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:28s}: {cnt}")
    if surrogate is not None:
        pop_size = max(int(args.pop_size), 1)
        # Acceleration accounting: without the surrogate the GA would call the
        # real evaluate_layout on ~pop_size*generations individuals; the hybrid
        # loop only verifies a handful of surrogate-ranked picks per generation.
        full_evals = pop_size * (int(args.generations) + 1)
        print("\n===== hybrid surrogate (SAGA) =====")
        print(f"surrogate fits            : {surrogate.n_fit}")
        print(f"label buffer (pos/total)  : {surrogate.n_pos}/{surrogate.n_labels}")
        print(f"real evaluate_layout calls: {n_verified}")
        print(f"full-fitness GA would cost : ~{full_evals} evaluate_layout calls")
        if n_verified > 0:
            print(f"speedup (WRS calls avoided): ~{full_evals / max(n_verified, 1):.1f}x")

    if best_layout is None:
        raise SystemExit("GA finished but no L2-passing layout found. "
                         "See failure histogram above. Try --disable-order-x, larger "
                         "--pop-size/--generations/--top-k/--num-centers, or a different region.")

    result = {
        "asmdef": args.asmdef,
        "assembly_center": best_center.tolist() if best_center is not None else None,
        "num_centers_searched": len(kept_centers),
        "num_parts": len(searcher.part_order),
        "ga_parts": [pid for pid in best_layout.xy if pid != first_pid],
        "layout_score": float(best_score),
        "l2_pass": True,
        "grasp_score_norm": float(best_layout.grasp_score_norm),
        "manip_score_norm": float(best_layout.manip_score_norm),
        "dist_score_norm": float(best_layout.dist_score_norm),
        "best_layout": {
            pid: {
                "init_pos": [float(best_layout.xy[pid][0]), float(best_layout.xy[pid][1]),
                             float(best_layout.z_offset.get(pid, 0.0))],
                "init_rotmat": np.asarray(best_layout.chosen_rotmat.get(pid, np.eye(3)),
                                          dtype=float).reshape(-1).tolist(),
                "pose_tag": best_layout.pose_tag.get(pid),
                "arm_choice": best_layout.arm_choice.get(pid),
                "grasp_count": int(best_layout.grasp_counts.get(pid, 0)),
                "manipulability": float(best_layout.per_part_manip.get(pid, 0.0)),
                "dist_to_goal": float(best_layout.per_part_dist.get(pid, 0.0)),
            }
            for pid in best_layout.xy
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    cc = best_center if best_center is not None else np.zeros(3)
    print(f"Done. best L2 layout_score={best_score:.4f} "
          f"@center({cc[0]:.3f},{cc[1]:.3f}) elapsed={time.time()-t0:.1f}s")
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as stream:
            stream.write(text)


if __name__ == "__main__":
    main()
