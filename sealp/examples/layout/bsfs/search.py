"""Backward OPEN-list controller: exact A*/branch-and-bound + anytime beam.

Both modes explore the SAME backward tree (decide part pi_n first, ..., pi_1
last) and differ only in node selection + pruning:

  * EXACT  : best-first A* on f = g + h with an admissible matching lower bound
             h. The first complete node popped is the discrete-domain global
             optimum (A* optimality). Sound domain propagation + Hall matching
             prune dead-end suffixes.
  * BEAM   : level-synchronous, keep the top-B partial suffixes by
             (cost, -clearance, -manip). Anytime; NO completeness/optimality
             guarantee (documented).

Every returned complete assignment is still re-checked by the caller with the
full forward ``evaluate_layout`` witness -> no false positives.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, Optional

import numpy as np

from .cost import lex_key
from .domain import (
    OccupancyBitmap,
    build_staging_grid,
    continuous_domain,
    discrete_domain,
)
from .oracle import HARD_FAIL, StepOracle
from .pruning import (
    hall_feasible,
    part_footprint_radius,
    propagate_domains,
    swept_segment_mask,
)
from .seeding import center_id, seed_everything, task_seed


def rec_jsonable(rec):
    """Serializable copy of one certify record (numpy -> list) for IPC."""
    if rec is None:
        return None
    out = {}
    for k, v in rec.items():
        out[k] = v.reshape(-1).tolist() if isinstance(v, np.ndarray) else v
    return out


def _goal_xy(searcher, pid: str) -> np.ndarray:
    gp, _ = searcher.world_poses[pid]
    return np.asarray(gp, dtype=float)[:2]


def _iter_rot(searcher, pid: str, poses_per_xy: int):
    cands = searcher.rot_cands.get(pid, []) or []
    return cands, max(int(poses_per_xy), 1)


# ------------------------------------------------------------------
# EXACT : backward A* / branch-and-bound
# ------------------------------------------------------------------
def _search_exact(searcher, oracle, args, pick_order, preassembled_pid,
                  stats, verbose, base_seed=0, cid="0") -> List[Dict]:
    n = len(pick_order)
    grid = build_staging_grid(searcher, args.grid_spacing)
    # SOUND occupancy: a placed part blocks only its OWN grid cell. Two parts in
    # the same cell would share an (x,y) and necessarily collide, so removing it
    # never discards a feasible layout; real adjacent-cell collisions are still
    # caught exactly by the oracle's mesh check. Using a larger (circumradius)
    # block here would OVER-prune valid adjacent placements and make the "exact"
    # optimum optimal only over an over-constrained domain (observed: exact cost
    # 1.2452 > beam 1.1923 on the same grid). foot_radius=0 restores true
    # grid-optimality; Hall matching (distinct cells) remains the sound prune.
    foot_radius = {p: 0.0 for p in pick_order}
    goal_xy = {p: _goal_xy(searcher, p) for p in pick_order}

    cell_cap = int(getattr(args, "exact_cell_cap", 0) or 0)
    part_cells0: Dict[str, List[int]] = {}
    for pid in pick_order:
        cells = discrete_domain(searcher, pid, grid, args.grid_spacing, preassembled_pid)
        if not cells:
            if verbose:
                print(f"[bsfs-exact] {pid}: empty discrete domain -> site infeasible")
            return []
        if cell_cap > 0 and len(cells) > cell_cap:
            # farthest-point subsample to bound A* branching (reduces the discrete
            # domain -> optimum is w.r.t. the REDUCED domain, documented tradeoff).
            from sealp.examples.layout.infer_assembly_ga import _farthest_point_order
            pts = [grid.cell_xy[c] for c in cells]
            order = _farthest_point_order(pts, cell_cap)
            cells = [cells[i] for i in order]
        part_cells0[pid] = cells
    if verbose:
        print("[bsfs-exact] grid cells=%d | per-part domain: %s" % (
            grid.n, {p: len(part_cells0[p]) for p in pick_order}))

    def h_of(remaining: List[str], cells: Dict[str, List[int]]) -> float:
        tot = 0.0
        for pid in remaining:
            gx = goal_xy[pid]
            best = min((float(np.linalg.norm(grid.cell_xy[c] - gx))
                        for c in cells[pid]), default=0.0)
            tot += best
        return tot

    root = {
        "k": n, "assign": {}, "g": 0.0,
        "min_clear": float("inf"), "min_manip": float("inf"),
        "occ": OccupancyBitmap(grid), "cells": part_cells0,
    }
    root["h"] = h_of(list(pick_order), part_cells0)

    # priority = (f, -depth, counter): A* by f, with a DEPTH-FIRST tiebreak among
    # equal-f nodes so the search dives to a complete layout quickly. Because the
    # tiebreak only reorders nodes of EQUAL f, the pop order is still
    # non-decreasing in f -> A* optimality is preserved.
    counter = itertools.count()
    openq = [(root["g"] + root["h"], -(n - root["k"]), next(counter), root)]
    nogood = set()
    complete: List[Dict] = []
    max_nodes = int(getattr(args, "max_nodes", 0)) or 10 ** 9
    expanded = 0

    while openq and len(complete) < max(int(args.beam_width), 1):
        f, _, _, node = heapq.heappop(openq)
        if node["k"] == 0:
            complete.append(node)            # A*: first popped goal is optimal
            if verbose:
                print(f"[bsfs-exact] complete layout g={node['g']:.4f} "
                      f"(optimum #{len(complete)})")
            continue
        expanded += 1
        stats["node_expansions"] = stats.get("node_expansions", 0) + 1
        if expanded > max_nodes:
            if verbose:
                print(f"[bsfs-exact] node cap {max_nodes} hit -> stop")
            break

        k = node["k"]
        pid = pick_order[k - 1]
        placed = set(pick_order[:k - 1])
        if preassembled_pid is not None:
            placed.add(preassembled_pid)
        oracle.apply_suffix(node["assign"], preassembled_pid)
        staged = ([preassembled_pid] if preassembled_pid else []) + list(node["assign"].keys())

        cands, poses_per_xy = _iter_rot(searcher, pid, args.poses_per_xy)
        for c_idx in node["cells"][pid]:
            xy = grid.cell_xy[c_idx]
            kept = 0
            for cand in cands:
                rot = str(getattr(cand, "rot_name", ""))
                key = (pid, c_idx, rot)
                if key in nogood:
                    continue
                stats["certifications"] += 1
                seed_everything(task_seed(base_seed, cid, k, pid, c_idx, rot, "certify"))
                rec, reason = oracle.certify(pid, xy, cand, placed, staged,
                                             level=2 if args.do_quick_check else 1)
                if rec is None:
                    if reason in HARD_FAIL:
                        nogood.add(key)
                        stats["hard_prunes"] = stats.get("hard_prunes", 0) + 1
                    continue
                rec["cell"] = c_idx
                # occupancy + sound domain propagation for the remaining parts
                # (earlier in assembly order). W_req / F_{k,j} applies to *later*
                # parts and is enforced inside StepOracle.certify — not here.
                occ = node["occ"].clone()
                occ.occupy(grid.footprint_mask(c_idx, foot_radius[pid]))
                remaining = list(pick_order[:k - 1])
                block = 0
                # Legacy optional XY transfer-corridor prune (NOT paper W_req).
                if getattr(args, "transfer_corridor_prune", False):
                    block = swept_segment_mask(
                        grid, xy, goal_xy[pid], part_footprint_radius(searcher, pid))
                cells2 = propagate_domains(remaining, node["cells"], occ,
                                           foot_radius, extra_block_mask=block)
                if cells2 is None:
                    stats["propagation_prunes"] = stats.get("propagation_prunes", 0) + 1
                    continue
                if not hall_feasible(remaining, cells2, occ):
                    stats["hall_pruned"] = stats.get("hall_pruned", 0) + 1
                    continue
                child = {
                    "k": k - 1,
                    "assign": {**node["assign"], pid: rec},
                    "g": node["g"] + rec["cost"],
                    "min_clear": min(node["min_clear"], rec["clearance"]),
                    "min_manip": min(node["min_manip"], rec["manipulability"]),
                    "occ": occ, "cells": cells2,
                }
                child["h"] = h_of(remaining, cells2)
                heapq.heappush(openq, (child["g"] + child["h"],
                                       -(n - child["k"]), next(counter), child))
                kept += 1
                if kept >= poses_per_xy:
                    break
        stats["max_depth_certified"] = max(stats["max_depth_certified"], n - (k - 1))

    complete.sort(key=lambda nd: lex_key(nd["g"], nd["min_clear"], nd["min_manip"]))
    return [nd["assign"] for nd in complete]


# ------------------------------------------------------------------
# BEAM : anytime backward beam search
# ------------------------------------------------------------------
def _search_beam(searcher, oracle, args, pick_order, preassembled_pid,
                 stats, verbose, base_seed=0, cid="0", cert_batch_fn=None) -> List[Dict]:
    n = len(pick_order)
    # For a fair optimality-gap vs the exact solver, --discrete-domain makes the
    # beam search the SAME discrete grid cells the exact A* uses (otherwise the
    # beam's continuous candidates form a different domain and can even undercut
    # the grid optimum, which is not a meaningful gap).
    use_discrete = bool(getattr(args, "discrete_domain", False))
    grid = build_staging_grid(searcher, args.grid_spacing) if use_discrete else None
    cand_xy: Dict[str, List[np.ndarray]] = {}
    for pid in pick_order:
        if use_discrete:
            cells = discrete_domain(searcher, pid, grid, args.grid_spacing, preassembled_pid)
            xys = [grid.cell_xy[c] for c in cells]
        else:
            xys = continuous_domain(searcher, pid, args.grid_spacing,
                                    args.cand_per_part, preassembled_pid)
        if not xys:
            if verbose:
                print(f"[bsfs-beam] {pid}: no candidates -> site infeasible")
            return []
        cand_xy[pid] = xys

    B = max(int(args.beam_width), 1)
    poses_per_xy = max(int(args.poses_per_xy), 1)
    nogood = set()
    beam: List[Dict] = [{
        "k": n, "assign": {}, "g": 0.0,
        "min_clear": float("inf"), "min_manip": float("inf"),
    }]

    level = 2 if args.do_quick_check else 1
    for k in range(n, 0, -1):
        pid = pick_order[k - 1]
        placed = set(pick_order[:k - 1])
        if preassembled_pid is not None:
            placed.add(preassembled_pid)
        children: List[Dict] = []
        cands, _ = _iter_rot(searcher, pid, poses_per_xy)

        if cert_batch_fn is None:
            # -------- serial certification (seeded per task) --------
            for node in beam:
                stats["node_expansions"] = stats.get("node_expansions", 0) + 1
                oracle.apply_suffix(node["assign"], preassembled_pid)
                staged = ([preassembled_pid] if preassembled_pid else []) \
                    + list(node["assign"].keys())
                suffix_xy = [np.asarray(r["xy"], dtype=float)
                             for r in node["assign"].values()]
                for xi, xy in enumerate(cand_xy[pid]):
                    if any(float(np.linalg.norm(np.asarray(xy)[:2] - s[:2])) < 0.02
                           for s in suffix_xy):
                        continue
                    kept = 0
                    for cand in cands:
                        rot = str(getattr(cand, "rot_name", ""))
                        key = (pid, round(float(xy[0]), 3), round(float(xy[1]), 3), rot)
                        if key in nogood:
                            continue
                        stats["certifications"] += 1
                        seed_everything(task_seed(base_seed, cid, k, pid, xi, rot, "certify"))
                        rec, reason = oracle.certify(pid, xy, cand, placed, staged,
                                                     level=level)
                        if rec is None:
                            if reason in HARD_FAIL:
                                nogood.add(key)
                                stats["hard_prunes"] = stats.get("hard_prunes", 0) + 1
                            continue
                        children.append({
                            "k": k - 1,
                            "assign": {**node["assign"], pid: rec},
                            "g": node["g"] + rec["cost"],
                            "min_clear": min(node["min_clear"], rec["clearance"]),
                            "min_manip": min(node["min_manip"], rec["manipulability"]),
                        })
                        kept += 1
                        if kept >= poses_per_xy:
                            break
        else:
            # -------- candidate-level parallel certification --------
            # Build all independent (node,xy,cand) tasks, evaluate in worker
            # processes, then reassemble children DETERMINISTICALLY (keep the
            # first poses_per_xy successes per (node,xy) in candidate order).
            tasks, meta = [], []
            for ni, node in enumerate(beam):
                stats["node_expansions"] = stats.get("node_expansions", 0) + 1
                staged = ([preassembled_pid] if preassembled_pid else []) \
                    + list(node["assign"].keys())
                suffix = [{"pid": q, "xy": np.asarray(r["xy"], dtype=float).tolist(),
                           "rot_name": r["rot_name"]} for q, r in node["assign"].items()]
                suffix_xy = [np.asarray(r["xy"], dtype=float)
                             for r in node["assign"].values()]
                for xi, xy in enumerate(cand_xy[pid]):
                    if any(float(np.linalg.norm(np.asarray(xy)[:2] - s[:2])) < 0.02
                           for s in suffix_xy):
                        continue
                    for ci, cand in enumerate(cands):
                        rot = str(getattr(cand, "rot_name", ""))
                        key = (pid, round(float(xy[0]), 3), round(float(xy[1]), 3), rot)
                        if key in nogood:
                            continue
                        tasks.append({
                            "suffix": suffix, "pid": pid,
                            "xy": np.asarray(xy, dtype=float).tolist(), "rot_name": rot,
                            "placed": sorted(placed), "staged": list(staged),
                            "level": level,
                            "seed": task_seed(base_seed, cid, k, pid, xi, rot, "certify"),
                        })
                        meta.append((ni, xi, ci, key))
            stats["certifications"] += len(tasks)
            results = cert_batch_fn(tasks) if tasks else []
            # group results by (ni, xi) preserving ci order
            grouped: Dict[tuple, List] = {}
            for (ni, xi, ci, key), res in zip(meta, results):
                if res.get("rec") is None:
                    if res.get("reason") in HARD_FAIL:
                        nogood.add(key)
                        stats["hard_prunes"] = stats.get("hard_prunes", 0) + 1
                    continue
                grouped.setdefault((ni, xi), []).append((ci, res["rec"]))
            for (ni, xi), recs in grouped.items():
                node = beam[ni]
                recs.sort(key=lambda t: t[0])
                for _ci, rec in recs[:poses_per_xy]:
                    children.append({
                        "k": k - 1,
                        "assign": {**node["assign"], pid: rec},
                        "g": node["g"] + rec["cost"],
                        "min_clear": min(node["min_clear"], rec["clearance"]),
                        "min_manip": min(node["min_manip"], rec["manipulability"]),
                    })
        if not children:
            if verbose:
                print(f"[bsfs-beam] depth {n - k + 1}/{n} ({pid}): beam emptied")
            return []
        children.sort(key=lambda nd: lex_key(nd["g"], nd["min_clear"], nd["min_manip"]))
        beam = children[:B]
        stats["max_depth_certified"] = max(stats["max_depth_certified"], n - (k - 1))
        if verbose:
            top = beam[0]
            print(f"[bsfs-beam] certified suffix depth {n - k + 1}/{n} (added {pid}): "
                  f"beam={len(beam)} best[g={top['g']:.4f} "
                  f"clr={top['min_clear']:.3f} manip={top['min_manip']:.4f}]")
    return [nd["assign"] for nd in beam]


# ------------------------------------------------------------------
# public entry
# ------------------------------------------------------------------
def search_site(searcher, args, station: np.ndarray, mode: str,
                stats: Dict, verbose: bool = True, cert_batch_fn=None) -> List[Dict]:
    """Run the backward search at one assembly site; return certified assigns.

    ``cert_batch_fn`` (optional) enables candidate-level parallelism for BEAM:
    given a list of certification task dicts it must return a same-length list of
    result dicts (from worker processes). When ``None`` the search certifies
    serially. Both paths seed each ``StepOracle.certify`` by stable task identity
    so results are independent of process scheduling.
    """
    from .cost import cost_params_from_args
    searcher._set_assembly_station(np.asarray(station, dtype=float))
    searcher._apply_first_part_as_assembled()
    first_pid = searcher._first_part_id()
    preassembled_pid = first_pid if searcher.preassemble_first_part else None
    pick_order = searcher._active_pick_part_order()
    if not pick_order:
        return []

    params = cost_params_from_args(args, mode=mode)
    oracle = StepOracle(searcher, params)
    base_seed = int(getattr(args, "seed", 0))
    cid = center_id(station)

    if mode == "exact":
        return _search_exact(searcher, oracle, args, pick_order,
                             preassembled_pid, stats, verbose, base_seed, cid)
    return _search_beam(searcher, oracle, args, pick_order,
                        preassembled_pid, stats, verbose, base_seed, cid,
                        cert_batch_fn=cert_batch_fn)
