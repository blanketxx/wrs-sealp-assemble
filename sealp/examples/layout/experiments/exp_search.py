"""Generalized beam search over (processing_order, state_model).

Reuses the frozen StepOracle / domain / cost / seeding UNCHANGED. Only two
controlled knobs are exposed:

  processing_order : "backward" (decide last-picked part first; the BSFS suffix
                     ordering) or "forward" (decide first-picked part first).
  state_model      : how already-decided / not-yet-decided parts appear as
                     obstacles when certifying a step:
                       "sequential"   -> physically correct step state: parts
                                         assembled BEFORE the current part are at
                                         GOAL, parts assembled AFTER are at
                                         STAGING. Forward processing cannot see
                                         the (undecided) later staging poses, so
                                         those are parked far away (absent) --
                                         this is exactly the myopia that makes a
                                         forward beam weaker than backward BSFS.
                       "static_start" -> every other part always at its STAGING
                                         pose (never assembled).
                       "static_final" -> every other part always at its GOAL
                                         pose (all assembled).

The (backward, sequential) configuration reproduces the frozen beam baseline;
this is asserted by the smoke test before any headline run.

Selection, beam width, per-task deterministic seeding, HARD_FAIL caching and the
candidate/serial certification paths mirror bsfs.search._search_beam exactly so
the ONLY differences are the two knobs above.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from sealp.examples.layout.bsfs.cost import CostParams, lex_key
from sealp.examples.layout.bsfs.domain import continuous_domain, discrete_domain, build_staging_grid
from sealp.examples.layout.bsfs.oracle import HARD_FAIL, StepOracle
from sealp.examples.layout.bsfs.search import _iter_rot
from sealp.examples.layout.bsfs.seeding import center_id, seed_everything, task_seed

# far "parked" pose offset (meters) for undecided/absent parts -- large enough to
# never interact with the assembly workspace or the robot.
_FAR_OFFSET = np.array([10.0, 10.0], dtype=float)


def _assembly_index(pick_order: List[str]) -> Dict[str, int]:
    return {pid: i for i, pid in enumerate(pick_order)}


def _scene_for_step(pid: str, decided: Dict[str, Dict], pick_order: List[str],
                    preassembled_pid: Optional[str], order: str, state_model: str,
                    station_xy: np.ndarray, searcher):
    """Return (placed_list, applied_poses, staged_list) describing the obstacle
    scene when certifying ``pid``.

    ``applied_poses`` is the list of {pid,xy,rot_name} the worker must apply as
    staging poses so that _step_obstacles reflects the intended state (decided
    parts at their real staging xy; parts that must be ABSENT are parked far).
    ``placed_list`` are parts pinned at GOAL. ``staged_list`` are the parts that
    count as *real* staging obstacles (for active collision / clearance).
    """
    aidx = _assembly_index(pick_order)
    others = [q for q in pick_order if q != pid]

    def _first_rot(q):
        cands = searcher.rot_cands.get(q, []) or []
        return str(getattr(cands[0], "rot_name", "identity")) if cands else "identity"

    far_xy = (np.asarray(station_xy, dtype=float)[:2] + _FAR_OFFSET).tolist()

    placed = set()
    if preassembled_pid is not None:
        placed.add(preassembled_pid)
    applied = []          # staging poses to apply in the worker
    staged = []           # real staging obstacles (decided)

    if state_model == "static_final":
        # every other pickable part at GOAL; nothing at staging.
        for q in others:
            placed.add(q)
        return sorted(placed), applied, staged

    if state_model == "static_start":
        # every other pickable part at STAGING; only decided ones are known.
        for q in others:
            if q in decided:
                applied.append({"pid": q, "xy": list(map(float, decided[q]["xy"])),
                                "rot_name": str(decided[q]["rot_name"])})
                staged.append(q)
            else:
                applied.append({"pid": q, "xy": far_xy, "rot_name": _first_rot(q)})
        return sorted(placed), applied, staged

    # ---- sequential (physically correct step state) ----
    for q in others:
        if aidx[q] < aidx[pid]:
            # assembled before pid -> at GOAL (goal is fixed/known regardless).
            placed.add(q)
        else:
            # assembled after pid -> at STAGING. Known only if already decided
            # (backward: yes; forward: no -> parked far / absent).
            if q in decided:
                applied.append({"pid": q, "xy": list(map(float, decided[q]["xy"])),
                                "rot_name": str(decided[q]["rot_name"])})
                staged.append(q)
            else:
                applied.append({"pid": q, "xy": far_xy, "rot_name": _first_rot(q)})
    return sorted(placed), applied, staged


def experimental_search_site(searcher, args, station, mode, stats, verbose=True,
                             cert_batch_fn=None, *, order="backward",
                             state_model="sequential"):
    """Beam search at one assembly site with a controlled (order, state_model).

    Signature is call-compatible with bsfs.search.search_site so it can be
    monkeypatched in for run.main()'s _site_best.
    """
    searcher._set_assembly_station(np.asarray(station, dtype=float))
    searcher._apply_first_part_as_assembled()
    first_pid = searcher._first_part_id()
    preassembled_pid = first_pid if searcher.preassemble_first_part else None
    pick_order = searcher._active_pick_part_order()
    if not pick_order:
        return []

    params = CostParams(
        lift=float(getattr(args, "lift", 0.10)),
        tau_clear=float(getattr(args, "tau_clear", 0.005)),
        tau_manip=float(getattr(args, "tau_manip", 1.0e-3)),
    )
    oracle = StepOracle(searcher, params)
    base_seed = int(getattr(args, "seed", 0))
    cid = center_id(station)
    station_xy = np.asarray(station, dtype=float)[:2]

    # candidate (x,y) domain per part -- IDENTICAL to bsfs.search._search_beam.
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
                print(f"[exp:{order}/{state_model}] {pid}: no candidates -> infeasible")
            return []
        cand_xy[pid] = xys

    proc_order = list(pick_order) if order == "forward" else list(reversed(pick_order))
    B = max(int(args.beam_width), 1)
    poses_per_xy = max(int(args.poses_per_xy), 1)
    level = 2 if getattr(args, "do_quick_check", False) else 1
    nogood = set()
    beam: List[Dict] = [{"assign": {}, "g": 0.0,
                         "min_clear": float("inf"), "min_manip": float("inf")}]

    for depth, pid in enumerate(proc_order):
        cands, _ = _iter_rot(searcher, pid, poses_per_xy)
        children: List[Dict] = []

        # build all (node, xy, cand) tasks then certify (parallel or serial),
        # reassembling children deterministically -- mirrors _search_beam.
        tasks, meta = [], []
        for ni, node in enumerate(beam):
            stats["node_expansions"] = stats.get("node_expansions", 0) + 1
            decided = node["assign"]
            suffix_xy = [np.asarray(r["xy"], dtype=float) for r in decided.values()]
            placed_list, applied, staged_list = _scene_for_step(
                pid, decided, pick_order, preassembled_pid, order, state_model,
                station_xy, searcher)
            for xi, xy in enumerate(cand_xy[pid]):
                if any(float(np.linalg.norm(np.asarray(xy)[:2] - s[:2])) < 0.02
                       for s in suffix_xy):
                    continue
                for ci, cand in enumerate(cands):
                    rot = str(getattr(cand, "rot_name", ""))
                    key = (pid, round(float(xy[0]), 3), round(float(xy[1]), 3), rot,
                           order, state_model, depth)
                    if key in nogood:
                        continue
                    tasks.append({
                        "suffix": applied, "pid": pid,
                        "xy": np.asarray(xy, dtype=float).tolist(), "rot_name": rot,
                        "placed": placed_list, "staged": list(staged_list),
                        "level": level,
                        "seed": task_seed(base_seed, cid, depth, pid, xi, rot, "certify"),
                    })
                    meta.append((ni, xi, ci, key))

        stats["certifications"] += len(tasks)
        if cert_batch_fn is not None:
            results = cert_batch_fn(tasks) if tasks else []
        else:
            results = [_certify_local(oracle, searcher, station, preassembled_pid, t)
                       for t in tasks]

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
                    "assign": {**node["assign"], pid: rec},
                    "g": node["g"] + rec["cost"],
                    "min_clear": min(node["min_clear"], rec["clearance"]),
                    "min_manip": min(node["min_manip"], rec["manipulability"]),
                })

        if not children:
            if verbose:
                print(f"[exp:{order}/{state_model}] depth {depth + 1}/{len(proc_order)} "
                      f"({pid}): beam emptied")
            return []
        children.sort(key=lambda nd: lex_key(nd["g"], nd["min_clear"], nd["min_manip"]))
        beam = children[:B]
        stats["max_depth_certified"] = max(stats.get("max_depth_certified", 0), depth + 1)
        if verbose:
            top = beam[0]
            print(f"[exp:{order}/{state_model}] depth {depth + 1}/{len(proc_order)} "
                  f"(+{pid}): beam={len(beam)} best[g={top['g']:.4f} "
                  f"clr={top['min_clear']:.3f} manip={top['min_manip']:.4f}]")

    return [nd["assign"] for nd in beam]


def _certify_local(oracle, searcher, station, preassembled_pid, task) -> Dict:
    """Serial (single-process) certify mirroring run._worker_certify's scene
    reconstruction, for --workers 1 / debugging."""
    from sealp.examples.layout.bsfs.run import _ensure_yaw_cand, _cand_by_rotname
    searcher._set_assembly_station(np.asarray(station, dtype=float))
    searcher._apply_first_part_as_assembled()
    for s in task["suffix"]:
        _ensure_yaw_cand(searcher, s["pid"], s["rot_name"])
        cand = _cand_by_rotname(searcher, s["pid"], s["rot_name"])
        if cand is not None:
            searcher._apply_staging_pose(s["pid"], np.asarray(s["xy"], dtype=float), cand)
    _ensure_yaw_cand(searcher, task["pid"], task["rot_name"])
    cand = _cand_by_rotname(searcher, task["pid"], task["rot_name"])
    seed_everything(int(task["seed"]))
    rec, reason = oracle.certify(task["pid"], np.asarray(task["xy"], dtype=float),
                                 cand, set(task["placed"]), list(task["staged"]),
                                 level=int(task["level"]))
    from sealp.examples.layout.bsfs.search import rec_jsonable
    return {"rec": rec_jsonable(rec), "reason": reason}
