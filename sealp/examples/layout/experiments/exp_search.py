"""Generalized beam search over (processing_order, state_model).

Reuses the frozen StepOracle / domain / cost / seeding UNCHANGED. Only two
controlled knobs are exposed:

  processing_order : "backward"     -> decide the last-picked part first; the
                                       BSFS suffix ordering.
                     "forward"      -> decide the first-picked part first, with
                                       the undecided suffix optimistically absent.
                     "forward_iter" -> forward, but repeated: each pass after the
                                       first certifies against the suffix staging
                                       poses chosen by the previous pass, so
                                       forward faces the same occupied staging
                                       area backward sees. Stops on a fixed point;
                                       every pass is billed to the shared counters.
  state_model      : how already-decided / not-yet-decided parts appear as
                     obstacles when certifying a step:
                       "sequential"   -> physically correct step state: parts
                                         assembled BEFORE the current part are at
                                         GOAL, parts assembled AFTER are at
                                         STAGING. Forward processing cannot see
                                         the (undecided) later staging poses, so
                                         those are parked far away (absent)
                                         unless "forward_iter" supplies an
                                         estimate from a previous pass -- this
                                         asymmetry is exactly what the backward
                                         suffix ordering removes by construction.
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

# Module-global side channel: experimental_search_site runs in the MAIN process
# (candidates parallelism), so run_one (same process) can read extra metrics not
# surfaced by the frozen run.main() result -- e.g. the number of complete leaves
# the beam returned. Reset per subprocess (module re-imported fresh each run).
LAST_RUN_STATS: Dict[str, float] = {"complete_leaves": 0}


def _assembly_index(pick_order: List[str]) -> Dict[str, int]:
    return {pid: i for i, pid in enumerate(pick_order)}


def _scene_for_step(pid: str, decided: Dict[str, Dict], pick_order: List[str],
                    preassembled_pid: Optional[str], order: str, state_model: str,
                    station_xy: np.ndarray, searcher,
                    prior: Optional[Dict[str, Dict]] = None):
    """Return (placed_list, applied_poses, staged_list, n_absent) describing the
    obstacle scene when certifying ``pid``.

    ``applied_poses`` is the list of {pid,xy,rot_name} the worker must apply as
    staging poses so that _step_obstacles reflects the intended state (decided
    parts at their real staging xy; parts that must be ABSENT are parked far).
    ``placed_list`` are parts pinned at GOAL. ``staged_list`` are the parts that
    count as *real* staging obstacles (for active collision / clearance).

    ``n_absent`` counts the parts that execution would have sitting at staging
    but that this scene omits. It is 0 for every backward step by construction,
    and positive for forward steps whose suffix is still undecided, which is what
    makes a certification issued under this scene optimistic.
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
    absent = 0            # parts execution would have at staging but this omits

    if state_model == "static_final":
        # every other pickable part at GOAL; nothing at staging.
        for q in others:
            placed.add(q)
        return sorted(placed), applied, staged, absent

    if state_model == "static_start":
        # every other pickable part at STAGING; only decided ones are known.
        for q in others:
            if q in decided:
                applied.append({"pid": q, "xy": list(map(float, decided[q]["xy"])),
                                "rot_name": str(decided[q]["rot_name"])})
                staged.append(q)
            else:
                applied.append({"pid": q, "xy": far_xy, "rot_name": _first_rot(q)})
                absent += 1
        return sorted(placed), applied, staged, absent

    # ---- sequential (physically correct step state) ----
    for q in others:
        if aidx[q] < aidx[pid]:
            # assembled before pid -> at GOAL (goal is fixed/known regardless).
            placed.add(q)
        else:
            # assembled after pid -> at STAGING. Known exactly only if already
            # decided (backward: always, by construction). A forward pass may
            # instead supply ``prior`` -- an estimate of the suffix staging poses
            # carried over from the previous pass -- so that it, too, certifies
            # against an occupied staging area. Without either, the part is
            # parked far away, i.e. optimistically absent.
            if q in decided:
                applied.append({"pid": q, "xy": list(map(float, decided[q]["xy"])),
                                "rot_name": str(decided[q]["rot_name"])})
                staged.append(q)
            elif prior is not None and q in prior:
                applied.append({"pid": q, "xy": list(map(float, prior[q]["xy"])),
                                "rot_name": str(prior[q]["rot_name"])})
                staged.append(q)
            else:
                applied.append({"pid": q, "xy": far_xy, "rot_name": _first_rot(q)})
                absent += 1
    return sorted(placed), applied, staged, absent


def _prior_key(prior: Optional[Dict[str, Dict]]) -> tuple:
    """Order-independent identity of a suffix estimate, for convergence testing."""
    if not prior:
        return ()
    return tuple(sorted(
        (q, round(float(r["xy"][0]), 4), round(float(r["xy"][1]), 4),
         str(r["rot_name"])) for q, r in prior.items()))


def experimental_search_site(searcher, args, station, mode, stats, verbose=True,
                             cert_batch_fn=None, *, order="backward",
                             state_model="sequential", forward_passes=1):
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

    proc_order = (list(pick_order) if order.startswith("forward")
                  else list(reversed(pick_order)))
    B = max(int(args.beam_width), 1)
    poses_per_xy = max(int(args.poses_per_xy), 1)
    level = 2 if getattr(args, "do_quick_check", False) else 1

    def _one_pass(prior: Optional[Dict[str, Dict]], seed_tag: str) -> List[Dict]:
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
                occupied = [np.asarray(r["xy"], dtype=float) for r in decided.values()]
                if prior:
                    occupied.extend(np.asarray(r["xy"], dtype=float)
                                    for q, r in prior.items()
                                    if q != pid and q not in decided)
                placed_list, applied, staged_list, n_absent = _scene_for_step(
                    pid, decided, pick_order, preassembled_pid, order, state_model,
                    station_xy, searcher, prior=prior)
                for xi, xy in enumerate(cand_xy[pid]):
                    if any(float(np.linalg.norm(np.asarray(xy)[:2] - s[:2])) < 0.02
                           for s in occupied):
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
                            "seed": task_seed(base_seed, cid, depth, pid, xi, rot,
                                              seed_tag),
                        })
                        meta.append((ni, xi, ci, key))
                        if n_absent:
                            LAST_RUN_STATS["optimistic_certifications"] = \
                                LAST_RUN_STATS.get("optimistic_certifications", 0) + 1

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
                    print(f"[exp:{order}/{state_model}] depth {depth + 1}/"
                          f"{len(proc_order)} ({pid}): beam emptied")
                return []
            children.sort(key=lambda nd: lex_key(nd["g"], nd["min_clear"],
                                                 nd["min_manip"]))
            beam = children[:B]
            stats["max_depth_certified"] = max(stats.get("max_depth_certified", 0),
                                               depth + 1)
            if verbose:
                top = beam[0]
                print(f"[exp:{order}/{state_model}] depth {depth + 1}/"
                      f"{len(proc_order)} (+{pid}): beam={len(beam)} "
                      f"best[g={top['g']:.4f} clr={top['min_clear']:.3f} "
                      f"manip={top['min_manip']:.4f}]")

        return [nd["assign"] for nd in beam]

    # Backward fixes each suffix part BEFORE any part that has to avoid it, so one
    # pass already certifies against the true sequential state. Forward cannot:
    # the later-assembled parts are still undecided while the early ones are being
    # certified. forward_passes > 1 lets forward re-certify against the suffix
    # estimate left by the previous pass, iterating toward a fixed point that is
    # not guaranteed to exist. All passes are billed to the same counters, so the
    # reported search cost includes the price of obtaining that estimate.
    n_passes = max(int(forward_passes), 1) if order.startswith("forward") else 1
    prior: Optional[Dict[str, Dict]] = None
    result: List[Dict] = []
    LAST_RUN_STATS["prior_converged"] = 0
    for p in range(n_passes):
        result = _one_pass(prior, "certify" if p == 0 else f"certify_pass{p}")
        LAST_RUN_STATS["passes_run"] = p + 1
        if not result:
            break
        nxt = {q: {"xy": list(map(float, r["xy"])), "rot_name": str(r["rot_name"])}
               for q, r in result[0].items()}
        if _prior_key(nxt) == _prior_key(prior):
            LAST_RUN_STATS["prior_converged"] = 1
            break
        prior = nxt

    # ---- suffix-soundness audit of the layout actually returned ----
    # Total certification counts saturate at the beam's capacity (beam_width x
    # domain size), so once both orders keep their beams full the counts are
    # identical and say nothing about search order. What still differs is whether
    # each certification was issued against the scene execution will really
    # present. This audit re-certifies every step of the FINAL layout under the
    # true sequential state -- suffix parts at the staging poses the layout
    # actually assigns them, prefix parts at goal -- and counts the steps whose
    # search-time verdict does not survive. Backward's search scene already IS
    # that scene at every step, so it must audit clean; a forward step certified
    # with its suffix absent need not. The cost is one certification per part
    # (4 for the chair, against ~928 for the search itself), and it is billed to
    # its own counter so the headline certification count stays comparable.
    if result:
        best = result[0]
        audit_tasks = []
        for pid in pick_order:
            if pid not in best:
                continue
            decided_all = {q: r for q, r in best.items() if q != pid}
            placed_list, applied, staged_list, n_absent = _scene_for_step(
                pid, decided_all, pick_order, preassembled_pid,
                "backward", "sequential", station_xy, searcher)
            rot = str(best[pid]["rot_name"])
            audit_tasks.append({
                "suffix": applied, "pid": pid,
                "xy": [float(best[pid]["xy"][0]), float(best[pid]["xy"][1])],
                "rot_name": rot, "placed": placed_list,
                "staged": list(staged_list), "level": level,
                "seed": task_seed(base_seed, cid, 0, pid, 0, rot, "audit"),
            })
        if audit_tasks:
            if cert_batch_fn is not None:
                audit_res = cert_batch_fn(audit_tasks)
            else:
                audit_res = [_certify_local(oracle, searcher, station,
                                            preassembled_pid, t)
                             for t in audit_tasks]
            unsound = [t["pid"] for t, r in zip(audit_tasks, audit_res)
                       if r.get("rec") is None]
            LAST_RUN_STATS["audit_certifications"] = len(audit_tasks)
            LAST_RUN_STATS["unsound_steps"] = len(unsound)
            LAST_RUN_STATS["unsound_parts"] = ",".join(unsound)
            if verbose:
                print(f"[exp:{order}/{state_model}] suffix-soundness audit: "
                      f"{len(unsound)}/{len(audit_tasks)} step(s) do NOT hold under "
                      f"the true sequential state"
                      + (f" -> {','.join(unsound)}" if unsound else ""))

    # complete leaves = number of complete assignments returned by this site.
    LAST_RUN_STATS["complete_leaves"] = (
        LAST_RUN_STATS.get("complete_leaves", 0) + len(result))
    return result


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
