"""Run ONE experiment configuration in an isolated subprocess.

Applies the requested experiment variant (search order / obstacle state model /
Hall+propagation ablation) by monkeypatching ONLY the search entry point that
run.main() calls -- the frozen core files are never edited -- then invokes the
normal BSFS pipeline (run.main) so all frozen orchestration (assembly-center
search, parallel witness, yaw refinement, robust full witness, metrics, JSON
output) is reused unchanged. Finally appends one row to the summary CSV.

The variants and how they are realized:

  --exp-order backward|forward + --exp-state sequential|static_start|static_final
      -> run.search_site is replaced by experiments.exp_search.experimental_search_site
         bound to (order, state_model). Runs the SAME StepOracle/domain/cost and
         dispatches certify to the SAME worker pool (candidates parallelism).
         Requires a single assembly center (--center-search single) so the search
         loop stays in the main process; workers still do certify (workers=4).

  --exp-hall off / --exp-prop off
      -> search.hall_feasible / search.propagate_domains are neutralized. These
         are only exercised by the EXACT A* solver, so use with --mode exact.

  (no exp-* flags) -> native run.main (frozen backward beam or exact A*).

  --l3-only PATH
      -> no search at all. Loads a previous run's result JSON, replays ONLY the
         staging_aware full-sequence L3 on the layout it already selected, and
         appends a CSV row carrying that run's original search metrics together
         with the fresh L3 verdict. Use it when an L3 check failed for a reason
         unrelated to the layout (e.g. a crash) so the expensive L2 search does
         not have to be repeated. The passthrough args must match the original
         run so the searcher is rebuilt identically.

Everything after ``--`` is passed verbatim to run.main (asmdef, grasp-dir,
part-order, mode, grid, beam-width, workers, ...). This module injects
--seed and --output-json; do not pass them in the passthrough.

BLAS threads are pinned to 1 before numpy import (mirrors run.py) so every
subprocess is deterministic and matches the frozen configuration.
"""
from __future__ import annotations

import os

for _thr_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thr_var] = "1"

import argparse
import csv
import datetime as _dt
import json
import sys
import time
import traceback
from typing import Dict, List, Optional

CSV_COLUMNS = [
    "timestamp", "priority", "variant", "task", "seed",
    "mode", "order", "state_model", "hall", "prop",
    "workers", "parallel_level", "success",
    "objective_cost", "ref_cost", "optimality_gap",
    "runtime_s", "time_to_first_feasible_s",
    "oracle_certifications", "node_expansions",
    "complete_leaves", "witness_calls", "deferred_witness_attempts",
    "hard_prunes", "hall_prunes", "propagation_prunes",
    "min_common_grasp", "num_parts",
    "l3_staging_aware", "l3_fail_step", "l3_fail_reason",
    "center_source", "assembly_center", "layout_fingerprint",
    "output_json", "fail_reason",
]


def _append_csv(csv_path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _unique_json_path(out_dir: str, run_id: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_id}.json")
    if os.path.exists(path):
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{run_id}_{stamp}.json")
    return path


def _merge_json(path: str, extra: Dict) -> None:
    """Fold extra bookkeeping into an already-written per-run JSON."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _apply_patches(order: Optional[str], state: Optional[str],
                   hall_off: bool, prop_off: bool, forward_passes: int = 1) -> None:
    import sealp.examples.layout.bsfs.run as run
    import sealp.examples.layout.bsfs.search as search

    if order or state:
        from sealp.examples.layout.experiments.exp_search import experimental_search_site
        _order = order or "backward"
        _state = state or "sequential"
        _passes = max(int(forward_passes), 1)

        def _patched_site(searcher, args, station, mode, stats,
                          verbose=True, cert_batch_fn=None):
            return experimental_search_site(
                searcher, args, station, mode, stats, verbose=verbose,
                cert_batch_fn=cert_batch_fn, order=_order, state_model=_state,
                forward_passes=_passes)

        run.search_site = _patched_site
        print(f"[run_one] patched search_site -> experimental "
              f"(order={_order}, state_model={_state}, passes={_passes})")

    if hall_off:
        search.hall_feasible = lambda remaining, cells, occ: True
        print("[run_one] Hall matching feasibility DISABLED (ablation)")
    if prop_off:
        search.propagate_domains = (
            lambda remaining, cells, occ, foot_radius, extra_block_mask=0: dict(cells))
        print("[run_one] domain propagation DISABLED (ablation)")


def _parse_l3_fail_step(reason: str) -> str:
    """Extract the failing step index from an l3_fail_reason of the form
    'L3 step={idx} {pid} ...: {err}'. Returns '' if not parseable."""
    import re
    m = re.search(r"step=(\d+)", reason or "")
    return m.group(1) if m else ""


def _swap_grasp_dir(passthrough: List[str], grasp_dir: str) -> List[str]:
    """Return a copy of passthrough with --grasp-dir replaced (for full-grasp L3)."""
    out = list(passthrough)
    if "--grasp-dir" in out:
        out[out.index("--grasp-dir") + 1] = grasp_dir
    else:
        out += ["--grasp-dir", grasp_dir]
    return out


def _staging_aware_l3(passthrough: List[str], out_json: str, seed: int,
                      l3_grasp_dir: str = "") -> Dict[str, str]:
    """Replay the selected layout (no search) and return the staging_aware L3
    diagnostics: {verdict: PASS/FAIL/NA, fail_step, fail_reason}. Reuses the
    frozen witness + validator; no algorithmic change. If ``l3_grasp_dir`` is
    set, the layout (found on the lean grasp set) is re-verified against the FULL
    grasp set for soundness."""
    out = {"verdict": "NA", "fail_step": "", "fail_reason": ""}
    try:
        import numpy as np
        import sealp.examples.layout.bsfs.run as run
        with open(out_json, encoding="utf-8") as fh:
            result = json.load(fh)
        best_layout = result.get("best_layout", {})
        preassembled = result.get("preassembled_part")
        center = np.asarray(result["assembly_center"], dtype=float)
        pt = _swap_grasp_dir(passthrough, l3_grasp_dir) if l3_grasp_dir else list(passthrough)
        args = run.parse_args(pt + ["--seed", str(seed)])
        searcher = run._build_searcher(args)
        assign = {}
        for pid in searcher.part_order:
            if pid == preassembled or pid not in best_layout:
                continue
            e = best_layout[pid]
            assign[pid] = {"xy": [float(e["init_pos"][0]), float(e["init_pos"][1])],
                           "rot_name": str(e["rot_name"]), "cost": 0.0}
        searcher._set_assembly_station(center)
        status, cand, _ = run.robust_evaluate_layout(
            searcher, assign, preassembled, center, args,
            list(searcher.part_order), run._new_stats())
        if status != "SUCCESS":
            out["fail_reason"] = "L2 reconstruction failed"
            return out
        ok = bool(searcher.validate_full_sequence_l3(
            cand, obstacle_mode="staging_aware", verbose=False))
        if ok:
            out["verdict"] = "PASS"
        else:
            out["verdict"] = "FAIL"
            reason = str(getattr(cand, "l3_fail_reason", "") or "")
            out["fail_reason"] = reason
            out["fail_step"] = _parse_l3_fail_step(reason)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[run_one] L3 check error: {type(exc).__name__}: {exc}")
        out["fail_reason"] = f"{type(exc).__name__}: {exc}"
        return out


def _extract_row(result: Optional[Dict], meta: Dict, ref_cost: Optional[float],
                 success: bool, fail_reason: str) -> Dict:
    row = {c: "" for c in CSV_COLUMNS}
    row.update({
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "priority": meta["priority"], "variant": meta["variant"],
        "task": meta["task"], "seed": meta["seed"],
        "order": meta.get("order") or "", "state_model": meta.get("state") or "",
        "hall": "off" if meta.get("hall_off") else "on",
        "prop": "off" if meta.get("prop_off") else "on",
        "ref_cost": "" if ref_cost is None else ref_cost,
        "success": success, "output_json": meta.get("output_json", ""),
        "fail_reason": fail_reason,
    })
    if result:
        cost = result.get("objective_cost")
        row.update({
            "mode": result.get("method", ""),
            "workers": result.get("execution", {}).get("workers", ""),
            "parallel_level": result.get("execution", {}).get("parallel_level", ""),
            "objective_cost": cost,
            "num_parts": result.get("num_parts", ""),
            "layout_fingerprint": result.get("layout_fingerprint", ""),
            "assembly_center": ";".join(
                f"{v:.4f}" for v in result.get("assembly_center", [])),
        })
        t = result.get("timing", {})
        row["runtime_s"] = t.get("total_runtime", "")
        row["time_to_first_feasible_s"] = t.get("time_to_first_feasible", "")
        b = result.get("bsfs_stats", {})
        row["oracle_certifications"] = b.get("oracle_certifications", "")
        row["node_expansions"] = b.get("node_expansions", "")
        row["hard_prunes"] = b.get("hard_prunes", "")
        row["hall_prunes"] = b.get("hall_prunes", "")
        row["propagation_prunes"] = b.get("propagation_prunes", "")
        w = result.get("witness", {})
        row["witness_calls"] = w.get("full_witness_calls", "")
        row["deferred_witness_attempts"] = w.get("witness_attempts_total", "")
        row["min_common_grasp"] = result.get(
            "grasp_robustness", {}).get("min_common_grasp_count", "")
        if ref_cost and cost:
            try:
                row["optimality_gap"] = (float(cost) - float(ref_cost)) / float(ref_cost)
            except Exception:
                pass
    return row


def _run_l3_only(a, passthrough: List[str]) -> int:
    """Replay ONLY the staging_aware L3 on an already-selected layout.

    The costly part of a run is the L2 search; an L3 verdict lost to a crash or
    an environment defect should not force it to be redone. The original search
    metrics are carried over verbatim from the source JSON so the new CSV row
    stays comparable with the rest of the suite, and the source run is left
    untouched.
    """
    src = os.path.abspath(a.l3_only)
    if not os.path.isfile(src):
        print(f"[run_one] --l3-only: no such result JSON: {src}")
        return 2
    with open(src, encoding="utf-8") as fh:
        result = json.load(fh)

    run_id = f"{a.priority}_{a.variant}_{a.task}_seed{a.seed}"
    out_json = _unique_json_path(a.out_dir, run_id)
    meta = {
        "priority": a.priority, "variant": a.variant, "task": a.task, "seed": a.seed,
        "order": a.exp_order, "state": a.exp_state,
        "hall_off": False, "prop_off": False, "output_json": out_json,
    }
    success = str(result.get("witness_status", "")).endswith("PASS")

    print("=" * 70)
    print(f"[run_one] {run_id}  (L3-ONLY replay, no search)")
    print(f"[run_one] source layout : {src}")
    print(f"[run_one] source cost   : {result.get('objective_cost')}")
    print("=" * 70)

    if not success:
        l3 = {"verdict": "NA", "fail_step": "",
              "fail_reason": "source run has no L2-passing layout"}
    else:
        gd = "FULL" if a.l3_grasp_dir else "search"
        print(f"[run_one] running staging_aware L3 (grasps={gd}) ...")
        l3 = _staging_aware_l3(passthrough, src, a.seed, a.l3_grasp_dir)

    row = _extract_row(result, meta, a.ref_cost, success, "")
    row["center_source"] = a.center_source
    row["l3_staging_aware"] = l3["verdict"]
    row["l3_fail_step"] = l3["fail_step"]
    row["l3_fail_reason"] = l3["fail_reason"]

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({
            "l3_only_replay_of": src,
            "l3_staging_aware": l3["verdict"],
            "l3_fail_step": l3["fail_step"],
            "l3_fail_reason": l3["fail_reason"],
            "l3_grasp_dir": a.l3_grasp_dir,
            "source_objective_cost": result.get("objective_cost"),
            "source_layout_fingerprint": result.get("layout_fingerprint"),
            "source_assembly_center": result.get("assembly_center"),
            "source_best_layout": result.get("best_layout"),
        }, fh, indent=2)

    _append_csv(a.csv, row)
    print(f"[run_one] staging_aware L3 = {l3['verdict']} "
          f"(step={l3['fail_step'] or '-'}, reason={l3['fail_reason'] or '-'})")
    print(f"[run_one] recorded -> {a.csv}")
    print(f"RUN_ONE_RESULT_JSON={out_json}")
    print(f"RUN_ONE_COST={result.get('objective_cost', '')}")
    print(f"RUN_ONE_SUCCESS={int(l3['verdict'] == 'PASS')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priority", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exp-order",
                    choices=["backward", "forward", "forward_iter"], default=None)
    ap.add_argument("--exp-forward-passes", type=int, default=2,
                    help="number of forward passes when --exp-order forward_iter: "
                         "each pass after the first re-certifies against the suffix "
                         "staging estimate from the previous one, and stops early on "
                         "a fixed point. Ignored for backward/forward.")
    ap.add_argument("--exp-state",
                    choices=["sequential", "static_start", "static_final"], default=None)
    ap.add_argument("--exp-hall", choices=["on", "off"], default="on")
    ap.add_argument("--exp-prop", choices=["on", "off"], default="on")
    ap.add_argument("--ref-cost", type=float, default=None,
                    help="reference optimum cost for optimality_gap (P3 beam vs exact).")
    ap.add_argument("--l3-check", choices=["none", "staging_aware"], default="none",
                    help="after a successful L2 run, replay the selected layout and "
                         "record the staging_aware full-sequence L3 verdict.")
    ap.add_argument("--center-source", default="",
                    help="human-readable provenance of the (predetermined) assembly "
                         "center; recorded verbatim in the CSV for auditability.")
    ap.add_argument("--l3-grasp-dir", default="",
                    help="if set, the staging_aware L3 recheck re-verifies the "
                         "layout against this (FULL) grasp dir for soundness, even "
                         "though search used a lean grasp set.")
    ap.add_argument("--l3-only", default="",
                    help="path to a previous run's result JSON. Skips the search "
                         "entirely and only replays the staging_aware L3 on the "
                         "layout that run already selected.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("run_args", nargs=argparse.REMAINDER,
                    help="everything after -- is forwarded to run.main verbatim.")
    a = ap.parse_args(argv)

    passthrough = list(a.run_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if a.l3_only:
        return _run_l3_only(a, passthrough)

    run_id = f"{a.priority}_{a.variant}_{a.task}_seed{a.seed}"
    out_json = _unique_json_path(a.out_dir, run_id)

    hall_off = (a.exp_hall == "off")
    prop_off = (a.exp_prop == "off")
    _apply_patches(a.exp_order, a.exp_state, hall_off, prop_off,
                   forward_passes=(a.exp_forward_passes
                                   if a.exp_order == "forward_iter" else 1))

    argv_run = passthrough + ["--seed", str(a.seed), "--output-json", out_json]

    meta = {
        "priority": a.priority, "variant": a.variant, "task": a.task, "seed": a.seed,
        "order": a.exp_order, "state": a.exp_state,
        "hall_off": hall_off, "prop_off": prop_off, "output_json": out_json,
    }

    print("=" * 70)
    print(f"[run_one] {run_id}")
    print(f"[run_one] passthrough run.py args: {' '.join(argv_run)}")
    print("=" * 70)

    import sealp.examples.layout.bsfs.run as run

    t0 = time.time()
    result, success, fail_reason = None, False, ""
    try:
        result = run.main(argv_run)
        success = bool(result) and result.get("witness_status", "").endswith("PASS")
        if not success and result:
            fail_reason = result.get("witness_status", "no witness pass")
    except SystemExit as exc:
        fail_reason = f"SystemExit: {exc}"
    except Exception as exc:  # noqa: BLE001 - record any failure and continue
        fail_reason = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    elapsed = time.time() - t0
    if not result:
        print(f"[run_one] FAILED after {elapsed:.1f}s: {fail_reason}")

    row = _extract_row(result, meta, a.ref_cost, success, fail_reason)

    # predetermined assembly-center provenance (recorded verbatim; the realized
    # numeric center is in the assembly_center column so the two can be compared
    # across backward/forward variants).
    row["center_source"] = a.center_source

    # extra metrics surfaced by the experimental search (same process as main).
    # The CSV schema is frozen so previously written rows stay aligned; the
    # multi-pass bookkeeping goes into the per-run JSON instead.
    try:
        import sealp.examples.layout.experiments.exp_search as exp_search
        cl = exp_search.LAST_RUN_STATS.get("complete_leaves")
        if cl:
            row["complete_leaves"] = int(cl)
        _merge_json(out_json, {
            "exp_order": a.exp_order,
            "exp_state": a.exp_state,
            "exp_passes_run": int(exp_search.LAST_RUN_STATS.get("passes_run", 1)),
            "exp_prior_converged": bool(exp_search.LAST_RUN_STATS.get(
                "prior_converged", 0)),
        })
    except Exception:
        pass

    # optional staging_aware L3 diagnostics on the final selected layout.
    if a.l3_check == "staging_aware" and success:
        gd = "FULL" if a.l3_grasp_dir else "search"
        print(f"[run_one] running staging_aware L3 check on selected layout "
              f"(grasps={gd}) ...")
        l3 = _staging_aware_l3(passthrough, out_json, a.seed, a.l3_grasp_dir)
        row["l3_staging_aware"] = l3["verdict"]
        row["l3_fail_step"] = l3["fail_step"]
        row["l3_fail_reason"] = l3["fail_reason"]
        print(f"[run_one] staging_aware L3 = {l3['verdict']} "
              f"(step={l3['fail_step'] or '-'}, reason={l3['fail_reason'] or '-'})")

    _append_csv(a.csv, row)
    print(f"[run_one] recorded -> {a.csv} (success={success})")
    # machine-readable markers so a suite runner can chain results (e.g. feed the
    # exact optimum into the beam run's optimality gap) without parsing the CSV.
    cost = result.get("objective_cost") if result else None
    print(f"RUN_ONE_RESULT_JSON={out_json}")
    print(f"RUN_ONE_COST={'' if cost is None else cost}")
    print(f"RUN_ONE_SUCCESS={int(success)}")
    # exit 0 even on experiment failure so a sequential suite keeps going;
    # the failure is captured in the CSV row.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
