"""Read-only staging_aware L3 validation of a SAVED BSFS layout JSON.

This harness does NOT run BSFS / beam / A* search. It loads a completed layout
result (e.g. ``bsfs_4leg_gate.json``), reconstructs the exact assignment
(assembly center + per-part staging xy + stable-pose rot_name) that BSFS already
selected, replays the deterministic L2 witness to rebuild the LayoutCandidate
(arm/grasp choices), and then runs the strict full-sequence L3 witness under the
project-recommended ``staging_aware`` obstacle mode, printing each assembly
step's verdict and the final full-sequence verdict.

Usage:
    python -m sealp.examples.layout.validate_saved_layout_l3 \
        --result-json bsfs_4leg_gate.json \
        --asmdef sealp/assembly_sequence/_demo_output/yuanchair.asmdef \
        --grasp-dir sealp/examples/grasp/yuanchair_grasp \
        --goal-pos 0.373,0.0,0.0 \
        --part-order seat,leg_bl,leg_br,leg_fl,leg_fr
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from sealp.examples.layout.bsfs.run import (
    parse_args, _build_searcher, robust_evaluate_layout,
)


def _fresh_stats() -> dict:
    return {
        "certifications": 0, "max_depth_certified": 0, "node_expansions": 0,
        "hard_prunes": 0, "hall_pruned": 0, "propagation_prunes": 0,
        "full_witness_calls": 0, "witness_attempts_total": 0,
        "sites_tried": 0, "sites_feasible": 0,
    }


def _write_back(path: str, obstacle_mode: str, ok: bool, cand, grasp_dir: str) -> None:
    """Record this verdict in the result JSON under its own key.

    The existing ``l3_pass`` field is left untouched: it is the in-run mesh-mode
    L3 flag, a deliberately over-strict check the pipeline does not use as a
    selection filter, so it reads False even for layouts that pass the
    staging_aware full-sequence witness. Overwriting it would destroy that
    distinction; a separate key with explicit provenance keeps both verdicts
    readable and makes clear which one was measured how.
    """
    import datetime as _dt

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["l3_staging_aware"] = {
        "verdict": "PASS" if ok else "FAIL",
        "obstacle_mode": obstacle_mode,
        "grasp_dir": grasp_dir,
        "fail_reason": "" if ok else str(getattr(cand, "l3_fail_reason", "")),
        "measured_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "measured_by": "sealp.examples.layout.validate_saved_layout_l3",
        "note": ("Full-sequence L3 replay of the saved layout; no search was "
                 "rerun. Distinct from the 'l3_pass' field, which is the "
                 "over-strict in-run mesh-mode flag and is not a selection "
                 "criterion in this pipeline."),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"[write-back] l3_staging_aware = {data['l3_staging_aware']['verdict']} "
          f"-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--asmdef", required=True)
    ap.add_argument("--grasp-dir", required=True)
    ap.add_argument("--goal-pos", required=True)
    ap.add_argument("--part-order", required=True)
    ap.add_argument("--obstacle-mode", default="staging_aware")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--write-back", action="store_true",
                    help="record the verdict in the result JSON under the "
                         "'l3_staging_aware' key (leaves the unrelated in-run "
                         "'l3_pass' flag untouched).")
    cli = ap.parse_args()

    with open(cli.result_json, "r", encoding="utf-8") as fh:
        result = json.load(fh)

    center = np.asarray(result["assembly_center"], dtype=float)
    best_layout = result["best_layout"]
    preassembled = result.get("preassembled_part")

    # Build the SAME searcher configuration used to produce the result; only the
    # loading/geometry-relevant flags matter because we replay a FIXED assign.
    args = parse_args([
        "--asmdef", cli.asmdef,
        "--grasp-dir", cli.grasp_dir,
        "--part-order", cli.part_order,
        "--goal-pos", cli.goal_pos,
        "--mode", "beam", "--seed", str(cli.seed), "--workers", "1",
        "--witness-retries", "5",
    ])
    searcher = _build_searcher(args)

    # Reconstruct the assign dict (picked parts only, exclude preassembled seat).
    assign = {}
    for pid in searcher.part_order:
        if pid == preassembled:
            continue
        if pid not in best_layout:
            continue
        e = best_layout[pid]
        assign[pid] = {
            "xy": [float(e["init_pos"][0]), float(e["init_pos"][1])],
            "rot_name": str(e["rot_name"]),
            "cost": 0.0,
        }

    print("=" * 70)
    print("READ-ONLY staging_aware L3 validation of saved layout")
    print("=" * 70)
    print(f"result-json    : {cli.result_json}")
    print(f"reported L2    : {result.get('witness_status')}  "
          f"cost={result.get('objective_cost')}")
    print(f"assembly center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"preassembled   : {preassembled}")
    print("picked parts (pick order = reverse assembly):")
    for pid in searcher._active_pick_part_order():
        if pid in assign:
            a = assign[pid]
            print(f"    {pid:10s} xy=({a['xy'][0]:.4f},{a['xy'][1]:.4f}) "
                  f"rot={a['rot_name']}")

    stats = _fresh_stats()
    searcher._set_assembly_station(center)
    part_order = list(searcher.part_order)

    print("\n--- replay deterministic L2 witness (rebuild LayoutCandidate) ---")
    status, cand, info = robust_evaluate_layout(
        searcher, assign, preassembled, center, args, part_order, stats)
    print(f"  [L2 witness] status={status}  "
          f"successful_retry={info.get('successful_retry_index')}")
    if status != "SUCCESS":
        print(f"  [L2 witness] FAIL part={getattr(cand, 'fail_part', None)} "
              f"reason={getattr(cand, 'fail_reason', '')}")
        print("\nFINAL L3 VERDICT: N/A (L2 reconstruction failed)")
        return

    ok = bool(searcher.validate_full_sequence_l3(
        cand, obstacle_mode=cli.obstacle_mode, verbose=True))

    print("\n" + "=" * 70)
    if ok:
        print(f"FINAL L3 VERDICT: PASS  (obstacle_mode={cli.obstacle_mode})")
    else:
        print(f"FINAL L3 VERDICT: FAIL  (obstacle_mode={cli.obstacle_mode})")
        print(f"  l3_fail_reason: {getattr(cand, 'l3_fail_reason', '')}")
    print("=" * 70)

    if cli.write_back:
        _write_back(cli.result_json, cli.obstacle_mode, ok, cand, cli.grasp_dir)


if __name__ == "__main__":
    main()
