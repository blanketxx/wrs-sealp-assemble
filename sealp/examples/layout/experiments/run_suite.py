"""Formal experiment suite driver (P1-P5).

Runs each experiment configuration STRICTLY SEQUENTIALLY, one isolated
subprocess per run (``run_one``), never two WRS processes at once. Every Beam run
uses workers=4; Exact A* is serial. Each run writes its own JSON and appends a
row to the single summary CSV; nothing is overwritten; failed runs are recorded.

This driver DOES NOT run smoke / reproducibility / validation tests. It only
launches the formal P1-P5 runs.

Usage:
    # everything, sequentially
    python -m sealp.examples.layout.experiments.run_suite --priority all

    # a single priority
    python -m sealp.examples.layout.experiments.run_suite --priority p1

    # just print the exact per-run commands without executing
    python -m sealp.examples.layout.experiments.run_suite --priority all --print-only

Output layout:
    sealp/examples/layout/experiments/_output/
        p1/  p2/  p3/  p4/  p5/            # one JSON per run
        experiments_summary.csv            # appended, never overwritten
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(_THIS_DIR, "_output")
CSV_PATH = os.path.join(OUT_ROOT, "experiments_summary.csv")

# ----------------------------------------------------------------------------
# shared run.py passthrough fragments
# ----------------------------------------------------------------------------
ASM_CHAIR = "sealp/assembly_sequence/_demo_output/yuanchair.asmdef"
ASM_TOWER = "sealp/assembly_sequence/_demo_output/topdown_tower.asmdef"

# FULL grasp sets (authoritative; used for the final staging_aware L3 recheck).
GRASP_CHAIR_FULL = "sealp/examples/grasp/yuanchair_grasp"
GRASP_TOWER_FULL = "sealp/examples/grasp/tower_grasp"
# LEAN grasp sets (~400/part) for SEARCH -- ~3x cheaper per certify. Pure input
# data produced by experiments.make_lean_grasps; the algorithm is unchanged.
GRASP_CHAIR = "sealp/examples/grasp/yuanchair_grasp_lean"
GRASP_TOWER = "sealp/examples/grasp/tower_grasp_lean"

CHAIR_PARTS_4LEG = "seat,leg_bl,leg_br,leg_fl,leg_fr"
CHAIR_PARTS_3LEG = "seat,leg_bl,leg_br,leg_fl"   # seat + 3 picked legs (P1 stat / P2)
CHAIR_PARTS_2LEG = "seat,leg_bl,leg_br"          # 2 legs (P3 exact-tractable discrete)
CHAIR_PARTS_1LEG = "seat,leg_bl"                 # 1 leg (scalability curve)
CHAIR_PARTS_REDUCED = CHAIR_PARTS_2LEG
TOWER_PARTS = "base_plate,post_br,post_fr,post_bl,post_fl,middle_plate,top_cross"

# Shared LEAN single-center beam fragment (grid 0.08 / cand=6 / beam=4 /
# retries=3, lean grasps, workers=16). Reused by the reduced-chair statistical
# runs and the one full 4-leg headline run so the ONLY difference between P1/P2
# cells is the experimental variable (search order / state model).
_LEAN_SINGLE = [
    "--mode", "beam", "--center-search", "single", "--goal-pos", "0.373,0.0,0.0",
    "--grid-spacing", "0.08", "--cand-per-part", "6", "--beam-width", "4",
    "--poses-per-xy", "1", "--yaw-step-deg", "20", "--witness-retries", "3",
    "--workers", "16", "--parallel-level", "auto",
]

# P1/P2 reduced statistical instance: seat + 3 picked legs, single fixed center.
CHAIR_3LEG_SINGLE = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
                     "--part-order", CHAIR_PARTS_3LEG] + _LEAN_SINGLE

# P1 headline instance: full 4-leg chair, single fixed center (seed 0 only).
CHAIR_4LEG_SINGLE = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
                     "--part-order", CHAIR_PARTS_4LEG] + _LEAN_SINGLE

# reduced discrete chair instance (P3: exact A* tractable; identical domain for
# exact vs beam). Lean grasps make each certify ~3x cheaper so a fresh P3 (if
# desired) finishes far faster; the already-collected P3 result can be reused.
CHAIR_REDUCED_DISCRETE = [
    "--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
    "--part-order", CHAIR_PARTS_REDUCED, "--goal-pos", "0.373,0.0,0.0",
    "--center-search", "single", "--discrete-domain",
    "--grid-spacing", "0.10", "--exact-cell-cap", "0", "--max-nodes", "500000",
    "--beam-width", "6", "--poses-per-xy", "1", "--yaw-step-deg", "0",
    "--witness-retries", "5", "--workers", "16", "--parallel-level", "auto",
]

# scalability curve (backward vs forward beam over 1..4 legs, single center,
# small lean domain so each point is cheap). Demonstrates how the suffix
# decomposition keeps backward feasible/low-cert as parts grow.
_SCALE_SINGLE = [
    "--mode", "beam", "--center-search", "single", "--goal-pos", "0.373,0.0,0.0",
    "--grid-spacing", "0.10", "--cand-per-part", "5", "--beam-width", "3",
    "--poses-per-xy", "1", "--yaw-step-deg", "20", "--witness-retries", "2",
    "--workers", "16", "--parallel-level", "auto",
]

# cross-assembly: frozen full pipeline (backward beam, coarse-to-fine), lean
# grasps for search; final L3 re-verified against FULL grasps.
_COARSE_TO_FINE = [
    "--mode", "beam", "--center-search", "coarse-to-fine",
    "--coarse-grid-n", "4", "--coarse-top-k", "4",
    "--refine-grid-n", "3", "--refine-spacing-factor", "0.5",
    "--grid-spacing", "0.07", "--cand-per-part", "8", "--beam-width", "5",
    "--poses-per-xy", "1", "--yaw-step-deg", "20", "--witness-retries", "5",
    "--workers", "16", "--parallel-level", "auto",
]
CHAIR_CROSS = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
               "--part-order", CHAIR_PARTS_4LEG, "--goal-pos", "0.373,0.0,0.0"] + _COARSE_TO_FINE
TOWER_CROSS = ["--asmdef", ASM_TOWER, "--grasp-dir", GRASP_TOWER,
               "--part-order", TOWER_PARTS, "--goal-pos", "0.373,0.0,0.0"] + _COARSE_TO_FINE

SEEDS_STAT = [0, 1, 2]   # reduced-instance statistical seeds

# Predetermined assembly center for the single-center P1/P2 runs: run.py's
# `--center-search single` derives the center as clip(goal_pos, feasible_bounds)
# BEFORE any search, so it is fixed and identical for Backward vs Forward (and
# across all state models) on the same instance -- never selected after
# observing a method's result. Recorded verbatim in the CSV `center_source`.
FIXED_CENTER_SRC = ("single@goal_pos=(0.373,0.0,0.0); predetermined "
                    "clip(goal_pos, feasible_bounds) before search; "
                    "identical for all variants of the same instance")


# ----------------------------------------------------------------------------
# config builders -- each returns a list of run specs
# ----------------------------------------------------------------------------
def _spec(priority, variant, task, seed, passthrough,
          order=None, state=None, hall="on", prop="on", mode=None,
          needs_ref=False, l3_check=False, center_source="",
          l3_grasp_dir="") -> Dict:
    pt = list(passthrough)
    if mode is not None:
        pt = _override_mode(pt, mode)
    return {
        "priority": priority, "variant": variant, "task": task, "seed": seed,
        "order": order, "state": state, "hall": hall, "prop": prop,
        "passthrough": pt, "needs_ref": needs_ref, "l3_check": l3_check,
        "center_source": center_source, "l3_grasp_dir": l3_grasp_dir,
    }


def _override_mode(passthrough: List[str], mode: str) -> List[str]:
    out = list(passthrough)
    if "--mode" in out:
        i = out.index("--mode")
        out[i + 1] = mode
    else:
        out += ["--mode", mode]
    return out


def build_p1() -> List[Dict]:
    # Statistical comparison on the reduced chair (seat + 3 legs), fixed center,
    # seeds 0/1/2, ONLY the search order varies. Plus ONE full 4-leg headline
    # case at seed 0 (backward vs forward).
    specs = []
    for seed in SEEDS_STAT:
        for order in ("backward", "forward"):
            specs.append(_spec("p1", order, "chair_3leg", seed, CHAIR_3LEG_SINGLE,
                               order=order, state="sequential",
                               center_source=FIXED_CENTER_SRC))
    for order in ("backward", "forward"):
        specs.append(_spec("p1", f"headline_{order}", "chair_4leg", 0, CHAIR_4LEG_SINGLE,
                           order=order, state="sequential",
                           center_source=FIXED_CENTER_SRC))
    return specs


def build_p2() -> List[Dict]:
    # Sequence-state comparison on the reduced chair (seat + 3 legs), seed 0,
    # Backward order fixed -- only the state/obstacle model varies. This TESTS
    # whether the three dynamic-state phenomena arise (growing partial assembly
    # blocks a later op; removed staging parts release space; remaining staged
    # parts block an earlier op); it does NOT presume they do. Which phenomena
    # actually occur is decided by inspecting the results. Each run records the
    # predicted feasibility (success) AND the final staging_aware L3 diagnostics
    # (verdict / fail step / fail reason), the latter re-verified against the FULL
    # grasp set.
    specs = []
    for state in ("sequential", "static_start", "static_final"):
        specs.append(_spec("p2", state, "chair_3leg", 0, CHAIR_3LEG_SINGLE,
                           order="backward", state=state, l3_check=True,
                           center_source=FIXED_CENTER_SRC,
                           l3_grasp_dir=GRASP_CHAIR_FULL))
    return specs


def build_p3() -> List[Dict]:
    # OPTIONAL fresh instance -- the primary P3 result is already collected and
    # should be REUSED (do not rerun the 10h version). This lean/coarse config
    # finishes far faster if a fresh exact-vs-beam point is wanted. exact first
    # (optimum), then beam with ref-cost = exact optimum (optimality gap).
    return [
        _spec("p3", "exact", "chair_reduced", 0, CHAIR_REDUCED_DISCRETE, mode="exact"),
        _spec("p3", "beam", "chair_reduced", 0, CHAIR_REDUCED_DISCRETE, mode="beam",
              needs_ref=True),
    ]


def build_p4() -> List[Dict]:
    # Full Chair + Tower (frozen backward beam, coarse-to-fine), seed 0, lean
    # grasps for search. The staging_aware full-sequence L3 is validated
    # separately on each final layout against the FULL grasp set.
    return [
        _spec("p4", "chair", "chair", 0, CHAIR_CROSS, l3_check=True,
              l3_grasp_dir=GRASP_CHAIR_FULL),
        _spec("p4", "tower", "tower", 0, TOWER_CROSS, l3_check=True,
              l3_grasp_dir=GRASP_TOWER_FULL),
    ]


def build_scale() -> List[Dict]:
    # Scalability curve: backward vs forward beam over 1..4 legs (single center,
    # small lean domain, seed 0). Records certifications / node_expansions /
    # complete_leaves / time-to-first-feasible / success / cost so a single figure
    # shows how the suffix decomposition keeps backward feasible with low
    # certification count as parts grow, while forward degrades.
    sizes = [("1leg", CHAIR_PARTS_1LEG), ("2leg", CHAIR_PARTS_2LEG),
             ("3leg", CHAIR_PARTS_3LEG), ("4leg", CHAIR_PARTS_4LEG)]
    specs = []
    for tag, parts in sizes:
        pt = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
              "--part-order", parts] + _SCALE_SINGLE
        for order in ("backward", "forward"):
            specs.append(_spec("scale", order, f"chair_{tag}", 0, pt,
                               order=order, state="sequential",
                               center_source=FIXED_CENTER_SRC))
    return specs


def build_p5() -> List[Dict]:
    # Hall / propagation are only active in the EXACT A* solver. NOTE: on the
    # reduced 2-leg instance these prunes typically never fire (hall/prop=0), so
    # this ablation is UNINFORMATIVE there; kept selectable but excluded from the
    # focused suite. A meaningful ablation needs a larger discrete instance where
    # exact remains tractable.
    return [
        _spec("p5", "full", "chair_reduced", 0, CHAIR_REDUCED_DISCRETE, mode="exact",
              hall="on", prop="on"),
        _spec("p5", "no_hall", "chair_reduced", 0, CHAIR_REDUCED_DISCRETE, mode="exact",
              hall="off", prop="on"),
        _spec("p5", "no_prop", "chair_reduced", 0, CHAIR_REDUCED_DISCRETE, mode="exact",
              hall="on", prop="off"),
        _spec("p5", "no_both", "chair_reduced", 0, CHAIR_REDUCED_DISCRETE, mode="exact",
              hall="off", prop="off"),
    ]


BUILDERS = {"p1": build_p1, "p2": build_p2, "p3": build_p3,
            "p4": build_p4, "p5": build_p5, "scale": build_scale}
# Focused suite for the deadline: P1 (core), P2 (state model), scale (scalability
# curve), P4 (cross-assembly). P3 is reused from the prior run; P5 is dropped
# (uninformative on the tractable instance). Run p3/p5 explicitly if desired.
FOCUSED = ["p1", "p2", "scale", "p4"]


# ----------------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------------
def _run_one_cmd(spec: Dict, ref_cost: Optional[float]) -> List[str]:
    out_dir = os.path.join(OUT_ROOT, spec["priority"])
    cmd = [sys.executable, "-X", "utf8", "-m",
           "sealp.examples.layout.experiments.run_one",
           "--priority", spec["priority"], "--variant", spec["variant"],
           "--task", spec["task"], "--seed", str(spec["seed"]),
           "--out-dir", out_dir, "--csv", CSV_PATH]
    if spec.get("order"):
        cmd += ["--exp-order", spec["order"]]
    if spec.get("state"):
        cmd += ["--exp-state", spec["state"]]
    if spec.get("hall") == "off":
        cmd += ["--exp-hall", "off"]
    if spec.get("prop") == "off":
        cmd += ["--exp-prop", "off"]
    if ref_cost is not None:
        cmd += ["--ref-cost", str(ref_cost)]
    if spec.get("l3_check"):
        cmd += ["--l3-check", "staging_aware"]
    if spec.get("l3_grasp_dir"):
        cmd += ["--l3-grasp-dir", spec["l3_grasp_dir"]]
    if spec.get("center_source"):
        cmd += ["--center-source", spec["center_source"]]
    cmd += ["--"] + list(spec["passthrough"])
    return cmd


def _execute(specs: List[Dict], print_only: bool) -> None:
    os.makedirs(OUT_ROOT, exist_ok=True)
    last_cost: Optional[float] = None
    for i, spec in enumerate(specs, 1):
        ref = last_cost if spec.get("needs_ref") else None
        cmd = _run_one_cmd(spec, ref)
        header = f"[{i}/{len(specs)}] {spec['priority']} {spec['variant']} " \
                 f"{spec['task']} seed={spec['seed']}"
        if print_only:
            print(f"# {header}")
            print(" ".join(_quote(c) for c in cmd))
            print()
            continue
        print("\n" + "#" * 72)
        print(header)
        print("#" * 72)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        # Stream stdout/stderr live so long runs show [1/9]..[9/9] progress.
        proc = subprocess.run(cmd, env=env)
        last_cost = _parse_cost_from_log(spec)


def _parse_cost(stdout: str) -> Optional[float]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RUN_ONE_COST="):
            val = line.split("=", 1)[1].strip()
            try:
                return float(val)
            except ValueError:
                return None
    return None


def _parse_cost_from_log(spec: Dict) -> Optional[float]:
    """Read RUN_ONE_COST from the JSON written by run_one (P3 beam ref chain)."""
    out_dir = os.path.join(OUT_ROOT, spec["priority"])
    run_id = f"{spec['priority']}_{spec['variant']}_{spec['task']}_seed{spec['seed']}"
    path = os.path.join(out_dir, f"{run_id}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("objective_cost")
    except Exception:
        return None


def _quote(s: str) -> str:
    return f'"{s}"' if (" " in s or "," in s) else s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", required=True,
                    choices=["p1", "p2", "p3", "p4", "p5", "scale", "focused", "all"],
                    help="focused/all = the deadline suite (p1,p2,scale,p4); p3 is "
                         "reused from the prior run and p5 is uninformative here, so "
                         "both are excluded from the suite but selectable explicitly.")
    ap.add_argument("--print-only", action="store_true",
                    help="print the exact per-run commands, do not execute.")
    a = ap.parse_args(argv)

    if a.priority in ("all", "focused"):
        specs: List[Dict] = []
        for key in FOCUSED:
            specs += BUILDERS[key]()
    else:
        specs = BUILDERS[a.priority]()

    print(f"Suite: {a.priority}  ({len(specs)} runs)  "
          f"{'PRINT-ONLY' if a.print_only else 'SEQUENTIAL EXECUTION'}")
    print(f"Output root : {OUT_ROOT}")
    print(f"Summary CSV : {CSV_PATH}")
    _execute(specs, a.print_only)
    if not a.print_only:
        print(f"\nDONE. Summary CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
