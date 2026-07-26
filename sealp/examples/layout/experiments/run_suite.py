"""Formal experiment suite driver (P1-P5).

Runs each experiment configuration STRICTLY SEQUENTIALLY, one isolated
subprocess per run (``run_one``), never two WRS processes at once. Every Beam run
uses workers=16; Exact A* is serial. Each run writes its own JSON and appends a
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
        p1/ p1b/ p1c/ p2/ p3/ p4/ p5/ scale/   # one JSON per run
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

# Generated 60x60 cm benchmark products (P4x). Unlike the chair/tower there is no
# lean/full grasp split: one set per distinct MESH serves both search and the L3
# recheck, so parts sharing a mesh share a grasp set (four post_* on one post.stl).
# Each asmdef ships a `<name>_contacts.json` sidecar listing the non-parent mates
# per step, which the searcher folds into its contact-exclusion table.
ASM_CRF = "sealp/assembly_sequence/_demo_output/cross_rail_frame_v1.asmdef"
ASM_CUBE = "sealp/assembly_sequence/_demo_output/stack_cube_v1.asmdef"
ASM_SPIRE = "sealp/assembly_sequence/_demo_output/buttressed_spire_tower_v1.asmdef"
GRASP_CRF = "sealp/examples/grasp/cross_rail_frame_grasp"
GRASP_CUBE = "sealp/examples/grasp/stack_cube_grasp"
GRASP_SPIRE = "sealp/examples/grasp/spire_tower_grasp"
CRF_PARTS = "base_plate,post_l,post_r,rail_l,rail_r,cap_plate"
CUBE_PARTS = ("base_plate,post_bl,post_br,post_fl,post_fr,mid_plate,"
              "clip_nx,clip_px,clip_ny,clip_py")
SPIRE_PARTS = ("cruciform_base,stepped_core,buttress_w,buttress_e,buttress_s,"
               "buttress_n,wing_w,wing_e,crown,spire")

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

# Matched-budget variant of _LEAN_SINGLE for the 4-leg headline (P1b). The
# candidate DOMAIN is unchanged (same grid 0.08) -- only the search BUDGET grows
# (cand-per-part 6->8, beam-width 4->6). Rationale: at beam=4 the backward order
# certifies against the physically-correct cluttered scene, so more candidates
# are rejected and the beam can be exhausted, while forward's optimistic scene
# (undecided suffix parked far away) keeps candidates alive. That is a budget
# artifact, not an algorithmic difference, so both orders are re-run at the
# wider budget. Yaw refinement is OFF: it is a post-search step applied to the
# already-selected layout, the objective is yaw-invariant (see run.py
# --yaw-step-deg help), and it consumed ~65% of wall-clock in earlier runs.
_LEAN_SINGLE_WIDE = [
    "--mode", "beam", "--center-search", "single", "--goal-pos", "0.373,0.0,0.0",
    "--grid-spacing", "0.08", "--cand-per-part", "8", "--beam-width", "6",
    "--poses-per-xy", "1", "--yaw-step-deg", "0", "--witness-retries", "3",
    "--workers", "16", "--parallel-level", "auto",
]

# P1/P2 reduced statistical instance: seat + 3 picked legs, single fixed center.
CHAIR_3LEG_SINGLE = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
                     "--part-order", CHAIR_PARTS_3LEG] + _LEAN_SINGLE

# P1 headline instance: full 4-leg chair, single fixed center (seed 0 only).
CHAIR_4LEG_SINGLE = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
                     "--part-order", CHAIR_PARTS_4LEG] + _LEAN_SINGLE

# P1b matched-budget headline instance: same 4-leg chair and same domain, wider
# beam/candidate budget, yaw refinement disabled.
CHAIR_4LEG_SINGLE_WIDE = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
                          "--part-order", CHAIR_PARTS_4LEG] + _LEAN_SINGLE_WIDE

# P1d: identical to CHAIR_4LEG_SINGLE_WIDE except yaw refinement is ON. Yaw is a
# post-search step on the already-selected layout and does not affect the
# objective, but it does decide the final heading each part is placed at, which
# in turn decides whether the full L3 motion plan exists -- the yaw-off layouts
# of P1b both fail L3. Use this preset for any executability claim.
CHAIR_4LEG_SINGLE_WIDE_YAW = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
                              "--part-order", CHAIR_PARTS_4LEG] + \
                             [("20" if p == "0" and
                               _LEAN_SINGLE_WIDE[i - 1] == "--yaw-step-deg" else p)
                              for i, p in enumerate(_LEAN_SINGLE_WIDE)]

# P1c statistical instance: the same reduced 3-leg chair and the same budget as
# CHAIR_3LEG_SINGLE, with yaw refinement disabled. The iterated forward baseline
# runs the beam several times, and yaw refinement -- a post-search step on the
# already-chosen layout, irrelevant to search order -- dominated wall-clock at
# ~4500 s/run. All three orders in P1c share this preset, so the comparison stays
# internally consistent; it is NOT comparable to the yaw-on P1 rows.
CHAIR_3LEG_SINGLE_NOYAW = ["--asmdef", ASM_CHAIR, "--grasp-dir", GRASP_CHAIR,
                           "--part-order", CHAIR_PARTS_3LEG] + \
                          [("0" if p == "20" and _LEAN_SINGLE[i - 1] == "--yaw-step-deg"
                            else p) for i, p in enumerate(_LEAN_SINGLE)]

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

# Order comparison on the CROWDED instance. The chair occupies ~1.7% of the
# staging area, so a forward pass that omits its undecided suffix is barely
# penalised and every order-sensitive metric saturates. The tower stages six
# large plates/posts instead of four thin legs, which is where omitting the
# suffix should actually cost something. Single fixed center and yaw off keep it
# affordable; the budget matches _LEAN_SINGLE so the only variable stays the
# search order.
TOWER_ORDER_SINGLE = [
    "--asmdef", ASM_TOWER, "--grasp-dir", GRASP_TOWER,
    "--part-order", TOWER_PARTS,
    "--mode", "beam", "--center-search", "single", "--goal-pos", "0.373,0.0,0.0",
    "--grid-spacing", "0.08", "--cand-per-part", "6", "--beam-width", "4",
    "--poses-per-xy", "1", "--yaw-step-deg", "0", "--witness-retries", "3",
    "--workers", "16", "--parallel-level", "auto",
]

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
          l3_grasp_dir="", l3_only="", forward_passes=None) -> Dict:
    pt = list(passthrough)
    if mode is not None:
        pt = _override_mode(pt, mode)
    return {
        "priority": priority, "variant": variant, "task": task, "seed": seed,
        "order": order, "state": state, "hall": hall, "prop": prop,
        "passthrough": pt, "needs_ref": needs_ref, "l3_check": l3_check,
        "center_source": center_source, "l3_grasp_dir": l3_grasp_dir,
        "l3_only": l3_only, "forward_passes": forward_passes,
    }


def _override_mode(passthrough: List[str], mode: str) -> List[str]:
    out = list(passthrough)
    if "--mode" in out:
        i = out.index("--mode")
        out[i + 1] = mode
    else:
        out += ["--mode", mode]
    return out


def _override_workers(passthrough: List[str], workers: int) -> List[str]:
    """Rewrite --workers in a passthrough (host-tuning only).

    Worker count changes wall-clock ONLY; oracle_certifications, node_expansions,
    complete_leaves, cost and layout_fingerprint are worker-invariant. Keep it
    identical across the cells of one comparison, and do not compare runtime
    across suites collected with different worker counts / on different hosts.
    """
    out = list(passthrough)
    if "--workers" in out:
        out[out.index("--workers") + 1] = str(workers)
    else:
        out += ["--workers", str(workers)]
    return out


def build_p1() -> List[Dict]:
    # Statistical comparison on the reduced chair (seat + 3 legs), fixed center,
    # seeds 0/1/2, ONLY the search order varies. Plus ONE full 4-leg headline
    # case at seed 0 (backward vs forward).
    #
    # LEFT UNCHANGED so the already-collected P1 results stay bit-reproducible.
    # The 4-leg headline here runs at beam=4/cand=6, where backward exhausts the
    # beam and fails; that run is kept on record as a documented budget-starvation
    # case. The matched-budget re-run lives in build_p1_matched (--priority p1b).
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


def build_p1_matched() -> List[Dict]:
    # P1b: the full 4-leg headline re-run at a MATCHED, wider search budget so
    # neither order is starved (see _LEAN_SINGLE_WIDE). Everything else -- domain,
    # oracle, cost model, grasp set, assembly center, seed, workers -- is
    # identical between the two cells; the search order remains the only variable.
    # Each run additionally reports the staging_aware full-sequence L3 verdict
    # against the FULL grasp set, so the layout each order commits to is checked
    # for soundness under the physically-correct sequential state, not just for
    # search cost.
    return [
        _spec("p1b", f"matched_{order}", "chair_4leg", 0, CHAIR_4LEG_SINGLE_WIDE,
              order=order, state="sequential", l3_check=True,
              center_source=FIXED_CENTER_SRC, l3_grasp_dir=GRASP_CHAIR_FULL)
        for order in ("backward", "forward")
    ]


def build_p1_dynamic() -> List[Dict]:
    # P1c: the comparison at EQUAL fidelity. P1/P1b let forward certify against an
    # emptier scene -- the suffix parts it has not decided yet are simply absent --
    # so its search cost is not comparable to backward's. "forward_iter" removes
    # that discount: after its first pass it re-certifies every step against the
    # suffix staging poses that pass chose, and keeps iterating until the layout
    # stops changing. All passes are billed to one certification counter.
    #
    # Both orders now certify against an occupied staging area, so cost is
    # comparable and, when the costs tie, so is search time. Backward needs one
    # pass by construction; whatever forward spends beyond that is the price of
    # not knowing its suffix. Budget, domain, oracle, grasps, center, seed and
    # workers are identical to P1b, so the P1b backward row is directly reusable
    # (it is re-run here anyway, which doubles as a determinism check).
    # 3-leg cells, all three orders at one shared preset (CHAIR_3LEG_SINGLE_NOYAW).
    # This is where P1 reported forward as "24% cheaper"; that gap was beam
    # occupancy under the relaxed scene, so it is re-measured here with forward
    # paying for its own suffix estimate.
    #
    # SEED 0 ONLY, deliberately. P1 already ran this instance at seeds 0/1/2 and
    # every cell came back bit-identical (backward 2.5380 / 336 certs / 9
    # expansions; forward 2.5380 / 256 / 7; same fingerprint at all three seeds).
    # The pipeline is deterministic given a config -- the seed only orders witness
    # retries -- so extra seeds add runtime and zero variance information. Cite the
    # P1 rows as the determinism evidence instead of re-collecting it.
    specs = [
        _spec("p1c", f"eqfid_{order}", "chair_3leg", 0, CHAIR_3LEG_SINGLE_NOYAW,
              order=order, state="sequential", center_source=FIXED_CENTER_SRC,
              forward_passes=3 if order == "forward_iter" else None)
        for order in ("backward", "forward", "forward_iter")
    ]
    # 4-leg headline at the matched wide budget, with the staging_aware L3 verdict
    # on whatever layout each order commits to.
    specs += [
        _spec("p1c", f"eqfid_{order}", "chair_4leg", 0, CHAIR_4LEG_SINGLE_WIDE,
              order=order, state="sequential", l3_check=True,
              center_source=FIXED_CENTER_SRC, l3_grasp_dir=GRASP_CHAIR_FULL,
              forward_passes=3 if order == "forward_iter" else None)
        for order in ("backward", "forward", "forward_iter")
    ]
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


def build_p4_l3() -> List[Dict]:
    # Replay ONLY the staging_aware L3 on the layouts P4 has already selected,
    # reusing each run's original passthrough so the searcher is rebuilt
    # identically. No search is repeated. Sources that do not exist yet are
    # skipped, so this is safe to run before the whole of P4 has finished.
    specs = []
    for variant, task, pt, grasps in (
            ("chair", "chair", CHAIR_CROSS, GRASP_CHAIR_FULL),
            ("tower", "tower", TOWER_CROSS, GRASP_TOWER_FULL)):
        src = os.path.join(OUT_ROOT, "p4", f"p4_{variant}_{task}_seed0.json")
        if not os.path.isfile(src):
            print(f"[p4l3] skip {variant}: no source result at {src}")
            continue
        specs.append(_spec("p4l3", variant, task, 0, pt, l3_check=True,
                           l3_grasp_dir=grasps, l3_only=src))
    return specs


#: P4x cells: (variant, asmdef, grasp dir, part order). Kept as data so the
#: search runs, the L3-only replay and the runtime estimate all read one list.
P4X_CELLS = (
    ("cross_rail_frame", ASM_CRF, GRASP_CRF, CRF_PARTS),
    ("stack_cube", ASM_CUBE, GRASP_CUBE, CUBE_PARTS),
    ("spire_tower", ASM_SPIRE, GRASP_SPIRE, SPIRE_PARTS),
)


# P4x budget. The claim being tested is only "the frozen backward beam returns a
# layout that passes L2 AND the staging_aware full-sequence L3 on these products",
# so it uses ONE predetermined center instead of P4's coarse-to-fine sweep -- the
# center search costs most of P4's wall-clock and is irrelevant to a feasibility
# claim. Yaw refinement stays ON: it is what decides each part's final heading,
# and P1b showed yaw-OFF layouts failing L3 at a mid-sequence step, so a yaw-off
# run could not support an executability claim.
_P4X_SINGLE = [
    "--mode", "beam", "--center-search", "single",
    "--goal-pos", "0.373,0.0,0.0",
    "--grid-spacing", "0.08", "--cand-per-part", "6", "--beam-width", "4",
    "--poses-per-xy", "1", "--yaw-step-deg", "20", "--witness-retries", "5",
    "--workers", "12", "--parallel-level", "auto",
]


def _p4x_passthrough(asmdef: str, grasp_dir: str, parts: str) -> List[str]:
    return ["--asmdef", asmdef, "--grasp-dir", grasp_dir,
            "--part-order", parts] + _P4X_SINGLE


def build_p4x() -> List[Dict]:
    # Cross-assembly feasibility on the generated 60x60 products: the frozen
    # backward beam, sequential state, seed 0, then the staging_aware
    # full-sequence L3 on whatever layout the search commits to. P4 itself is
    # untouched so its chair/tower rows stay reproducible.
    #
    # These products are harder staging instances than the chair or tower: the
    # chair stages four thin legs and the tower six parts, whereas stack_cube and
    # spire_tower stage nine each, and both mate along five distinct directions
    # instead of top-down only. Ordered cheapest first (6, then 10, then 10).
    return [
        _spec("p4x", variant, variant, 0,
              _p4x_passthrough(asmdef, grasps, parts),
              order="backward", state="sequential", l3_check=True,
              l3_grasp_dir=grasps)
        for variant, asmdef, grasps, parts in P4X_CELLS
    ]


def build_p4x_l3() -> List[Dict]:
    # Replay ONLY the staging_aware L3 on the layouts P4x already selected,
    # reusing each run's original passthrough so the searcher is rebuilt
    # identically. No search is repeated; missing sources are skipped.
    specs = []
    for variant, asmdef, grasps, parts in P4X_CELLS:
        src = os.path.join(OUT_ROOT, "p4x", f"p4x_{variant}_{variant}_seed0.json")
        if not os.path.isfile(src):
            print(f"[p4xl3] skip {variant}: no source result at {src}")
            continue
        specs.append(_spec("p4xl3", variant, variant, 0,
                           _p4x_passthrough(asmdef, grasps, parts),
                           l3_check=True, l3_grasp_dir=grasps, l3_only=src))
    return specs


def build_p1_tower() -> List[Dict]:
    # P1t: the order comparison on the crowded tower. Everything the chair runs
    # measure is reported here too, but this is the instance where
    # optimistic_certifications and unsound_steps can actually diverge, because
    # six large staged parts leave little slack for a forward pass that pretends
    # its undecided suffix is not there. Three orders, seed 0.
    return [
        _spec("p1t", f"tower_{order}", "tower", 0, TOWER_ORDER_SINGLE,
              order=order, state="sequential", center_source=FIXED_CENTER_SRC,
              forward_passes=3 if order == "forward_iter" else None)
        for order in ("backward", "forward", "forward_iter")
    ]


def build_p1_l3() -> List[Dict]:
    # Replay ONLY the staging_aware L3 (against the FULL grasp set) on the yaw-ON
    # 4-leg layout P1 already selected. P1 ran no L3 at all, so forward's yaw-on
    # layout has never been checked for executability -- the one backward yaw-on
    # layout known to pass L3 (bsfs_4leg_gate.json) came from a different,
    # pre-experiment configuration and is NOT a matched counterpart. No search is
    # repeated. P1's backward cell has no layout to replay: it exhausted the beam
    # at that budget, which is why only forward appears here.
    specs = []
    src = os.path.join(OUT_ROOT, "p1",
                       "p1_headline_forward_chair_4leg_seed0.json")
    if os.path.isfile(src):
        specs.append(_spec("p1l3", "headline_forward_yawon", "chair_4leg", 0,
                           CHAIR_4LEG_SINGLE, l3_check=True,
                           l3_grasp_dir=GRASP_CHAIR_FULL, l3_only=src,
                           center_source=FIXED_CENTER_SRC))
    else:
        print(f"[p1l3] skip: no source result at {src}")
    return specs


def build_p1_yawon() -> List[Dict]:
    # P1d: a YAW ABLATION, not an order comparison. P1b runs the matched wide
    # budget with yaw refinement OFF and its layout fails staging_aware L3 at step
    # 3, so P1b alone cannot exhibit a deployable layout. This run is P1b with the
    # single change --yaw-step-deg 0 -> 20, closing that loop: same search, same
    # budget, yaw on, staging_aware L3 against the full grasp set.
    #
    # ONE cell (backward) on purpose. P1b's two orders returned the SAME
    # layout_fingerprint, i.e. yaw refinement would receive a byte-identical
    # assignment in both cells; yaw refinement is a deterministic post-search step
    # (BLAS pinned to one thread, near-ties broken by smallest yaw angle), so a
    # forward cell here is provably identical to this one. Yaw dominates wall-clock
    # (~97%: 6955 s vs 164 s on this instance), so measuring a provable tie would
    # cost ~2 h for no information. If a reviewer asks for the forward cell, add
    # "forward" back to the tuple below -- the preset is already shared.
    return [
        _spec("p1d", f"yawon_{order}", "chair_4leg", 0, CHAIR_4LEG_SINGLE_WIDE_YAW,
              order=order, state="sequential", l3_check=True,
              center_source=FIXED_CENTER_SRC, l3_grasp_dir=GRASP_CHAIR_FULL)
        for order in ("backward",)
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


BUILDERS = {"p1": build_p1, "p1b": build_p1_matched, "p1c": build_p1_dynamic,
            "p1d": build_p1_yawon, "p1l3": build_p1_l3, "p1t": build_p1_tower,
            "p2": build_p2, "p3": build_p3, "p4": build_p4, "p4l3": build_p4_l3,
            "p4x": build_p4x, "p4xl3": build_p4x_l3,
            "p5": build_p5, "scale": build_scale}
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
    if spec.get("forward_passes"):
        cmd += ["--exp-forward-passes", str(spec["forward_passes"])]
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
    if spec.get("l3_only"):
        cmd += ["--l3-only", spec["l3_only"]]
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
                    choices=["p1", "p1b", "p1c", "p1d", "p1l3", "p1t", "p2", "p3",
                             "p4", "p4l3", "p4x", "p4xl3", "p5", "scale",
                             "focused", "all"],
                    help="focused/all = the deadline suite (p1,p2,scale,p4); p1b is "
                         "the matched-budget 4-leg headline re-run; p1c is the "
                         "equal-fidelity re-run where forward also certifies against "
                         "an occupied staging area; p1t is that same comparison on "
                         "the CROWDED tower, where the order-sensitive metrics can "
                         "actually diverge; p1d is the yaw ablation that turns P1b's "
                         "L3-failing layout into a deployable one; p1l3 replays only "
                         "the L3 check on the "
                         "yaw-on layout P1 already found; p4l3 replays only the "
                         "staging_aware L3 on layouts P4 already found (no search); "
                         "p4x is the P4 recipe on the generated 60x60 products "
                         "(cross_rail_frame / stack_cube / spire_tower), p4xl3 "
                         "replays only their L3; "
                         "p3 is reused from the prior run and p5 is uninformative "
                         "here. p1b/p1c/p4l3/p4x/p4xl3/p3/p5 are excluded from the "
                         "suite but selectable explicitly.")
    ap.add_argument("--print-only", action="store_true",
                    help="print the exact per-run commands, do not execute.")
    ap.add_argument("--workers", type=int, default=None,
                    help="override the worker count of every run (host tuning). "
                         "Each worker is a spawned process that loads WRS + meshes "
                         "+ grasps independently, so RAM -- not core count -- is "
                         "usually the binding constraint. Affects wall-clock only.")
    a = ap.parse_args(argv)

    if a.priority in ("all", "focused"):
        specs: List[Dict] = []
        for key in FOCUSED:
            specs += BUILDERS[key]()
    else:
        specs = BUILDERS[a.priority]()

    if a.workers is not None:
        for s in specs:
            s["passthrough"] = _override_workers(s["passthrough"], a.workers)

    print(f"Suite: {a.priority}  ({len(specs)} runs)  "
          f"{'PRINT-ONLY' if a.print_only else 'SEQUENTIAL EXECUTION'}")
    if a.workers is not None:
        print(f"Workers override: {a.workers} (wall-clock only; certification "
              f"counts and cost are worker-invariant)")
    print(f"Output root : {OUT_ROOT}")
    print(f"Summary CSV : {CSV_PATH}")
    _execute(specs, a.print_only)
    if not a.print_only:
        print(f"\nDONE. Summary CSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
