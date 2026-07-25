#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ModularFixtureStackV1 -- six-part low-profile stacked fixture benchmark.

The assembly is intentionally NOT tower-like.  It resembles a modular
industrial fixture built by stacking broad, shallow modules:

    1) base plate (preassembled),
    2) lower fixture block,
    3) locating spacer plate,
    4) upper fixture block,
    5) transverse clamp bridge,
    6) top pressure cap.

Every part is inserted vertically along world -Z.  Successive parts use
matching feet / pockets, so the geometry is simple but still has explicit
state-dependent mating constraints.

Suggested project location:
    sealp/assets/models/ModularFixtureStackV1/gen_meshes.py

Run:
    python -m sealp.assets.models.ModularFixtureStackV1.gen_meshes
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.abspath(os.path.join(_MODELS_DIR, "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from sealp.assets.models._mesh_kit import (
    DOWN, OPEN_FACE_EPS,
    AssemblySpec, Part, Step,
    bnds, build_solid, centered, generate,
)

# ── base plate ────────────────────────────────────────────────
BASE_X = 0.160
BASE_Y = 0.130
BASE_H = 0.016

BASE_POCKET_X = 0.120
BASE_POCKET_Y = 0.090
BASE_POCKET_DEPTH = 0.006

# ── lower fixture ─────────────────────────────────────────────
LOWER_FOOT_X = 0.116
LOWER_FOOT_Y = 0.086
LOWER_FOOT_H = BASE_POCKET_DEPTH

LOWER_BODY_X = 0.108
LOWER_BODY_Y = 0.078
LOWER_H = 0.030

# Small side ears make the module look like a fixture block rather than
# a plain rectangular brick.
LOWER_EAR_X = 0.010
LOWER_EAR_Y = 0.040
LOWER_EAR_Z = (0.012, 0.024)

SPACER_POCKET_X = 0.096
SPACER_POCKET_Y = 0.066
SPACER_POCKET_DEPTH = 0.006

# ── locating spacer ───────────────────────────────────────────
SPACER_FOOT_X = 0.092
SPACER_FOOT_Y = 0.062
SPACER_FOOT_H = SPACER_POCKET_DEPTH

SPACER_BODY_X = 0.100
SPACER_BODY_Y = 0.070
SPACER_H = 0.014

UPPER_POCKET_X = 0.084
UPPER_POCKET_Y = 0.054
UPPER_POCKET_DEPTH = 0.005

# ── upper fixture ─────────────────────────────────────────────
UPPER_FOOT_X = 0.080
UPPER_FOOT_Y = 0.050
UPPER_FOOT_H = UPPER_POCKET_DEPTH

UPPER_BODY_X = 0.088
UPPER_BODY_Y = 0.058
UPPER_H = 0.025

# Two shallow side shoulders create a keyed / fixture-like outline.
UPPER_SHOULDER_X = 0.036
UPPER_SHOULDER_Y = 0.008
UPPER_SHOULDER_Z = (0.010, 0.022)

CLAMP_POCKET_X = 0.070
CLAMP_POCKET_Y = 0.022
CLAMP_POCKET_DEPTH = 0.005

# ── transverse clamp bridge ───────────────────────────────────
CLAMP_FOOT_X = 0.066
CLAMP_FOOT_Y = 0.018
CLAMP_FOOT_H = CLAMP_POCKET_DEPTH

CLAMP_BODY_X = 0.118
CLAMP_BODY_Y = 0.028
CLAMP_H = 0.014

CAP_POCKET_X = 0.028
CAP_POCKET_Y = 0.024
CAP_POCKET_DEPTH = 0.004

# ── top pressure cap ──────────────────────────────────────────
CAP_FOOT_X = 0.024
CAP_FOOT_Y = 0.020
CAP_FOOT_H = CAP_POCKET_DEPTH

CAP_BODY_X = 0.040
CAP_BODY_Y = 0.034
CAP_BODY_Z = (CAP_FOOT_H, 0.012)

CAP_PAD_X = 0.026
CAP_PAD_Y = 0.020
CAP_H = 0.018
CAP_PAD_Z = (0.012, CAP_H)


def base_plate() -> Part:
    """Broad base with one large locating pocket."""
    body = centered(BASE_X, BASE_Y, (0.0, BASE_H))
    cuts = [
        centered(
            BASE_POCKET_X,
            BASE_POCKET_Y,
            (BASE_H - BASE_POCKET_DEPTH, BASE_H + OPEN_FACE_EPS),
        )
    ]
    return Part("base_plate", build_solid(body, cuts))


def lower_fixture() -> Part:
    """Low fixture block with two lateral ears and an upper locating pocket."""
    foot = centered(
        LOWER_FOOT_X, LOWER_FOOT_Y,
        (0.0, LOWER_FOOT_H),
    )
    upper = centered(
        LOWER_BODY_X, LOWER_BODY_Y,
        (LOWER_FOOT_H, LOWER_H),
    )

    half_body_x = LOWER_BODY_X / 2.0
    ear_l = bnds(
        (-half_body_x - LOWER_EAR_X, -half_body_x),
        (-LOWER_EAR_Y / 2.0, LOWER_EAR_Y / 2.0),
        LOWER_EAR_Z,
    )
    ear_r = bnds(
        (half_body_x, half_body_x + LOWER_EAR_X),
        (-LOWER_EAR_Y / 2.0, LOWER_EAR_Y / 2.0),
        LOWER_EAR_Z,
    )

    cuts = [
        centered(
            SPACER_POCKET_X,
            SPACER_POCKET_Y,
            (LOWER_H - SPACER_POCKET_DEPTH, LOWER_H + OPEN_FACE_EPS),
        )
    ]
    return Part(
        "lower_fixture",
        build_solid(foot, cuts, pegs=[upper, ear_l, ear_r]),
    )


def spacer_plate() -> Part:
    """Wide thin locating plate with a smaller keyed foot and top pocket."""
    foot = centered(
        SPACER_FOOT_X, SPACER_FOOT_Y,
        (0.0, SPACER_FOOT_H),
    )
    body = centered(
        SPACER_BODY_X, SPACER_BODY_Y,
        (SPACER_FOOT_H, SPACER_H),
    )
    cuts = [
        centered(
            UPPER_POCKET_X,
            UPPER_POCKET_Y,
            (SPACER_H - UPPER_POCKET_DEPTH, SPACER_H + OPEN_FACE_EPS),
        )
    ]
    return Part("spacer_plate", build_solid(foot, cuts, pegs=[body]))


def upper_fixture() -> Part:
    """Upper block with two side shoulders and a narrow clamp-seat pocket."""
    foot = centered(
        UPPER_FOOT_X, UPPER_FOOT_Y,
        (0.0, UPPER_FOOT_H),
    )
    body = centered(
        UPPER_BODY_X, UPPER_BODY_Y,
        (UPPER_FOOT_H, UPPER_H),
    )

    half_body_y = UPPER_BODY_Y / 2.0
    shoulder_f = bnds(
        (-UPPER_SHOULDER_X / 2.0, UPPER_SHOULDER_X / 2.0),
        (-half_body_y - UPPER_SHOULDER_Y, -half_body_y),
        UPPER_SHOULDER_Z,
    )
    shoulder_b = bnds(
        (-UPPER_SHOULDER_X / 2.0, UPPER_SHOULDER_X / 2.0),
        (half_body_y, half_body_y + UPPER_SHOULDER_Y),
        UPPER_SHOULDER_Z,
    )

    cuts = [
        centered(
            CLAMP_POCKET_X,
            CLAMP_POCKET_Y,
            (UPPER_H - CLAMP_POCKET_DEPTH, UPPER_H + OPEN_FACE_EPS),
        )
    ]
    return Part(
        "upper_fixture",
        build_solid(foot, cuts, pegs=[body, shoulder_f, shoulder_b]),
    )


def clamp_bridge() -> Part:
    """Broad transverse bridge with a narrow locating foot and top cap socket."""
    foot = centered(
        CLAMP_FOOT_X, CLAMP_FOOT_Y,
        (0.0, CLAMP_FOOT_H),
    )
    bridge = centered(
        CLAMP_BODY_X, CLAMP_BODY_Y,
        (CLAMP_FOOT_H, CLAMP_H),
    )
    cuts = [
        centered(
            CAP_POCKET_X,
            CAP_POCKET_Y,
            (CLAMP_H - CAP_POCKET_DEPTH, CLAMP_H + OPEN_FACE_EPS),
        )
    ]
    return Part("clamp_bridge", build_solid(foot, cuts, pegs=[bridge]))


def top_cap() -> Part:
    """Small pressure cap with a keyed foot and raised central pad."""
    foot = centered(
        CAP_FOOT_X, CAP_FOOT_Y,
        (0.0, CAP_FOOT_H),
    )
    body = centered(
        CAP_BODY_X, CAP_BODY_Y,
        CAP_BODY_Z,
    )
    pad = centered(
        CAP_PAD_X, CAP_PAD_Y,
        CAP_PAD_Z,
    )
    return Part("top_cap", build_solid(foot, pegs=[body, pad]))


def build_spec() -> AssemblySpec:
    parts = {
        p.mesh_name: p
        for p in (
            base_plate(),
            lower_fixture(),
            spacer_plate(),
            upper_fixture(),
            clamp_bridge(),
            top_cap(),
        )
    }

    # Each child is expressed in the coordinate frame of its direct parent.
    lower_z = BASE_H - BASE_POCKET_DEPTH
    spacer_z = LOWER_H - SPACER_POCKET_DEPTH
    upper_z = SPACER_H - UPPER_POCKET_DEPTH
    clamp_z = UPPER_H - CLAMP_POCKET_DEPTH
    cap_z = CLAMP_H - CAP_POCKET_DEPTH

    steps = [
        Step(
            "base_plate", "base_plate", "fixture",
            (0.0, 0.0, 0.0), DOWN,
            name="Fixture Base Plate",
            notes="Broad base plate; treated as preassembled during layout search.",
        ),
        Step(
            "lower_fixture", "lower_fixture", "base_plate",
            (0.0, 0.0, lower_z), DOWN,
            name="Lower Fixture Module",
            notes="Drops into the large base locating pocket along world -Z.",
        ),
        Step(
            "spacer_plate", "spacer_plate", "lower_fixture",
            (0.0, 0.0, spacer_z), DOWN,
            name="Locating Spacer Plate",
            notes="Thin locating plate stacked into the lower fixture pocket.",
        ),
        Step(
            "upper_fixture", "upper_fixture", "spacer_plate",
            (0.0, 0.0, upper_z), DOWN,
            name="Upper Fixture Module",
            notes="Keyed upper block seated into the spacer plate.",
        ),
        Step(
            "clamp_bridge", "clamp_bridge", "upper_fixture",
            (0.0, 0.0, clamp_z), DOWN,
            name="Transverse Clamp Bridge",
            notes="Wide clamp bridge lowered into the narrow upper-fixture seat.",
        ),
        Step(
            "top_cap", "top_cap", "clamp_bridge",
            (0.0, 0.0, cap_z), DOWN,
            name="Top Pressure Cap",
            notes="Final pressure cap inserted into the bridge socket.",
        ),
    ]

    return AssemblySpec(
        name="ModularFixtureStackV1",
        description=(
            "Six-part low-profile modular fixture stack: broad base, lower "
            "fixture, locating spacer, upper fixture, transverse clamp bridge, "
            "and top pressure cap. All insertions are vertical."
        ),
        out_dir=_HERE,
        parts=parts,
        steps=steps,
    )


if __name__ == "__main__":
    raise SystemExit(0 if generate(build_spec()) else 1)
