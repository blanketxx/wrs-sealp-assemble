#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ButtressedSpireTowerV1 -- irregular stepped/spire tower benchmark.

This benchmark deliberately avoids a collection of plain cuboids.  Its parts
are composite solids made from multiple intersecting boxes:

    * cruciform base plinth,
    * stepped central tower core,
    * four identical L/T-like buttresses,
    * two horizontal side wings,
    * cross-shaped crown,
    * stepped spire.

The geometry is still generated through the same _mesh_kit API used by the
other SEALP model generators, so it remains easy to export to STL and to build
an AssemblySpec.

Suggested project location:
    sealp/assets/models/ButtressedSpireTowerV1/gen_meshes.py

Run:
    python -m sealp.assets.models.ButtressedSpireTowerV1.gen_meshes
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
    DOWN, MINUS_X, OPEN_FACE_EPS, PLUS_X,
    AssemblySpec, Part, Step,
    bnds, build_solid, centered, generate,
)

# ── cruciform base ────────────────────────────────────────────
BASE_CENTER = 0.085
BASE_ARM_LEN = 0.150
BASE_ARM_W = 0.040
BASE_H = 0.016

CORE_SOCKET = 0.052
CORE_SOCKET_DEPTH = 0.010

BUTTRESS_OFFSET = 0.048
BUTTRESS_SOCKET = 0.026
BUTTRESS_SOCKET_DEPTH = 0.009

# ── central stepped tower core ────────────────────────────────
CORE_FOOT = 0.050
CORE_FOOT_H = CORE_SOCKET_DEPTH

CORE_LOW = 0.044
CORE_LOW_Z = (CORE_FOOT_H, 0.050)

CORE_MID = 0.034
CORE_MID_Z = (0.050, 0.090)

CORE_HIGH = 0.024
CORE_HIGH_Z = (0.090, 0.125)

CORE_H = 0.125

# Horizontal wing slots near the upper core.
WING_SLOT_Y = 0.014
WING_SLOT_Z = (0.082, 0.098)

# ── irregular buttress ────────────────────────────────────────
BUTT_FOOT_X = 0.026
BUTT_FOOT_Y = 0.034
BUTT_FOOT_H = BUTTRESS_SOCKET_DEPTH

BUTT_SHAFT_X = 0.018
BUTT_SHAFT_Y = 0.022
BUTT_SHAFT_H = 0.072

BUTT_FLANGE_X = 0.032
BUTT_FLANGE_Y = 0.012
BUTT_FLANGE_Z = (0.030, 0.055)

BUTT_CAP_X = 0.025
BUTT_CAP_Y = 0.028
BUTT_CAP_Z = (0.064, 0.078)

# ── horizontal wing ───────────────────────────────────────────
WING_BODY_LEN = 0.050
WING_BODY_Y = 0.020
WING_BODY_H = 0.018

WING_PLUG_LEN = 0.018
WING_PLUG_Y = 0.011
WING_PLUG_H = 0.012

WING_FIN_LEN = 0.030
WING_FIN_Y = 0.010
WING_FIN_H = 0.012

WING_SEAT_GAP = 0.001

# ── crown ─────────────────────────────────────────────────────
CROWN_CENTER = 0.050
CROWN_ARM_LEN = 0.100
CROWN_ARM_W = 0.022
CROWN_H = 0.014

CROWN_CORE_SOCKET = 0.027
CROWN_CORE_SOCKET_DEPTH = 0.008
SPIRE_SOCKET = 0.022
SPIRE_SOCKET_DEPTH = 0.008

# ── spire ─────────────────────────────────────────────────────
SPIRE_FOOT = 0.020
SPIRE_FOOT_H = SPIRE_SOCKET_DEPTH
SPIRE_LOW = 0.015
SPIRE_MID = 0.010
SPIRE_TIP = 0.006

SPIRE_Z1 = 0.030
SPIRE_Z2 = 0.055
SPIRE_H = 0.085


def cruciform_base() -> Part:
    """Cross-shaped base built as a center block plus orthogonal arms."""
    center = centered(BASE_CENTER, BASE_CENTER, (0.0, BASE_H))
    x_arm = centered(BASE_ARM_LEN, BASE_ARM_W, (0.0, BASE_H))
    y_arm = centered(BASE_ARM_W, BASE_ARM_LEN, (0.0, BASE_H))

    cuts = [
        # Central socket for the stepped core.
        centered(
            CORE_SOCKET, CORE_SOCKET,
            (BASE_H - CORE_SOCKET_DEPTH, BASE_H + OPEN_FACE_EPS)
        ),

        # Four small sockets for the buttresses.
        centered(
            BUTTRESS_SOCKET, BUTTRESS_SOCKET,
            (BASE_H - BUTTRESS_SOCKET_DEPTH, BASE_H + OPEN_FACE_EPS),
            cx=-BUTTRESS_OFFSET,
        ),
        centered(
            BUTTRESS_SOCKET, BUTTRESS_SOCKET,
            (BASE_H - BUTTRESS_SOCKET_DEPTH, BASE_H + OPEN_FACE_EPS),
            cx=BUTTRESS_OFFSET,
        ),
        centered(
            BUTTRESS_SOCKET, BUTTRESS_SOCKET,
            (BASE_H - BUTTRESS_SOCKET_DEPTH, BASE_H + OPEN_FACE_EPS),
            cy=-BUTTRESS_OFFSET,
        ),
        centered(
            BUTTRESS_SOCKET, BUTTRESS_SOCKET,
            (BASE_H - BUTTRESS_SOCKET_DEPTH, BASE_H + OPEN_FACE_EPS),
            cy=BUTTRESS_OFFSET,
        ),
    ]
    return Part("cruciform_base", build_solid(center, cuts, pegs=[x_arm, y_arm]))


def stepped_core() -> Part:
    """Three-stage tapered-looking tower core made from stacked boxes."""
    foot = centered(CORE_FOOT, CORE_FOOT, (0.0, CORE_FOOT_H))
    low = centered(CORE_LOW, CORE_LOW, CORE_LOW_Z)
    mid = centered(CORE_MID, CORE_MID, CORE_MID_Z)
    high = centered(CORE_HIGH, CORE_HIGH, CORE_HIGH_Z)

    # Slots pass completely through x, allowing the same wing STL to enter
    # from +X and -X.
    cuts = [
        bnds(
            (-CORE_HIGH / 2.0 - OPEN_FACE_EPS,
             CORE_HIGH / 2.0 + OPEN_FACE_EPS),
            (-WING_SLOT_Y / 2.0, WING_SLOT_Y / 2.0),
            WING_SLOT_Z,
        )
    ]
    return Part("stepped_core", build_solid(foot, cuts, pegs=[low, mid, high]))


def buttress() -> Part:
    """Irregular L/T-like buttress assembled from four overlapping solids."""
    foot = centered(
        BUTT_FOOT_X, BUTT_FOOT_Y,
        (0.0, BUTT_FOOT_H)
    )

    shaft = centered(
        BUTT_SHAFT_X, BUTT_SHAFT_Y,
        (BUTT_FOOT_H, BUTT_SHAFT_H),
        cx=0.004,
    )

    # Side flange makes the part clearly asymmetric / non-cuboid.
    flange = centered(
        BUTT_FLANGE_X, BUTT_FLANGE_Y,
        BUTT_FLANGE_Z,
        cx=-0.006,
        cy=0.009,
    )

    cap = centered(
        BUTT_CAP_X, BUTT_CAP_Y,
        BUTT_CAP_Z,
        cx=0.003,
    )

    return Part("buttress", build_solid(foot, pegs=[shaft, flange, cap]))


def wing() -> Part:
    """T-shaped horizontal wing with a narrow insertion plug."""
    total_len = WING_BODY_LEN + WING_PLUG_LEN
    x_lo = -total_len / 2.0
    body_hi = x_lo + WING_BODY_LEN

    body = bnds(
        (x_lo, body_hi),
        (-WING_BODY_Y / 2.0, WING_BODY_Y / 2.0),
        (0.0, WING_BODY_H),
    )

    plug_z_lo = (WING_BODY_H - WING_PLUG_H) / 2.0
    plug = bnds(
        (body_hi, body_hi + WING_PLUG_LEN),
        (-WING_PLUG_Y / 2.0, WING_PLUG_Y / 2.0),
        (plug_z_lo, plug_z_lo + WING_PLUG_H),
    )

    # A transverse fin near the outer end makes the wing T-shaped in plan view.
    fin_center_x = x_lo + 0.012
    fin = centered(
        WING_FIN_LEN, WING_FIN_Y,
        ((WING_BODY_H - WING_FIN_H) / 2.0,
         (WING_BODY_H + WING_FIN_H) / 2.0),
        cx=fin_center_x,
        cy=0.014,
    )

    fin2 = centered(
        WING_FIN_LEN, WING_FIN_Y,
        ((WING_BODY_H - WING_FIN_H) / 2.0,
         (WING_BODY_H + WING_FIN_H) / 2.0),
        cx=fin_center_x,
        cy=-0.014,
    )

    return Part("wing", build_solid(body, pegs=[plug, fin, fin2]))


def crown() -> Part:
    """Cross-shaped crown with underside core pocket and top spire socket."""
    center = centered(CROWN_CENTER, CROWN_CENTER, (0.0, CROWN_H))
    x_arm = centered(CROWN_ARM_LEN, CROWN_ARM_W, (0.0, CROWN_H))
    y_arm = centered(CROWN_ARM_W, CROWN_ARM_LEN, (0.0, CROWN_H))

    cuts = [
        centered(
            CROWN_CORE_SOCKET,
            CROWN_CORE_SOCKET,
            (-OPEN_FACE_EPS, CROWN_CORE_SOCKET_DEPTH),
        ),
        centered(
            SPIRE_SOCKET,
            SPIRE_SOCKET,
            (CROWN_H - SPIRE_SOCKET_DEPTH, CROWN_H + OPEN_FACE_EPS),
        ),
    ]
    return Part("crown", build_solid(center, cuts, pegs=[x_arm, y_arm]))


def spire() -> Part:
    """Four-stage stepped spire."""
    foot = centered(SPIRE_FOOT, SPIRE_FOOT, (0.0, SPIRE_FOOT_H))
    low = centered(SPIRE_LOW, SPIRE_LOW, (SPIRE_FOOT_H, SPIRE_Z1))
    mid = centered(SPIRE_MID, SPIRE_MID, (SPIRE_Z1, SPIRE_Z2))
    tip = centered(SPIRE_TIP, SPIRE_TIP, (SPIRE_Z2, SPIRE_H))
    return Part("spire", build_solid(foot, pegs=[low, mid, tip]))


def build_spec() -> AssemblySpec:
    parts = {
        p.mesh_name: p
        for p in (
            cruciform_base(),
            stepped_core(),
            buttress(),
            wing(),
            crown(),
            spire(),
        )
    }

    core_z = BASE_H - CORE_SOCKET_DEPTH
    butt_z = BASE_H - BUTTRESS_SOCKET_DEPTH

    slot_center_z = (WING_SLOT_Z[0] + WING_SLOT_Z[1]) / 2.0
    wing_z = slot_center_z - WING_BODY_H / 2.0

    wing_body_hi = WING_BODY_LEN - (WING_BODY_LEN + WING_PLUG_LEN) / 2.0
    wing_x = -(CORE_HIGH / 2.0 + WING_SEAT_GAP) - wing_body_hi

    crown_z = core_z + CORE_H - CROWN_CORE_SOCKET_DEPTH
    spire_z = CROWN_H - SPIRE_SOCKET_DEPTH

    steps = [
        Step(
            "cruciform_base", "cruciform_base", "fixture",
            (0.0, 0.0, 0.0), DOWN,
            name="Cruciform Base Plinth",
            notes="Cross-shaped base treated as preassembled.",
        ),

        Step(
            "stepped_core", "stepped_core", "cruciform_base",
            (0.0, 0.0, core_z), DOWN,
            name="Stepped Tower Core",
            notes="Central tapered-looking core inserted vertically.",
        ),

        Step(
            "buttress_w", "buttress", "cruciform_base",
            (-BUTTRESS_OFFSET, 0.0, butt_z), DOWN,
            yaw_deg=0.0,
            name="West Buttress",
            notes="Asymmetric buttress seated in the west socket.",
        ),
        Step(
            "buttress_e", "buttress", "cruciform_base",
            (BUTTRESS_OFFSET, 0.0, butt_z), DOWN,
            yaw_deg=180.0,
            name="East Buttress",
            notes="Same STL rotated 180 deg.",
        ),
        Step(
            "buttress_s", "buttress", "cruciform_base",
            (0.0, -BUTTRESS_OFFSET, butt_z), DOWN,
            yaw_deg=90.0,
            name="South Buttress",
            notes="Same STL rotated 90 deg.",
        ),
        Step(
            "buttress_n", "buttress", "cruciform_base",
            (0.0, BUTTRESS_OFFSET, butt_z), DOWN,
            yaw_deg=270.0,
            name="North Buttress",
            notes="Same STL rotated 270 deg.",
        ),

        Step(
            "wing_w", "wing", "stepped_core",
            (wing_x, 0.0, wing_z), PLUS_X,
            name="West Horizontal Wing",
            notes="T-shaped wing pushed horizontally along world +X.",
        ),
        Step(
            "wing_e", "wing", "stepped_core",
            (-wing_x, 0.0, wing_z), MINUS_X,
            yaw_deg=180.0,
            name="East Horizontal Wing",
            notes="Same STL yawed 180 deg and pushed along world -X.",
        ),

        Step(
            "crown", "crown", "stepped_core",
            (0.0, 0.0, CORE_H - CROWN_CORE_SOCKET_DEPTH), DOWN,
            name="Cross Crown",
            notes="Cross-shaped crown lowered over the upper core.",
        ),

        Step(
            "spire", "spire", "crown",
            (0.0, 0.0, spire_z), DOWN,
            name="Stepped Spire",
            notes="Final tapered spire inserted into the crown socket.",
        ),
    ]

    return AssemblySpec(
        name="ButtressedSpireTowerV1",
        description=(
            "Ten-step irregular tower benchmark with a cruciform base, "
            "stepped core, four asymmetric buttresses, two horizontal "
            "T-shaped wings, a cross crown, and a stepped spire."
        ),
        out_dir=_HERE,
        parts=parts,
        steps=steps,
        symmetry_groups={
            "buttresses": [
                "buttress_w", "buttress_e", "buttress_s", "buttress_n"
            ],
            "wings": ["wing_w", "wing_e"],
        },
    )


if __name__ == "__main__":
    raise SystemExit(0 if generate(build_spec()) else 1)
