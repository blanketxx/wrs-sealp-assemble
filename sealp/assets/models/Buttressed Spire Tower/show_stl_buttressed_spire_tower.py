#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""View ButtressedSpireTowerV1.

    # assembled product
    python -m sealp.assets.models.ButtressedSpireTowerV1.show_stl

    # explode every part along its own insertion axis, with arrows
    python -m sealp.assets.models.ButtressedSpireTowerV1.show_stl --explode 0.05 --arrows

    # one irregular buttress on its own; SPACE steps a 90 deg yaw
    python -m sealp.assets.models.ButtressedSpireTowerV1.show_stl --part buttress_w

    # one horizontal wing on its own
    python -m sealp.assets.models.ButtressedSpireTowerV1.show_stl --part wing_w

    # parts / steps as text
    python -m sealp.assets.models.ButtressedSpireTowerV1.show_stl --list
"""

from __future__ import annotations

import os
import sys

_PROJ_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from sealp.assets.models._viewer import main

ASMDEF = "sealp/assembly_sequence/_demo_output/buttressed_spire_tower_v1.asmdef"

if __name__ == "__main__":
    raise SystemExit(main(ASMDEF))
