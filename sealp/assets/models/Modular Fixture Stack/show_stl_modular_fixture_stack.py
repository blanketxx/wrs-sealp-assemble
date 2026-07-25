#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""View ModularFixtureStackV1.

    # assembled product
    python -m sealp.assets.models.ModularFixtureStackV1.show_stl

    # pull every part back along its insertion axis, with arrows
    python -m sealp.assets.models.ModularFixtureStackV1.show_stl --explode 0.05 --arrows

    # inspect one part; SPACE steps a 90 deg yaw
    python -m sealp.assets.models.ModularFixtureStackV1.show_stl --part lower_fixture

    # parts / steps as text
    python -m sealp.assets.models.ModularFixtureStackV1.show_stl --list
"""

from __future__ import annotations

import os
import sys

_PROJ_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from sealp.assets.models._viewer import main

ASMDEF = "sealp/assembly_sequence/_demo_output/modular_fixture_stack_v1.asmdef"

if __name__ == "__main__":
    raise SystemExit(main(ASMDEF))
