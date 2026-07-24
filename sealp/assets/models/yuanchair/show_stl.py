#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/4/20 16:40
# @Author : ZhangXi
import os
import sys

import numpy as np

# Repo root (…/wrs-sealp-assemble) must be on PYTHONPATH for ``import wrs``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wrs import wd, rm, mgm, mcm, cbt, gg, ppp, rrtc

SEAT_STL = os.path.join(_HERE, "yuanchair-part1.stl")
LEG_STL = os.path.join(_HERE, "yuanchair-part2.stl")

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
holder_1 = mcm.CollisionModel(SEAT_STL)
holder_1.attach_to(base)
leg1 = mcm.CollisionModel(LEG_STL)
leg1.pos = np.array([0.07, -0.07, 0.02])
leg1.attach_to(base)
leg2 = mcm.CollisionModel(LEG_STL)
leg2.pos = np.array([0.07, 0.07, 0.02])
leg2.attach_to(base)
leg3 = mcm.CollisionModel(LEG_STL)
leg3.pos = np.array([-0.07, -0.07, 0.02])
leg3.attach_to(base)
leg4 = mcm.CollisionModel(LEG_STL)
leg4.pos = np.array([-0.07, 0.07, 0.02])
leg4.attach_to(base)
base.run()
