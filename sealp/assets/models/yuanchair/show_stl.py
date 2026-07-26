#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/4/20 16:40
# @Author : ZhangXi
import numpy as np

from wrs import wd, rm, mgm, mcm, cbt, gg, ppp, rrtc
base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
holder_1 = mcm.CollisionModel(r"D:\Project\wrs-sealp\sealp\assets\models\yuanchair\yuanchair-part1.stl")
holder_1.attach_to(base)
leg1 = mcm.CollisionModel(r"D:\Project\wrs-sealp\sealp\assets\models\yuanchair\yuanchair-part2.stl")
leg1.pos = np.array([0.07,-0.07,0.02])
leg1.attach_to(base)
leg2 = mcm.CollisionModel(r"D:\Project\wrs-sealp\sealp\assets\models\yuanchair\yuanchair-part2.stl")
leg2.pos = np.array([0.07,0.07,0.02])
leg2.attach_to(base)
leg3 = mcm.CollisionModel(r"D:\Project\wrs-sealp\sealp\assets\models\yuanchair\yuanchair-part2.stl")
leg3.pos = np.array([-0.07,-0.07,0.02])
leg3.attach_to(base)
leg4 = mcm.CollisionModel(r"D:\Project\wrs-sealp\sealp\assets\models\yuanchair\yuanchair-part2.stl")
leg4.pos = np.array([-0.07,0.07,0.02])
leg4.attach_to(base)
base.run()