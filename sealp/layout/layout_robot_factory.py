"""Create layout-search robots for single-arm or dual-arm workspace setups."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

import wrs.robot_sim.robots.robot_panthera_ht.panthera_ht_dual_arm as pda
from wrs.robot_sim.robots.robot_panthera_ht.panthera_ht import PantheraHTSglArm


@dataclass
class LayoutRobotBundle:
    robot: object
    single_arm: bool
    dual_arm_y_offset: float

    @property
    def primary_arm(self):
        return self.robot.lft_arm


def create_layout_robot(
    *,
    single_arm: bool,
    robot_base_pos: np.ndarray,
    robot_base_rotmat: np.ndarray,
    dual_arm_y_offset: float = 0.62,
    enable_cc: bool = True,
) -> LayoutRobotBundle:
    pos = np.asarray(robot_base_pos, dtype=float)
    rotmat = np.asarray(robot_base_rotmat, dtype=float)
    if single_arm:
        arm = PantheraHTSglArm(pos=pos, rotmat=rotmat, enable_cc=enable_cc)
        try:
            arm.setup_cc()
        except Exception:
            pass
        robot = _SingleArmRobotShell(arm)
        return LayoutRobotBundle(
            robot=robot,
            single_arm=True,
            dual_arm_y_offset=float(dual_arm_y_offset),
        )

    robot = pda.DualPantheraHTNoBody(
        pos=pos,
        rotmat=rotmat,
        arm_y_offset=float(dual_arm_y_offset),
        enable_cc=enable_cc,
    )
    try:
        robot.setup_cc()
    except Exception:
        pass
    return LayoutRobotBundle(
        robot=robot,
        single_arm=False,
        dual_arm_y_offset=float(dual_arm_y_offset),
    )


class _SingleArmRobotShell:
    """Minimal dual-arm-shaped wrapper around one Panthera-HT arm."""

    def __init__(self, arm: PantheraHTSglArm):
        self.lft_arm = arm
        self.rgt_arm = None
        self.delegator = arm
        self.cc = getattr(arm, "cc", None)

    def use_lft(self):
        self.delegator = self.lft_arm
        self.cc = self.lft_arm.cc
        return self.lft_arm

    def use_rgt(self):
        return self.use_lft()

    def use_all(self):
        self.delegator = None


def iter_layout_arms(robot, *, single_arm: bool) -> List[object]:
    if single_arm:
        return [robot.lft_arm]
    return [robot.lft_arm, robot.rgt_arm]


def layout_arm_pairs(robot, *, single_arm: bool) -> Sequence[Tuple[str, object]]:
    if single_arm:
        return (("lft", robot.lft_arm),)
    return (("lft", robot.lft_arm), ("rgt", robot.rgt_arm))


def get_layout_arm(robot, arm_tag: str, *, single_arm: bool):
    if single_arm:
        return robot.lft_arm
    return robot.rgt_arm if arm_tag == "rgt" else robot.lft_arm


def arm_base_xy_map(
    robot_base_pos: np.ndarray,
    *,
    single_arm: bool,
    dual_arm_y_offset: float,
) -> dict:
    rb_x = float(np.asarray(robot_base_pos, dtype=float)[0])
    rb_y = float(np.asarray(robot_base_pos, dtype=float)[1])
    if single_arm:
        return {"arm_base": (rb_x, rb_y)}
    return {
        "lft_arm_base": (rb_x, rb_y),
        "rgt_arm_base": (rb_x, rb_y - float(dual_arm_y_offset)),
    }


def goto_home_joints(robot, home_jv: np.ndarray, *, single_arm: bool) -> None:
    for arm in iter_layout_arms(robot, single_arm=single_arm):
        try:
            arm.goto_given_conf(jnt_values=home_jv)
        except Exception:
            pass


def reset_robot_for_l3(robot, home_jv: np.ndarray, *, single_arm: bool) -> None:
    for arm in iter_layout_arms(robot, single_arm=single_arm):
        ee = getattr(arm, "end_effector", None)
        if ee is not None:
            try:
                ee.oiee_list = []
                ee.oiee_list_bk.clear()
                ee.oiee_pose_list_bk.clear()
            except Exception:
                pass
        try:
            arm.goto_given_conf(jnt_values=home_jv)
        except Exception:
            pass
