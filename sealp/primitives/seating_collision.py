"""Mesh-accurate seating collision for parts that mate with already-assembled parts.

Removing mating partners from the placement obstacle set is what lets a plate be seated between
posts at all, but doing it wholesale also lets the path sweep *through* those posts.  The two are
separated here: partners stay checked, only with triangle mesh instead of the box cdprim, so a
plate with holes is not treated as a solid slab and a gripper finger cannot pass through a post.

Which parts count as mating is decided geometrically by
:mod:`sealp.assembly_sequence.mating_detection`; nothing in this module is assembly specific.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

import wrs.modeling.constant as const

Pose = Tuple[np.ndarray, np.ndarray]


def to_triangle_cdmesh(cmodel):
    """Copy a collision model set up for triangle-mesh (not box) collision queries."""
    out = cmodel.copy()
    try:
        out.change_cdmesh_type(cdmesh_type=const.CDMeshType.DEFAULT)
    except Exception:
        pass
    for attr in ("_sealp_part_id", "_sealp_role"):
        value = getattr(cmodel, attr, None)
        if value is not None:
            setattr(out, attr, value)
    out._sealp_cdprim_type = "triangles"
    return out


def mesh_obstacles_for_parts(obstacle_list: Sequence, part_ids: Iterable[str]) -> List:
    """Triangle-mesh copies of the obstacles whose ``_sealp_part_id`` is in ``part_ids``."""
    want = {p for p in part_ids}
    if not want:
        return []
    return [to_triangle_cdmesh(obs) for obs in obstacle_list
            if getattr(obs, "_sealp_part_id", None) in want]


def direct_transport_seating_kwargs(mating_part_ids: Iterable[str],
                                    obstacle_list: Sequence) -> dict:
    """``DirectTransportPrimitive.plan`` kwargs enabling phased mesh seating."""
    mesh_obs = mesh_obstacles_for_parts(obstacle_list, mating_part_ids)
    if not mesh_obs:
        return {}
    return {"seating_mesh_obstacles": mesh_obs}


def _end_effector(robot):
    ee = getattr(robot, "end_effector", None)
    if ee is not None:
        return ee
    delegator = getattr(robot, "delegator", None)
    return getattr(delegator, "end_effector", None) if delegator is not None else None


def _held_objects(robot) -> List:
    ee = _end_effector(robot)
    if ee is None:
        return []
    return [lnk.cmodel for lnk in getattr(ee, "oiee_list", []) or []]


def gripper_mesh_collides(robot, mesh_obstacles: Sequence) -> bool:
    """The gripper may never intersect a mating partner, not even at the seated pose."""
    if not mesh_obstacles:
        return False
    ee = _end_effector(robot)
    if ee is None:
        return False
    try:
        return bool(ee.is_mesh_collided(cmodel_list=list(mesh_obstacles)))
    except Exception:
        return False


def held_object_mesh_collides(robot, mesh_obstacles: Sequence) -> bool:
    """True when a held object's triangle mesh intersects a mating partner's triangle mesh."""
    if not mesh_obstacles:
        return False
    for obj in _held_objects(robot):
        try:
            if obj.is_mcdwith(list(mesh_obstacles)):
                return True
        except Exception:
            continue
    return False


def contact_flags_form_tail(flags: Sequence[bool]) -> bool:
    """Return True when contact is absent or forms one contiguous tail ending at the goal.

    This is the intended rule for insertion/seating: the part may begin touching its mating
    partner before the exact final waypoint, but once contact starts it must remain continuous
    through the end of the segment.  Scattered/mid-path contact is rejected.
    """
    flags = [bool(v) for v in flags]
    idx = [i for i, flag in enumerate(flags) if flag]
    if not idx:
        return True
    contiguous = (idx[-1] - idx[0] + 1) == len(idx)
    return bool(contiguous and idx[-1] == len(flags) - 1)


def object_pose_mesh_collides(obj_cmodel, pos, rotmat, mesh_obstacles: Sequence) -> bool:
    """Triangle-mesh test for a free object placed at ``(pos, rotmat)``."""
    if not mesh_obstacles:
        return False
    obj = to_triangle_cdmesh(obj_cmodel)
    obj.pos = np.asarray(pos, dtype=float)
    obj.rotmat = np.asarray(rotmat, dtype=float)
    try:
        return bool(obj.is_mcdwith(list(mesh_obstacles)))
    except Exception:
        return False


def seating_waypoint_valid(robot,
                           mesh_obstacles: Sequence,
                           *,
                           is_final: bool) -> Tuple[bool, str]:
    """Mesh checks for one waypoint of the stand-off -> goal seating segment.

    Backward-compatible single-waypoint check.  The gripper is never allowed to intersect a
    mating partner.  For a complete insertion segment, prefer ``contact_flags_form_tail`` so
    legitimate sliding/press-fit contact may start shortly before the final waypoint while
    scattered mid-path penetration is still rejected.
    """
    if not mesh_obstacles:
        return True, ""
    if gripper_mesh_collides(robot, mesh_obstacles):
        return False, "gripper mesh intersects a mating partner"
    if not is_final and held_object_mesh_collides(robot, mesh_obstacles):
        return False, "held object mesh passes through a mating partner"
    return True, ""
