"""Mesh-accurate seating collision for parts that mate through holes/slots.

Tower ``middle_plate`` rests on four posts whose corners pass through plate holes.
AABB/box cdprims treat the plate as a solid slab and falsely allow paths that sweep
through post bodies.  This module upgrades mating partners to triangle cdmesh and
checks the *held object* during the final seating segment.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

import wrs.modeling.constant as const

Pose = Tuple[np.ndarray, np.ndarray]

TOWER_MIDDLE_PLATE_POSTS: Tuple[str, ...] = (
    "post_bl", "post_fl", "post_br", "post_fr",
)


def mesh_mating_partner_ids(part_id: str, asm_part_ids: Iterable[str]) -> Tuple[str, ...]:
    """Return mating partner part ids that need triangle mesh checks for ``part_id``."""
    ids = set(asm_part_ids)
    if part_id == "middle_plate":
        return tuple(p for p in TOWER_MIDDLE_PLATE_POSTS if p in ids)
    return ()


def to_triangle_cdmesh(cmodel) -> object:
    """Copy a collision model and use triangle cdmesh for accurate mesh-mesh tests."""
    out = cmodel.copy()
    try:
        out.change_cdmesh_type(cdmesh_type=const.CDMeshType.DEFAULT)
    except Exception:
        pass
    try:
        out.change_cdprim_type(cdprim_type=const.CDPrimType.AABB)
    except Exception:
        pass
    if not hasattr(out, "_sealp_cdprim_type"):
        out._sealp_cdprim_type = "triangles"
    else:
        out._sealp_cdprim_type = "triangles"
    pid = getattr(cmodel, "_sealp_part_id", None)
    if pid is not None:
        out._sealp_part_id = pid
    role = getattr(cmodel, "_sealp_role", None)
    if role is not None:
        out._sealp_role = role
    return out


def mesh_mating_obstacles(obstacle_list: Sequence,
                          partner_ids: Sequence[str]) -> List:
    """Build triangle-mesh copies of obstacles whose ``_sealp_part_id`` is a mating partner."""
    want = set(partner_ids)
    if not want:
        return []
    out: List = []
    for obs in obstacle_list:
        pid = getattr(obs, "_sealp_part_id", None)
        if pid is None or pid not in want:
            continue
        out.append(to_triangle_cdmesh(obs))
    return out


def _held_objects(robot) -> List:
    ee = getattr(robot, "end_effector", None)
    if ee is None:
        delegator = getattr(robot, "delegator", None)
        ee = getattr(delegator, "end_effector", None) if delegator is not None else None
    if ee is None:
        return []
    return [lnk.cmodel for lnk in getattr(ee, "oiee_list", []) or []]


def held_object_mesh_collides(robot, mesh_obstacles: Sequence) -> bool:
    """True when any held object triangle mesh intersects an obstacle triangle mesh."""
    if not mesh_obstacles:
        return False
    for obj in _held_objects(robot):
        try:
            if obj.is_mcdwith(list(mesh_obstacles)):
                return True
        except Exception:
            return True
    return False


def object_pose_mesh_collides(obj_cmodel, pos, rotmat, mesh_obstacles: Sequence) -> bool:
    """Mesh-mesh test for a free object at ``(pos, rotmat)``."""
    if not mesh_obstacles:
        return False
    obj = obj_cmodel.copy()
    obj.pos = np.asarray(pos, dtype=float)
    obj.rotmat = np.asarray(rotmat, dtype=float)
    try:
        return bool(obj.is_mcdwith(list(mesh_obstacles)))
    except Exception:
        return True


def goal_mating_pose_valid(obj_cmodel, goal_pose: Pose,
                           mesh_obstacles: Sequence,
                           *,
                           pos_tol: float = 2e-3,
                           rot_tol: float = 5e-2) -> bool:
    """At the certified goal pose the held mesh should clear mating partners (holes/slots)."""
    if not mesh_obstacles:
        return True
    gp, gr = goal_pose
    return not object_pose_mesh_collides(obj_cmodel, gp, gr, mesh_obstacles)


def waypoint_seating_valid(robot,
                           mesh_obstacles: Sequence,
                           *,
                           is_final: bool,
                           obj_template,
                           goal_pose: Optional[Pose] = None) -> bool:
    """Held-object mesh check for one seating waypoint.

    Non-final waypoints must have zero mesh intersection with mating partners.
    The final waypoint must match the goal mating pose (mesh clear at goal).
    """
    if not mesh_obstacles:
        return True
    if is_final:
        if goal_pose is None:
            return not held_object_mesh_collides(robot, mesh_obstacles)
        held = _held_objects(robot)
        if not held:
            return goal_mating_pose_valid(obj_template, goal_pose, mesh_obstacles)
        gp, gr = goal_pose
        obj = held[0]
        return not object_pose_mesh_collides(obj, obj.pos, obj.rotmat, mesh_obstacles)
    return not held_object_mesh_collides(robot, mesh_obstacles)


def direct_transport_seating_kwargs(part_id: str,
                                    asm_part_ids: Iterable[str],
                                    transit_obstacles: Sequence) -> dict:
    """Keyword args for phased mesh seating (e.g. tower middle_plate vs four posts)."""
    partners = mesh_mating_partner_ids(part_id, asm_part_ids)
    if not partners:
        return {}
    mesh_obs = mesh_mating_obstacles(transit_obstacles, partners)
    if not mesh_obs:
        return {}
    return {"seating_mesh_obstacles": mesh_obs}
