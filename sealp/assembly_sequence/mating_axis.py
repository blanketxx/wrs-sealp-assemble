"""The direction a part is inserted along, taken from the asmdef step.

One implementation shared by the layout searcher and the executor so both aim the Cartesian
approach/depart segments at the axis the assembly actually declares, instead of guessing with a
fan of world-frame directions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def step_for_part(asm, pid: str):
    for step in getattr(asm, "steps", []) or []:
        if getattr(step, "part_id", None) == pid:
            return step
    return None


def assembly_mating_dirs(asm, pid: str, goal_rotmat) -> Tuple[Optional[np.ndarray],
                                                              Optional[np.ndarray]]:
    """``(approach, depart)`` unit vectors in world coordinates, or ``(None, None)``.

    Sign convention
    ---------------
        approach = d_insert  = world direction the end effector travels while the part is fed
                               into its parent;
        depart   = -d_insert = the way back out once it is seated.

    Priority, highest first:

      1) explicit ``insertion_axis`` on the step, already in world coordinates::

             d_insert = normalize(insertion_axis)

         ``[0, 0, 1]`` therefore means "inserted along world +Z", and the retreat is ``-Z``.
      1') legacy ``insertion_axis_local``, expressed in the part's goal frame::

             d_insert = normalize(goal_rotmat @ insertion_axis_local)

      2) the part's local +Z at the goal: ``axis = goal_rotmat[:, 2]``. The sign of ``rel_pos``
         projected on that axis says which side of the parent the part sits on, and insertion runs
         from that side along ``-axis``. A projection of ~0 (offset perpendicular to the axis)
         falls back to approaching from above.
      3) undetermined -- the caller decides what to do (usually plain world -Z/+Z).

    The axis is deliberately never inferred from the staging -> goal transport direction: where the
    part happens to be stored says nothing about how it mates.
    """
    step = step_for_part(asm, pid)
    if step is None:
        return None, None
    try:
        goal_rotmat = np.asarray(goal_rotmat, dtype=float).reshape(3, 3)

        d_world = getattr(step, "insertion_axis", None)
        if d_world is not None:
            d_world = np.asarray(d_world, dtype=float).reshape(3)
            norm = float(np.linalg.norm(d_world))
            if norm >= 1e-9:
                d_insert = d_world / norm
                return d_insert, -d_insert

        d_local = getattr(step, "insertion_axis_local", None)
        if d_local is not None:
            d_local = np.asarray(d_local, dtype=float).reshape(3)
            if float(np.linalg.norm(d_local)) >= 1e-9:
                d_insert = goal_rotmat @ d_local
                norm = float(np.linalg.norm(d_insert))
                if norm >= 1e-9:
                    d_insert = d_insert / norm
                    return d_insert, -d_insert

        axis = goal_rotmat[:, 2].astype(float)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return None, None
        axis = axis / norm
        rel_rot = np.asarray(step.rel_rotmat, dtype=float).reshape(3, 3)
        rel_pos = np.asarray(step.rel_pos, dtype=float).reshape(3)
        parent_rot = goal_rotmat @ rel_rot.T
        offset = float(np.dot(parent_rot @ rel_pos, axis))
        if abs(offset) < 1e-6:
            approach = axis if axis[2] < 0 else -axis
        else:
            approach = -(1.0 if offset > 0 else -1.0) * axis
        return approach, -approach
    except Exception:
        return None, None
