"""
Single-Arm Direct Transport Primitive
=====================================

``pick -> transport -> place`` with one arm and one grasp, connected by RRT in joint space
instead of by mandatory Cartesian straight-line approach/depart segments::

    common grasp -> q_pick (IK) -> RRT to q_pick -> close -> held-object RRT to q_place
                 -> open -> RRT to q_end

Why the Cartesian segments are the problem
------------------------------------------
``PickPlacePlanner.gen_pick_and_place()`` always brackets the transport with four straight-line
segments (pick approach, pick depart, place approach, place depart). Each one is produced by
``InterplatedMotion.gen_linear_motion`` / ``gen_rel_linear_motion_with_given_conf``, and each one
imposes requirements that are much stronger than reaching the grasp itself:

* IK must solve at *every* interpolated pose along the line, not just at the grasp pose. When it
  does not, the log line is ``IK not solvable in gen_linear_motion!`` and the grasp is discarded --
  which is how every common grasp for the tower's ``middle_plate`` was eliminated.
* ``gen_rel_linear_motion_with_given_conf`` additionally rejects any step whose joint delta exceeds
  45 degrees, to keep the segment straight. Near a wrist or elbow singularity a perfectly reachable
  pose pair can violate this even when a smooth joint-space path exists.
* ``gen_linear_motion`` re-solves IK with ``seed_jnt_values=None`` on its first pose, so it can land
  on a *different* IK branch than the one ``reason_common_gids`` certified. The certified
  configuration is then never actually used, and the branch that is used may be collided.

Shortening the distance does not avoid any of this, and neither does setting it to zero: with
``distance=0`` the "sink" branch computes ``start_tcp_pos = goal_tcp_pos - unit_vector(dir) * 0``,
i.e. start equals goal, and ``gen_linear_motion`` is still called and still re-solves IK on the
interpolated (degenerate) pose list. The segment has to be removed, not shrunk.

What replaces them
------------------
This planner anchors on the configurations it certified -- ``q_pick`` from IK at the staging grasp
pose and ``q_place`` from IK at the goal grasp pose, seeded from ``q_pick`` so both lie on the same
branch -- and connects them with RRT. Nothing about the certified grasp is re-derived, so the
motion is executed by exactly the configuration that was validated.

Removing the straight lines must not remove what they guaranteed, namely that the object only
touches anything at the very end of the reach and at the moment of seating. That property is kept
directly instead of via the shape of the path: each RRT segment is planned against the obstacle set
that makes its endpoint admissible, and is then audited waypoint by waypoint against the *strict*
set. Contact is accepted only when the offending waypoints form a contiguous tail at the end of an
approach (or a contiguous head at the start of a retract). A path that sweeps through the object or
through the mating parts mid-flight has its contact waypoints scattered in the middle, fails the
audit, and is rejected. So the arm is still forbidden from passing through anything; it is merely
no longer forced to arrive along a straight line.

IK, joint limits and robot/object collisions stay fully enforced: configurations come from
``robot.ik`` (which respects joint limits), RRT samples within ``jnt_ranges``, and every collision
query goes through ``robot.is_collided`` with the held object attached.

Single arm only. No second arm, no receiver, no handover, no putting the object down on the way.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd
import wrs.motion.primitives.interpolated as mpi
import wrs.motion.probabilistic.rrt_connect as rrtc

from .base import MotionPrimitive, PrimitiveResult
from .seating_collision import (
    goal_mating_pose_valid,
    to_triangle_cdmesh,
    waypoint_seating_valid,
)

Pose = Tuple[np.ndarray, np.ndarray]

# Stand-off distances tried when looking for the configuration just before contact. Each entry
# costs one IK call at a single displaced pose -- not a pose-by-pose IK chain along a line, and not
# subject to the 45-degree straightness rule that kills the Cartesian segments.
DEFAULT_STANDOFF_SCHEDULE: Tuple[float, ...] = (0.06, 0.045, 0.03, 0.02, 0.012, 0.006)


def _summarise(reasons: Sequence[str]) -> str:
    """Group per-grasp failure messages by kind, so the dominant cause is visible at a glance."""
    counts: dict = {}
    for reason in reasons:
        key = reason.split(" ", 1)[1] if reason.startswith("gid=") else reason
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return "; ".join(f"{n}x {key}" for key, n in ordered)


class _GraspCandidate:
    __slots__ = ("gid", "grasp", "q_pick", "q_place")

    def __init__(self, gid, grasp, q_pick, q_place):
        self.gid = gid
        self.grasp = grasp
        self.q_pick = q_pick
        self.q_place = q_place


class DirectTransportPrimitive(MotionPrimitive):
    """Single-arm pick/transport/place without mandatory Cartesian approach or depart.

    Parameters
    ----------
    robot : SglArmRobotInterface
        One robot arm. The same arm and the same grasp are used for the whole motion.
    rrt_ext_dist
        RRT-Connect extension step.
    rrt_max_time
        Per-segment RRT time limit, seconds.
    max_grasps
        Upper bound on how many certified grasps are tried before giving up.
    """

    def __init__(self,
                 robot,
                 rrt_ext_dist: float = 0.1,
                 rrt_max_time: float = 100.0,
                 max_grasps: int = 24,
                 standoff_schedule: Sequence[float] = DEFAULT_STANDOFF_SCHEDULE,
                 interp_granularity: float = 0.02):
        self.robot = robot
        self.rrt_ext_dist = float(rrt_ext_dist)
        self.rrt_max_time = float(rrt_max_time)
        self.max_grasps = int(max_grasps)
        self.standoff_schedule = tuple(standoff_schedule)
        self.interp_granularity = float(interp_granularity)
        self._rrtc = rrtc.RRTConnect(robot)
        self._interp = mpi.InterplatedMotion(robot)

    # ------------------------------------------------------------------
    def plan(self,
             obj_cmodel,
             grasp_collection,
             goal_pose_list: List[Pose],
             start_jnt_values: Optional[np.ndarray] = None,
             end_jnt_values: Optional[np.ndarray] = None,
             obstacle_list: Optional[List] = None,
             approach_distance: float = 0.05,
             depart_distance: float = 0.05,
             use_rrt: bool = True,
             **kwargs) -> PrimitiveResult:
        """Plan the motion. ``goal_pose_list`` must contain exactly one goal pose.

        Recognised keyword arguments
        ----------------------------
        grasp_obstacle_list
            Legacy contact-exempt obstacle set kept for API compatibility with
            :class:`TransportPrimitive`.  When ``seating_mesh_obstacles`` is supplied, mating
            partners stay in the strict obstacle set for pick/transport/stand-off and are checked
            with triangle mesh during seating instead of being removed wholesale.

        seating_mesh_obstacles
            Triangle-mesh copies of mating partners (e.g. the four tower posts for
            ``middle_plate``).  When non-empty, enables phased seating collision: strict obstacles
            everywhere, mesh held-object checks on the stand-off -> goal segment, goal pose
            certified with mesh geometry rather than box exclusion.

        place_depart_direction_list
            Mating axis of the assembly. The stand-off configuration above the goal is placed along
            it, so the object leaves and enters its seat the way the assembly requires.
        pick_standoff_direction
            Where to put the stand-off before the grasp. Defaults to straight back out of the
            gripper, which is what clears a side grasp.

        ``approach_distance``/``depart_distance`` are accepted for interface compatibility with
        :class:`TransportPrimitive` and ignored: there is no Cartesian segment to parameterise, only
        a stand-off point chosen from ``standoff_schedule``.
        """
        if obstacle_list is None:
            obstacle_list = []
        if len(goal_pose_list) != 1:
            return PrimitiveResult(
                success=False,
                error_msg=f"DirectTransport supports one goal pose, got {len(goal_pose_list)}.")
        if grasp_collection is None or len(grasp_collection) == 0:
            return PrimitiveResult(success=False, error_msg="empty grasp collection")

        strict_obs = list(obstacle_list)
        grasp_obs = kwargs.get("grasp_obstacle_list")
        relaxed_obs = list(grasp_obs) if grasp_obs is not None else list(strict_obs)
        mating_mesh_obs = list(kwargs.get("seating_mesh_obstacles") or [])
        phased_seating = bool(mating_mesh_obs)
        # Stand-off directions. At the goal it is the assembly's mating axis, so the object enters
        # its seat the way the assembly requires. At the pick it is straight back out of the
        # gripper, which is the only direction guaranteed to clear a side grasp; note this is
        # deliberately NOT ``pick_depart_direction`` (that is the lift *after* the grasp, and for a
        # part held by its edge a lift direction makes a poor pre-grasp stand-off).
        depart_dirs = kwargs.get("place_depart_direction_list") or [None]
        place_standoff_dir = depart_dirs[-1]
        pick_standoff_dir = kwargs.get("pick_standoff_direction")

        start_pose = (np.asarray(obj_cmodel.pos, dtype=float).copy(),
                      np.asarray(obj_cmodel.rotmat, dtype=float).copy())
        goal_pose = (np.asarray(goal_pose_list[0][0], dtype=float),
                     np.asarray(goal_pose_list[0][1], dtype=float))
        if start_jnt_values is None:
            start_jnt_values = self.robot.get_jnt_values()
        if end_jnt_values is None:
            end_jnt_values = self.robot.get_jnt_values()

        # The object at its staging pose and at its goal pose. Neither is in ``obstacle_list`` (the
        # caller excludes the part currently being moved), so they are tracked separately and used
        # by the audits below.
        obj_at_start = obj_cmodel.copy()
        obj_at_start.pose = start_pose
        obj_at_goal = obj_cmodel.copy()
        obj_at_goal.pose = goal_pose

        self.robot.backup_state()
        try:
            candidates = self._certify_grasps(
                grasp_collection, start_pose, goal_pose, strict_obs, relaxed_obs,
                obj_cmodel=obj_cmodel, mating_mesh_obs=mating_mesh_obs,
                phased_seating=phased_seating,
            )
        finally:
            self.robot.restore_state()
        if not candidates:
            return PrimitiveResult(
                success=False,
                error_msg="no grasp is IK-feasible and collision-free at both the staging "
                          "and the goal pose")
        print(f"  [direct] certified grasps={len(candidates)} "
              f"(gids={[c.gid for c in candidates[:8]]}{'...' if len(candidates) > 8 else ''})")

        last_err = "no rrt path"
        reasons: List[str] = []
        for cand in candidates[:self.max_grasps]:
            depth = self._backup_depth()
            self._empty_hand()
            self.robot.backup_state()
            try:
                mot_data, err = self._plan_with_grasp(
                    cand=cand,
                    obj_cmodel=obj_cmodel,
                    obj_at_start=obj_at_start,
                    obj_at_goal=obj_at_goal,
                    start_jnt_values=start_jnt_values,
                    end_jnt_values=end_jnt_values,
                    strict_obs=strict_obs,
                    relaxed_obs=relaxed_obs,
                    mating_mesh_obs=mating_mesh_obs,
                    phased_seating=phased_seating,
                    goal_pose=goal_pose,
                    pick_standoff_dir=pick_standoff_dir,
                    place_standoff_dir=place_standoff_dir,
                )
            except Exception as e:  # a bad grasp must not abort the whole search
                mot_data, err = None, f"gid={cand.gid} exception {type(e).__name__}: {e!r}"
            finally:
                self._reset_to(depth)
            if mot_data is not None:
                print(f"  [direct] solved with gid={cand.gid}, {len(mot_data.jv_list)} waypoints")
                return PrimitiveResult(success=True,
                                       mot_data=mot_data,
                                       end_jnt_values=mot_data.jv_list[-1])
            last_err = err
            reasons.append(err)
            print(f"  [direct] {err}")
        print(f"  [direct] all {len(reasons)} attempts failed; breakdown: {_summarise(reasons)}")
        return PrimitiveResult(success=False, error_msg=f"DirectTransport failed: {last_err}")

    # ------------------------------------------------------------------
    # State hygiene. WRS's ``keep_states_decorator`` has no try/finally, so any exception raised
    # inside a decorated planner call leaves an extra frame on the backup stacks. Later restores
    # then pop the wrong frame and the arm can come back still "holding" an object, which makes the
    # next ``hold`` raise "The hand is holding objects!". Trying many grasps in a row makes that
    # inevitable, so each attempt records the stack depth and unwinds to it afterwards.
    def _backup_depth(self) -> Tuple[int, int]:
        ee = self._ee()
        return (len(getattr(ee, "oiee_list_bk", [])) if ee is not None else 0,
                len(getattr(self.robot.manipulator, "jnt_values_bk", [])))

    def _reset_to(self, depth: Tuple[int, int]) -> None:
        ee_depth, arm_depth = depth
        try:
            self.robot.restore_state()
        except Exception:
            pass
        ee = self._ee()
        if ee is not None:
            del ee.oiee_list_bk[ee_depth:]
            del ee.oiee_pose_list_bk[ee_depth:]
        self._empty_hand()
        jnt_bk = getattr(self.robot.manipulator, "jnt_values_bk", None)
        if jnt_bk is not None:
            del jnt_bk[arm_depth:]

    def _ee(self):
        """The end effector that actually owns ``oiee_list``.

        ``hold`` may be delegated, in which case ``self.robot.end_effector`` is not the object whose
        ``oiee_list`` the assertion in ``hold`` inspects, and clearing the wrong one leaves the hand
        looking full forever.
        """
        ee = getattr(self.robot, "end_effector", None)
        if ee is not None:
            return ee
        delegator = getattr(self.robot, "delegator", None)
        return getattr(delegator, "end_effector", None) if delegator is not None else None

    def _empty_hand(self) -> None:
        ee = self._ee()
        if ee is not None and len(ee.oiee_list) > 0:
            ee.release_all()

    def _hand_is_full(self) -> bool:
        ee = self._ee()
        return ee is not None and len(ee.oiee_list) > 0

    def _goto(self, jnt_values, ee_values=None) -> None:
        """Move the arm, commanding the jaw width only when that is legal.

        ``change_ee_values`` carries the same ``assert_oiee_decorator`` as ``hold``, so sending a
        width while an object is grasped raises "The hand is holding objects!". Once the object is
        held its width is fixed by the grasp anyway, so the value is simply dropped.
        """
        if ee_values is not None and self._hand_is_full():
            ee_values = None
        self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)

    def _set_jaw(self, ee_values) -> None:
        if ee_values is not None and not self._hand_is_full():
            self.robot.change_ee_values(ee_values=ee_values)

    def _within_limits(self, jnt_values) -> bool:
        """Explicit joint-limit check.

        ``robot.ik`` can return values marginally outside ``jnt_ranges``, and ``goto_given_conf``
        reacts to those by printing "The given joint angles are out of joint limits." and returning
        *without moving the arm*. A collision query issued afterwards would then be evaluating the
        previous configuration, and an infeasible grasp could be accepted. Rejecting out-of-range
        configurations up front closes that hole.
        """
        ranges = self.robot.jnt_ranges
        return bool(np.all(jnt_values >= ranges[:, 0] - 1e-9) and
                    np.all(jnt_values <= ranges[:, 1] + 1e-9))

    # ------------------------------------------------------------------
    def _certify_grasps(self, grasp_collection, start_pose, goal_pose,
                        strict_obs, relaxed_obs, *, obj_cmodel, mating_mesh_obs,
                        phased_seating) -> List[_GraspCandidate]:
        """Grasps that are IK-feasible and collision-free at the staging pose and at the goal pose.

        Both configurations are kept. ``q_place`` is solved with ``q_pick`` as its IK seed so the two
        lie on the same branch, which is what makes the joint-space transport short and is also the
        continuity the Cartesian segments were trying to obtain by interpolating.

        With ``phased_seating``, pick uses ``strict_obs`` and goal uses ``strict_obs`` for the arm
        plus a triangle-mesh mating check for the object (posts stay in the scene; holes are not
        treated as solid box slabs).
        """
        out: List[_GraspCandidate] = []
        sp, sr = start_pose
        gp, gr = goal_pose
        pick_obs = strict_obs if phased_seating else relaxed_obs
        place_obs = strict_obs if phased_seating else relaxed_obs
        for gid in range(len(grasp_collection)):
            grasp = grasp_collection[gid]
            q_pick = self._ik_at(sp, sr, grasp, pick_obs, seed=None)
            if q_pick is None:
                continue
            q_place = self._ik_at_place_phased(
                gp, gr, grasp, strict_obs, mating_mesh_obs,
                to_triangle_cdmesh(obj_cmodel), seed=q_pick) if phased_seating else self._ik_at(
                gp, gr, grasp, place_obs, seed=q_pick)
            out.append(_GraspCandidate(gid, grasp, q_pick, q_place))
        # Prefer grasps whose two configurations are closest in joint space: the arm has to
        # reorient the object the least, so the transport is easiest to connect.
        out.sort(key=lambda c: float(np.max(np.abs(c.q_place - c.q_pick))))
        return out

    def _ik_at(self, pos, rotmat, grasp, obstacle_list, seed):
        tcp_pos = pos + rotmat.dot(grasp.ac_pos)
        tcp_rotmat = rotmat.dot(grasp.ac_rotmat)
        jnt_values = self.robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rotmat, seed_jnt_values=seed)
        if jnt_values is None or not self._within_limits(jnt_values):
            return None
        self._goto(jnt_values, ee_values=grasp.ee_values)
        if self.robot.is_collided(obstacle_list=obstacle_list):
            return None
        if self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
            return None
        return jnt_values

    def _ik_at_place_phased(self, pos, rotmat, grasp, strict_obs, mating_mesh_obs,
                            obj_mesh, seed):
        """Goal grasp IK: arm checked against strict obstacles; object mating via triangle mesh.

        Post box cdprims can false-positive against the gripper at the seated goal even when the
        plate mesh clears the post holes.  When strict IK succeeds and mesh mating is valid, accept.
        Otherwise retry arm IK with mating-partner *boxes* removed (mesh still required).
        """
        q = self._ik_at(pos, rotmat, grasp, strict_obs, seed=seed)
        if q is not None and goal_mating_pose_valid(obj_mesh, (pos, rotmat), mating_mesh_obs):
            return q
        partner_ids = {getattr(o, "_sealp_part_id", None) for o in mating_mesh_obs}
        arm_obs = [o for o in strict_obs if getattr(o, "_sealp_part_id", None) not in partner_ids]
        q = self._ik_at(pos, rotmat, grasp, arm_obs, seed=seed)
        if q is None:
            return None
        if goal_mating_pose_valid(obj_mesh, (pos, rotmat), mating_mesh_obs):
            return q
        return None

    # ------------------------------------------------------------------
    def _plan_with_grasp(self, cand, obj_cmodel, obj_at_start, obj_at_goal,
                         start_jnt_values, end_jnt_values, strict_obs, relaxed_obs,
                         mating_mesh_obs, phased_seating, goal_pose,
                         pick_standoff_dir, place_standoff_dir):
        """Plan one grasp end to end. Returns ``(MotionData, "")`` or ``(None, reason)``.

        Each of the three phases is built as *free RRT up to a stand-off configuration* plus *a
        short joint-space interpolation across the contact*. The stand-off configuration comes from
        a single IK call at a displaced pose, seeded from the certified configuration, and the
        interpolation needs no IK at all. That is what replaces the Cartesian segment: the long
        part of the motion stays under the strict obstacle set.  When ``phased_seating`` is active,
        the final centimetres keep posts in ``strict_obs`` for the arm and add triangle-mesh held-
        object checks so the plate cannot sweep through post bodies.
        """
        jaw_open = float(self.robot.end_effector.jaw_range[1])
        sp, sr = obj_at_start.pos, obj_at_start.rotmat
        gp, gr = obj_at_goal.pos, obj_at_goal.rotmat
        # Retreating along the mating axis is what the assembly dictates at the goal; retreating
        # back out of the gripper is the other direction that is guaranteed to clear a side grasp.
        pick_dirs = [pick_standoff_dir] if pick_standoff_dir is not None else [None]
        place_dirs = [place_standoff_dir] if place_standoff_dir is not None else [None]
        retract_dirs = list(place_dirs) + [None]

        # -- reach: q_start -> stand-off -> q_pick, gripper open, object not yet held -------------
        reach_obs = strict_obs + [obj_at_start]
        q_pre_pick, d_pick, detail = self._standoff_conf(
            (sp, sr), cand.grasp, pick_dirs, reach_obs, seed=cand.q_pick, ee_values=jaw_open)
        if q_pre_pick is None:
            return None, f"gid={cand.gid} no stand-off before the grasp ({detail})"
        reach = self._rrt(start_jnt_values, q_pre_pick, reach_obs, ee_values=jaw_open)
        if reach is None:
            return None, f"gid={cand.gid} no rrt path to the pre-grasp stand-off"
        # closing in on the object: it is excluded here because the gripper has to enclose it
        close_in = self._interpolate(q_pre_pick, cand.q_pick, strict_obs, ee_values=jaw_open)
        if close_in is None:
            return None, f"gid={cand.gid} cannot close in on the object from the stand-off"
        ok, n_contact = self._contact_is_tail(close_in, reach_obs, ee_values=jaw_open, want="tail")
        if not ok:
            return None, (f"gid={cand.gid} closing in touches the object before the grasp "
                          f"({n_contact} waypoints)")

        # -- grasp, transport, seat --------------------------------------------------------------
        obj_held = to_triangle_cdmesh(obj_cmodel.copy()) if phased_seating else obj_cmodel.copy()
        obj_held.pose = (sp, sr)
        self.robot.goto_given_conf(cand.q_pick)
        self._empty_hand()
        self.robot.hold(obj_cmodel=obj_held, jaw_width=cand.grasp.ee_values)
        try:
            q_pre_place, d_place, detail = self._standoff_conf(
                (gp, gr), cand.grasp, place_dirs, strict_obs, seed=cand.q_place,
                ee_values=cand.grasp.ee_values)
            if q_pre_place is None:
                return None, f"gid={cand.gid} no stand-off above the goal ({detail})"
            transport = self._rrt(cand.q_pick, q_pre_place, strict_obs,
                                  ee_values=cand.grasp.ee_values)
            if transport is None:
                return None, f"gid={cand.gid} no rrt path from q_pick to the pre-place stand-off"
            if phased_seating:
                seat, seat_err = self._interpolate_seating(
                    q_pre_place, cand.q_place, strict_obs, mating_mesh_obs,
                    ee_values=cand.grasp.ee_values,
                    obj_template=to_triangle_cdmesh(obj_cmodel),
                    goal_pose=goal_pose)
                if seat is None:
                    return None, f"gid={cand.gid} {seat_err}"
            else:
                # Legacy path: contact-exempt relaxed obstacles + tail audit.
                seat = self._interpolate(q_pre_place, cand.q_place, relaxed_obs,
                                         ee_values=cand.grasp.ee_values)
                if seat is None:
                    return None, f"gid={cand.gid} cannot seat the object from the stand-off"
                ok, n_contact = self._contact_is_tail(
                    seat, strict_obs, ee_values=cand.grasp.ee_values, want="tail")
                if not ok:
                    return None, (f"gid={cand.gid} seating collides before it lands "
                                  f"({n_contact} waypoints)")
        finally:
            # Always give the object back, including on the early returns above, so the next grasp
            # does not start with a hand that still thinks it is full.
            self.robot.goto_given_conf(cand.q_place)
            try:
                self.robot.release(obj_cmodel=obj_held)
            except ValueError:
                self._empty_hand()

        # -- retract: q_place -> stand-off -> q_end, object left at the goal -----------------------
        retract_obs = strict_obs + [obj_at_goal]
        q_post_place = None
        detail = "no opening tried"
        for width in self._jaw_release_widths(cand.grasp):
            self._set_jaw(width)
            q_post_place, _, detail = self._standoff_conf(
                (gp, gr), cand.grasp, retract_dirs, retract_obs, seed=cand.q_place,
                ee_values=width)
            if q_post_place is not None:
                jaw_release = width
                break
        if q_post_place is None:
            return None, f"gid={cand.gid} nowhere to retract to after releasing ({detail})"
        retract_obs_list = strict_obs if phased_seating else relaxed_obs
        back_off = self._interpolate(cand.q_place, q_post_place, retract_obs_list,
                                     ee_values=jaw_release)
        if back_off is None:
            return None, f"gid={cand.gid} cannot back off from the placed object"
        ok, n_contact = self._contact_is_tail(back_off, retract_obs, ee_values=jaw_release,
                                              want="head")
        if not ok:
            return None, (f"gid={cand.gid} backing off re-enters the placed object "
                          f"({n_contact} waypoints)")
        home = self._rrt(q_post_place, end_jnt_values, retract_obs, ee_values=jaw_release)
        if home is None:
            return None, f"gid={cand.gid} no rrt path from the stand-off back to the end configuration"

        print(f"  [direct] gid={cand.gid} stand-off pick={d_pick * 1000:.0f}mm "
              f"place={d_place * 1000:.0f}mm, release opening={jaw_release * 1000:.1f}mm")
        return self._assemble(cand, obj_cmodel, obj_at_start, obj_at_goal,
                              reach_jv=list(reach.jv_list) + list(close_in.jv_list),
                              carry_jv=list(transport.jv_list) + list(seat.jv_list),
                              retract_jv=list(back_off.jv_list) + list(home.jv_list),
                              jaw_open=jaw_open, jaw_release=jaw_release), ""

    # ------------------------------------------------------------------
    def _standoff_conf(self, pose, grasp, directions, obstacle_list, seed, ee_values):
        """Configuration at the grasp pose displaced clear of contact, nearest one that works.

        Tries each direction at each distance in ``standoff_schedule``: one IK call per combination,
        seeded from the certified configuration so the result stays on the same branch. This is the
        only place a Cartesian offset appears, and it is a *point*, not a path -- there is no
        requirement that a straight line to it be feasible, which is precisely the requirement that
        made the original approach/depart segments fail.

        Returns ``(jnt_values, distance, detail)``; ``detail`` counts why the rejected candidates
        were rejected, so a failure says whether the arm could not reach or could not fit.
        """
        pos, rotmat = pose
        tcp_pos = pos + rotmat.dot(grasp.ac_pos)
        tcp_rotmat = rotmat.dot(grasp.ac_rotmat)
        no_ik = 0
        collided = 0
        for direction in directions:
            vec = -tcp_rotmat[:, 2] if direction is None else np.asarray(direction, dtype=float)
            unit = rm.unit_vector(vec)
            for distance in self.standoff_schedule:
                jnt_values = self.robot.ik(tgt_pos=tcp_pos + unit * distance,
                                           tgt_rotmat=tcp_rotmat,
                                           seed_jnt_values=seed)
                if jnt_values is None or not self._within_limits(jnt_values):
                    no_ik += 1
                    continue
                self._goto(jnt_values, ee_values=ee_values)
                if self.robot.is_collided(obstacle_list=obstacle_list):
                    collided += 1
                    continue
                return jnt_values, distance, ""
        return None, None, f"no ik x{no_ik}, collided x{collided}"

    def _jaw_release_widths(self, grasp) -> List[float]:
        """Openings to try when letting go, smallest first.

        Opening to the maximum is the obvious choice and the one WRS defaults to, but a plate seated
        between four posts leaves very little room, and fully spread fingers tend to hit a post. The
        gripper only has to open far enough to lose contact with the part, so the smaller openings
        are tried first and the maximum last.
        """
        jaw_max = float(self.robot.end_effector.jaw_range[1])
        grasp_width = float(grasp.ee_values)
        widths = [grasp_width + 0.006, grasp_width + 0.015, jaw_max]
        return [w for i, w in enumerate(widths)
                if w <= jaw_max + 1e-9 and w not in widths[:i]]

    def _interpolate_seating(self, start_conf, goal_conf, strict_obs, mating_mesh_obs,
                             ee_values, obj_template, goal_pose):
        """Joint-space seating with posts always in the arm obstacle set + mesh held-object checks."""
        if np.allclose(start_conf, goal_conf):
            self._goto(goal_conf, ee_values=ee_values)
            if not self._robot_pose_clear(strict_obs, ee_values):
                return None, "seating: arm collided at goal"
            if not waypoint_seating_valid(
                    self.robot, mating_mesh_obs, is_final=True,
                    obj_template=obj_template, goal_pose=goal_pose):
                return None, "seating: held object mesh penetrates mating partners at goal"
            mot_data = motd.MotionData(robot=self.robot)
            mot_data.extend(jv_list=[np.asarray(goal_conf, dtype=float)])
            return mot_data, ""

        interpolated = rm.interpolate_vectors(
            start_vector=start_conf, end_vector=goal_conf, granularity=self.interp_granularity)
        clipped = np.clip(interpolated, self.robot.jnt_ranges[:, 0], self.robot.jnt_ranges[:, 1])
        jv_list = []
        n = len(clipped)
        for i, jnt_values in enumerate(clipped):
            is_final = (i == n - 1)
            self._goto(jnt_values, ee_values=ee_values)
            if not self._robot_pose_clear(strict_obs, ee_values):
                return None, "seating: arm/gripper collided with obstacles (posts included)"
            if not waypoint_seating_valid(
                    self.robot, mating_mesh_obs, is_final=is_final,
                    obj_template=obj_template, goal_pose=goal_pose if is_final else None):
                return None, ("seating: held object penetrates mating partner mesh "
                              f"at waypoint {i + 1}/{n}")
            jv_list.append(jnt_values)
        mot_data = motd.MotionData(robot=self.robot)
        mot_data.extend(jv_list=jv_list)
        return mot_data, ""

    def _robot_pose_clear(self, obstacle_list, ee_values) -> bool:
        if self.robot.is_collided(obstacle_list=obstacle_list):
            return False
        ee = self.robot.end_effector
        if ee is not None and ee.is_mesh_collided(cmodel_list=obstacle_list):
            return False
        return True

    def _interpolate(self, start_conf, goal_conf, obstacle_list, ee_values):
        """Straight line in *joint* space. No IK, so none of the Cartesian failure modes apply."""
        if np.allclose(start_conf, goal_conf):
            mot_data = motd.MotionData(robot=self.robot)
            self._set_jaw(ee_values)
            mot_data.extend(jv_list=[np.asarray(goal_conf, dtype=float)])
            return mot_data
        # While the object is held its width is already fixed, and passing it on would trip the
        # "hand is holding objects" assertion inside change_ee_values.
        return self._interp.gen_interplated_between_given_conf(
            start_jnt_values=start_conf,
            end_jnt_values=goal_conf,
            obstacle_list=obstacle_list,
            granularity=self.interp_granularity,
            ee_values=None if self._hand_is_full() else ee_values)

    def _rrt(self, start_conf, goal_conf, obstacle_list, ee_values):
        ee_bk = self.robot.get_ee_values()
        self._set_jaw(ee_values)
        try:
            return self._rrtc.plan(start_conf=start_conf,
                                   goal_conf=goal_conf,
                                   obstacle_list=obstacle_list,
                                   ext_dist=self.rrt_ext_dist,
                                   max_time=self.rrt_max_time,
                                   toggle_dbg=False)
        finally:
            self._set_jaw(ee_bk)

    def _contact_is_tail(self, mot_data, obstacle_list, ee_values, want: str):
        """Require the colliding waypoints to be a contiguous run at one end of the segment.

        ``want="tail"`` allows contact only while arriving (approach, seating); ``want="head"``
        allows it only while leaving (retract). Anything else -- contact in the middle, or contact
        that starts, stops and starts again -- means the path goes through something, and is
        rejected. This is what the mandatory straight-line segments were there to prevent, kept as
        an explicit property of the path rather than as a constraint on its shape.
        """
        jv_list = mot_data.jv_list
        if not jv_list:
            return False, 0
        ee_bk = self.robot.get_ee_values()
        flags = []
        try:
            for jnt_values in jv_list:
                self._goto(jnt_values, ee_values=ee_values)
                flags.append(bool(self.robot.is_collided(obstacle_list=obstacle_list)))
        finally:
            self._set_jaw(ee_bk)
        n_contact = sum(flags)
        if n_contact == 0:
            return True, 0
        idx = [i for i, f in enumerate(flags) if f]
        contiguous = (idx[-1] - idx[0] + 1) == len(idx)
        if not contiguous:
            return False, n_contact
        if want == "tail":
            return idx[-1] == len(flags) - 1, n_contact
        return idx[0] == 0, n_contact

    def _assemble(self, cand, obj_cmodel, obj_at_start, obj_at_goal,
                  reach_jv, carry_jv, retract_jv, jaw_open, jaw_release):
        """Concatenate the phases into one MotionData with the object drawn along the way."""
        self.robot.goto_given_conf(reach_jv[0])
        self._set_jaw(jaw_open)
        mot_data = motd.MotionData(robot=self.robot)
        mot_data.extend(jv_list=reach_jv)
        for mesh in mot_data.mesh_list:
            if mesh is not None:
                obj_at_start.copy().attach_to(mesh)
        # carry: the object is held, so the robot meshes already include it
        obj_held = obj_cmodel.copy()
        obj_held.pose = (obj_at_start.pos, obj_at_start.rotmat)
        self.robot.goto_given_conf(cand.q_pick)
        self._empty_hand()
        self.robot.hold(obj_cmodel=obj_held, jaw_width=cand.grasp.ee_values)
        carry = motd.MotionData(robot=self.robot)
        carry.extend(jv_list=carry_jv)
        mot_data += carry
        # retract: the object stays behind at the goal pose
        self.robot.goto_given_conf(cand.q_place)
        self.robot.release(obj_cmodel=obj_held)
        self._set_jaw(jaw_release)
        tail = motd.MotionData(robot=self.robot)
        tail.extend(jv_list=retract_jv)
        for mesh in tail.mesh_list:
            if mesh is not None:
                obj_at_goal.copy().attach_to(mesh)
        mot_data += tail
        return mot_data
