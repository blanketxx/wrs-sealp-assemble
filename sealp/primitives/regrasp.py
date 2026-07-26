"""
Single-Arm Regrasp Primitive
=============================

Same interface as :class:`~sealp.primitives.transport.TransportPrimitive`, but it can change
grasp part-way by putting the object down on the table and picking it up again. One arm does
everything: ``pick -> (carry -> put down -> re-grasp)* -> carry -> place``.

Why this exists
---------------
``TransportPrimitive`` wraps ``PickPlacePlanner.gen_pick_and_place()``, which starts from
``reason_common_gids([staging_pose, goal_pose])`` -- the *intersection* of the grasps that are
IK-feasible and collision-free at the staging pose **and** at the goal pose. A single grasp then
has to survive the whole chain. When the staging orientation and the goal orientation differ by a
large rotation the intersection is empty (or the survivors cannot complete the linear seating and
release segments), and the planner reports "no valid grasp/path found" without ever considering
that the object could be set down and picked up differently. The tower's ``middle_plate`` is
exactly this case: it is staged standing on a narrow edge but must land flat on four posts.

This planner uses the regrasp graph from :mod:`wrs.manipulation.flatsurface_regrasp` instead:

* nodes are ``(object pose, grasp, arm configuration)`` triples;
* the staging pose and the goal pose contribute the *union* of their feasible grasps, so no grasp
  has to serve both ends;
* every re-grasp spot contributes each stable flat-surface orientation of the object;
* ``transit`` edges join two grasps at the same object pose -- release, re-approach, re-grasp;
* ``transfer`` edges join one grasp at two object poses -- carry the object;
* a graph search returns a sequence of carries and re-grasps, and whenever a segment turns out to
  be unplannable the offending node or edge is deleted and the search runs again.

The reorientation is therefore split across several carries, each of which only needs one grasp
feasible at its own two poses. That is a strictly weaker requirement than one grasp for the whole
task, which is why paths exist here that ``gen_pick_and_place`` cannot find.

Nothing is relaxed to get there: every segment is planned against the same obstacle sets that
``TransportPrimitive`` uses, the initial pick approach and the final release depart are both
planned (``toggle_start_approach``/``toggle_end_depart``), re-grasp spots are rejected when the
object would intersect the scene there, and the arm holds the object with a real grasp throughout.

This is a single-arm planner. There is no sender/receiver pair, no handover edge, and no second
robot anywhere in the graph.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import networkx
import numpy as np

import wrs.basis.robot_math as rm
import wrs.manipulation.flatsurface_regrasp as fsreg
import wrs.manipulation.placement.flatsurface as mp_fsp

from .base import MotionPrimitive, PrimitiveResult

Pose = Tuple[np.ndarray, np.ndarray]

# Re-grasp spot offsets relative to the staging position, in metres, tried in this order. The
# first one is the staging cell itself: the object just left it, so it is guaranteed clear and
# known to be reachable, which makes it the cheapest place to turn the object over.
DEFAULT_SPOT_OFFSETS: Tuple[Tuple[float, float], ...] = (
    (0.00, 0.00),
    (0.00, -0.07),
    (0.00, 0.07),
    (-0.07, 0.00),
    (0.07, 0.00),
)
# Yaw of the re-grasp spot frame. Two values give the arm a second wrist solution for the same
# stable orientation without doubling the number of spots.
DEFAULT_SPOT_ROTZ: Tuple[float, ...] = (0.0, rm.pi / 2)


class _QuietRegraspGraph(fsreg.FSRegraspPlanner):
    """``FSRegraspPlanner`` with the interactive matplotlib graph plots removed.

    ``plan_by_obj_poses`` calls ``show_graph()`` on every iteration, which opens a blocking
    matplotlib window. That is unusable inside a batch L3 sweep and crashes on a headless host.
    """

    def show_graph(self):
        pass

    def show_graph_with_path(self, path):
        pass


class SingleArmRegraspPrimitive(MotionPrimitive):
    """Single-arm pick / carry / re-grasp / place primitive.

    Parameters
    ----------
    robot : SglArmRobotInterface
        One robot arm. The same arm performs every segment.
    spot_offsets
        Re-grasp spot positions as ``(dx, dy)`` offsets from the staging position.
    spot_rotz
        Yaw values applied to each spot frame.
    max_repairs
        How many times a failed node/edge may be pruned before giving up.
    time_budget_s
        Wall-clock ceiling for one ``plan()`` call.
    """

    def __init__(self,
                 robot,
                 spot_offsets: Sequence[Tuple[float, float]] = DEFAULT_SPOT_OFFSETS,
                 spot_rotz: Sequence[float] = DEFAULT_SPOT_ROTZ,
                 max_repairs: int = 12,
                 time_budget_s: float = 240.0):
        self.robot = robot
        self.spot_offsets = tuple(spot_offsets)
        self.spot_rotz = tuple(spot_rotz)
        self.max_repairs = int(max_repairs)
        self.time_budget_s = float(time_budget_s)
        self._fs_pose_cache = {}

    # ------------------------------------------------------------------
    # stable flat-surface orientations of the object (cached per mesh)
    # ------------------------------------------------------------------
    def _fs_reference_poses(self, obj_cmodel):
        key = getattr(obj_cmodel, "name", None) or id(obj_cmodel)
        cached = self._fs_pose_cache.get(key)
        if cached is None:
            cached = mp_fsp.FSReferencePoses(obj_cmodel=obj_cmodel.copy())
            self._fs_pose_cache[key] = cached
        return cached

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
        """Plan one single-arm transport that may re-grasp on the way.

        ``goal_pose_list`` must hold exactly one pose: a re-grasp sequence has no meaning for a
        chain of intermediate goals, and every caller in this project passes a single goal.

        Recognised keyword arguments
        ----------------------------
        grasp_obstacle_list
            Contact-exempt obstacle set for grasp reasoning and for the linear seating/release
            segments, matching ``TransportPrimitive``. Transit and RRT segments keep
            ``obstacle_list``.
        place_approach_direction_list, place_depart_direction_list
            Assembly mating axis for the final seating carry and the release retract. Without
            them the generic top-down put-down is used, which would not verify the real
            insertion direction.
        """
        if obstacle_list is None:
            obstacle_list = []
        if len(goal_pose_list) != 1:
            return PrimitiveResult(
                success=False,
                error_msg=("SingleArmRegraspPrimitive supports exactly one goal pose, "
                           f"got {len(goal_pose_list)}."),
            )
        if grasp_collection is None or len(grasp_collection) == 0:
            return PrimitiveResult(success=False, error_msg="empty grasp collection")

        goal_pose = (np.asarray(goal_pose_list[0][0], dtype=float),
                     np.asarray(goal_pose_list[0][1], dtype=float))
        start_pose = (np.asarray(obj_cmodel.pos, dtype=float).copy(),
                      np.asarray(obj_cmodel.rotmat, dtype=float).copy())
        if end_jnt_values is None:
            end_jnt_values = self.robot.get_jnt_values()
        if start_jnt_values is None:
            start_jnt_values = self.robot.get_jnt_values()

        grasp_obstacle_list = kwargs.get("grasp_obstacle_list")
        goal_obs = grasp_obstacle_list if grasp_obstacle_list is not None else None
        approach_dirs = kwargs.get("place_approach_direction_list") or [None]
        depart_dirs = kwargs.get("place_depart_direction_list") or [None]
        goal_approach_direction = approach_dirs[-1]
        goal_depart_direction = depart_dirs[-1]
        # Re-grasp spots sit on the same surface the staged parts stand on.
        spot_z = float(start_pose[0][2])
        linear_distance = float(max(depart_distance, 0.01))

        t0 = time.time()
        try:
            planner = _QuietRegraspGraph(robot=self.robot,
                                         obj_cmodel=obj_cmodel.copy(),
                                         fs_reference_poses=self._fs_reference_poses(obj_cmodel),
                                         reference_gc=grasp_collection)
        except Exception as e:
            return PrimitiveResult(success=False,
                                   error_msg=f"regrasp graph init failed: {type(e).__name__}: {e!r}")

        # ------------------------------------------------------------------
        # re-grasp spots: each contributes every stable orientation of the object
        # ------------------------------------------------------------------
        n_spot_poses = 0
        for dx, dy in self.spot_offsets:
            if time.time() - t0 > self.time_budget_s:
                break
            spot_pos = np.array([start_pose[0][0] + dx, start_pose[0][1] + dy, spot_z], dtype=float)
            for rotz in self.spot_rotz:
                try:
                    spot = planner.create_add_fsregspot(spot_pos=spot_pos,
                                                        spot_rotz=float(rotz),
                                                        barrier_z_offset=-0.01,
                                                        consider_robot=True,
                                                        toggle_dbg=False,
                                                        extra_obstacle_list=obstacle_list)
                except Exception as e:
                    print(f"  [regrasp] spot {np.round(spot_pos, 3).tolist()} rotz={rotz:.2f} "
                          f"failed: {type(e).__name__}: {e!r}")
                    continue
                n_spot_poses += len(spot.fspg_list)
        if n_spot_poses == 0:
            return PrimitiveResult(
                success=False,
                error_msg="no reachable re-grasp spot pose (object blocked or out of reach)",
            )

        # ------------------------------------------------------------------
        # endpoints: union of feasible grasps, no intersection requirement
        # ------------------------------------------------------------------
        obs_for_grasp = goal_obs if goal_obs is not None else obstacle_list
        start_nodes = planner.add_start_pose(obj_pose=start_pose, obstacle_list=obstacle_list) or []
        goal_nodes = planner.add_goal_pose(obj_pose=goal_pose, obstacle_list=obs_for_grasp) or []
        if not start_nodes:
            return PrimitiveResult(success=False, error_msg="no feasible grasp at the staging pose")
        if not goal_nodes:
            return PrimitiveResult(success=False, error_msg="no feasible grasp at the goal pose")

        print(f"  [regrasp] spot poses={n_spot_poses} start grasps={len(start_nodes)} "
              f"goal grasps={len(goal_nodes)} nodes={planner.graph.number_of_nodes()}")

        # ------------------------------------------------------------------
        # search, then repair the graph wherever a segment cannot be planned
        # ------------------------------------------------------------------
        start_nodes = list(start_nodes)
        goal_nodes = list(goal_nodes)
        last_err = "no regrasp path"
        for attempt in range(self.max_repairs):
            if time.time() - t0 > self.time_budget_s:
                last_err = f"time budget {self.time_budget_s:.0f}s exhausted after {attempt} repairs"
                break
            path = self._shortest_path(planner.graph, start_nodes, goal_nodes)
            if path is None:
                last_err = "no regrasp path" if attempt == 0 else f"graph exhausted after {attempt} repairs"
                break
            n_regrasps = sum(1 for k in range(1, len(path))
                             if planner.graph.edges[(path[k - 1], path[k])]["type"].endswith("transit"))
            try:
                result = planner.gen_regrasp_motion(
                    path=path,
                    obstacle_list=obstacle_list,
                    start_jnt_values=start_jnt_values,
                    linear_distance=linear_distance,
                    toggle_start_approach=True,
                    toggle_end_depart=True,
                    toggle_dbg=False,
                    goal_approach_direction=goal_approach_direction,
                    goal_depart_direction=goal_depart_direction,
                    goal_obstacle_list=goal_obs,
                )
            except Exception as e:
                last_err = f"gen_regrasp_motion raised: {type(e).__name__}: {e!r}"
                break

            tag = result[0]
            if tag.startswith("s"):
                mot_data = result[1]
                print(f"  [regrasp] solved with {n_regrasps} re-grasp(s), "
                      f"{len(path)} graph nodes, {len(mot_data.jv_list)} waypoints")
                return PrimitiveResult(success=True,
                                       mot_data=mot_data,
                                       end_jnt_values=mot_data.jv_list[-1])
            last_err = f"regrasp {tag}"
            if tag.startswith("n"):
                self._drop_nodes(planner.graph, [result[1]], start_nodes, goal_nodes)
            elif tag.startswith("e"):
                self._drop_nodes(planner.graph, list(result[1]), start_nodes, goal_nodes)
            else:
                break
            if not start_nodes or not goal_nodes:
                last_err = f"{last_err}; all endpoint grasps exhausted"
                break
        return PrimitiveResult(success=False, error_msg=f"SingleArmRegrasp failed: {last_err}")

    # ------------------------------------------------------------------
    @staticmethod
    def _shortest_path(graph, start_nodes, goal_nodes):
        """Fewest-hops start->goal path, i.e. the fewest carries and re-grasps."""
        best = None
        for start in start_nodes:
            if start not in graph:
                continue
            try:
                paths = networkx.single_source_shortest_path(graph, start)
            except Exception:
                continue
            for goal in goal_nodes:
                path = paths.get(goal)
                if path is not None and (best is None or len(path) < len(best)):
                    best = path
        return best

    @staticmethod
    def _drop_nodes(graph, nodes, start_nodes, goal_nodes):
        for node in nodes:
            if node in graph:
                graph.remove_node(node)
            if node in start_nodes:
                start_nodes.remove(node)
            if node in goal_nodes:
                goal_nodes.remove(node)
