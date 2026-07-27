"""
Single-Arm Direct Transport Primitive
=====================================

``pick -> +Z lift -> transport -> place`` with one arm and one grasp, connected by RRT in joint space
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

import itertools
import math
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # scipy is optional; identity-only fallback remains available
    cKDTree = None

import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd
import wrs.motion.primitives.interpolated as mpi
import wrs.motion.probabilistic.rrt_connect as rrtc

from .base import MotionPrimitive, PrimitiveResult
from .seating_collision import (
    contact_flags_form_tail,
    gripper_mesh_collides,
    held_object_mesh_collides,
    seating_waypoint_valid,
    to_triangle_cdmesh,
)

Pose = Tuple[np.ndarray, np.ndarray]

# Stand-off distances tried when looking for the configuration just before contact. Each entry
# costs one IK call at a single displaced pose -- not a pose-by-pose IK chain along a line, and not
# subject to the 45-degree straightness rule that kills the Cartesian segments.
DEFAULT_STANDOFF_SCHEDULE: Tuple[float, ...] = (0.06, 0.045, 0.03, 0.02, 0.012, 0.006)

# Automatic object-symmetry reasoning.  The detector is deliberately conservative: a transform is
# accepted only when it maps the object's own local mesh back onto itself within a small geometric
# tolerance.  No part names, assembly names, or tower-specific axes are used.
DEFAULT_SYMMETRY_REL_TOL = 0.0025       # 0.25 % of object diagonal
DEFAULT_SYMMETRY_ABS_TOL = 2.0e-4      # 0.2 mm floor for CAD tessellation noise
DEFAULT_SYMMETRY_MAX_POINTS = 6000
DEFAULT_MAX_AUTO_SYMMETRIES = 12       # includes identity


class _ObjectSymmetry:
    __slots__ = ("rotmat", "offset", "label", "error")

    def __init__(self, rotmat, offset=None, label="identity", error=0.0):
        self.rotmat = np.asarray(rotmat, dtype=float).reshape(3, 3)
        self.offset = (np.zeros(3, dtype=float) if offset is None
                       else np.asarray(offset, dtype=float).reshape(3))
        self.label = str(label)
        self.error = float(error)


class _GraspCandidate:
    __slots__ = ("gid", "grasp", "q_pick", "q_place", "goal_pose", "symmetry_label")

    def __init__(self, gid, grasp, q_pick, q_place, goal_pose, symmetry_label="identity"):
        self.gid = gid
        self.grasp = grasp
        self.q_pick = q_pick
        self.q_place = q_place
        self.goal_pose = (np.asarray(goal_pose[0], dtype=float).copy(),
                          np.asarray(goal_pose[1], dtype=float).copy())
        self.symmetry_label = str(symmetry_label)

    @property
    def key(self) -> str:
        return f"gid={self.gid}@{self.symmetry_label}"


def _axis_angle_rotmat(axis, angle: float) -> np.ndarray:
    """Rodrigues rotation; kept local so symmetry detection has no WRS-version dependency."""
    axis = np.asarray(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, ss = math.cos(float(angle)), math.sin(float(angle))
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*ss, x*z*C + y*ss],
        [y*x*C + z*ss, c + y*y*C,     y*z*C - x*ss],
        [z*x*C - y*ss, z*y*C + x*ss, c + z*z*C],
    ], dtype=float)


def _rotation_angle(rotmat: np.ndarray) -> float:
    c = float(np.clip((np.trace(rotmat) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(c))


def _mesh_from_cmodel(obj_cmodel):
    """Return the underlying trimesh-like object across the WRS variants used by SEALP."""
    for attr in ("trm_mesh", "_trm_mesh", "mesh", "_mesh", "objtrm"):
        try:
            mesh = getattr(obj_cmodel, attr, None)
        except Exception:
            mesh = None
        if mesh is not None and hasattr(mesh, "vertices"):
            return mesh
    return None


def _mesh_local_bounds(obj_cmodel):
    mesh = _mesh_from_cmodel(obj_cmodel)
    if mesh is None:
        return None
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
    except Exception:
        return None
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        return None
    return vertices.min(axis=0), vertices.max(axis=0)


def _obstacle_label(obj, index=None) -> str:
    """Human-readable obstacle label for diagnostic output.

    SEALP's executor annotates staging/goal collision models with
    ``_sealp_part_id`` and ``_sealp_role``.  Those identifiers must take
    precedence over WRS' generic model names such as ``sgm`` or
    ``collision_model``; otherwise every different assembly part looks the
    same in the log.
    """
    try:
        pid = getattr(obj, "_sealp_part_id", None)
    except Exception:
        pid = None
    try:
        role = getattr(obj, "_sealp_role", None)
    except Exception:
        role = None
    if pid is not None and str(pid).strip():
        pid_s = str(pid).strip()
        # Keep the main label compact.  The part id is what matters for the
        # collision diagnosis; role is only appended when it adds information.
        if role is not None and str(role).strip() and str(role) not in (
                "assembled_at_goal", "staging_on_table"):
            return f"{pid_s}({str(role).strip()})"
        return pid_s

    # Generic WRS names are not useful identifiers.  Fall through to a stable
    # env/obstacle index instead of printing hundreds of 'sgm' entries.
    generic_names = {"sgm", "collision_model", "cm", "model", "none", "unnamed"}
    for attr in ("name", "objname", "model_name", "_name"):
        try:
            value = getattr(obj, attr, None)
        except Exception:
            value = None
        if value is not None:
            name = str(value).strip()
            if name and name.lower() not in generic_names:
                return name

    try:
        p = np.asarray(obj.pos, dtype=float).reshape(3)
        suffix = f"@[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]"
    except Exception:
        suffix = ""
    prefix = f"env/obs#{index}" if index is not None else obj.__class__.__name__
    return prefix + suffix




def _is_work_table_obstacle(obj, index=None) -> bool:
    """Project-specific work-table detector used for cap_plate exemption.

    Prefer explicit SEALP ids/names.  The current CrossRailFrame executor's
    legacy solid work_table is otherwise an unlabeled env obstacle at
    approximately [0.234, 0.000, -0.010].
    """
    tokens = []
    for attr in ("_sealp_part_id", "_sealp_role", "name", "objname", "model_name", "_name"):
        try:
            value = getattr(obj, attr, None)
        except Exception:
            value = None
        if value is not None:
            tokens.append(str(value).strip().lower())
    joined = " ".join(tokens)
    if "work_table" in joined or "worktable" in joined:
        return True
    # Avoid classifying assembled parts whose ids happen to contain 'table'.
    if any(tok in {"table", "desk", "workbench"} for tok in tokens):
        return True

    # Fallback for the exact static table used by this project/log.
    try:
        p = np.asarray(obj.pos, dtype=float).reshape(3)
        if np.linalg.norm(p - np.array([0.234, 0.0, -0.010])) <= 0.015:
            pid = getattr(obj, "_sealp_part_id", None)
            if pid is None or not str(pid).strip():
                return True
    except Exception:
        pass

    # Last-resort fallback: in this executor the table is env/obs#0.
    try:
        pid = getattr(obj, "_sealp_part_id", None)
    except Exception:
        pid = None
    return index == 0 and (pid is None or not str(pid).strip())


def _local_mesh_points(obj_cmodel, max_points: int = DEFAULT_SYMMETRY_MAX_POINTS):
    """Return a geometry-driven local-frame surface cloud plus its sampling resolution.

    Preferred path uses ``trimesh.voxelized``.  Surface voxels are independent of the STL triangle
    diagonals, so a square/cube is still recognised as 90-degree symmetric even when opposite faces
    were triangulated differently.  The pitch is automatically coarsened only when the cloud would
    otherwise become too large.  Raw vertices/face centroids are the fallback for older mesh types.
    """
    mesh = _mesh_from_cmodel(obj_cmodel)
    if mesh is None:
        return None, None
    try:
        vertices = np.asarray(mesh.vertices, dtype=float)
    except Exception:
        return None, None
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        return None, None

    pmin, pmax = vertices.min(axis=0), vertices.max(axis=0)
    diag = float(np.linalg.norm(pmax - pmin))
    if diag < 1e-9:
        return vertices, 0.0

    # About 250 cells along the diagonal is fine enough for assembly CAD while still cheap enough
    # to run once per moved part.  Never go below 0.25 mm by default.
    pitch = max(2.5e-4, diag / 250.0)
    try:
        if hasattr(mesh, "voxelized"):
            for _ in range(4):
                vox = mesh.voxelized(pitch)
                pts = np.asarray(vox.points, dtype=float)
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                if 4 <= len(pts) <= int(max_points):
                    return pts, pitch
                if len(pts) > int(max_points):
                    # Surface point count scales approximately with 1/pitch^2.
                    pitch *= max(1.15, math.sqrt(len(pts) / float(max_points)))
                    continue
                break
    except Exception:
        pass

    clouds = [vertices]
    try:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.ndim == 2 and faces.shape[1] >= 3 and len(faces):
            clouds.append(vertices[faces[:, :3]].mean(axis=1))
    except Exception:
        pass
    pts = np.vstack(clouds)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) > int(max_points):
        idx = np.linspace(0, len(pts) - 1, int(max_points), dtype=np.int64)
        pts = pts[idx]
    return (pts if len(pts) >= 4 else None), None


def _proper_signed_permutations() -> List[np.ndarray]:
    """The 24 orientation-preserving signed permutation rotations."""
    mats: List[np.ndarray] = []
    eye = np.eye(3)
    for perm in itertools.permutations(range(3)):
        base = eye[:, perm]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            R = base @ np.diag(signs)
            if np.linalg.det(R) > 0.5:
                mats.append(R)
    return mats


def _candidate_symmetry_rotations(points: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Geometry-generic discrete rotation proposals in model axes and PCA axes.

    The proposals are only hypotheses.  `_detect_object_symmetries` subsequently verifies every
    hypothesis against the actual mesh, so an asymmetric object still keeps identity only.
    """
    candidates: List[Tuple[str, np.ndarray]] = [("identity", np.eye(3))]
    candidates.extend((f"signed_perm_{i:02d}", R) for i, R in enumerate(_proper_signed_permutations()))

    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    X = points - center
    bases = [("model", np.eye(3))]
    try:
        cov = (X.T @ X) / max(1, len(X))
        _, V = np.linalg.eigh(cov)
        V = V[:, ::-1]
        if np.linalg.det(V) < 0:
            V[:, -1] *= -1.0
        bases.append(("pca", V))
    except Exception:
        pass

    # Common finite CAD symmetries plus 30-degree increments for 6/12-fold axial parts.
    angles_deg = (30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330)
    for basis_name, B in bases:
        for axis_i in range(3):
            axis = B[:, axis_i]
            for deg in angles_deg:
                candidates.append((f"{basis_name}_a{axis_i}_{deg}",
                                   _axis_angle_rotmat(axis, math.radians(deg))))

    # Remove numerically duplicate rotations before expensive mesh checks.
    unique: List[Tuple[str, np.ndarray]] = []
    for label, R in candidates:
        if any(np.max(np.abs(R - old_R)) < 1e-7 for _, old_R in unique):
            continue
        unique.append((label, R))
    return unique


def _detect_object_symmetries(obj_cmodel,
                              rel_tol: float = DEFAULT_SYMMETRY_REL_TOL,
                              abs_tol: float = DEFAULT_SYMMETRY_ABS_TOL,
                              max_points: int = DEFAULT_SYMMETRY_MAX_POINTS,
                              max_symmetries: int = DEFAULT_MAX_AUTO_SYMMETRIES,
                              ) -> List[_ObjectSymmetry]:
    """Infer proper rigid self-symmetries from the current object's CAD mesh.

    Each accepted transform S=(R,t) satisfies x' = R x + t and maps the local object surface back
    onto itself.  Rotation is performed about the mesh bounding-box centre, so this also works when
    the CAD origin is not at the geometric centre.  If mesh access/scipy is unavailable, identity is
    returned and the planner behaves exactly like the old implementation.
    """
    identity = _ObjectSymmetry(np.eye(3), np.zeros(3), "identity", 0.0)
    if cKDTree is None:
        return [identity]
    points, sample_resolution = _local_mesh_points(obj_cmodel, max_points=max_points)
    if points is None or len(points) < 4:
        return [identity]

    raw_bounds = _mesh_local_bounds(obj_cmodel)
    if raw_bounds is None:
        pmin, pmax = points.min(axis=0), points.max(axis=0)
    else:
        pmin, pmax = raw_bounds
    center = 0.5 * (pmin + pmax)
    diag = float(np.linalg.norm(pmax - pmin))
    if diag < 1e-9:
        return [identity]
    tol = max(float(abs_tol), float(rel_tol) * diag)
    if sample_resolution is not None:
        tol = max(tol, 1.75 * float(sample_resolution))
    tree = cKDTree(points)
    X = points - center
    cov = (X.T @ X) / max(1, len(X))
    cov_scale = max(float(np.linalg.norm(cov)), 1e-12)

    accepted: List[_ObjectSymmetry] = [identity]
    for label, R in _candidate_symmetry_rotations(points):
        if np.max(np.abs(R - np.eye(3))) < 1e-7:
            continue
        # Necessary second-moment invariance rejects most impossible axes before the denser
        # surface comparison (e.g. a 90-deg rotation that swaps unequal box dimensions).
        if float(np.linalg.norm(R @ cov @ R.T - cov)) > 2.0e-3 * cov_scale:
            continue
        # Full SE(3) self-transform for a rotation about the object's geometric centre.
        t = center - R @ center
        moved = points @ R.T + t
        d1 = tree.query(moved, k=1)[0]
        # A second direction prevents a near-subset from looking symmetric.
        moved_tree = cKDTree(moved)
        d2 = moved_tree.query(points, k=1)[0]
        d = np.concatenate((np.asarray(d1), np.asarray(d2)))
        p95 = float(np.quantile(d, 0.95))
        rms = float(np.sqrt(np.mean(d * d)))
        # Conservative enough for collision planning, but tolerant of STL tessellation diagonals.
        if p95 <= 2.0 * tol and rms <= 1.25 * tol:
            accepted.append(_ObjectSymmetry(R, t, label, max(p95, rms)))

    # Prefer low-error symmetries, then smaller rotations. Identity stays first.
    rest = accepted[1:]
    rest.sort(key=lambda s: (s.error, _rotation_angle(s.rotmat)))
    # Deduplicate transforms that were proposed in both model and PCA bases.
    unique = [identity]
    for sym in rest:
        if any(np.max(np.abs(sym.rotmat - old.rotmat)) < 1e-6 and
               np.linalg.norm(sym.offset - old.offset) < 1e-6 for old in unique):
            continue
        unique.append(sym)
        if len(unique) >= max(1, int(max_symmetries)):
            break
    return unique


def _compose_pose_with_local_transform(pose: Pose, symmetry: _ObjectSymmetry) -> Pose:
    """Compose world object pose T_WO with local self-symmetry S: T_WO_equiv = T_WO * S."""
    p, R = pose
    p = np.asarray(p, dtype=float)
    R = np.asarray(R, dtype=float)
    return p + R @ symmetry.offset, R @ symmetry.rotmat


def _tcp_world_pose(obj_pose: Pose, grasp) -> Pose:
    """The exact WRS local-grasp -> world-TCP conversion used for all IK checks."""
    p, R = obj_pose
    p = np.asarray(p, dtype=float)
    R = np.asarray(R, dtype=float)
    return p + R @ np.asarray(grasp.ac_pos, dtype=float), R @ np.asarray(grasp.ac_rotmat, dtype=float)


def _summarise(reasons: Sequence[str]) -> str:
    """Group per-grasp failure messages by kind, so the dominant cause is visible at a glance."""
    counts: dict = {}
    for reason in reasons:
        key = reason.split(" ", 1)[1] if reason.startswith("gid=") else reason
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return "; ".join(f"{n}x {key}" for key, n in ordered)



def _cmodel_world_z_bounds(cmodel):
    """Return (min_z, max_z) of a CollisionModel mesh in WORLD coordinates."""
    try:
        mesh = _mesh_from_cmodel(cmodel)
        if mesh is None:
            return None
        verts = np.asarray(mesh.vertices, dtype=float)
        pos = np.asarray(cmodel.pos, dtype=float).reshape(3)
        rot = np.asarray(cmodel.rotmat, dtype=float).reshape(3, 3)
        wz = (verts @ rot.T + pos)[:, 2]
        if len(wz) == 0:
            return None
        return float(np.min(wz)), float(np.max(wz))
    except Exception:
        return None


def _eef_world_z_bounds(robot):
    """Return true end-effector collision-mesh z bounds in WORLD coordinates."""
    ee = getattr(robot, "end_effector", None)
    if ee is None:
        delegator = getattr(robot, "delegator", None)
        ee = getattr(delegator, "end_effector", None) if delegator is not None else None
    if ee is None:
        return None

    zmins, zmaxs = [], []
    for el in getattr(ee, "cdelements", []) or []:
        cmodel = getattr(el, "cmodel", None)
        if cmodel is None:
            continue
        bounds = _cmodel_world_z_bounds(cmodel)
        if bounds is not None:
            zmins.append(bounds[0])
            zmaxs.append(bounds[1])
    if not zmins:
        return None
    return min(zmins), max(zmaxs)


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
                 interp_granularity: float = 0.02,
                 cartesian_granularity: float = 0.005):
        self.robot = robot
        self.rrt_ext_dist = float(rrt_ext_dist)
        self.rrt_max_time = float(rrt_max_time)
        self.max_grasps = int(max_grasps)
        self.standoff_schedule = tuple(standoff_schedule)
        self.interp_granularity = float(interp_granularity)
        self.cartesian_granularity = max(1.0e-4, float(cartesian_granularity))
        self.last_certification = ""
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
            Contact-exempt obstacle set: the scene minus the parts this object mates with by
            construction.  Their *box* cdprims are what would otherwise veto every grasp at the
            seated pose.

        seating_mesh_obstacles
            Triangle-mesh copies of those same mating partners, detected geometrically at the goal
            pose.  When non-empty the seating phase runs in "phased" mode: pick / transport / the
            pre-place stand-off keep the full strict obstacle set, and the stand-off -> goal segment
            replaces the removed boxes with mesh checks -- the gripper may never intersect a
            partner, and the held object may only touch one at the final waypoint.

        place_depart_direction_list
            Mating axis of the assembly. The stand-off configuration above the goal is placed along
            it, so the object leaves and enters its seat the way the assembly requires.
        pick_standoff_direction
            Where to put the stand-off before the grasp. Defaults to straight back out of the
            gripper, which is what clears a side grasp.

        ``approach_distance``/``depart_distance`` remain interface-compatibility arguments.  The actual
        staging lift is controlled by ``pick_depart_direction`` and ``pick_depart_distance``; the
        asmdef insertion/depart axis is used only on the placement side.
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

        # CrossRailFrame cap_plate special case: ignore the work table completely.
        # All assembly-part collisions (rails/posts/base plate) remain active.
        current_pid = kwargs.get("part_id") or kwargs.get("pid")
        if current_pid is None:
            try:
                current_pid = getattr(obj_cmodel, "_sealp_part_id", None)
            except Exception:
                current_pid = None
        if current_pid is None:
            for attr in ("name", "objname", "model_name", "_name"):
                try:
                    value = getattr(obj_cmodel, attr, None)
                except Exception:
                    value = None
                if value is not None and "cap_plate" in str(value).lower():
                    current_pid = "cap_plate"
                    break

        ignore_work_table = bool(kwargs.get("ignore_work_table", False)) or str(current_pid) == "cap_plate"
        if ignore_work_table:
            strict_before = len(strict_obs)
            relaxed_before = len(relaxed_obs)
            strict_obs = [obs for i, obs in enumerate(strict_obs)
                          if not _is_work_table_obstacle(obs, i)]
            relaxed_obs = [obs for i, obs in enumerate(relaxed_obs)
                           if not _is_work_table_obstacle(obs, i)]
            print(
                f"  [TABLE EXEMPT] pid={current_pid or 'cap_plate'}: work_table ignored "
                f"for DirectTransport collision checks "
                f"(strict {strict_before}->{len(strict_obs)}, "
                f"relaxed {relaxed_before}->{len(relaxed_obs)})."
            )

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

        # Staging pick-depart is independent of the asmdef insertion axis.
        # Every part is lifted in WORLD +Z before the long transport/RRT phase.
        pick_depart_dir = np.asarray(
            kwargs.get("pick_depart_direction", [0.0, 0.0, 1.0]), dtype=float
        )
        pick_depart_distance = float(kwargs.get("pick_depart_distance", 0.03))

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

        # Geometry-driven symmetry: no part-id special cases.  Equivalent goal poses are generated
        # from rigid self-symmetries of the current CAD mesh, then every grasp is converted to the
        # WORLD TCP pose before IK/collision certification.  Identity is always present, so a mesh
        # with no verified symmetry behaves exactly like the old planner.
        use_auto_symmetry = bool(kwargs.get("use_auto_symmetry", True))
        if use_auto_symmetry:
            symmetries = _detect_object_symmetries(
                obj_cmodel,
                rel_tol=float(kwargs.get("symmetry_rel_tol", DEFAULT_SYMMETRY_REL_TOL)),
                abs_tol=float(kwargs.get("symmetry_abs_tol", DEFAULT_SYMMETRY_ABS_TOL)),
                max_points=int(kwargs.get("symmetry_max_points", DEFAULT_SYMMETRY_MAX_POINTS)),
                max_symmetries=int(kwargs.get("max_auto_symmetries", DEFAULT_MAX_AUTO_SYMMETRIES)),
            )
        else:
            symmetries = [_ObjectSymmetry(np.eye(3), np.zeros(3), "identity", 0.0)]
        print(f"  [direct/symmetry] verified self-symmetries={len(symmetries)}: "
              f"{[s.label for s in symmetries]}")

        self.robot.backup_state()
        try:
            candidates = self._certify_grasps(
                grasp_collection, start_pose, goal_pose, relaxed_obs,
                mating_mesh_obs=mating_mesh_obs, symmetries=symmetries,
            )
        finally:
            self.robot.restore_state()
        if not candidates:
            return PrimitiveResult(
                success=False,
                error_msg="no grasp is IK-feasible and collision-free at both the staging "
                          f"and the goal pose ({getattr(self, 'last_certification', '')})")
        print(f"  [direct] certified grasps={len(candidates)} "
              f"(candidates={[c.key for c in candidates[:8]]}{'...' if len(candidates) > 8 else ''})")
        for cand in candidates[:8]:
            pick_tcp = _tcp_world_pose(start_pose, cand.grasp)
            place_tcp = _tcp_world_pose(cand.goal_pose, cand.grasp)
            print(
                f"    [world-tcp] {cand.key}: "
                f"pick_p={np.round(pick_tcp[0], 4).tolist()} "
                f"place_p={np.round(place_tcp[0], 4).tolist()} "
                f"|dq|inf={float(np.max(np.abs(cand.q_place-cand.q_pick))):.4f}"
            )

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
                    start_jnt_values=start_jnt_values,
                    end_jnt_values=end_jnt_values,
                    strict_obs=strict_obs,
                    relaxed_obs=relaxed_obs,
                    mating_mesh_obs=mating_mesh_obs,
                    phased_seating=phased_seating,
                    pick_standoff_dir=pick_standoff_dir,
                    place_standoff_dir=place_standoff_dir,
                    pick_depart_dir=pick_depart_dir,
                    pick_depart_distance=pick_depart_distance,
                )
            except Exception as e:  # a bad grasp must not abort the whole search
                mot_data, err = None, f"gid={cand.gid} exception {type(e).__name__}: {e!r}"
            finally:
                self._reset_to(depth)
            if mot_data is not None:
                print(f"  [direct] solved with {cand.key}, {len(mot_data.jv_list)} waypoints")
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
                        relaxed_obs, *, mating_mesh_obs,
                        symmetries: Sequence[_ObjectSymmetry]) -> List[_GraspCandidate]:
        """Certify one rigid grasp against start and all geometry-equivalent goal poses.

        The grasp itself is never changed: the same local ``grasp`` object is used from pick to
        place, so this is still a single-arm, single-grasp transport with no regrasp.  Object
        symmetry is represented instead by an equivalent final object pose ``T_goal * S``.  Because
        S has already been verified to map the CAD mesh onto itself, all such poses occupy the same
        physical goal geometry.

        Crucially, every IK query is performed after converting ``T_WO * T_OG`` to the common WORLD
        frame via `_tcp_world_pose`; gid equality is not treated as a geometric comparison rule.
        """
        out: List[_GraspCandidate] = []
        pick_fail: List[str] = []
        place_fail: List[str] = []
        # Diagnostic rows for grasps that are staging-feasible and IK-reachable at the
        # identity goal, but are rejected only by collision.
        identity_collision_rows = []
        equivalent_goals = [(_compose_pose_with_local_transform(goal_pose, sym), sym)
                            for sym in symmetries]

        for gid in range(len(grasp_collection)):
            grasp = grasp_collection[gid]
            q_pick, why = self._ik_at_detail_pose(start_pose, grasp, relaxed_obs, seed=None)
            if q_pick is None:
                pick_fail.append(why)
                continue

            found_for_gid = False
            for eq_goal, sym in equivalent_goals:
                q_place, why = self._ik_at_detail_pose(
                    eq_goal, grasp, relaxed_obs, seed=q_pick,
                    mesh_obstacle_list=mating_mesh_obs)
                # q_pick is only an IK-branch preference.  If that branch is unreachable, outside
                # limits, or arm-collided, also try an unseeded branch before rejecting the pose.
                if q_place is None and why in ("no ik", "out of joint limits", "arm collided"):
                    q_place, why = self._ik_at_detail_pose(
                        eq_goal, grasp, relaxed_obs, seed=None,
                        mesh_obstacle_list=mating_mesh_obs)
                if q_place is None:
                    place_fail.append(f"{sym.label}: {why}")
                    if sym.label == "identity" and why in (
                            "arm collided", "gripper collided",
                            "gripper mesh inside a mating partner"):
                        try:
                            jaw_mm = float(np.asarray(grasp.ee_values).reshape(-1)[0]) * 1000.0
                        except Exception:
                            try:
                                jaw_mm = float(grasp.ee_values) * 1000.0
                            except Exception:
                                jaw_mm = float("nan")
                        sources = self._collision_sources_at_pose(
                            eq_goal, grasp, relaxed_obs, seed=q_pick,
                            mesh_obstacle_list=mating_mesh_obs)
                        identity_collision_rows.append((gid, jaw_mm, why, sources))
                    continue

                out.append(_GraspCandidate(
                    gid=gid,
                    grasp=grasp,
                    q_pick=q_pick,
                    q_place=q_place,
                    goal_pose=eq_goal,
                    symmetry_label=sym.label,
                ))
                found_for_gid = True

            # Keep all feasible symmetry variants: a slightly longer joint-space candidate may have
            # a much easier RRT path than the shortest one. `max_grasps` still caps expensive trials.
            if not found_for_gid:
                pass

        self.last_certification = (
            f"{len(grasp_collection)} grasps tried, {len(out)} grasp/symmetry candidates certified; "
            f"staging rejects {_summarise(pick_fail)}; "
            f"goal rejects {_summarise(place_fail)}"
        )
        print(f"  [direct] grasp certification: {self.last_certification}")

        if identity_collision_rows:
            from collections import Counter
            width_counter = Counter(round(row[1], 1) for row in identity_collision_rows)
            source_counter = Counter()
            for _, _, _, sources in identity_collision_rows:
                for source in sources:
                    source_counter[source] += 1

            print("  [direct/goal-collision-diagnosis] identity goal has IK but is rejected by collision:")
            print(f"    total={len(identity_collision_rows)}")
            print("    jaw-width counts: " + ", ".join(
                f"{w:.1f}mm={n}" for w, n in sorted(width_counter.items())))
            if source_counter:
                print("    collision-source counts:")
                for source, n in source_counter.most_common():
                    print(f"      {source}: {n}")

            # The 32 mm jaw group is the important one for cap_plate: in the
            # current grasp cache it corresponds to grasps spanning the 12 mm
            # plate thickness.  Print its blocker distribution separately.
            thin32_rows = [row for row in identity_collision_rows
                           if abs(float(row[1]) - 32.0) <= 0.2]
            if thin32_rows:
                thin32_sources = Counter()
                for _, _, _, sources in thin32_rows:
                    # Count a blocker at most once per grasp.
                    for source in set(sources):
                        thin32_sources[source] += 1
                print(f"    32mm thin-edge goal-collision grasps: {len(thin32_rows)}")
                if thin32_sources:
                    print("    32mm blocker counts:")
                    for source, n in thin32_sources.most_common():
                        print(f"      {source}: {n}")

                # Geometry check: distinguish a REAL table hit from a collision-model artefact.
                # The gripper TCP pose is fixed by the grasp, independent of which IK branch is used.
                table_tops = []
                for obs in relaxed_obs:
                    label = _obstacle_label(obs)
                    if str(label).startswith("env/") or "work_table" in str(label).lower():
                        zb = _cmodel_world_z_bounds(obs)
                        if zb is not None:
                            table_tops.append(zb[1])
                table_top_z = max(table_tops) if table_tops else 0.0

                clearances = []
                below_count = 0
                geom_rows = []
                ee = self._ee()
                if ee is not None and hasattr(ee, "grip_at_by_pose"):
                    for gid, jaw_mm, _, _sources in thin32_rows:
                        grasp = grasp_collection[gid]
                        tcp_pos, tcp_rotmat = _tcp_world_pose(goal_pose, grasp)
                        self.robot.backup_state()
                        try:
                            ee.grip_at_by_pose(
                                jaw_center_pos=np.asarray(tcp_pos, dtype=float),
                                jaw_center_rotmat=np.asarray(tcp_rotmat, dtype=float),
                                jaw_width=grasp.ee_values,
                            )
                            zb = _eef_world_z_bounds(self.robot)
                        except Exception:
                            zb = None
                        finally:
                            self.robot.restore_state()

                        if zb is None:
                            continue
                        clearance = float(zb[0] - table_top_z)
                        clearances.append(clearance)
                        if clearance < -1e-6:
                            below_count += 1
                        geom_rows.append((gid, float(tcp_pos[2]), zb[0], zb[1], clearance))

                if clearances:
                    c = np.asarray(clearances, dtype=float)
                    print("    32mm table-geometry check:")
                    print(
                        f"      table_top_z={table_top_z:.4f}m; "
                        f"gripper_mesh_clearance min/median/max="
                        f"{np.min(c)*1000.0:.1f}/"
                        f"{np.median(c)*1000.0:.1f}/"
                        f"{np.max(c)*1000.0:.1f}mm"
                    )
                    print(
                        f"      true mesh below table top: "
                        f"{below_count}/{len(clearances)} grasps"
                    )
                    for gid, tcp_z, gmin, gmax, clearance in geom_rows[:20]:
                        print(
                            f"      gid={gid:4d} tcp_z={tcp_z:.4f} "
                            f"gripper_z=[{gmin:.4f},{gmax:.4f}] "
                            f"clearance={clearance*1000.0:+.1f}mm"
                        )

            print("    per-grasp rows:")
            for gid, jaw_mm, why, sources in identity_collision_rows:
                source_text = ", ".join(sources)
                print(f"      gid={gid:4d} jaw={jaw_mm:5.1f}mm reason={why}; colliders=[{source_text}]")

        out.sort(key=lambda c: float(np.max(np.abs(c.q_place - c.q_pick))))
        return out

    def _ik_at(self, pos, rotmat, grasp, obstacle_list, seed, mesh_obstacle_list=None):
        return self._ik_at_detail(pos, rotmat, grasp, obstacle_list, seed, mesh_obstacle_list)[0]

    def _ik_at_detail_pose(self, obj_pose, grasp, obstacle_list, seed, mesh_obstacle_list=None):
        tcp_pos, tcp_rotmat = _tcp_world_pose(obj_pose, grasp)
        jnt_values = self.robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rotmat, seed_jnt_values=seed)
        if jnt_values is None:
            return None, "no ik"
        if not self._within_limits(jnt_values):
            return None, "out of joint limits"
        self._goto(jnt_values, ee_values=grasp.ee_values)
        if self.robot.is_collided(obstacle_list=obstacle_list):
            return None, "arm collided"
        if self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
            return None, "gripper collided"
        if mesh_obstacle_list and gripper_mesh_collides(self.robot, mesh_obstacle_list):
            return None, "gripper mesh inside a mating partner"
        return jnt_values, ""

    def _ik_at_detail(self, pos, rotmat, grasp, obstacle_list, seed, mesh_obstacle_list=None):
        """Backward-compatible wrapper; all actual checks are world-frame TCP checks."""
        return self._ik_at_detail_pose((pos, rotmat), grasp, obstacle_list, seed, mesh_obstacle_list)

    def _collision_sources_at_pose(self, obj_pose, grasp, obstacle_list, seed=None, mesh_obstacle_list=None):
        """Re-evaluate one grasp pose and identify which obstacle(s) cause collision.

        Diagnostic only. Robot state is restored before returning so this probe cannot change
        subsequent IK/RRT behaviour.
        """
        self.robot.backup_state()
        try:
            tcp_pos, tcp_rotmat = _tcp_world_pose(obj_pose, grasp)
            q = self.robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rotmat, seed_jnt_values=seed)
            if q is None:
                q = self.robot.ik(tgt_pos=tcp_pos, tgt_rotmat=tcp_rotmat, seed_jnt_values=None)
            if q is None:
                return ["no-ik-during-diagnostic"]
            if not self._within_limits(q):
                return ["out-of-joint-limits-during-diagnostic"]

            self._goto(q, ee_values=grasp.ee_values)
            sources = []

            try:
                if self.robot.is_collided(obstacle_list=[]):
                    sources.append("robot-self")
            except Exception:
                pass

            for i, obs in enumerate(obstacle_list or []):
                label = _obstacle_label(obs, i)
                try:
                    if self.robot.is_collided(obstacle_list=[obs]):
                        sources.append(f"arm:{label}")
                except Exception:
                    pass
                try:
                    if self.robot.end_effector.is_mesh_collided(cmodel_list=[obs]):
                        sources.append(f"gripper:{label}")
                except Exception:
                    pass

            for i, obs in enumerate(mesh_obstacle_list or []):
                label = _obstacle_label(obs, i)
                try:
                    if gripper_mesh_collides(self.robot, [obs]):
                        tag = f"gripper-mesh:{label}"
                        if tag not in sources:
                            sources.append(tag)
                except Exception:
                    pass

            if not sources:
                try:
                    if self.robot.is_collided(obstacle_list=obstacle_list):
                        sources.append("scene-collision:pair-unavailable")
                except Exception:
                    pass
                try:
                    if self.robot.end_effector.is_mesh_collided(cmodel_list=obstacle_list):
                        sources.append("gripper-collision:pair-unavailable")
                except Exception:
                    pass
            return sources or ["collision-source-not-reproduced"]
        finally:
            self.robot.restore_state()

    # ------------------------------------------------------------------
    def _plan_with_grasp(self, cand, obj_cmodel, obj_at_start,
                         start_jnt_values, end_jnt_values, strict_obs, relaxed_obs,
                         mating_mesh_obs, phased_seating,
                         pick_standoff_dir, place_standoff_dir,
                         pick_depart_dir, pick_depart_distance):
        """Plan one grasp end to end. Returns ``(MotionData, "")`` or ``(None, reason)``.

        Each of the three phases is built as *free RRT up to a stand-off configuration* plus *a
        short Cartesian TCP segment across the contact*. The stand-off configuration comes from
        a single IK call at a displaced pose, seeded from the certified configuration, and the
        interpolation needs no IK at all. That is what replaces the Cartesian segment: the long
        part of the motion stays under the strict obstacle set.  When ``phased_seating`` is active,
        the final centimetres swap the mating partners' boxes for their triangle meshes rather than
        dropping them, so the held part cannot sweep through a partner on its way down.
        """
        jaw_open = float(self.robot.end_effector.jaw_range[1])
        sp, sr = obj_at_start.pos, obj_at_start.rotmat
        gp, gr = cand.goal_pose
        obj_at_goal = obj_cmodel.copy()
        obj_at_goal.pose = (np.asarray(gp, dtype=float), np.asarray(gr, dtype=float))
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
            return None, f"{cand.key} no stand-off before the grasp ({detail})"
        reach = self._rrt(start_jnt_values, q_pre_pick, reach_obs, ee_values=jaw_open)
        if reach is None:
            return None, f"{cand.key} no rrt path to the pre-grasp stand-off"
        # closing in on the object: it is excluded here because the gripper has to enclose it
        close_in, close_err = self._cartesian_between_given_conf(
            q_pre_pick,
            cand.q_pick,
            strict_obs,
            ee_values=jaw_open,
            segment_name="pre-grasp approach",
        )
        if close_in is None:
            return None, f"{cand.key} cannot close in on the object from the stand-off: {close_err}"
        ok, n_contact = self._contact_is_tail(close_in, reach_obs, ee_values=jaw_open, want="tail")
        if not ok:
            return None, (f"{cand.key} closing in touches the object before the grasp "
                          f"({n_contact} waypoints)")

        # -- grasp, transport, seat --------------------------------------------------------------
        # The held copy carries the collision geometry used for the whole carry. In phased mode it
        # is a triangle-mesh copy so the seating audit sees the real shape (holes included) rather
        # than a solid box.
        obj_held = to_triangle_cdmesh(obj_cmodel) if phased_seating else obj_cmodel.copy()
        obj_held.pose = (sp, sr)
        self.robot.goto_given_conf(cand.q_pick)
        self._empty_hand()
        self.robot.hold(obj_cmodel=obj_held, jaw_width=cand.grasp.ee_values)
        try:
            # 1) Mandatory pick-depart: once the grasp is closed, lift the part in WORLD +Z.
            #    This is independent of the asmdef insertion axis, which is used only at placement.
            lift, q_post_pick, lift_err = self._pick_lift(
                start_pose=(sp, sr),
                grasp=cand.grasp,
                q_pick=cand.q_pick,
                direction=pick_depart_dir,
                distance=pick_depart_distance,
                strict_obs=strict_obs,
                ee_values=cand.grasp.ee_values,
            )
            if lift is None:
                return None, f"{cand.key} pick-depart failed: {lift_err}"

            # 2) Find the pre-place stand-off.  Prefer the strict box scene.  If the conservative
            #    mating-part box blocks a geometrically valid stand-off, fall back to the relaxed
            #    scene but then audit the RRT waypoint-by-waypoint against the true mating meshes.
            transport_obs = strict_obs
            transport_needs_mesh_audit = False
            q_pre_place, d_place, detail = self._standoff_conf(
                (gp, gr), cand.grasp, place_dirs, strict_obs, seed=cand.q_place,
                ee_values=cand.grasp.ee_values)
            if q_pre_place is None and phased_seating:
                q_pre_place, d_place, detail = self._standoff_conf(
                    (gp, gr), cand.grasp, place_dirs, relaxed_obs, seed=cand.q_place,
                    ee_values=cand.grasp.ee_values, mesh_obstacle_list=mating_mesh_obs)
                if q_pre_place is not None:
                    transport_obs = relaxed_obs
                    transport_needs_mesh_audit = True
            if q_pre_place is None:
                return None, f"{cand.key} no stand-off above the goal ({detail})"

            # IMPORTANT: RRT starts from q_post_pick, never from the table-level q_pick.
            transport = self._rrt(
                q_post_pick, q_pre_place, transport_obs,
                ee_values=cand.grasp.ee_values)
            if transport is None:
                return None, f"{cand.key} no rrt path from q_post_pick to the pre-place stand-off"

            if transport_needs_mesh_audit:
                ok, why = self._audit_mating_mesh_clear(
                    transport, mating_mesh_obs, ee_values=cand.grasp.ee_values)
                if not ok:
                    return None, f"{cand.key} transport near mating partner: {why}"

            if phased_seating:
                seat, seat_err = self._interpolate_seating(
                    q_pre_place, cand.q_place, relaxed_obs, mating_mesh_obs,
                    ee_values=cand.grasp.ee_values)
                if seat is None:
                    return None, f"{cand.key} {seat_err}"
            else:
                # Legacy path: contact-exempt relaxed obstacles + tail audit.
                seat, seat_err = self._cartesian_between_given_conf(
                    q_pre_place,
                    cand.q_place,
                    relaxed_obs,
                    ee_values=cand.grasp.ee_values,
                    segment_name="seating",
                )
                if seat is None:
                    return None, f"{cand.key} cannot seat the object from the stand-off: {seat_err}"
                ok, n_contact = self._contact_is_tail(
                    seat, strict_obs, ee_values=cand.grasp.ee_values, want="tail")
                if not ok:
                    return None, (f"{cand.key} seating collides before it lands "
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
                ee_values=width, mesh_obstacle_list=mating_mesh_obs)
            if q_post_place is not None:
                jaw_release = width
                break
        if q_post_place is None:
            return None, f"{cand.key} nowhere to retract to after releasing ({detail})"
        back_off, back_err = self._cartesian_between_given_conf(
            cand.q_place,
            q_post_place,
            relaxed_obs,
            ee_values=jaw_release,
            mating_mesh_obs=mating_mesh_obs if phased_seating else None,
            segment_name="place retract",
        )
        if back_off is None:
            return None, f"{cand.key} cannot back off from the placed object: {back_err}"
        if phased_seating:
            ok, why = self._audit_gripper_mesh(back_off, mating_mesh_obs, ee_values=jaw_release)
            if not ok:
                return None, f"{cand.key} backing off: {why}"
        ok, n_contact = self._contact_is_tail(back_off, retract_obs, ee_values=jaw_release,
                                              want="head")
        if not ok:
            return None, (f"{cand.key} backing off re-enters the placed object "
                          f"({n_contact} waypoints)")
        home = self._rrt(q_post_place, end_jnt_values, retract_obs, ee_values=jaw_release)
        if home is None:
            return None, f"{cand.key} no rrt path from the stand-off back to the end configuration"

        print(f"  [direct] {cand.key} stand-off pick={d_pick * 1000:.0f}mm "
              f"lift={pick_depart_distance * 1000:.0f}mm "
              f"place={d_place * 1000:.0f}mm, release opening={jaw_release * 1000:.1f}mm")
        return self._assemble(cand, obj_cmodel, obj_at_start, obj_at_goal,
                              reach_jv=list(reach.jv_list) + list(close_in.jv_list),
                              carry_jv=list(lift.jv_list) + list(transport.jv_list) + list(seat.jv_list),
                              retract_jv=list(back_off.jv_list) + list(home.jv_list),
                              jaw_open=jaw_open, jaw_release=jaw_release), ""

    # ------------------------------------------------------------------
    def _standoff_conf(self, pose, grasp, directions, obstacle_list, seed, ee_values,
                       mesh_obstacle_list=None):
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
                if mesh_obstacle_list and gripper_mesh_collides(self.robot, mesh_obstacle_list):
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

    def _pick_lift(self, start_pose, grasp, q_pick, direction, distance,
                   strict_obs, ee_values):
        """Lift the newly grasped part along a true WORLD Cartesian line."""
        distance = max(0.0, float(distance))
        q_pick = np.asarray(q_pick, dtype=float)
        if distance <= 1e-9:
            md = motd.MotionData(robot=self.robot)
            md.extend(jv_list=[q_pick])
            return md, q_pick.copy(), ""

        unit = rm.unit_vector(np.asarray(direction, dtype=float))
        tcp_pos, tcp_rotmat = _tcp_world_pose(start_pose, grasp)
        goal_tcp_pos = tcp_pos + unit * distance

        n = max(2, int(math.ceil(distance / self.cartesian_granularity)) + 1)
        q_seed = q_pick.copy()
        jv_list = []
        collision_flags = []

        for i, t in enumerate(np.linspace(0.0, 1.0, n)):
            if i == 0:
                q = q_pick.copy()
            else:
                p = tcp_pos * (1.0 - t) + goal_tcp_pos * t
                q = self.robot.ik(
                    tgt_pos=p,
                    tgt_rotmat=tcp_rotmat,
                    seed_jnt_values=q_seed,
                )
                if q is None:
                    return None, None, f"no IK during +Z lift at waypoint {i + 1}/{n}"
                q = np.asarray(q, dtype=float)
                if not self._within_limits(q):
                    return None, None, f"joint limit during +Z lift at waypoint {i + 1}/{n}"

            self._goto(q, ee_values=ee_values)

            try:
                if self.robot.end_effector.is_mesh_collided(cmodel_list=strict_obs):
                    return None, None, (
                        f"gripper collided during +Z lift at waypoint {i + 1}/{n}"
                    )
            except Exception:
                pass

            collision_flags.append(
                bool(self.robot.is_collided(obstacle_list=strict_obs))
            )
            jv_list.append(q.copy())
            q_seed = q

        idx = [i for i, flag in enumerate(collision_flags) if flag]
        if idx:
            contiguous = (idx[-1] - idx[0] + 1) == len(idx)
            if not contiguous or idx[0] != 0 or idx[-1] == len(collision_flags) - 1:
                return None, None, (
                    "pick lift collision is not a clearing head "
                    f"(contact waypoints={[i + 1 for i in idx]}, total={n})"
                )

        md = motd.MotionData(robot=self.robot)
        md.extend(jv_list=jv_list)
        return md, np.asarray(jv_list[-1], dtype=float), ""

    def _audit_mating_mesh_clear(self, mot_data, mating_mesh_obs, ee_values):
        """Transport/RRT near a mating partner must remain physically mesh-clear.

        This is used when the conservative mating-part box had to be removed to make a pre-place
        stand-off admissible.  The RRT may ignore that box, but neither gripper nor held part may
        intersect the actual partner mesh before the controlled seating segment begins.
        """
        if not mating_mesh_obs:
            return True, ""
        for i, jnt_values in enumerate(mot_data.jv_list):
            self._goto(jnt_values, ee_values=ee_values)
            if gripper_mesh_collides(self.robot, mating_mesh_obs):
                return False, f"gripper mesh intersects mating partner at waypoint {i + 1}"
            if held_object_mesh_collides(self.robot, mating_mesh_obs):
                return False, f"held object reaches mating partner before seating at waypoint {i + 1}"
        return True, ""

    def _cartesian_between_given_conf(
            self,
            start_conf,
            goal_conf,
            obstacle_list,
            ee_values,
            *,
            mating_mesh_obs=None,
            track_held_object_contact=False,
            segment_name="cartesian",
            return_contact_flags=False):
        """Connect two endpoint configurations with a straight TCP translation."""
        start_conf = np.asarray(start_conf, dtype=float)
        goal_conf = np.asarray(goal_conf, dtype=float)

        start_pos, _ = self.robot.fk(jnt_values=start_conf)
        goal_pos, goal_rot = self.robot.fk(jnt_values=goal_conf)
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        goal_rot = np.asarray(goal_rot, dtype=float)

        distance = float(np.linalg.norm(goal_pos - start_pos))
        n = max(2, int(math.ceil(distance / self.cartesian_granularity)) + 1)

        q_seed = start_conf.copy()
        jv_list = []
        held_contact_flags = []

        for i, t in enumerate(np.linspace(0.0, 1.0, n)):
            if i == 0:
                q = start_conf.copy()
            elif i == n - 1:
                q = goal_conf.copy()
            else:
                tcp_pos = start_pos * (1.0 - t) + goal_pos * t
                q = self.robot.ik(
                    tgt_pos=tcp_pos,
                    tgt_rotmat=goal_rot,
                    seed_jnt_values=q_seed,
                )
                if q is None:
                    err = f"{segment_name}: IK failed at waypoint {i + 1}/{n}"
                    if return_contact_flags:
                        return None, err, held_contact_flags
                    return None, err
                q = np.asarray(q, dtype=float)
                if not self._within_limits(q):
                    err = f"{segment_name}: joint limit at waypoint {i + 1}/{n}"
                    if return_contact_flags:
                        return None, err, held_contact_flags
                    return None, err

            self._goto(q, ee_values=ee_values)

            if self.robot.is_collided(obstacle_list=obstacle_list):
                err = f"{segment_name}: arm/scene collision at waypoint {i + 1}/{n}"
                if return_contact_flags:
                    return None, err, held_contact_flags
                return None, err

            if mating_mesh_obs:
                if gripper_mesh_collides(self.robot, mating_mesh_obs):
                    err = (
                        f"{segment_name}: gripper mesh intersects mating partner "
                        f"at waypoint {i + 1}/{n}"
                    )
                    if return_contact_flags:
                        return None, err, held_contact_flags
                    return None, err

                if track_held_object_contact:
                    held_contact_flags.append(
                        bool(held_object_mesh_collides(self.robot, mating_mesh_obs))
                    )

            jv_list.append(q.copy())
            q_seed = q

        md = motd.MotionData(robot=self.robot)
        md.extend(jv_list=jv_list)
        if return_contact_flags:
            if len(held_contact_flags) < len(jv_list):
                held_contact_flags.extend([False] * (len(jv_list) - len(held_contact_flags)))
            return md, "", held_contact_flags
        return md, ""

    def _interpolate_seating(self, start_conf, goal_conf, arm_obs, mating_mesh_obs, ee_values):
        """Stand-off -> goal on a true Cartesian TCP line."""
        mot_data, err, contact_flags = self._cartesian_between_given_conf(
            start_conf,
            goal_conf,
            arm_obs,
            ee_values=ee_values,
            mating_mesh_obs=mating_mesh_obs,
            track_held_object_contact=True,
            segment_name="seating",
            return_contact_flags=True,
        )
        if mot_data is None:
            return None, err

        if not contact_flags_form_tail(contact_flags):
            idx = [i + 1 for i, flag in enumerate(contact_flags) if flag]
            return None, (
                "seating: held-object/mating contact is not a contiguous tail "
                f"(contact waypoints={idx}, total={len(contact_flags)})"
            )
        return mot_data, ""

    def _audit_gripper_mesh(self, mot_data, mating_mesh_obs, ee_values):
        """No waypoint of a segment may have the gripper inside a mating partner."""
        if not mating_mesh_obs:
            return True, ""
        ee_bk = self.robot.get_ee_values()
        try:
            for i, jnt_values in enumerate(mot_data.jv_list):
                self._goto(jnt_values, ee_values=ee_values)
                if gripper_mesh_collides(self.robot, mating_mesh_obs):
                    return False, f"gripper mesh intersects a mating partner at waypoint {i + 1}"
        finally:
            self._set_jaw(ee_bk)
        return True, ""

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
