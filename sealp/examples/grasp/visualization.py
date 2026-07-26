"""
Grasp Visualization Example
=============================

Demonstrates loading a saved ``GraspCollection`` from disk and
visualizing it with statistics (jaw width distribution, height
distribution, orientation analysis).

Usage::

    python -m sealp.examples.grasp.visualization

Adapted from tiaozhanbei/grasp/visualize_graspcollection.py
"""

import os
from collections import defaultdict

import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
# import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
import wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper as pg
from wrs.grasping.grasp import GraspCollection


def print_grasp_statistics(grasp_collection):
    """Print summary statistics of a grasp collection.

    Parameters
    ----------
    grasp_collection : GraspCollection
    """
    n = len(grasp_collection)
    print(f"\n  Total grasps: {n}")
    if n == 0:
        return

    # Jaw width distribution
    width_counts = defaultdict(int)
    positions = []
    for grasp in grasp_collection:
        width_counts[round(grasp.ee_values, 4)] += 1
        positions.append(grasp.ac_pos)

    print(f"\n  Jaw width distribution:")
    for width, count in sorted(width_counts.items()):
        print(f"    width={width:.4f}m : {count} grasps")

    # Position statistics
    positions = np.array(positions)
    print(f"\n  Position ranges:")
    for i, axis in enumerate(["X", "Y", "Z"]):
        lo, hi = positions[:, i].min(), positions[:, i].max()
        print(f"    {axis}: [{lo:.4f}, {hi:.4f}]m")

    # Orientation analysis — count z-axis-down grasps
    n_down = sum(1 for g in grasp_collection
                 if g.ac_rotmat[2, 2] < 0.0)
    print(f"\n  Gripper z-axis pointing down: {n_down}/{n}")


def compute_approach_point(grasp, axis_idx=2, sign=-1.0, dist=0.02):
    """Compute an approach point in the object frame.

    The approach point is a point along the gripper's ``axis_idx``
    column of the rotation matrix, offset by ``dist``.

    Parameters
    ----------
    grasp : Grasp
        A grasp from a GraspCollection.
    axis_idx : int
        Column index of ac_rotmat to use (0=x, 1=y, 2=z).
    sign : float
        Direction multiplier (-1.0 or 1.0).
    dist : float
        Offset distance in meters.

    Returns
    -------
    approach_dir : np.ndarray
        Approach direction vector (unit).
    approach_point : np.ndarray
        The approach point position.
    """
    approach_dir = sign * grasp.ac_rotmat[:, axis_idx]
    approach_point = grasp.ac_pos + dist * approach_dir
    return approach_dir, approach_point


def main():
    """Run the grasp visualization demo."""
    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
    # mgm.gen_frame(ax_length=0.3).attach_to(base)

    gripper = pg.PantheraGripper()

    # ------------------------------------------------------------------
    # 2. Load grasps (plan first if not cached)
    # ------------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "first_stack_grasp")
    pickle_path = os.path.join(out_dir, "part_4_grasps.pickle")

    if os.path.isfile(pickle_path):
        print(f"Loading grasps from {pickle_path}...")
        grasp_collection = GraspCollection.load_from_disk(
            file_name=pickle_path)
    else:
        # Plan grasps on a demo box
        from sealp.examples.grasp.planning import plan_grasps

        obj_cmodel = mcm.gen_box(xyz_lengths=np.array([0.06, 0.04, 0.03]),
                                 pos=np.array([0, 0, 0.015]))
        grasp_collection, gripper = plan_grasps(obj_cmodel, max_samples=50)
        os.makedirs(out_dir, exist_ok=True)
        grasp_collection.save_to_disk(file_name=pickle_path)

    # ------------------------------------------------------------------
    # 3. Print statistics
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Grasp Visualization")
    print("=" * 50)
    print_grasp_statistics(grasp_collection)

    # ------------------------------------------------------------------
    # 4. Print approach points for first 5 grasps
    # ------------------------------------------------------------------
    print(f"\n  Approach points (first 5):")
    for i, g in enumerate(grasp_collection):
        if i >= 5:
            break
        a_dir, a_pt = compute_approach_point(g, axis_idx=2, sign=-1.0)
        print(f"    grasp {i}: ac_pos={g.ac_pos}, approach={a_pt}")

    # ------------------------------------------------------------------
    # 5. Visualize
    # ------------------------------------------------------------------
    # Show the demo object
    obj_cmodel = mcm.CollisionModel(r"D:\Project\wrs-sealp\sealp\assets\models\first\part_4.stl")
    obj_cmodel.rgba = np.array([0.4, 0.4, 0.4, 1.0])
    obj_cmodel.attach_to(base)

    # Show grasps (limited for performance)
    max_show = min(100, len(grasp_collection))
    print(f"\n  Showing {max_show} grasps...")

    for i, grasp in enumerate(grasp_collection):
        if i >= max_show:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat,
                                grasp.ee_values)
        gripper.gen_meshmodel(np.array([0.7,0.7,0.7]),alpha=0.1).attach_to(base)

        # Draw approach direction arrow
        a_dir, a_pt = compute_approach_point(grasp)
        mgm.gen_arrow(spos=grasp.ac_pos, epos=a_pt,
                      rgb=np.array([1, 0.3, 0.3]),
                      stick_radius=0.001).attach_to(base)

    print("=" * 50)
    print("Press ESC to close.")
    base.run()


if __name__ == "__main__":
    main()
