"""Regenerate YuanChair grasps for the CURRENT meshes (PantheraGripper).

The layout robot ``PantheraHT`` uses ``PantheraGripper`` as its end effector, so
grasp collections MUST be planned with the SAME gripper (the old
``leg_model_grasps.pickle`` was planned for the 33 cm leg and is stale after the
leg was shortened to 16 cm).

This script plans antipodal grasps for the leg (and optionally the seat) on the
current STL files and writes them into ``sealp/examples/grasp/yuanchair_grasp/``
with the exact names the layout searcher looks up:

    leg_model  -> leg_model_grasps.pickle   (model id ``leg_model`` in asmdef)
    seat       -> seat_grasps.pickle        (part id ``seat``; preassembled)

Run::

    python -m sealp.examples.grasp.gen_yuanchair_grasps           # leg only
    python -m sealp.examples.grasp.gen_yuanchair_grasps --seat    # leg + seat
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.grasping.planning.antipodal as gpa
from wrs.robot_sim.end_effectors.grippers.panthera_gripper.panthera_gripper import (
    PantheraGripper,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "..", "assets", "models", "yuanchair"))
_OUT_DIR = os.path.join(_HERE, "yuanchair_grasp")

LEG_STL = os.path.join(_ASSET_DIR, "yuanchair-part2.stl")
SEAT_STL = os.path.join(_ASSET_DIR, "yuanchair-part1.stl")


def _plan(obj_path: str, *, max_samples: int, rot_deg: float,
          min_dist: float, contact_offset: float):
    if not os.path.isfile(obj_path):
        raise SystemExit(f"mesh not found: {obj_path}")
    obj = mcm.CollisionModel(initor=obj_path)
    gripper = PantheraGripper()
    ext = obj.trm_mesh.bounds[1] - obj.trm_mesh.bounds[0]
    print(f"  mesh bbox(m) = [{ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f}]")
    gc = gpa.plan_gripper_grasps(
        gripper,
        obj,
        angle_between_contact_normals=rm.radians(175),
        rotation_interval=rm.radians(float(rot_deg)),
        max_samples=int(max_samples),
        min_dist_between_sampled_contact_points=float(min_dist),
        contact_offset=float(contact_offset),
        toggle_dbg=False,
    )
    return gc


def _topdown_count(gc) -> int:
    """Grasps whose approach is roughly top-down (-Z), i.e. ac z-axis points down."""
    n = 0
    for g in gc:
        z_axis = np.asarray(g.ac_rotmat, dtype=float)[:, 2]
        if float(z_axis[2]) < -0.5:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate YuanChair grasps (PantheraGripper).")
    ap.add_argument("--seat", action="store_true", help="Also regenerate the seat grasps.")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--rot-deg", type=float, default=20.0,
                    help="Rotation sampling interval in degrees (smaller = more grasps).")
    ap.add_argument("--min-dist", type=float, default=0.006,
                    help="Min distance between sampled contact points (m).")
    ap.add_argument("--contact-offset", type=float, default=0.008)
    args = ap.parse_args()

    os.makedirs(_OUT_DIR, exist_ok=True)
    print("========== YuanChair grasp regeneration (PantheraGripper) ==========")

    print(f"[leg ] planning on {os.path.relpath(LEG_STL, _HERE)}")
    leg_gc = _plan(LEG_STL, max_samples=args.max_samples, rot_deg=args.rot_deg,
                   min_dist=args.min_dist, contact_offset=args.contact_offset)
    leg_out = os.path.join(_OUT_DIR, "leg_model_grasps.pickle")
    leg_gc.save_to_disk(file_name=leg_out)
    print(f"[leg ] {len(leg_gc)} grasps (topdown(-Z)={_topdown_count(leg_gc)}) "
          f"-> {os.path.relpath(leg_out, _HERE)}")

    if args.seat:
        print(f"[seat] planning on {os.path.relpath(SEAT_STL, _HERE)}")
        seat_gc = _plan(SEAT_STL, max_samples=args.max_samples, rot_deg=args.rot_deg,
                        min_dist=args.min_dist, contact_offset=args.contact_offset)
        seat_out = os.path.join(_OUT_DIR, "seat_grasps.pickle")
        seat_gc.save_to_disk(file_name=seat_out)
        print(f"[seat] {len(seat_gc)} grasps (topdown(-Z)={_topdown_count(seat_gc)}) "
              f"-> {os.path.relpath(seat_out, _HERE)}")

    print("Done. NOTE: seat is preassembled (not picked), so leg grasps are what "
          "the GA/L2 actually rely on.")


if __name__ == "__main__":
    main()
