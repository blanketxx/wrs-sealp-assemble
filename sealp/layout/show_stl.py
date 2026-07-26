#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Show a Specified STL Mesh
============================

在 Panda3D 窗口中加载并显示指定 STL 模型，支持设置位置/姿态，
按 **空格** 绕 z 轴每次旋转 90°。

运行::

    python -m sealp.layout.show_stl ^
      --stl sealp/assets/models/Toy/model/middle_plate.stl

指定位置与欧拉角（度，sxyz 顺序）::

    python -m sealp.layout.show_stl ^
      --stl sealp/assets/models/Toy/model/post.stl ^
      --pos 0.1,-0.2,0.05 ^
      --rot-deg 0,90,0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from wrs import mcm, mgm, rm, wd

_THIS_FILE = os.path.abspath(__file__)
_SEALP_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
_PROJECT_ROOT = os.path.dirname(_SEALP_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sealp.layout._viz_common import hide_rot_center_marker


def _resolve_stl_path(stl: str) -> str:
    path = Path(stl)
    if not path.is_absolute():
        for base in (_PROJECT_ROOT, _SEALP_ROOT, Path.cwd()):
            candidate = (base / path).resolve()
            if candidate.is_file():
                return str(candidate)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"STL not found: {stl}")
    return str(path)


def _parse_vec3(text: str, default: Sequence[float]) -> np.ndarray:
    parts = [p.strip() for p in str(text).replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got {text!r}")
    return np.asarray([float(v) for v in parts], dtype=float)


def _auto_frame_length(obj: mcm.CollisionModel) -> float:
    try:
        extent = float(np.max(obj.trm_mesh.aabb_bound.extents))
        return max(extent * 0.45, 0.03)
    except Exception:
        return 0.08


def _auto_camera(
    obj: mcm.CollisionModel,
    pos: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        aabb = obj.trm_mesh.aabb_bound
        extent = float(np.max(aabb.extents))
        center = np.asarray(pos, dtype=float) + np.asarray(aabb.centroid, dtype=float)
    except Exception:
        extent = 0.2
        center = np.asarray(pos, dtype=float)
    cam = center + np.array([extent * 1.8, -extent * 1.8, extent * 1.2], dtype=float)
    return cam, center


def _yaw_rotmats() -> List[np.ndarray]:
    return [
        np.eye(3),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(180.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(-90.0)),
    ]


class StlViewer:
    """Interactive STL viewer: SPACE cycles z-axis 90° rotations."""

    def __init__(
        self,
        obj_model: mcm.CollisionModel,
        *,
        pos: np.ndarray,
        base_rotmat: np.ndarray,
        rgb: np.ndarray,
        alpha: float,
        frame_length: Optional[float] = None,
        show_world_frame: bool = True,
        cam_pos: Optional[np.ndarray] = None,
        lookat_pos: Optional[np.ndarray] = None,
    ) -> None:
        self.obj_model = obj_model
        self.pos = np.asarray(pos, dtype=float)
        self.base_rotmat = np.asarray(base_rotmat, dtype=float)
        self.rgb = np.asarray(rgb, dtype=float)
        self.alpha = float(alpha)
        self.frame_length = float(frame_length or _auto_frame_length(obj_model))
        self.show_world_frame = bool(show_world_frame)
        self.yaw_rotmats = _yaw_rotmats()
        self.counter = 0
        self.onscreen: List = []

        if cam_pos is None or lookat_pos is None:
            auto_cam, auto_look = _auto_camera(obj_model, self.pos)
            cam_pos = auto_cam if cam_pos is None else cam_pos
            lookat_pos = auto_look if lookat_pos is None else lookat_pos

        self.base = wd.World(cam_pos=cam_pos, lookat_pos=lookat_pos)
        hide_rot_center_marker(self.base)

        # if self.show_world_frame:
        #     world_frame = mgm.gen_frame(ax_length=max(self.frame_length * 0.8, 0.05))
        #     world_frame.attach_to(self.base)
        #     self.onscreen.append(world_frame)

    def _clear_mesh_and_pose_frame(self) -> None:
        keep = []
        for model in self.onscreen:
            if self.show_world_frame and len(keep) == 0:
                keep.append(model)
                continue
            model.detach()
        self.onscreen = keep

    def _show_pose(self, yaw_index: int) -> None:
        yaw_index = int(yaw_index) % len(self.yaw_rotmats)
        self._clear_mesh_and_pose_frame()

        rot = self.yaw_rotmats[yaw_index] @ self.base_rotmat
        self.obj_model.pose = (self.pos, rot)
        self.obj_model.rgb = self.rgb
        self.obj_model.alpha = self.alpha
        self.obj_model.attach_to(self.base)
        self.onscreen.append(self.obj_model)

        # frame = mgm.gen_frame(
        #     pos=self.pos,
        #     rotmat=rot,
        #     ax_length=self.frame_length,
        #     ax_radius=max(self.frame_length * 0.02, 0.0015),
        # )
        # frame.attach_to(self.base)
        # self.onscreen.append(frame)

        print(
            f"[POSE {yaw_index + 1}/{len(self.yaw_rotmats)}] "
            f"yaw_step={yaw_index * 90}° pos={np.round(self.pos, 4).tolist()}"
        )

    def _update(self, task):
        if self.base.inputmgr.keymap.get("space"):
            time.sleep(0.08)
            self._show_pose(self.counter)
            self.counter += 1
        return task.cont

    def run(self) -> None:
        print(f"[INFO] Press SPACE to rotate 90° about +Z. Close the window to exit.")
        self._show_pose(0)
        self.counter = 1
        taskMgr.doMethodLater(0.01, self._update, "stl_viewer", appendTask=True)
        self.base.run()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a specified STL mesh in Panda3D."
    )
    parser.add_argument(
        "--stl",
        required=True,
        help="Path to the STL mesh (absolute or relative to project/sealp/cwd).",
    )
    parser.add_argument(
        "--pos",
        default="0,0,0",
        help="Object position x,y,z in meters (default: 0,0,0).",
    )
    parser.add_argument(
        "--rot-deg",
        default="0,0,0",
        help="Base orientation as Euler angles in degrees, sxyz order (default: 0,0,0).",
    )
    parser.add_argument(
        "--rgb",
        default="0.45,0.45,0.48",
        help="Mesh color r,g,b in [0,1] (default: 0.45,0.45,0.48).",
    )
    parser.add_argument("--alpha", type=float, default=1, help="Mesh alpha (default: 0.88).")
    parser.add_argument(
        "--frame-length",
        type=float,
        default=None,
        help="Object coordinate frame axis length. Default: auto from mesh size.",
    )
    parser.add_argument(
        "--no-world-frame",
        action="store_true",
        help="Hide the fixed world coordinate frame at the origin.",
    )
    parser.add_argument(
        "--cam-pos",
        default=None,
        help="Camera position x,y,z. Default: auto from mesh bounds.",
    )
    parser.add_argument(
        "--lookat",
        default=None,
        help="Camera look-at x,y,z. Default: auto from mesh bounds.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stl_path = _resolve_stl_path(args.stl)
    pos = _parse_vec3(args.pos, (0.0, 0.0, 0.0))
    rot_deg = _parse_vec3(args.rot_deg, (0.0, 0.0, 0.0))
    rgb = _parse_vec3(args.rgb, (0.45, 0.45, 0.48))
    base_rotmat = rm.rotmat_from_euler(
        np.deg2rad(rot_deg[0]),
        np.deg2rad(rot_deg[1]),
        np.deg2rad(rot_deg[2]),
        order="sxyz",
    )

    cam_pos = _parse_vec3(args.cam_pos, (0.0, 0.0, 0.0)) if args.cam_pos else None
    lookat_pos = _parse_vec3(args.lookat, (0.0, 0.0, 0.0)) if args.lookat else None

    print(f"[INFO] STL: {stl_path}")
    print(f"[INFO] pos={pos.tolist()} rot_deg(sxyz)={rot_deg.tolist()}")

    obj_model = mcm.CollisionModel(stl_path)
    viewer = StlViewer(
        obj_model,
        pos=pos,
        base_rotmat=base_rotmat,
        rgb=rgb,
        alpha=float(args.alpha),
        frame_length=args.frame_length,
        show_world_frame=not args.no_world_frame,
        cam_pos=cam_pos,
        lookat_pos=lookat_pos,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
