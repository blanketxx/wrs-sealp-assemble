#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Show Stable Placement Poses for an STL Mesh
==============================================

读取 STL，计算稳定摆放姿态与常用 90° 旋转候选，
在 Panda3D 窗口中按 **空格** 切换到下一个姿态。

运行::

    python -m sealp.layout.show_stable_poses ^
      --stl sealp/assets/models/Toy/model/middle_plate.stl

可选保存 pickle::

    python -m sealp.layout.show_stable_poses ^
      --stl sealp/assets/models/Toy/model/post.stl ^
      --save-pickle sealp/examples/layout/_output/post_fsref.pickle
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
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
from wrs.manipulation.placement.flatsurface import FSReferencePoses


@dataclass(frozen=True)
class PoseEntry:
    label: str
    pos: np.ndarray
    rotmat: np.ndarray
    stability: Optional[float] = None


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


def _rotmat_90_candidates() -> List[np.ndarray]:
    """常用 90° 旋转候选（与 layout 搜索脚本一致）。"""
    base_rots = [
        np.eye(3),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.x_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.y_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(-90.0)),
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(180.0)),
    ]
    yaw_rots = [
        rm.rotmat_from_axangle(rm.const.z_ax, np.deg2rad(a))
        for a in (0.0, 90.0, -90.0, 180.0)
    ]

    out: List[np.ndarray] = []
    for base_rot in base_rots:
        for yaw_rot in yaw_rots:
            rot = yaw_rot @ base_rot
            if not any(np.allclose(rot, old, atol=1e-6) for old in out):
                out.append(rot)

    out.sort(key=lambda rot: 0 if np.allclose(rot, np.eye(3), atol=1e-6) else 1)
    return out


def _rotation_exists(rot: np.ndarray, existing: Sequence[np.ndarray]) -> bool:
    return any(np.allclose(rot, old, atol=1e-6) for old in existing)


def _auto_frame_length(obj_cmodel) -> float:
    try:
        extent = np.asarray(obj_cmodel.get_extents(), dtype=float)
        return max(float(np.max(extent)) * 0.45, 0.03)
    except Exception:
        return 0.08


def _build_pose_catalog(
    stl_path: str,
    stability_threshold: float,
    include_90_rot: bool,
) -> Tuple[mcm.CollisionModel, List[PoseEntry]]:
    obj = mcm.CollisionModel(stl_path)

    pose_entries: List[PoseEntry] = []
    existing_rots: List[np.ndarray] = []

    raw_poses = FSReferencePoses.compute_reference_poses(
        obj_cmodel=obj,
        stability_threshhold=float(stability_threshold),
        toggle_support_facets=False,
    )
    for i, pose in enumerate(raw_poses):
        pos, rot = pose
        rot = np.asarray(rot, dtype=float)
        if _rotation_exists(rot, existing_rots):
            continue
        existing_rots.append(rot)
        pose_entries.append(
            PoseEntry(
                label=f"fs_{i:02d}",
                pos=np.asarray(pos, dtype=float),
                rotmat=rot,
            )
        )

    if include_90_rot:
        for i, rot in enumerate(_rotmat_90_candidates()):
            if _rotation_exists(rot, existing_rots):
                continue
            existing_rots.append(rot)
            name = "identity" if np.allclose(rot, np.eye(3), atol=1e-6) else f"rot90_{i:02d}"
            pose_entries.append(
                PoseEntry(
                    label=name,
                    pos=np.zeros(3, dtype=float),
                    rotmat=rot,
                )
            )

    return obj, pose_entries


class StablePoseViewer:
    """Interactive viewer: SPACE cycles through placement/rotation poses."""

    def __init__(
        self,
        obj_model: mcm.CollisionModel,
        pose_entries: Sequence[PoseEntry],
        alpha: float = 0.85,
        frame_length: Optional[float] = None,
    ) -> None:
        self.pose_entries = list(pose_entries)
        self.alpha = float(alpha)
        self.frame_length = float(frame_length or _auto_frame_length(obj_model))
        self.counter = 0
        self.onscreen: List = []
        self.obj_model = obj_model

        self.base = wd.World(cam_pos=[0.8, 0.8, 0.6], lookat_pos=[0.0, 0.0, 0.05])
        hide_rot_center_marker(self.base)

    def _clear_onscreen(self) -> None:
        for model in self.onscreen:
            model.detach()
        self.onscreen.clear()

    def _show_pose(self, index: int) -> None:
        if not self.pose_entries:
            print("[WARN] No poses to display.")
            return

        index = int(index) % len(self.pose_entries)
        self._clear_onscreen()

        entry = self.pose_entries[index]
        pos = np.asarray(entry.pos, dtype=float)
        rot = np.asarray(entry.rotmat, dtype=float)

        self.obj_model.pose = (pos, rot)
        self.obj_model.rgb = np.array([0.4, 0.4, 0.4])
        self.obj_model.alpha = self.alpha
        self.obj_model.attach_to(self.base)
        self.onscreen.append(self.obj_model)

        frame = mgm.gen_frame(
            pos=pos,
            rotmat=rot,
            ax_length=self.frame_length,
            ax_radius=max(self.frame_length * 0.02, 0.0015),
        )
        frame.attach_to(self.base)
        self.onscreen.append(frame)

        msg = f"[POSE {index + 1}/{len(self.pose_entries)}] {entry.label} pos={np.round(pos, 4).tolist()}"
        if entry.stability is not None:
            msg += f" stability={entry.stability:.4f}"
        print(msg)

    def _update(self, task):
        if self.base.inputmgr.keymap.get("space"):
            time.sleep(0.08)
            self._show_pose(self.counter)
            self.counter += 1
        return task.cont

    def run(self) -> None:
        n_poses = len(self.pose_entries)
        print(f"[INFO] Found {n_poses} pose(s) (flatsurface + optional 90° rotations).")
        print("[INFO] Press SPACE to show the next pose. Close the window to exit.")
        if n_poses > 0:
            self._show_pose(0)
            self.counter = 1

        taskMgr.doMethodLater(0.01, self._update, "stable_pose_viewer", appendTask=True)
        self.base.run()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize stable flat-surface placement poses for an STL mesh."
    )
    parser.add_argument(
        "--stl",
        required=True,
        help="Path to the STL mesh (absolute or relative to project/sealp/cwd).",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.10,
        help="Stability threshold for flatsurface poses (default: 0.10).",
    )
    parser.add_argument(
        "--no-90-rot",
        action="store_true",
        help="Do not append common 90° rotation candidates.",
    )
    parser.add_argument(
        "--frame-length",
        type=float,
        default=None,
        help="Coordinate frame axis length. Default: auto from mesh size.",
    )
    parser.add_argument(
        "--save-pickle",
        default=None,
        help="Optional output path to save flatsurface poses only.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stl_path = _resolve_stl_path(args.stl)
    print(f"[INFO] STL: {stl_path}")

    obj_model, pose_entries = _build_pose_catalog(
        stl_path=stl_path,
        stability_threshold=args.stability_threshold,
        include_90_rot=not args.no_90_rot,
    )

    if args.save_pickle:
        fs_only = [(entry.pos, entry.rotmat) for entry in pose_entries if entry.label.startswith("fs_")]
        out_path = Path(args.save_pickle)
        if not out_path.is_absolute():
            out_path = (_PROJECT_ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        FSReferencePoses(poses=fs_only).save_to_disk(str(out_path))
        print(f"[INFO] Saved {len(fs_only)} flatsurface pose(s) to {out_path}")

    viewer = StablePoseViewer(
        obj_model=obj_model,
        pose_entries=pose_entries,
        frame_length=args.frame_length,
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
