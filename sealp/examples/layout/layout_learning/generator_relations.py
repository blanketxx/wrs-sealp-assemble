"""Static and partial-layout relations for RelSeqGen.

Independent from ``dynamic_relations.py`` (DynEdge scorer).  Generators must not
consume target staging_xy as input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from . import features as F

if TYPE_CHECKING:
    import torch

STATIC_EDGE_DIM = 10
PARTIAL_DYN_EDGE_DIM = 14
MAX_POSE_CANDIDATES = 8
MAX_ROTATIONS = 4
POSE_CAND_DIM = 12


def _goal_xy(parts: List[Dict]) -> np.ndarray:
    return np.array(
        [np.asarray(p.get("goal_pos", [0, 0, 0]), dtype=np.float32)[:2] for p in parts],
        dtype=np.float32,
    )


def build_static_edge_index(sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Assembly-order, parent-child, goal-space, same-model edges."""
    parts = F._ordered_parts(sample)
    n = len(parts)
    pid_to_idx = {p["part_id"]: i for i, p in enumerate(parts)}
    model_to_parts: Dict[str, List[int]] = {}
    for i, p in enumerate(parts):
        model_to_parts.setdefault(str(p.get("model_alias", p["part_id"])), []).append(i)

    edges: Dict[Tuple[int, int], np.ndarray] = {}

    def _add(a: int, b: int, slot: int):
        if a == b or a < 0 or b < 0:
            return
        for u, v in ((a, b), (b, a)):
            flags = edges.setdefault((u, v), np.zeros(4, dtype=np.float32))
            flags[slot] = 1.0

    order_sorted = sorted(range(n), key=lambda i: float(parts[i].get("order_index", i)))
    for a, b in zip(order_sorted[:-1], order_sorted[1:]):
        _add(a, b, 0)
    for i, p in enumerate(parts):
        par = p.get("parent")
        if par and str(par) in pid_to_idx:
            _add(i, pid_to_idx[str(par)], 1)
    goal_xy = _goal_xy(parts)
    if n >= 2:
        for i in range(n):
            d = np.linalg.norm(goal_xy - goal_xy[i], axis=1)
            nn = np.argsort(d)[1:3]
            for j in nn:
                _add(i, int(j), 2)
    for indices in model_to_parts.values():
        if len(indices) >= 2:
            for a in indices:
                for b in indices:
                    if a != b:
                        _add(a, b, 3)

    if not edges:
        edges[(0, 0)] = np.zeros(4, dtype=np.float32)
    edge_index = np.asarray(list(edges.keys()), dtype=np.int64).T
    relation_flags = np.stack(list(edges.values())).astype(np.float32)
    return edge_index, relation_flags


def build_static_edge_feature(
    sample: Dict,
    edge_index: np.ndarray,
    relation_flags: np.ndarray,
) -> np.ndarray:
    parts = F._ordered_parts(sample)
    goal_xy = _goal_xy(parts)
    goal_z = np.array(
        [float(np.asarray(p.get("goal_pos", [0, 0, 0]), dtype=np.float32)[2]) for p in parts],
        dtype=np.float32,
    )
    scale = max(float(F._table_diag(sample)), 1e-9)
    e_count = int(edge_index.shape[1])
    feat = np.zeros((e_count, STATIC_EDGE_DIM), dtype=np.float32)
    for e in range(e_count):
        s, t = int(edge_index[0, e]), int(edge_index[1, e])
        dx, dy = goal_xy[t] - goal_xy[s]
        dz = goal_z[t] - goal_z[s]
        feat[e, 0] = dx / scale
        feat[e, 1] = dy / scale
        feat[e, 2] = float(np.hypot(dx, dy)) / scale
        feat[e, 3] = dz / scale
        feat[e, 4] = float(parts[t].get("order_index", 0) - parts[s].get("order_index", 0)) / max(len(parts), 1)
        feat[e, 5:9] = relation_flags[e]
        feat[e, 9] = 1.0 if str(parts[t].get("parent")) == str(parts[s].get("part_id")) else 0.0
    return feat


def build_static_graph(sample: Dict) -> Dict[str, np.ndarray]:
    edge_index, relation_flags = build_static_edge_index(sample)
    edge_feat = build_static_edge_feature(sample, edge_index, relation_flags)
    return {
        "static_edge_index": edge_index,
        "static_edge_feat": edge_feat,
        "relation_flags": relation_flags,
    }


def build_partial_dynamic_edge_feature(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    source_fp: np.ndarray,
    target_fp: np.ndarray,
    source_gen: float,
    target_gen: float,
    table_bounds: Tuple[float, float, float, float],
    arm_bases: np.ndarray,
    order_delta: float,
    parent_flag: float,
    scale: float,
) -> np.ndarray:
    """Single directed partial-layout edge (14-D)."""
    feat = np.zeros(PARTIAL_DYN_EDGE_DIM, dtype=np.float32)
    if source_gen < 0.5 or target_gen < 0.5:
        return feat
    dx, dy = target_xy - source_xy
    feat[0] = dx / scale
    feat[1] = dy / scale
    feat[2] = float(np.hypot(dx, dy)) / scale
    cx = abs(dx) - (source_fp[0] + target_fp[0]) / 2.0
    cy = abs(dy) - (source_fp[1] + target_fp[1]) / 2.0
    feat[3] = cx / scale
    feat[4] = cy / scale
    feat[5] = max(0.0, -cx) / scale
    feat[6] = max(0.0, -cy) / scale
    xlo, xhi, ylo, yhi = table_bounds
    for xy, fp, slot in ((target_xy, target_fp, 7), (source_xy, source_fp, 8)):
        ml = (xy[0] - fp[0] / 2.0 - xlo) / scale
        mr = (xhi - (xy[0] + fp[0] / 2.0)) / scale
        mb = (xy[1] - fp[1] / 2.0 - ylo) / scale
        mt = (yhi - (xy[1] + fp[1] / 2.0)) / scale
        feat[slot] = min(ml, mr, mb, mt)
    if arm_bases.size:
        dists = np.linalg.norm(arm_bases - target_xy[None, :], axis=1)
        feat[9] = float(dists.min()) / scale
    feat[10] = source_gen
    feat[11] = target_gen
    feat[12] = order_delta
    feat[13] = parent_flag
    return feat


def build_partial_dynamic_edge_feature_torch(
    source_xy: "torch.Tensor",
    target_xy: "torch.Tensor",
    source_fp: "torch.Tensor",
    target_fp: "torch.Tensor",
    source_gen: "torch.Tensor",
    target_gen: "torch.Tensor",
    table_bounds: "torch.Tensor",
    order_delta: "torch.Tensor",
    parent_flag: "torch.Tensor",
    scale: "torch.Tensor",
    arm_dist: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Batched partial-layout edge features [B, 14], GPU-friendly."""
    import torch  # local: numpy helpers above must work without torch installed

    feat = torch.zeros(
        source_xy.shape[0], PARTIAL_DYN_EDGE_DIM,
        device=source_xy.device, dtype=source_xy.dtype)
    valid = (source_gen > 0.5) & (target_gen > 0.5)
    if not bool(valid.any()):
        return feat

    dx = target_xy[:, 0] - source_xy[:, 0]
    dy = target_xy[:, 1] - source_xy[:, 1]
    feat[:, 0] = dx / scale
    feat[:, 1] = dy / scale
    feat[:, 2] = torch.sqrt(dx * dx + dy * dy).clamp_min(0.0) / scale
    cx = dx.abs() - (source_fp[:, 0] + target_fp[:, 0]) / 2.0
    cy = dy.abs() - (source_fp[:, 1] + target_fp[:, 1]) / 2.0
    feat[:, 3] = cx / scale
    feat[:, 4] = cy / scale
    feat[:, 5] = torch.relu(-cx) / scale
    feat[:, 6] = torch.relu(-cy) / scale

    xlo = table_bounds[:, 0]
    xhi = table_bounds[:, 1]
    ylo = table_bounds[:, 2]
    yhi = table_bounds[:, 3]
    for xy, fp, slot in ((target_xy, target_fp, 7), (source_xy, source_fp, 8)):
        ml = (xy[:, 0] - fp[:, 0] / 2.0 - xlo) / scale
        mr = (xhi - (xy[:, 0] + fp[:, 0] / 2.0)) / scale
        mb = (xy[:, 1] - fp[:, 1] / 2.0 - ylo) / scale
        mt = (yhi - (xy[:, 1] + fp[:, 1] / 2.0)) / scale
        feat[:, slot] = torch.minimum(
            torch.minimum(ml, mr), torch.minimum(mb, mt))

    if arm_dist is not None:
        feat[:, 9] = arm_dist / scale
    feat[:, 10] = source_gen
    feat[:, 11] = target_gen
    feat[:, 12] = order_delta
    feat[:, 13] = parent_flag
    return feat * valid.unsqueeze(-1).to(feat.dtype)


def build_pose_candidate_tensor(
    part: Dict,
    sample: Dict,
    max_pose: int = MAX_POSE_CANDIDATES,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build pose candidate features + mask + rot_names list."""
    scale = max(float(F._table_diag(sample)), 1e-9)
    cands = list(part.get("pose_candidates") or [])
    if not cands:
        fp = np.asarray(part.get("footprint", [0.0, 0.0]), dtype=np.float32)
        rot = part.get("goal_rotmat", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        R = np.asarray(rot, dtype=np.float32).reshape(3, 3)
        rot6 = np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)
        cands = [{
            "pose_id": str(part.get("pose_tag", "default")),
            "pose_tag": part.get("pose_tag", "default"),
            "rot_name": part.get("rot_name", "unknown"),
            "rotmat": rot,
            "footprint": fp.tolist(),
            "support_area": float(fp[0] * fp[1]),
            "support_area_ratio": 1.0,
            "center_of_mass_height": float(part.get("extent", [0, 0, 0])[2]) * 0.5,
            "stable_probability": 1.0,
            "grasp_total": float(part.get("grasp_total", 0)),
            "topdown_count": float(part.get("topdown_count", 0)),
        }]
    feats = np.zeros((max_pose, POSE_CAND_DIM), dtype=np.float32)
    mask = np.zeros(max_pose, dtype=np.float32)
    rot_names: List[str] = []
    for i, c in enumerate(cands[:max_pose]):
        fp = np.asarray(c.get("footprint", part.get("footprint", [0, 0])), dtype=np.float32)
        rot = c.get("rotmat", part.get("goal_rotmat", [1, 0, 0, 0, 1, 0, 0, 0, 1]))
        R = np.asarray(rot, dtype=np.float32).reshape(3, 3)
        rot6 = np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)
        feats[i, 0:2] = fp[:2] / scale
        feats[i, 2] = float(c.get("support_area", fp[0] * fp[1])) / max(scale ** 2, 1e-12)
        feats[i, 3] = float(c.get("support_area_ratio", 1.0))
        feats[i, 4] = float(c.get("center_of_mass_height", 0.0)) / scale
        feats[i, 5] = float(c.get("stable_probability", 1.0))
        feats[i, 6] = float(c.get("grasp_total", 0)) / 100.0
        feats[i, 7] = float(c.get("topdown_count", 0)) / 30.0
        feats[i, 8:12] = rot6[:4]
        mask[i] = 1.0
        rot_names.append(str(c.get("rot_name", "unknown")))
    return feats, mask, rot_names
