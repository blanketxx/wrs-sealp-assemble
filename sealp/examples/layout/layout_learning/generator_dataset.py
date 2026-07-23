"""Generator dataset helpers and geometry-holdout utilities."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import features as F
from .geometry_features import (
    FEATURE_SCHEMA_VERSION,
    geometry_domain_key,
    load_or_compute_geometry,
    mesh_sha256_for_sample,
)
from .generator_relations import (
    MAX_POSE_CANDIDATES,
    MAX_ROTATIONS,
    build_pose_candidate_tensor,
    build_static_graph,
)

GENERATOR_SCHEMA_VERSION = "generator_v1"


def _target_pose_index(part: Dict) -> int:
    if "target_pose_index" in part:
        return int(part["target_pose_index"])
    return 0


def _target_rotation_index(part: Dict, rot_names: List[str]) -> int:
    if "target_rotation_index" in part:
        return int(part["target_rotation_index"])
    name = str(part.get("rot_name", "unknown"))
    if name in rot_names:
        return rot_names.index(name)
    return 0


def enrich_part_for_generator(part: Dict, sample: Dict, norm_scale: float) -> Dict:
    """Attach geometry / pose-candidate tensors to one part dict (numpy side)."""
    geo, sha, gpath = load_or_compute_geometry(part, norm_scale=norm_scale)
    pose_feat, pose_mask, rot_names = build_pose_candidate_tensor(part, sample)
    fp = np.asarray(part.get("footprint", [0.05, 0.05]), dtype=np.float32)
    return {
        "geometry_feat": geo.astype(np.float32),
        "mesh_sha256": sha,
        "geometry_feature_path": gpath,
        "pose_candidate_feat": pose_feat.astype(np.float32),
        "pose_candidate_mask": pose_mask.astype(np.float32),
        "target_pose_index": np.int64(_target_pose_index(part)),
        "target_rotation_index": np.int64(_target_rotation_index(part, rot_names)),
        "selected_footprint": fp.astype(np.float32),
        "rot_names": rot_names,
        "is_first_flag": np.float32(1.0 if part.get("is_first", False) else 0.0),
        "order_index_val": np.float32(part.get("order_index", 0)),
    }


def generator_item_fields(sample: Dict, feature_version: str = F.DEFAULT_FEATURE_VERSION) -> Dict:
    """Extra collate fields for RelSeqGen."""
    parts = F._ordered_parts(sample)
    n = max(1, len(parts))
    scale = F._len_scale(sample, feature_version)
    static_graph = build_static_graph(sample)

    geo = np.zeros((n, 20), dtype=np.float32)
    pose_feat = np.zeros((n, MAX_POSE_CANDIDATES, 12), dtype=np.float32)
    pose_mask = np.zeros((n, MAX_POSE_CANDIDATES), dtype=np.float32)
    target_pose = np.zeros(n, dtype=np.int64)
    target_rot = np.zeros(n, dtype=np.int64)
    selected_fp = np.zeros((n, 2), dtype=np.float32)
    is_first = np.zeros(n, dtype=np.float32)
    order_index = np.zeros(n, dtype=np.float32)

    for i, p in enumerate(parts):
        extra = enrich_part_for_generator(p, sample, scale)
        geo[i] = extra["geometry_feat"]
        pose_feat[i] = extra["pose_candidate_feat"]
        pose_mask[i] = extra["pose_candidate_mask"]
        target_pose[i] = extra["target_pose_index"]
        target_rot[i] = extra["target_rotation_index"]
        selected_fp[i] = extra["selected_footprint"]
        is_first[i] = extra["is_first_flag"]
        order_index[i] = extra["order_index_val"]

    n_max = max(n, 1)
    adj = np.zeros((n_max, n_max), dtype=np.float32)
    edge_attr = np.zeros((n_max, n_max, 10), dtype=np.float32)
    ei = static_graph["static_edge_index"]
    ef = static_graph["static_edge_feat"]
    for e in range(ei.shape[1]):
        u, v = int(ei[0, e]), int(ei[1, e])
        if u < n_max and v < n_max:
            adj[u, v] = 1.0
            edge_attr[u, v] = ef[e]

    elite = bool(sample.get("l2_pass", False))
    score = float(sample.get("layout_score", 0.0) if elite else 0.0)
    return {
        "geometry_feat": geo,
        "pose_candidate_feat": pose_feat,
        "pose_candidate_mask": pose_mask,
        "target_pose_index": target_pose,
        "target_rotation_index": target_rot,
        "selected_footprint": selected_fp,
        "is_first_mask": is_first,
        "order_index": order_index,
        "static_adj": adj,
        "static_edge_attr": edge_attr,
        "mesh_sha256_bundle": mesh_sha256_for_sample(sample),
        "geometry_domain": geometry_domain_key(sample),
        "schema_version": sample.get("schema_version", "legacy"),
        "is_elite_flag": np.float32(
            1.0 if sample.get("is_elite", elite) else 0.0),
        "proposal_mask_hint": np.float32(1.0 if elite else 0.0),
        "layout_score_val": np.float32(score),
    }


def geometry_holdout_split(
    samples: List[Dict],
    val_ratio: float,
    seed: int,
    holdout_domains: Optional[Set[str]] = None,
) -> Tuple[List[int], List[int], Dict]:
    """Split by mesh SHA256 / geometry domain without leakage."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(samples))
    domains = [geometry_domain_key(s) for s in samples]
    mesh_keys = [mesh_sha256_for_sample(s) for s in samples]

    if holdout_domains:
        hold = set(holdout_domains)
        val = [int(i) for i in idx if domains[i] in hold]
        train = [int(i) for i in idx if domains[i] not in hold]
        if not train or not val:
            raise RuntimeError("geometry holdout produced empty train or val split")
        _assert_no_mesh_leakage(samples, train, val)
        return sorted(train), sorted(val), {
            "split_mode": "geometry_holdout",
            "holdout_domains": sorted(hold),
        }

    uniq_mesh = sorted(set(mesh_keys))
    rng.shuffle(uniq_mesh)
    n_hold = max(1, int(round(len(uniq_mesh) * val_ratio)))
    hold_mesh = set(uniq_mesh[:n_hold])
    val = [int(i) for i in idx if mesh_keys[i] in hold_mesh]
    train = [int(i) for i in idx if mesh_keys[i] not in hold_mesh]
    if not train or not val:
        raise RuntimeError("geometry_holdout mesh split failed")
    _assert_no_mesh_leakage(samples, train, val)
    return sorted(train), sorted(val), {
        "split_mode": "geometry_holdout",
        "val_mesh_sha256": sorted(hold_mesh),
        "train_mesh_sha256": sorted(set(mesh_keys) - hold_mesh),
    }


def _assert_no_mesh_leakage(samples: List[Dict], train: List[int], val: List[int]) -> None:
    train_mesh = {mesh_sha256_for_sample(samples[i]) for i in train}
    val_mesh = {mesh_sha256_for_sample(samples[i]) for i in val}
    overlap = train_mesh & val_mesh
    if overlap:
        raise RuntimeError(
            f"geometry leakage detected: shared mesh SHA256 between train/val: {sorted(overlap)}")


def sample_schema_extensions(sample: Dict) -> Dict:
    """Optional fields to add when writing new dataset rows (non-breaking)."""
    return {
        "schema_version": sample.get("schema_version", GENERATOR_SCHEMA_VERSION),
        "assembly_id": sample.get("assembly_id", sample.get("task_id", "unknown")),
        "geometry_domain": geometry_domain_key(sample),
        "mesh_sha256_bundle": mesh_sha256_for_sample(sample),
        "condition_signature": sample.get("condition_signature"),
        "layout_signature": sample.get("layout_signature"),
        "is_elite": bool(sample.get("is_elite", sample.get("l2_pass", False))),
        "seen_geometry_during_training": sample.get("seen_geometry_during_training"),
    }


def rot_names_per_part(sample: Dict) -> List[List[str]]:
    """Rotation name lists per part for structured inference."""
    out: List[List[str]] = []
    for p in F._ordered_parts(sample):
        _, _, names = build_pose_candidate_tensor(p, sample)
        out.append(names or ["unknown"])
    return out
