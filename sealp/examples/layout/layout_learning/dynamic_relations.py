"""Candidate-dependent staging-space relations for DynaSeqRel-DynEdge.

This module is deliberately independent from ``features.build_graph_feature``:
legacy models keep their goal-space graph and checkpoint semantics unchanged.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from . import features as F


DYN_EDGE_FEATURE_DIM = 11
DYN_RELATION_MODES = ("staging_dynedge", "staging_topology_only")


def build_active_layout_geometry(
    sample: Dict,
) -> Dict[str, np.ndarray]:
    """Return physical active XY, footprints, and per-part geometry validity.

    The preassembled first part uses goal XY.  Other parts must have staging XY;
    missing staging is explicitly invalid and never falls back to goal XY.
    """
    parts = F._ordered_parts(sample)
    n_real = len(parts)
    n = max(n_real, 1)
    active_xy = np.zeros((n, 2), dtype=np.float32)
    footprint = np.zeros((n, 2), dtype=np.float32)
    geometry_valid = np.zeros(n, dtype=np.float32)

    for i, part in enumerate(parts):
        fp = np.asarray(part.get("footprint", [0.0, 0.0]), dtype=np.float32)
        footprint[i, :min(2, fp.size)] = fp[:2]
        if bool(part.get("is_first", False)):
            goal = np.asarray(part.get("goal_pos", [0.0, 0.0, 0.0]),
                              dtype=np.float32)
            active_xy[i] = goal[:2]
            geometry_valid[i] = 1.0
        else:
            staging = part.get("staging_xy")
            if staging is not None:
                active_xy[i] = np.asarray(staging, dtype=np.float32)[:2]
                geometry_valid[i] = 1.0

    return {
        "active_layout_xy": active_xy,
        "footprint": footprint,
        "geometry_valid": geometry_valid,
        "num_parts": np.int64(n_real),
    }


def _add_bidirectional(
    edges: Dict[Tuple[int, int], np.ndarray],
    a: int,
    b: int,
    relation_slot: int,
) -> None:
    if a == b or a < 0 or b < 0:
        return
    for source, target in ((a, b), (b, a)):
        flags = edges.setdefault(
            (source, target), np.zeros(3, dtype=np.float32))
        flags[relation_slot] = 1.0


def build_dynamic_edge_index(
    sample: Dict,
    active_geometry: Dict[str, np.ndarray] | None = None,
    k_spatial: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build directed order/parent/staging-kNN edges with multi-hot flags.

    Returns ``edge_index[2,E]`` and ``relation_flags[E,3]`` ordered as
    ``is_order, is_parent, is_spatial``.  Spatial topology uses active layout
    positions only; goal XY is never consulted except for the first part's
    explicitly defined active position.
    """
    if k_spatial < 0:
        raise ValueError("k_spatial must be >= 0")
    parts = F._ordered_parts(sample)
    n = len(parts)
    geometry = active_geometry or build_active_layout_geometry(sample)
    active_xy = geometry["active_layout_xy"]
    valid = geometry["geometry_valid"] > 0.5
    edges: Dict[Tuple[int, int], np.ndarray] = {}

    order_sorted = sorted(
        range(n), key=lambda i: float(parts[i].get("order_index", i)))
    for a, b in zip(order_sorted[:-1], order_sorted[1:]):
        _add_bidirectional(edges, a, b, 0)

    pid_to_idx = {
        str(part.get("part_id")): i for i, part in enumerate(parts)
    }
    for i, part in enumerate(parts):
        parent = part.get("parent")
        if parent is not None and str(parent) in pid_to_idx:
            _add_bidirectional(edges, i, pid_to_idx[str(parent)], 1)

    valid_idx = np.flatnonzero(valid[:n])
    if k_spatial > 0 and valid_idx.size >= 2:
        for i in valid_idx:
            candidates = valid_idx[valid_idx != i]
            distances = np.linalg.norm(
                active_xy[candidates] - active_xy[i], axis=1)
            nearest = candidates[
                np.argsort(distances, kind="stable")[:min(k_spatial, len(candidates))]
            ]
            for j in nearest:
                _add_bidirectional(edges, int(i), int(j), 2)

    # Keep downstream sparse operations well-defined for empty/singleton graphs.
    if not edges:
        edges[(0, 0)] = np.zeros(3, dtype=np.float32)

    edge_index = np.asarray(list(edges.keys()), dtype=np.int64).T
    relation_flags = np.stack(list(edges.values())).astype(np.float32)
    return edge_index, relation_flags


def build_dynamic_edge_feature(
    sample: Dict,
    edge_index: np.ndarray,
    relation_flags: np.ndarray,
    active_geometry: Dict[str, np.ndarray] | None = None,
    relation_mode: str = "staging_dynedge",
) -> np.ndarray:
    """Build the fixed 11-D dynamic relation attributes for existing edges."""
    if relation_mode not in DYN_RELATION_MODES:
        raise ValueError(
            f"unknown relation_mode={relation_mode!r}; "
            f"expected one of {DYN_RELATION_MODES}")
    geometry = active_geometry or build_active_layout_geometry(sample)
    active_xy = geometry["active_layout_xy"]
    footprint = geometry["footprint"]
    valid = geometry["geometry_valid"] > 0.5
    scale = max(float(F._table_diag(sample)), 1e-9)
    e_count = int(edge_index.shape[1])
    edge_feat = np.zeros((e_count, DYN_EDGE_FEATURE_DIM), dtype=np.float32)

    for e in range(e_count):
        source, target = int(edge_index[0, e]), int(edge_index[1, e])
        edge_feat[e, 7:10] = relation_flags[e]
        geometry_valid = (
            source < len(valid) and target < len(valid)
            and bool(valid[source]) and bool(valid[target])
        )
        edge_feat[e, 10] = float(geometry_valid)
        if not geometry_valid or relation_mode == "staging_topology_only":
            continue

        dx, dy = active_xy[target] - active_xy[source]
        clearance_x = abs(float(dx)) - (
            float(footprint[source, 0]) + float(footprint[target, 0])) / 2.0
        clearance_y = abs(float(dy)) - (
            float(footprint[source, 1]) + float(footprint[target, 1])) / 2.0
        edge_feat[e, 0] = float(dx) / scale
        edge_feat[e, 1] = float(dy) / scale
        edge_feat[e, 2] = float(np.hypot(dx, dy)) / scale
        edge_feat[e, 3] = clearance_x / scale
        edge_feat[e, 4] = clearance_y / scale
        edge_feat[e, 5] = max(0.0, -clearance_x) / scale
        edge_feat[e, 6] = max(0.0, -clearance_y) / scale

    return edge_feat


def build_dynamic_graph(
    sample: Dict,
    k_spatial: int = 2,
    relation_mode: str = "staging_dynedge",
) -> Dict[str, np.ndarray]:
    """Build a complete local-index dynamic graph for one sample."""
    geometry = build_active_layout_geometry(sample)
    edge_index, relation_flags = build_dynamic_edge_index(
        sample, geometry, k_spatial=k_spatial)
    edge_feat = build_dynamic_edge_feature(
        sample, edge_index, relation_flags, geometry,
        relation_mode=relation_mode)
    return {
        **geometry,
        "dynamic_edge_index": edge_index,
        "dynamic_edge_feat": edge_feat,
        "relation_flags": relation_flags,
    }
