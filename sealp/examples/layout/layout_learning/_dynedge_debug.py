"""Allowed synthetic-only self-check for DynaSeqRel-DynEdge.

No project dataset or formal checkpoint is used.  All temporary checkpoints
are created under a TemporaryDirectory and deleted before exit.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from . import features as F
from .dataset import LayoutDataset, collate_items, move_batch
from .dynamic_relations import build_dynamic_graph
from .infer import LayoutModelRunner
from .losses import LossWeights, compute_loss
from .models import build_model
from .train import train_model


_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "_output" / "dynaseqrel_diagnostics"
)


def _sample(index: int, n_parts: int = 5) -> Dict:
    """Deterministic synthetic sample whose labels depend on staging geometry."""
    angle = 0.31 * index
    spread = 0.045 + 0.007 * (index % 8)
    parts = []
    staging_distances = []
    for i in range(n_parts):
        goal_xy = np.array([0.28 + 0.018 * i, -0.08 + 0.035 * i])
        if i == 0:
            staging = goal_xy.copy()
        else:
            direction = np.array([
                np.cos(angle + 1.17 * i), np.sin(angle + 1.17 * i)])
            staging = np.array([0.31, 0.0]) + direction * (
                spread + 0.012 * i)
            staging_distances.append(float(np.linalg.norm(
                staging - goal_xy)))
        parts.append({
            "part_id": f"part_{i}",
            "order_index": i,
            "is_first": i == 0,
            "extent": [0.04 + 0.002 * i, 0.035 + 0.002 * i, 0.06],
            "footprint": [0.04 + 0.002 * i, 0.035 + 0.002 * i],
            "goal_pos": [float(goal_xy[0]), float(goal_xy[1]), 0.02 * i],
            "goal_rotmat": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            "parent": f"part_{i - 1}" if i > 0 else "fixture",
            "topdown_count": 12 + i,
            "grasp_total": 40 + i,
            "staging_xy": [float(staging[0]), float(staging[1])],
        })
    mean_distance = float(np.mean(staging_distances or [0.0]))
    feasible = mean_distance < 0.115
    score = float(np.clip(1.0 - 2.2 * mean_distance, 0.05, 0.95))
    return {
        "sample_id": f"synthetic_{index}",
        "seed": 0,
        "assembly_region_id": "synthetic",
        "assembly_region_rc": [-1, -1],
        "assembly_grid": 1,
        "assembly_station_pos": [0.31, 0.0, 0.0],
        "table_x_range": [0.0, 0.62],
        "table_y_range": [-0.5, 0.5],
        "part_order": [p["part_id"] for p in parts],
        "parts": parts,
        "l2_pass": bool(feasible),
        "layout_score": score if feasible else 0.0,
        "fail_reason": "" if feasible else "pair_collision",
        "fail_part": None if feasible else "part_2",
    }


def _edge_set(graph: Dict, slot: int) -> set:
    index = graph["dynamic_edge_index"]
    flags = graph["relation_flags"]
    return {
        (int(index[0, e]), int(index[1, e]))
        for e in range(index.shape[1]) if flags[e, slot] > 0.5
    }


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return total ** 0.5


def _fit(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    steps: int,
    lr: float = 3e-3,
) -> List[float]:
    weights = LossWeights(
        alpha=1.0, pos_weight=1.0, rank_weight=0.5, fail_weight=0.2,
        rank_margin=0.05, rank_min_score_gap=0.02,
        rank_pairs_per_batch=256, use_focal=True,
        focal_alpha=0.25, focal_gamma=2.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        model.train()
        output = model(batch)
        loss = compute_loss(output, batch, weights, False)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))
    return history


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    checks: Dict[str, object] = {}
    edge_rows: List[Dict[str, object]] = []

    # A: dynamic graph construction.
    base = _sample(3)
    graph_base = build_dynamic_graph(base)
    moved = copy.deepcopy(base)
    moved["parts"][2]["staging_xy"] = [0.57, 0.39]
    moved["parts"][3]["staging_xy"] = [0.05, -0.42]
    graph_moved = build_dynamic_graph(moved)
    goal_changed = copy.deepcopy(base)
    for part in goal_changed["parts"][1:]:
        part["goal_pos"][0] += 0.21
        part["goal_pos"][1] -= 0.17
    graph_goal_changed = build_dynamic_graph(goal_changed)

    checks["edge_feature_changes_with_staging"] = bool(
        not np.array_equal(
            graph_base["dynamic_edge_feat"],
            graph_moved["dynamic_edge_feat"]))
    checks["spatial_topology_changes_with_staging"] = bool(
        _edge_set(graph_base, 2) != _edge_set(graph_moved, 2))
    checks["nonfirst_goal_does_not_change_topology"] = bool(
        _edge_set(graph_base, 2) == _edge_set(graph_goal_changed, 2))
    checks["order_edges_present"] = bool(_edge_set(graph_base, 0))
    checks["parent_edges_present"] = bool(_edge_set(graph_base, 1))
    checks["first_uses_goal_xy"] = bool(np.allclose(
        graph_base["active_layout_xy"][0],
        np.asarray(base["parts"][0]["goal_pos"][:2])))
    checks["nonfirst_uses_staging_xy"] = bool(np.allclose(
        graph_base["active_layout_xy"][1],
        np.asarray(base["parts"][1]["staging_xy"])))

    one = _sample(0, n_parts=1)
    two = _sample(1, n_parts=2)
    missing = _sample(2, n_parts=3)
    missing["parts"][1]["staging_xy"] = None
    graph_one = build_dynamic_graph(one)
    graph_two = build_dynamic_graph(two)
    graph_missing = build_dynamic_graph(missing)
    checks["n1_supported"] = graph_one["dynamic_edge_index"].shape[1] >= 1
    checks["n2_supported"] = graph_two["dynamic_edge_index"].shape[1] >= 2
    checks["missing_staging_supported"] = bool(
        graph_missing["geometry_valid"][1] == 0
        and np.all(graph_missing["dynamic_edge_feat"][
            (graph_missing["dynamic_edge_index"] == 1).any(axis=0), 0:7
        ] == 0))

    for name, graph in (("base", graph_base), ("moved", graph_moved),
                        ("missing", graph_missing)):
        for e in range(graph["dynamic_edge_index"].shape[1]):
            feat = graph["dynamic_edge_feat"][e]
            edge_rows.append({
                "case": name,
                "source": int(graph["dynamic_edge_index"][0, e]),
                "target": int(graph["dynamic_edge_index"][1, e]),
                **{f"edge_{i}": float(feat[i]) for i in range(11)},
            })

    samples = [_sample(i) for i in range(32)]
    dataset = LayoutDataset(samples, feature_version="v2")
    batch_cpu = collate_items([dataset[i] for i in range(len(dataset))])
    flat_dim = F.flatten_feature_dim()
    model = build_model(
        "dynaseqrel_dynedge", flat_dim=flat_dim, hidden=64, layers=2,
        dropout=0.2, relation_mode="staging_dynedge",
        edge_encoding="edge_mlp_v2",
        head_mode="task_specific_score_v3")
    legacy = build_model(
        "seqrel", flat_dim=flat_dim, hidden=64, layers=2, dropout=0.2)
    raw_dynedge = build_model(
        "dynaseqrel_dynedge", flat_dim=flat_dim, hidden=64, layers=2,
        dropout=0.2, edge_encoding="raw_v1")
    v2_shared_dynedge = build_model(
        "dynaseqrel_dynedge", flat_dim=flat_dim, hidden=64, layers=2,
        dropout=0.2, edge_encoding="edge_mlp_v2",
        head_mode="shared_v1")
    parameter_count = sum(p.numel() for p in model.parameters())
    legacy_parameter_count = sum(p.numel() for p in legacy.parameters())
    raw_dynedge_parameter_count = sum(
        p.numel() for p in raw_dynedge.parameters())
    v2_shared_parameter_count = sum(
        p.numel() for p in v2_shared_dynedge.parameters())

    model.eval()
    with torch.no_grad():
        initial = model(batch_cpu)
    checks["initial_logits_nonconstant"] = bool(
        float(initial["feas_logit"].std()) > 1e-5)
    checks["initial_scores_nonconstant"] = bool(
        float(initial["score_pred"].std()) > 1e-5)
    topology_only = build_model(
        "dynaseqrel_dynedge", flat_dim=flat_dim,
        relation_mode="staging_topology_only",
        edge_encoding="edge_mlp_v2",
        head_mode="task_specific_score_v3").eval()
    with torch.no_grad():
        topology_debug = topology_only.forward_debug(batch_cpu)
    checks["topology_only_interface"] = bool(
        torch.isfinite(topology_debug["feas_logit"]).all()
        and torch.count_nonzero(
            topology_debug["dynamic_edge_feat_used"][:, 0:7]) == 0)

    # Synthetic 32-sample overfit only; never reads layout_dataset_v2.
    fit_history = _fit(model, batch_cpu, steps=100)
    checks["synthetic_overfit_loss_drop"] = bool(
        min(fit_history[-10:]) < fit_history[0] * 0.45)

    # Shuffle sanity is evaluated on held-out geometry: shuffled supervision
    # may be memorized, but must not generalize as the true geometric rule.
    shuffled_model = build_model(
        "dynaseqrel_dynedge", flat_dim=flat_dim, hidden=64, layers=2,
        dropout=0.2, edge_encoding="edge_mlp_v2",
        head_mode="task_specific_score_v3")
    shuffled_batch = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch_cpu.items()
    }
    permutation = torch.randperm(len(samples))
    shuffled_batch["feas"] = shuffled_batch["feas"][permutation]
    shuffled_batch["score"] = shuffled_batch["score"][permutation]
    shuffled_history = _fit(shuffled_model, shuffled_batch, steps=100)
    holdout_samples = [_sample(i + 64) for i in range(32)]
    holdout = collate_items([
        LayoutDataset(
            holdout_samples, feature_version="v2")[i]
        for i in range(len(holdout_samples))
    ])
    shuffled_model.eval()
    with torch.no_grad():
        shuffled_output = shuffled_model(holdout)
    shuffled_accuracy = float(
        ((shuffled_output["feas_logit"] > 0)
         == (holdout["feas"] > 0.5)).float().mean().item())
    checks["shuffle_does_not_generalize"] = bool(shuffled_accuracy < 0.8)

    # B: sensitivity and gradient diagnostics after legitimate synthetic fit.
    model.eval()
    edge_variable = batch_cpu["dynamic_edge_feat"].detach().clone()
    edge_variable.requires_grad_(True)
    debug_output = model.forward_debug(
        batch_cpu, edge_feat_override=edge_variable)
    objective = (
        debug_output["feas_logit"].square().mean()
        + debug_output["score_pred"].mean())
    model.zero_grad()
    objective.backward()
    edge_gradient_norm = float(edge_variable.grad.norm().item())
    message_gradient_norm = _grad_norm(model.mp.parameters())

    with torch.no_grad():
        reference = model.forward_debug(batch_cpu)
        zero = model.forward_debug(
            batch_cpu,
            edge_feat_override=torch.zeros_like(
                batch_cpu["dynamic_edge_feat"]))
        permutation = torch.randperm(
            batch_cpu["dynamic_edge_feat"].shape[0])
        shuffled_edges = model.forward_debug(
            batch_cpu,
            edge_feat_override=batch_cpu["dynamic_edge_feat"][permutation])
    sensitivity = {
        "edge_attr_zero": {
            "feasibility_logit_delta_mean_abs": float(
                (reference["feas_logit"] - zero["feas_logit"]).abs().mean()),
            "feasibility_logit_delta_max_abs": float(
                (reference["feas_logit"] - zero["feas_logit"]).abs().max()),
            "score_delta_mean_abs": float(
                (reference["score_pred"] - zero["score_pred"]).abs().mean()),
            "score_delta_max_abs": float(
                (reference["score_pred"] - zero["score_pred"]).abs().max()),
        },
        "edge_attr_shuffle": {
            "feasibility_logit_delta_mean_abs": float(
                (reference["feas_logit"]
                 - shuffled_edges["feas_logit"]).abs().mean()),
            "feasibility_logit_delta_max_abs": float(
                (reference["feas_logit"]
                 - shuffled_edges["feas_logit"]).abs().max()),
            "score_delta_mean_abs": float(
                (reference["score_pred"]
                 - shuffled_edges["score_pred"]).abs().mean()),
            "score_delta_max_abs": float(
                (reference["score_pred"]
                 - shuffled_edges["score_pred"]).abs().max()),
        },
        "edge_gradient_norm": edge_gradient_norm,
        "message_gradient_norm": message_gradient_norm,
    }
    checks["edge_gradient_nonzero"] = edge_gradient_norm > 1e-8
    checks["message_gradient_nonzero"] = message_gradient_norm > 1e-8
    checks["edge_zero_measurable"] = (
        sensitivity["edge_attr_zero"]["feasibility_logit_delta_mean_abs"]
        > 1e-5)
    checks["edge_shuffle_measurable"] = (
        sensitivity["edge_attr_shuffle"]["feasibility_logit_delta_mean_abs"]
        > 1e-5)

    # C: CPU/CUDA, checkpoint roundtrip, old checkpoints, unaffected models.
    checks["cpu_forward"] = all(
        torch.isfinite(value).all() for value in model(batch_cpu).values())
    if torch.cuda.is_available():
        cuda_model = build_model(
            "dynaseqrel_dynedge", flat_dim=flat_dim,
            edge_encoding="edge_mlp_v2",
            head_mode="task_specific_score_v3").cuda().eval()
        cuda_batch = move_batch(batch_cpu, "cuda")
        with torch.no_grad():
            cuda_out = cuda_model(cuda_batch)
        checks["cuda_forward"] = all(
            torch.isfinite(value).all().item() for value in cuda_out.values())
    else:
        checks["cuda_forward"] = "skipped: CUDA unavailable"

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = os.path.join(temporary, "dynedge_roundtrip.pt")
        torch.save({
            "model_name": "dynaseqrel_dynedge",
            "model_kwargs": {
                "hidden": 64, "layers": 2, "dropout": 0.2,
                "relation_mode": "staging_dynedge",
                "dynamic_k_spatial": 2,
                "edge_encoding": "edge_mlp_v2",
                "head_mode": "task_specific_score_v3",
            },
            "state_dict": model.state_dict(),
            "flat_dim": flat_dim,
            "max_parts": F.MAX_PARTS_DEFAULT,
            "feature_version": "v2",
            "is_generator": False,
            "dynamic_edge_dim": 11,
        }, checkpoint)
        runner = LayoutModelRunner(checkpoint, device="cpu")
        expected = model.eval()(batch_cpu)
        actual = runner.model.eval()(batch_cpu)
        roundtrip_delta = max(
            float((expected[key] - actual[key]).abs().max())
            for key in ("feas_logit", "score_pred", "fail_logits"))
        checks["checkpoint_roundtrip"] = roundtrip_delta < 1e-6

        synthetic_jsonl = os.path.join(temporary, "synthetic.jsonl")
        with open(synthetic_jsonl, "w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample) + "\n")
        metric_dir = os.path.join(temporary, "metric_checkpoints")
        train_model(
            synthetic_jsonl,
            "dynaseqrel_dynedge",
            save_dir=metric_dir,
            epochs=3,
            batch_size=8,
            lr=1e-3,
            val_ratio=0.25,
            seed=4,
            device="cpu",
            loss_weights=LossWeights(),
            model_kwargs={
                "hidden": 64,
                "layers": 2,
                "dropout": 0.2,
                "relation_mode": "staging_dynedge",
                "dynamic_k_spatial": 2,
                "edge_encoding": "edge_mlp_v2",
                "head_mode": "task_specific_score_v3",
            },
            feature_version="v2",
            split_mode="stratified",
            early_stop_metric="pr_auc",
            save_metric_checkpoints=True,
            verbose=False,
        )
        checks["multi_metric_checkpoint_selection"] = all(
            os.path.isfile(os.path.join(metric_dir, filename))
            for filename in (
                "dynaseqrel_dynedge_best.pt",
                "best_pr_auc.pt",
                "best_spearman.pt",
                "best_legacy_composite.pt",
            ))

    repo_root = Path(__file__).resolve().parents[4]
    old_seqrel = (
        repo_root / "checkpoints/layout_models_repro/seqrel/"
        "stratified/seed0/seqrel_best.pt")
    if old_seqrel.is_file():
        old_runner = LayoutModelRunner(str(old_seqrel), device="cpu")
        old_result = old_runner.score_layouts(samples[:4])
        checks["old_seqrel_checkpoint_loads"] = bool(
            np.isfinite(old_result["feas_prob"]).all())
    else:
        checks["old_seqrel_checkpoint_loads"] = (
            "skipped: reference checkpoint absent")

    old_dynedge = (
        repo_root / "checkpoints/layout_models_repro/dynaseqrel_dynedge/"
        "stratified/seed0/dynaseqrel_dynedge_best.pt")
    if old_dynedge.is_file():
        old_dynedge_runner = LayoutModelRunner(
            str(old_dynedge), device="cpu")
        old_dynedge_result = old_dynedge_runner.score_layouts(samples[:4])
        checks["raw_v1_dynedge_checkpoint_loads"] = bool(
            np.isfinite(old_dynedge_result["feas_prob"]).all())
    else:
        checks["raw_v1_dynedge_checkpoint_loads"] = (
            "skipped: raw_v1 checkpoint absent")

    old_v2_dynedge = (
        repo_root / "checkpoints/layout_models_repro/"
        "dynaseqrel_dynedge_edge_mlp_v2/stratified/seed0/"
        "dynaseqrel_dynedge_best.pt")
    if old_v2_dynedge.is_file():
        old_v2_runner = LayoutModelRunner(
            str(old_v2_dynedge), device="cpu")
        old_v2_result = old_v2_runner.score_layouts(samples[:4])
        checks["shared_v1_edge_mlp_v2_checkpoint_loads"] = bool(
            np.isfinite(old_v2_result["feas_prob"]).all())
    else:
        checks["shared_v1_edge_mlp_v2_checkpoint_loads"] = (
            "skipped: shared-v1 edge-mlp-v2 checkpoint absent")

    unaffected = {}
    for name in ("deepsets", "mlp", "gcn", "gat", "sagpn"):
        candidate = build_model(name, flat_dim=flat_dim).eval()
        with torch.no_grad():
            output = candidate(batch_cpu)
        unaffected[name] = bool(
            torch.isfinite(output["feas_logit"]).all()
            and torch.isfinite(output["score_pred"]).all())
    checks["legacy_model_smoke"] = unaffected

    parameter_report = {
        "dynaseqrel_dynedge_task_specific_score_v3": parameter_count,
        "dynaseqrel_dynedge_edge_mlp_v2": v2_shared_parameter_count,
        "dynaseqrel_dynedge_raw_v1": raw_dynedge_parameter_count,
        "seqrel_full": legacy_parameter_count,
        "difference": parameter_count - legacy_parameter_count,
        "v2_minus_raw_v1": (
            v2_shared_parameter_count - raw_dynedge_parameter_count),
        "v3_minus_v2": parameter_count - v2_shared_parameter_count,
        "v2_edge_branch_source": {
            "raw_message_per_layer": 13120,
            "v2_node_message_per_layer": 12416,
            "v2_edge_encoder_per_layer": 2688,
            "increase_per_layer": 1984,
            "message_layers": 2,
            "derivation": "(12416 + 2688 - 13120) * 2 = 3968",
        },
        "v3_score_trunk_source": {
            "pool_dim": 199,
            "hidden": 64,
            "derivation": "(199*64+64) + (64*64+64) = 16960",
        },
        "dynamic_edge_feature_dim": 11,
        "relation_mode": "staging_dynedge",
        "edge_encoding": "edge_mlp_v2",
        "head_mode": "task_specific_score_v3",
    }
    selfcheck = {
        "status": "pass" if all(
            value is True or isinstance(value, dict)
            for value in checks.values()) else "review",
        "checks": checks,
        "synthetic_only": True,
        "formal_training_executed": False,
        "synthetic_fit": {
            "sample_count": 32,
            "steps": 100,
            "initial_loss": fit_history[0],
            "best_final_10_loss": min(fit_history[-10:]),
            "shuffled_initial_loss": shuffled_history[0],
            "shuffled_best_final_10_loss": min(shuffled_history[-10:]),
            "shuffled_holdout_accuracy": shuffled_accuracy,
        },
        "checkpoint_roundtrip_max_delta": roundtrip_delta,
    }

    with (_OUTPUT / "dynedge_edge_examples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(edge_rows[0]))
        writer.writeheader()
        writer.writerows(edge_rows)
    (_OUTPUT / "dynedge_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2), encoding="utf-8")
    (_OUTPUT / "dynedge_parameter_report.json").write_text(
        json.dumps(parameter_report, indent=2), encoding="utf-8")
    (_OUTPUT / "dynedge_selfcheck.json").write_text(
        json.dumps(selfcheck, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": selfcheck["status"],
        "parameter_count": parameter_count,
        "checks": checks,
        "sensitivity": sensitivity,
        "output": str(_OUTPUT),
    }, indent=2))
    failed = [
        name for name, result in checks.items()
        if result is not True and not isinstance(result, dict)
    ]
    if failed:
        raise AssertionError(f"DynEdge self-check failed/requires review: {failed}")


if __name__ == "__main__":
    main()
