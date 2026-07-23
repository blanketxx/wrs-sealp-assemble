"""Synthetic and checkpoint-only checks for strict-v3 and adapter-v4.

This module never starts a training epoch and never reads or modifies the
project dataset.  It uses deterministic synthetic samples plus existing
read-only checkpoints.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Subset

from . import features as F
from ._dynedge_debug import _sample
from .dataset import LayoutDataset, collate_items
from .infer import LayoutModelRunner
from .losses import LossWeights, compute_loss
from .models import build_model
from .train import (
    _build_strict_shared_v3,
    _load_frozen_adapter_base,
    _state_hash,
    train_model,
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "dynaseqrel_diagnostics"
_BASE_CHECKPOINT = (
    _REPO_ROOT / "checkpoints/layout_models_repro/"
    "dynaseqrel_dynedge_edge_mlp_v2/stratified/seed0/"
    "dynaseqrel_dynedge_best.pt"
)


def _max_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _parameter_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _changed_parameters(
    before: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> List[str]:
    return [
        name for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    ]


def _loader_order(dataset: LayoutDataset, seed: int) -> List[int]:
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        Subset(dataset, list(range(len(dataset)))),
        batch_size=5,
        shuffle=True,
        generator=generator,
        collate_fn=collate_items,
    )
    order: List[int] = []
    for batch_index, batch in enumerate(loader):
        order.extend(int(value) for value in batch["sample_index"].tolist())
        if batch_index >= 2:
            break
    return order


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = [_sample(index) for index in range(32)]
    dataset = LayoutDataset(samples, feature_version="v2")
    batch = collate_items([dataset[index] for index in range(16)])
    flat_dim = F.flatten_feature_dim()
    common_kwargs = {
        "hidden": 64,
        "layers": 2,
        "dropout": 0.2,
        "relation_mode": "staging_dynedge",
        "dynamic_k_spatial": 2,
        "edge_encoding": "edge_mlp_v2",
    }

    # ---- strict-v3 initialization ----
    model_init_seed = 0
    score_trunk_init_seed = 100003
    dataloader_seed = 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_init_seed)
        v2_reference = build_model(
            "dynaseqrel_dynedge", flat_dim=flat_dim,
            **common_kwargs, head_mode="shared_v1")
    strict_v3, strict_meta = _build_strict_shared_v3(
        flat_dim,
        {**common_kwargs, "head_mode": "task_specific_score_v3"},
        model_init_seed,
        score_trunk_init_seed,
    )
    strict_v3_repeat, strict_repeat_meta = _build_strict_shared_v3(
        flat_dim,
        {**common_kwargs, "head_mode": "task_specific_score_v3"},
        model_init_seed,
        score_trunk_init_seed,
    )

    reference_params = dict(v2_reference.named_parameters())
    strict_params = dict(strict_v3.named_parameters())
    shared_rows = []
    mismatches = []
    max_shared_delta = 0.0
    for name, parameter in strict_params.items():
        if name in reference_params:
            delta = _max_delta(parameter, reference_params[name])
            max_shared_delta = max(max_shared_delta, delta)
            if delta != 0.0:
                mismatches.append(name)
            category = "shared"
        else:
            delta = None
            category = "v3_only"
        shared_rows.append({
            "parameter_name": name,
            "category": category,
            "shape": str(tuple(parameter.shape)),
            "numel": parameter.numel(),
            "max_abs_diff": "" if delta is None else delta,
        })
    v3_only = sorted(set(strict_params) - set(reference_params))
    if any(not name.startswith("score_trunk.") for name in v3_only):
        raise AssertionError(f"unexpected v3-only parameters: {v3_only}")
    if max_shared_delta != 0.0 or mismatches:
        raise AssertionError(
            f"strict shared parameters differ: max={max_shared_delta}, "
            f"names={mismatches}")

    v2_reference.eval()
    strict_v3.eval()
    strict_v3_repeat.eval()
    with torch.no_grad():
        v2_initial = v2_reference.forward_debug(batch)
        v3_initial = strict_v3.forward_debug(batch)
        v3_repeat_initial = strict_v3_repeat.forward_debug(batch)
    loader_order_v2 = _loader_order(dataset, dataloader_seed)
    loader_order_v3 = _loader_order(dataset, dataloader_seed)
    loader_order_repeat = _loader_order(dataset, dataloader_seed)
    repeat_state_equal = all(
        torch.equal(value, strict_v3_repeat.state_dict()[name])
        for name, value in strict_v3.state_dict().items())
    strict_result = {
        "status": "pass",
        "formal_training_executed": False,
        "model_init_seed": model_init_seed,
        "score_trunk_init_seed": score_trunk_init_seed,
        "dataloader_seed": dataloader_seed,
        "matched_parameter_tensor_count": strict_meta[
            "shared_parameter_tensor_count"],
        "matched_parameter_count": strict_meta["shared_parameter_count"],
        "shared_parameter_hash": strict_meta["shared_parameter_hash"],
        "repeat_shared_parameter_hash": strict_repeat_meta[
            "shared_parameter_hash"],
        "max_abs_diff": max_shared_delta,
        "mismatched_parameter_names": mismatches,
        "v3_only_parameter_names": v3_only,
        "v3_only_all_score_trunk": all(
            name.startswith("score_trunk.") for name in v3_only),
        "feasibility_logits_max_abs_diff": _max_delta(
            v2_initial["feas_logit"], v3_initial["feas_logit"]),
        "fail_logits_max_abs_diff": _max_delta(
            v2_initial["fail_logits"], v3_initial["fail_logits"]),
        "score_predictions_max_abs_diff": _max_delta(
            v2_initial["score_pred"], v3_initial["score_pred"]),
        "first_three_batches_v2": loader_order_v2,
        "first_three_batches_v3": loader_order_v3,
        "first_three_batches_repeat": loader_order_repeat,
        "dataloader_order_equal": (
            loader_order_v2 == loader_order_v3 == loader_order_repeat),
        "strict_v3_repeat_state_equal": repeat_state_equal,
        "strict_v3_repeat_output_max_abs_diff": max(
            _max_delta(v3_initial[key], v3_repeat_initial[key])
            for key in ("feas_logit", "fail_logits", "score_pred")),
    }
    if (strict_result["feasibility_logits_max_abs_diff"] != 0.0
            or strict_result["fail_logits_max_abs_diff"] != 0.0
            or not strict_result["dataloader_order_equal"]
            or not repeat_state_equal):
        raise AssertionError(f"strict-v3 check failed: {strict_result}")
    (_OUTPUT / "strict_v3_initialization_check.json").write_text(
        json.dumps(strict_result, indent=2), encoding="utf-8")
    _write_csv(_OUTPUT / "strict_v3_shared_parameters.csv", shared_rows)

    # ---- frozen residual score adapter ----
    if not _BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing base checkpoint: {_BASE_CHECKPOINT}")
    checkpoint = torch.load(
        _BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    base_model = build_model(
        "dynaseqrel_dynedge", flat_dim=flat_dim,
        **checkpoint["model_kwargs"]).eval()
    base_model.load_state_dict(checkpoint["state_dict"], strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        adapter_model = build_model(
            "dynaseqrel_dynedge", flat_dim=flat_dim, **common_kwargs,
            head_mode="frozen_residual_score_adapter_v4",
            adapter_hidden_dim=64, adapter_dropout=0.2)
    adapter_meta = _load_frozen_adapter_base(
        adapter_model, str(_BASE_CHECKPOINT), "v2", {
            **common_kwargs,
            "head_mode": "frozen_residual_score_adapter_v4",
            "adapter_hidden_dim": 64,
            "adapter_dropout": 0.2,
        })
    base_model.eval()
    adapter_model.eval()
    with torch.no_grad():
        base_initial = base_model.forward_debug(batch)
        adapter_initial = adapter_model.forward_debug(batch)
    output_equivalence = {
        "feasibility_logits_max_abs_diff": _max_delta(
            base_initial["feas_logit"], adapter_initial["feas_logit"]),
        "fail_logits_max_abs_diff": _max_delta(
            base_initial["fail_logits"], adapter_initial["fail_logits"]),
        "score_logits_max_abs_diff": _max_delta(
            base_initial["score_logit"], adapter_initial["score_logit"]),
        "score_predictions_max_abs_diff": _max_delta(
            base_initial["score_pred"], adapter_initial["score_pred"]),
        "delta_score_logit_max_abs": float(
            adapter_initial["delta_score_logit"].abs().max().item()),
    }
    if any(value != 0.0 for value in output_equivalence.values()):
        raise AssertionError(
            f"adapter initialization is not exactly equivalent: "
            f"{output_equivalence}")

    optimizer_parameters = [
        parameter for parameter in adapter_model.parameters()
        if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        optimizer_parameters, lr=5e-4, weight_decay=1e-4)
    optimizer_names = {
        name for name, parameter in adapter_model.named_parameters()
        if any(parameter is candidate for group in optimizer.param_groups
               for candidate in group["params"])
    }
    before = _parameter_snapshot(adapter_model)
    adapter_model.train()
    mode_check = {
        "adapter_training": adapter_model.score_adapter.training,
        "frozen_modules_eval": all(
            not module.training
            for module in adapter_model.frozen_base_modules().values()),
    }
    with torch.no_grad():
        first_train_forward = adapter_model.forward_debug(batch)
        second_train_forward = adapter_model.forward_debug(batch)
    mode_check.update({
        "consecutive_feasibility_equal": torch.equal(
            first_train_forward["feas_logit"],
            second_train_forward["feas_logit"]),
        "consecutive_frozen_representation_equal": torch.equal(
            first_train_forward["encoded"],
            second_train_forward["encoded"]),
    })

    output = adapter_model(batch)
    weights = LossWeights(
        alpha=1.0, rank_weight=0.5, fail_weight=0.0,
        pos_weight=1.0, score_only=True)
    loss = compute_loss(output, batch, weights, False)["loss"]
    optimizer.zero_grad()
    loss.backward()
    adapter_gradient_norm = sum(
        float(parameter.grad.detach().pow(2).sum().item())
        for parameter in optimizer_parameters if parameter.grad is not None
    ) ** 0.5
    frozen_gradients_none = all(
        parameter.grad is None
        for name, parameter in adapter_model.named_parameters()
        if not name.startswith("score_adapter."))
    optimizer.step()
    changed = _changed_parameters(before, adapter_model)
    frozen_changed = [
        name for name in changed if not name.startswith("score_adapter.")]
    adapter_changed = [
        name for name in changed if name.startswith("score_adapter.")]

    adapter_model.eval()
    with torch.no_grad():
        after_step = adapter_model.forward_debug(batch)
    after_step_checks = {
        "feasibility_logits_max_abs_diff": _max_delta(
            base_initial["feas_logit"], after_step["feas_logit"]),
        "fail_logits_max_abs_diff": _max_delta(
            base_initial["fail_logits"], after_step["fail_logits"]),
        "delta_score_logit_max_abs": float(
            after_step["delta_score_logit"].abs().max().item()),
        "score_predictions_changed": not torch.equal(
            base_initial["score_pred"], after_step["score_pred"]),
    }

    with tempfile.TemporaryDirectory() as temporary:
        roundtrip_path = os.path.join(temporary, "adapter_v4.pt")
        torch.save({
            "model_name": "dynaseqrel_dynedge",
            "model_kwargs": {
                **common_kwargs,
                "head_mode": "frozen_residual_score_adapter_v4",
                "adapter_hidden_dim": 64,
                "adapter_dropout": 0.2,
            },
            "state_dict": adapter_model.state_dict(),
            "flat_dim": flat_dim,
            "feature_version": "v2",
        }, roundtrip_path)
        reloaded = LayoutModelRunner(roundtrip_path, device="cpu").model.eval()
        with torch.no_grad():
            roundtrip_output = reloaded.forward_debug(batch)
        roundtrip_delta = max(
            _max_delta(after_step[key], roundtrip_output[key])
            for key in ("feas_logit", "fail_logits", "score_pred",
                        "score_logit", "delta_score_logit"))
        synthetic_jsonl = os.path.join(temporary, "synthetic.jsonl")
        with open(synthetic_jsonl, "w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps(sample) + "\n")
        integration_dir = os.path.join(temporary, "adapter_train_integration")
        train_model(
            synthetic_jsonl,
            "dynaseqrel_dynedge",
            save_dir=integration_dir,
            epochs=1,
            batch_size=8,
            lr=5e-4,
            weight_decay=1e-4,
            val_ratio=0.25,
            seed=0,
            device="cpu",
            loss_weights=LossWeights(
                alpha=1.0, rank_weight=0.5, fail_weight=0.0),
            model_kwargs={
                **common_kwargs,
                "head_mode": "frozen_residual_score_adapter_v4",
                "adapter_hidden_dim": 64,
                "adapter_dropout": 0.2,
            },
            feature_version="v2",
            split_mode="stratified",
            early_stop_metric="score_spearman",
            save_metric_checkpoints=True,
            base_checkpoint_path=str(_BASE_CHECKPOINT),
            model_init_seed=0,
            dataloader_seed=0,
            verbose=False,
        )
        integration_config = json.loads(
            Path(integration_dir, "config.json").read_text(encoding="utf-8"))
        integration_check = {
            "training_mode": integration_config.get("training_mode"),
            "base_checkpoint_epoch": integration_config.get(
                "base_checkpoint_epoch"),
            "frozen_parameter_count": integration_config.get(
                "frozen_parameter_count"),
            "trainable_parameter_count": integration_config.get(
                "trainable_parameter_count"),
            "primary_metric": integration_config.get("primary_metric"),
            "best_rmse_checkpoint_saved": os.path.isfile(
                os.path.join(integration_dir, "best_rmse.pt")),
            "default_checkpoint_saved": os.path.isfile(
                os.path.join(
                    integration_dir, "dynaseqrel_dynedge_best.pt")),
        }

    compatibility = {}
    checkpoint_paths = {
        "v2": _BASE_CHECKPOINT,
        "v3": (
            _REPO_ROOT / "checkpoints/layout_models_repro/"
            "dynaseqrel_dynedge_scorehead_v3/stratified/seed0/"
            "dynaseqrel_dynedge_best.pt"),
        "deepsets": (
            _REPO_ROOT / "checkpoints/layout_models_repro/deepsets/"
            "stratified/seed0/deepsets_best.pt"),
        "seqrel": (
            _REPO_ROOT / "checkpoints/layout_models_repro/seqrel/"
            "stratified/seed0/seqrel_best.pt"),
    }
    for name, path in checkpoint_paths.items():
        if path.is_file():
            runner = LayoutModelRunner(str(path), device="cpu")
            result = runner.score_layouts(samples[:4])
            compatibility[name] = bool(
                torch.as_tensor(result["feas_prob"]).isfinite().all())
        else:
            compatibility[name] = f"skipped: missing {path}"

    adapter_rows = []
    for name, parameter in adapter_model.named_parameters():
        adapter_rows.append({
            "parameter_name": name,
            "shape": str(tuple(parameter.shape)),
            "numel": parameter.numel(),
            "requires_grad": parameter.requires_grad,
            "optimizer_member": name in optimizer_names,
            "changed_after_step": name in changed,
            "gradient_is_none": parameter.grad is None,
        })

    adapter_result = {
        "status": "pass",
        "formal_training_executed": False,
        **adapter_meta,
        "total_parameter_count": sum(
            parameter.numel() for parameter in adapter_model.parameters()),
        "optimizer_parameter_count": sum(
            parameter.numel() for parameter in optimizer_parameters),
        "optimizer_parameter_names": sorted(optimizer_names),
        "optimizer_only_adapter": all(
            name.startswith("score_adapter.") for name in optimizer_names),
        "initialization_output_equivalence": output_equivalence,
        "mode_check": mode_check,
        "adapter_gradient_norm": adapter_gradient_norm,
        "frozen_gradients_none": frozen_gradients_none,
        "adapter_parameters_changed": adapter_changed,
        "frozen_parameters_changed": frozen_changed,
        "frozen_state_hash_before": _state_hash({
            name: value for name, value in before.items()
            if not name.startswith("score_adapter.")}),
        "frozen_state_hash_after": _state_hash({
            name: parameter for name, parameter in adapter_model.state_dict().items()
            if not name.startswith("score_adapter.")}),
        "after_one_step": after_step_checks,
        "checkpoint_roundtrip_max_abs_diff": roundtrip_delta,
        "training_entrypoint_integration": integration_check,
        "checkpoint_compatibility": compatibility,
    }
    required = [
        adapter_result["optimizer_only_adapter"],
        frozen_gradients_none,
        adapter_gradient_norm > 0.0,
        bool(adapter_changed),
        not frozen_changed,
        mode_check["adapter_training"],
        mode_check["frozen_modules_eval"],
        mode_check["consecutive_feasibility_equal"],
        mode_check["consecutive_frozen_representation_equal"],
        after_step_checks["feasibility_logits_max_abs_diff"] == 0.0,
        after_step_checks["fail_logits_max_abs_diff"] == 0.0,
        after_step_checks["delta_score_logit_max_abs"] > 0.0,
        after_step_checks["score_predictions_changed"],
        roundtrip_delta == 0.0,
        integration_check["training_mode"]
        == "frozen_residual_score_adapter",
        integration_check["base_checkpoint_epoch"] == 48,
        integration_check["frozen_parameter_count"] == 78472,
        integration_check["trainable_parameter_count"] == 4225,
        integration_check["primary_metric"] == "score_spearman",
        integration_check["best_rmse_checkpoint_saved"],
        integration_check["default_checkpoint_saved"],
        all(value is True for value in compatibility.values()),
    ]
    if not all(required):
        adapter_result["status"] = "fail"
        raise AssertionError(f"adapter-v4 check failed: {adapter_result}")
    (_OUTPUT / "frozen_score_adapter_selfcheck.json").write_text(
        json.dumps(adapter_result, indent=2), encoding="utf-8")
    (_OUTPUT / "frozen_score_adapter_output_equivalence.json").write_text(
        json.dumps({
            "initialization": output_equivalence,
            "after_one_optimizer_step": after_step_checks,
            "checkpoint_roundtrip_max_abs_diff": roundtrip_delta,
        }, indent=2), encoding="utf-8")
    _write_csv(_OUTPUT / "frozen_score_adapter_parameters.csv", adapter_rows)
    print(json.dumps({
        "strict_v3": strict_result,
        "adapter_v4": adapter_result,
        "outputs": str(_OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
