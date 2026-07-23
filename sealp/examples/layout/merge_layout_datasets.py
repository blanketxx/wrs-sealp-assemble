#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并多个 layout jsonl 数据集，并写出 train/val split。

典型用法（tower repro + totem）::

    python -m sealp.examples.layout.merge_layout_datasets ^
      --inputs sealp/examples/layout/_output/datasets/layout_dataset_v2_repro_2293.jsonl ^
               sealp/examples/layout/_output/totem_layout_v1.jsonl ^
      --output-jsonl sealp/examples/layout/_output/datasets/layout_dataset_tower_totem_v1.jsonl ^
      --split-mode stratified --val-ratio 0.2 --seed 0

split-mode:
    - stratified       : 按 l2_pass 分层随机划分（快速实验）
    - geometry_holdout : 按 mesh SHA256 划分，避免几何泄漏
    - task_holdout     : 指定 assembly 整包进 val（测跨任务泛化）
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from layout_learning.dataset import load_jsonl
from layout_learning.generator_dataset import (
    GENERATOR_SCHEMA_VERSION,
    geometry_holdout_split,
    sample_schema_extensions,
)
from layout_learning.repro_data import sha256_file, sha256_lines
from layout_learning.train import _split_indices, _save_split_indices


def _infer_assembly_id(record: Dict, source_path: str) -> str:
    existing = record.get("assembly_id") or record.get("task_id")
    if existing:
        return str(existing)
    base = os.path.basename(source_path).lower()
    if "totem" in base:
        return "totem"
    if "yuanchair" in base or "chair" in base:
        return "yuanchair"
    if "tower" in base or "repro" in base or "layout_dataset" in base:
        return "topdown_tower"
    return os.path.splitext(os.path.basename(source_path))[0]


def _normalize_record(record: Dict, source_path: str) -> Dict:
    out = dict(record)
    asm_id = _infer_assembly_id(out, source_path)
    out.setdefault("task_id", asm_id)
    out.setdefault("assembly_id", asm_id)
    out.setdefault("assembly_type", asm_id.split("_")[0] if "_" in asm_id else asm_id)
    out.setdefault("schema_version", GENERATOR_SCHEMA_VERSION)
    out.setdefault("geometry_domain", asm_id)
    out.setdefault("num_parts", len(out.get("parts", [])))
    ext = sample_schema_extensions(out)
    for key, value in ext.items():
        out.setdefault(key, value)
    return out


def _load_inputs(paths: List[str]) -> Tuple[List[Dict], List[str]]:
    merged: List[Dict] = []
    tags: List[str] = []
    for path in paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"input not found: {path}")
        tag = os.path.splitext(os.path.basename(path))[0]
        rows = load_jsonl(path)
        for row in rows:
            merged.append(_normalize_record(row, path))
        tags.append(tag)
    return merged, tags


def _dedupe_by_sample_id(samples: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for row in samples:
        sid = row.get("sample_id")
        key = str(sid) if sid is not None else None
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        out.append(row)
    return out


def _task_holdout_split(
    samples: List[Dict],
    holdout_tasks: List[str],
) -> Tuple[List[int], List[int], Dict]:
    hold = {t.strip() for t in holdout_tasks if t.strip()}
    if not hold:
        raise ValueError("task_holdout requires --holdout-tasks")
    val, train = [], []
    for i, row in enumerate(samples):
        asm = str(row.get("assembly_id") or row.get("task_id") or "unknown")
        if asm in hold:
            val.append(i)
        else:
            train.append(i)
    if not train or not val:
        raise RuntimeError(
            f"task_holdout produced empty split: holdout={sorted(hold)} "
            f"train={len(train)} val={len(val)}")
    return sorted(train), sorted(val), {
        "split_mode": "task_holdout",
        "holdout_tasks": sorted(hold),
    }


def _build_split(
    samples: List[Dict],
    split_mode: str,
    val_ratio: float,
    seed: int,
    holdout_tasks: List[str],
) -> Tuple[List[int], List[int], Dict]:
    if split_mode == "geometry_holdout":
        train_idx, val_idx, meta = geometry_holdout_split(samples, val_ratio, seed)
        return train_idx, val_idx, meta
    if split_mode == "task_holdout":
        return _task_holdout_split(samples, holdout_tasks)
    train_idx, val_idx = _split_indices(samples, val_ratio, seed, split_mode)
    return train_idx, val_idx, {"split_mode": split_mode, "val_ratio": val_ratio, "seed": seed}


def _summary(samples: List[Dict]) -> Dict:
    asm = Counter(str(s.get("assembly_id") or "unknown") for s in samples)
    feas = Counter(bool(s.get("l2_pass", False)) for s in samples)
    return {
        "total": len(samples),
        "feasible": int(feas.get(True, 0)),
        "infeasible": int(feas.get(False, 0)),
        "by_assembly_id": dict(sorted(asm.items())),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Merge layout jsonl datasets")
    p.add_argument(
        "--inputs", nargs="+", required=True,
        help="一个或多个 jsonl 路径，按顺序拼接",
    )
    p.add_argument("--output-jsonl", required=True, help="合并后的 jsonl 输出路径")
    p.add_argument(
        "--split-json",
        default="",
        help="split_indices.json 输出路径；默认与 output-jsonl 同目录同名",
    )
    p.add_argument(
        "--manifest-json",
        default="",
        help="manifest 输出路径；默认同目录 *_manifest.json",
    )
    p.add_argument(
        "--split-mode",
        default="stratified",
        choices=["stratified", "random", "seed_holdout", "region_holdout",
                 "geometry_holdout", "task_holdout"],
    )
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--holdout-tasks",
        default="totem",
        help="task_holdout 时整包放入 val 的 assembly_id，逗号分隔",
    )
    p.add_argument(
        "--dedupe-sample-id",
        action="store_true",
        help="按 sample_id 去重（默认保留全部行）",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    samples, input_tags = _load_inputs(args.inputs)
    if args.dedupe_sample_id:
        before = len(samples)
        samples = _dedupe_by_sample_id(samples)
        print(f"[merge] dedupe sample_id: {before} -> {len(samples)}")

    out_path = os.path.abspath(args.output_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in samples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    holdout_tasks = [t.strip() for t in args.holdout_tasks.split(",") if t.strip()]
    train_idx, val_idx, split_meta = _build_split(
        samples, args.split_mode, args.val_ratio, args.seed, holdout_tasks)

    split_path = args.split_json or out_path.replace(".jsonl", "_split_indices.json")
    manifest_path = args.manifest_json or out_path.replace(".jsonl", "_manifest.json")
    split_path = os.path.abspath(split_path)
    manifest_path = os.path.abspath(manifest_path)

    split_payload = {
        "dataset_path": out_path,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        **split_meta,
    }
    _save_split_indices(split_path, train_idx, val_idx, split_payload)

    stats = _summary(samples)
    sample_ids = [row.get("sample_id") for row in samples]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [os.path.abspath(p) for p in args.inputs],
        "input_tags": input_tags,
        "output_jsonl": out_path,
        "split_indices": split_path,
        "split_mode": args.split_mode,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "holdout_tasks": holdout_tasks if args.split_mode == "task_holdout" else [],
        "line_count": len(samples),
        "feasible_count": stats["feasible"],
        "infeasible_count": stats["infeasible"],
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "dataset_sha256": sha256_file(out_path),
        "split_sha256": sha256_file(split_path),
        "all_sample_ids_sha256": sha256_lines(sample_ids),
        "train_sample_ids_sha256": sha256_lines(sample_ids[i] for i in train_idx),
        "val_sample_ids_sha256": sha256_lines(sample_ids[i] for i in val_idx),
        "feature_version": "v2",
        "stats": stats,
        "train_stats": _summary([samples[i] for i in train_idx]),
        "val_stats": _summary([samples[i] for i in val_idx]),
        "generator_script": "sealp/examples/layout/merge_layout_datasets.py",
        "notes": "Merged multi-task layout dataset; hashes lock jsonl + split for training.",
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print("=" * 70)
    print("[merge] done")
    print(f"  output     = {out_path}")
    print(f"  split      = {split_path}")
    print(f"  manifest   = {manifest_path}")
    print(f"  total      = {stats['total']}")
    print(f"  feasible   = {stats['feasible']} / {stats['total']}")
    print(f"  assemblies = {stats['by_assembly_id']}")
    print(f"  train/val  = {len(train_idx)} / {len(val_idx)} ({args.split_mode})")
    print("=" * 70)


if __name__ == "__main__":
    main()
