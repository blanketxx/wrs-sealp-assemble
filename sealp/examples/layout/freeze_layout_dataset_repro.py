"""Create the immutable 2293-sample layout dataset reproduction snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .layout_learning.repro_data import sha256_file, sha256_lines


REPRO_LINE_COUNT = 2293
SOURCE_REL = Path("sealp/examples/layout/_output/layout_dataset_v2.jsonl")
SPLIT_REL = Path(
    "checkpoints/layout_models_repro/_splits/stratified/seed0/"
    "split_indices.json")
OUTPUT_REL = Path(
    "sealp/examples/layout/_output/datasets/"
    "layout_dataset_v2_repro_2293.jsonl")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _git_head_bytes(repo_root: Path) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{SOURCE_REL.as_posix()}"],
        cwd=repo_root,
    )


def main() -> None:
    args = _parse_args()
    root = Path(args.repo_root).resolve()
    source = root / SOURCE_REL
    split_source = root / SPLIT_REL
    output = root / OUTPUT_REL
    split_output = output.with_name(
        "layout_dataset_v2_repro_2293_split_indices.json")
    manifest_output = output.with_name(
        "layout_dataset_v2_repro_2293_manifest.json")

    source_lines = source.read_bytes().splitlines(keepends=True)
    if len(source_lines) < REPRO_LINE_COUNT:
        raise RuntimeError(
            f"source has only {len(source_lines)} lines; "
            f"need {REPRO_LINE_COUNT}")
    prefix = b"".join(source_lines[:REPRO_LINE_COUNT])
    head = _git_head_bytes(root)
    if prefix != head:
        raise RuntimeError(
            "current first 2293 lines do not match git HEAD exactly; "
            "refusing to claim historical identity")

    if not args.verify_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(prefix)
        shutil.copyfile(split_source, split_output)

    if not output.is_file() or output.read_bytes() != prefix:
        raise RuntimeError("snapshot content verification failed")
    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split = json.loads(split_output.read_text(encoding="utf-8"))
    train_indices = [int(index) for index in split["train_indices"]]
    val_indices = [int(index) for index in split["val_indices"]]
    all_indices = train_indices + val_indices
    if (len(records) != REPRO_LINE_COUNT
            or len(set(all_indices)) != REPRO_LINE_COUNT
            or min(all_indices) != 0
            or max(all_indices) != REPRO_LINE_COUNT - 1):
        raise RuntimeError("split does not partition exactly indices 0..2292")

    sample_ids = [record.get("sample_id") for record in records]
    if any(sample_id is None for sample_id in sample_ids):
        raise RuntimeError("snapshot contains records without sample_id")
    if len(set(map(str, sample_ids))) != len(sample_ids):
        raise RuntimeError("snapshot sample_id values are not unique")

    manifest = {
        "snapshot_path": OUTPUT_REL.as_posix(),
        "source_path": SOURCE_REL.as_posix(),
        "line_count": len(records),
        "feasible_count": sum(
            bool(record.get("l2_pass", False)) for record in records),
        "infeasible_count": sum(
            not bool(record.get("l2_pass", False)) for record in records),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "maximum_split_index": max(all_indices),
        "dataset_sha256": sha256_file(str(output)),
        "split_sha256": sha256_file(str(split_output)),
        "all_sample_ids_sha256": sha256_lines(sample_ids),
        "train_sample_ids_sha256": sha256_lines(
            sample_ids[index] for index in train_indices),
        "val_sample_ids_sha256": sha256_lines(
            sample_ids[index] for index in val_indices),
        "first_2293_content_sha256": sha256_file(str(output)),
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "generator_script": (
            "sealp/examples/layout/generate_layout_dataset.py"),
        "append_only_verified": True,
        "feature_version": "v2",
        "git_head_matches_first_2293": True,
        "sample_id_hash_encoding": "utf-8 value plus LF, in listed order",
        "notes": (
            "Immutable reproduction snapshot for seed0 experiments. "
            "Content is byte-identical to git HEAD and preserves original "
            "line order. Split was copied without repartitioning."),
    }
    if not args.verify_only:
        manifest_output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.chmod(output, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(split_output, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
