#!/usr/bin/env python
"""Generate reproducible planar perturbations of a saved SEALP layout.

The preassembled part and assembly station are kept fixed. Each movable staging
part receives independent x/y offsets sampled uniformly from [-range_m, range_m].
A CSV manifest records all offsets so every trial is reproducible.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--range-m", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preassembled-part", default="base_plate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.layout.is_file():
        raise FileNotFoundError(args.layout)
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.range_m < 0:
        raise ValueError("--range-m must be non-negative")

    with args.layout.open("r", encoding="utf-8") as stream:
        source: Dict[str, Any] = yaml.safe_load(stream)

    staging = source.get("staging")
    if not isinstance(staging, dict) or not staging:
        raise ValueError("layout has no staging map")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest_rows: list[dict[str, Any]] = []

    for trial in range(args.trials):
        record = deepcopy(source)
        offsets: Dict[str, list[float]] = {}
        row: dict[str, Any] = {"trial": trial, "layout": f"perturb_{trial:02d}.layout"}

        for part_id, pose in record["staging"].items():
            if part_id == args.preassembled_part:
                continue
            pos = pose.get("pos")
            if not isinstance(pos, list) or len(pos) < 2:
                raise ValueError(f"invalid staging position for {part_id}")

            delta = rng.uniform(-args.range_m, args.range_m, size=2)
            dx, dy = float(delta[0]), float(delta[1])
            pos[0] = float(pos[0]) + dx
            pos[1] = float(pos[1]) + dy
            offsets[part_id] = [dx, dy]
            row[f"{part_id}_dx_m"] = dx
            row[f"{part_id}_dy_m"] = dy

        record["name"] = f"{source.get('name', 'layout')}_perturb_{trial:02d}"
        metadata = record.setdefault("metadata", {})
        metadata["perturbation_trial"] = trial
        metadata["perturbation_range_m"] = float(args.range_m)
        metadata["perturbation_seed"] = int(args.seed)
        metadata["perturbation_offsets_xy"] = offsets
        metadata["perturbation_source_layout"] = str(args.layout.resolve())

        target = args.output_dir / f"perturb_{trial:02d}.layout"
        with target.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(record, stream, sort_keys=False, allow_unicode=True)
        manifest_rows.append(row)

    fieldnames: list[str] = []
    for row in manifest_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (args.output_dir / "perturbation_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Generated {args.trials} layouts in {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'perturbation_manifest.csv'}")


if __name__ == "__main__":
    main()
