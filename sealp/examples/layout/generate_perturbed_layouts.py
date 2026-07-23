#!/usr/bin/env python
"""Generate reproducible planar perturbations of a saved SEALP layout."""
from __future__ import annotations

import argparse
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
    parser.add_argument(
        "--preassembled-part",
        default="base_plate",
        help="This staging entry is not perturbed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.range_m < 0:
        raise ValueError("--range-m must be non-negative")

    with args.layout.open("r", encoding="utf-8") as stream:
        source: Dict[str, Any] = yaml.safe_load(stream)

    staging = source.get("staging")
    if not isinstance(staging, dict):
        raise ValueError("layout has no staging map")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for trial in range(args.trials):
        record = deepcopy(source)
        offsets: Dict[str, list[float]] = {}
        for part_id, pose in record["staging"].items():
            if part_id == args.preassembled_part:
                continue
            pos = pose.get("pos")
            if not isinstance(pos, list) or len(pos) < 2:
                raise ValueError(f"invalid staging position for {part_id}")
            delta = rng.uniform(-args.range_m, args.range_m, size=2)
            pos[0] = float(pos[0]) + float(delta[0])
            pos[1] = float(pos[1]) + float(delta[1])
            offsets[part_id] = [float(delta[0]), float(delta[1])]

        record["name"] = f"{source.get('name', 'layout')}_perturb_{trial:02d}"
        metadata = record.setdefault("metadata", {})
        metadata["perturbation_trial"] = trial
        metadata["perturbation_range_m"] = float(args.range_m)
        metadata["perturbation_seed"] = int(args.seed)
        metadata["perturbation_offsets_xy"] = offsets

        target = args.output_dir / f"perturb_{trial:02d}.layout"
        with target.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(record, stream, sort_keys=False, allow_unicode=True)

    print(f"Generated {args.trials} layouts in {args.output_dir}")


if __name__ == "__main__":
    main()
