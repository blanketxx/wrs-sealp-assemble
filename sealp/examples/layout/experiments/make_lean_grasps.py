"""Subsample full grasp pickles into a lean search set (~400 / file).

Matches the convention used by ``yuanchair_grasp_lean`` / ``tower_grasp_lean``:
keep the first ``--keep`` grasps of each pickle (search-only subsample; the
full set stays untouched for L3 / execution).

Usage::

    python -m sealp.examples.layout.experiments.make_lean_grasps \
        --src sealp/examples/grasp/first_stack_grasp \
        --dst sealp/examples/grasp/first_stack_grasp_lean \
        --keep 400
"""

from __future__ import annotations

import argparse
import os
import shutil

from wrs.grasping.grasp import GraspCollection


def lean_subsample(gc: GraspCollection, n: int) -> GraspCollection:
    grasps = list(gc)[: min(int(n), len(gc))]
    return GraspCollection(
        end_effector=getattr(gc, "end_effector", None),
        grasp_list=grasps,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory of full grasp pickles")
    ap.add_argument("--dst", required=True, help="output lean directory")
    ap.add_argument("--keep", type=int, default=400,
                    help="max grasps kept per pickle (default 400)")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if not os.path.isdir(src):
        raise SystemExit(f"src not found: {src}")
    os.makedirs(dst, exist_ok=True)

    pickles = sorted(f for f in os.listdir(src) if f.endswith(".pickle"))
    if not pickles:
        raise SystemExit(f"no .pickle files in {src}")

    print(f"src  = {src}")
    print(f"dst  = {dst}")
    print(f"keep = {args.keep}")
    for name in pickles:
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        gc = GraspCollection.load_from_disk(file_name=src_path)
        if len(gc) <= args.keep:
            shutil.copyfile(src_path, dst_path)
            print(f"  {name}: {len(gc)} -> {len(gc)} (copy, already lean)")
            continue
        lean = lean_subsample(gc, args.keep)
        lean.save_to_disk(file_name=dst_path)
        print(f"  {name}: {len(gc)} -> {len(lean)}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
