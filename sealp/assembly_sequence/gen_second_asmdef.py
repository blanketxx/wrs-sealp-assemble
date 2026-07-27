"""Generate ``second_assembly.asmdef`` for ``sealp/assets/models/second``.

Parts (mesh XY offsets are already baked into ``ao`` / ``tu``)::

    rectangular  (70x70x25 mm, centered)     -- base on fixture
      -> cylinder   (Ø40 x 100 mm, centered) -- stands on base
      -> ao         (40x70x30 mm, +X biased) -- seats on base
      -> tu         (50x70x30 mm, -X biased) -- seats on base

All insertions use world ``insertion_axis = [0, 0, -1]``.
Child ``rel_pos.z`` = parent (rectangular) height so bottoms sit on the base top.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import trimesh as trm

from sealp.assembly_sequence import AssemblyDef, PartDef, StepDef

ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "models", "second"))
OUT_DIR = os.path.join(os.path.dirname(__file__), "_demo_output")

# Assembly order: base first, then the three parts that seat on it.
PART_ORDER = ("rectangular", "cylinder", "ao", "tu")
BASE = "rectangular"


def _mesh_info(stl_path: str):
    mesh = trm.load_mesh(stl_path)
    bounds = np.asarray(mesh.bounds, dtype=float)
    height = float(bounds[1, 2] - bounds[0, 2])
    volume = float(mesh.volume)
    return height, volume, bounds


def generate() -> AssemblyDef:
    asm = AssemblyDef(
        name="SecondAssembly",
        description=(
            "Assembly from sealp/assets/models/second: rectangular base with "
            "cylinder / ao / tu seated on top along world -Z."
        ),
    )

    heights = {}
    for pid in PART_ORDER:
        stl = os.path.join(ASSET_DIR, f"{pid}.stl")
        if not os.path.isfile(stl):
            raise FileNotFoundError(stl)
        model_id = f"{pid}_model"
        asm.add_model(model_id, stl)
        h, vol, bounds = _mesh_info(stl)
        heights[pid] = h
        print(f"  {pid}: height={h:.4f} bounds_min={np.round(bounds[0],4).tolist()} "
              f"bounds_max={np.round(bounds[1],4).tolist()}")
        asm.add_part(PartDef(
            part_id=pid,
            name=pid.replace("_", " ").title(),
            model=model_id,
            mass=max(0.01, vol * 1000.0),
        ))

    base_h = heights[BASE]
    for i, pid in enumerate(PART_ORDER):
        if pid == BASE:
            parent, rel_z, deps = "fixture", 0.0, []
        else:
            parent, rel_z, deps = BASE, base_h, [0]
        asm.add_step(StepDef(
            step_id=i,
            part_id=pid,
            parent_id=parent,
            rel_pos=np.array([0.0, 0.0, rel_z], dtype=float),
            rel_rotmat=np.eye(3),
            insertion_axis=np.array([0.0, 0.0, -1.0], dtype=float),
            deps=deps,
            notes=(
                f"Seat {pid} along world -Z onto "
                f"{'the fixture' if parent == 'fixture' else parent}."
            ),
        ))

    errors = asm.validate(strict=False)
    if errors:
        print("Validation warnings:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation passed.")
    return asm


def main() -> int:
    print("========== Second assembly meshes ==========")
    asm = generate()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "second_assembly.asmdef")
    asm.save(out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
