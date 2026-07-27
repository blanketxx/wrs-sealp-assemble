"""Generate ``first_stack.asmdef`` for ``sealp/assets/models/first``.

Four plates stacked by geometric-center XY alignment::

    part_10 (base, 100x100x30 mm)
      -> part_8  (80x80x30 mm)
      -> part_6  (60x60x30 mm)
      -> part_4  (40x40x30 mm)

Every mesh already has its XY origin at the plate center and Z from 0 (bottom)
to height (top), so a stack is just ``rel_pos = [0, 0, parent_height]`` with
``insertion_axis = [0, 0, -1]`` (seat by translating along world -Z).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import trimesh as trm

from sealp.assembly_sequence import AssemblyDef, PartDef, StepDef

ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "models", "first"))
OUT_DIR = os.path.join(os.path.dirname(__file__), "_demo_output")

# Assembly order (bottom -> top).
STACK = ("part_10", "part_8", "part_6", "part_4")


def _mesh_height(stl_path: str) -> float:
    mesh = trm.load_mesh(stl_path)
    bounds = np.asarray(mesh.bounds, dtype=float)
    return float(bounds[1, 2] - bounds[0, 2])


def generate() -> AssemblyDef:
    asm = AssemblyDef(
        name="FirstStack",
        description=(
            "Four centered plates from sealp/assets/models/first, stacked along "
            "world -Z (center-aligned XY)."
        ),
    )

    heights = {}
    for pid in STACK:
        stl = os.path.join(ASSET_DIR, f"{pid}.stl")
        if not os.path.isfile(stl):
            raise FileNotFoundError(stl)
        model_id = f"{pid}_model"
        asm.add_model(model_id, stl)
        h = _mesh_height(stl)
        heights[pid] = h
        # ~1 g/cm^3 plastic; volumes are exact boxes.
        vol = float(trm.load_mesh(stl).volume)
        asm.add_part(PartDef(
            part_id=pid,
            name=pid.replace("_", " ").title(),
            model=model_id,
            mass=max(0.01, vol * 1000.0),
        ))

    for i, pid in enumerate(STACK):
        if i == 0:
            parent = "fixture"
            rel_z = 0.0
            deps = []
        else:
            parent = STACK[i - 1]
            rel_z = heights[parent]
            deps = [i - 1]
        asm.add_step(StepDef(
            step_id=i,
            part_id=pid,
            parent_id=parent,
            rel_pos=np.array([0.0, 0.0, rel_z], dtype=float),
            rel_rotmat=np.eye(3),
            insertion_axis=np.array([0.0, 0.0, -1.0], dtype=float),
            deps=deps,
            notes=(
                f"Seat {pid} by translating along world -Z onto "
                f"{'the fixture' if parent == 'fixture' else parent}; "
                f"XY centers coincide."
            ),
        ))

    errors = asm.validate(strict=False)
    if errors:
        print("Validation warnings:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation passed.")

    print("Stack heights (m):")
    z = 0.0
    for pid in STACK:
        print(f"  {pid}: height={heights[pid]:.4f}  world_z_bottom={z:.4f}  "
              f"world_z_top={z + heights[pid]:.4f}")
        z += heights[pid]
    return asm


def main() -> int:
    asm = generate()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "first_stack.asmdef")
    asm.save(out_path)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
