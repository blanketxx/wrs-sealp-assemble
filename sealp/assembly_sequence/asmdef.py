"""
Assembly Definition Format (.asmdef)
=====================================

Product-level assembly specification.  YAML-backed, custom ``.asmdef``
extension.  Contains only what defines the *product*:

- Shared model library (alias → path)
- Parts (reference model aliases)
- Symmetry groups (interchangeable parts)
- Assembly graph (relative poses, dependency DAG)

Task-specific data (staging positions, grasp IDs, motion primitives)
lives in ``.tplan`` files (not implemented here).

Usage::

    from sealp.assembly_sequence import AssemblyDef

    asm = AssemblyDef.load("chair.asmdef")
    asm.parts              # dict[str, PartDef]
    asm.models             # dict[str, str]  (alias → path)
    asm.symmetry_groups    # dict[str, list[str]]
    asm.steps              # list[StepDef]

    asm.save("output.asmdef")
"""

from __future__ import annotations

import copy
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


# ══════════════════════════════════════════════════════════════
#  YAML helpers — clean float output
# ══════════════════════════════════════════════════════════════
def _float_representer(dumper: yaml.Dumper, value: float):
    if value != value:
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".nan")
    if value == float("inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".inf")
    if value == float("-inf"):
        return dumper.represent_scalar("tag:yaml.org,2002:float", "-.inf")
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.6g}")


yaml.add_representer(float, _float_representer)

FORMAT_VERSION = "1.0"


# ══════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════
@dataclass
class PartDef:
    """A part in the assembly.

    Attributes
    ----------
    part_id : str
        Unique identifier (e.g. ``"leg_fl"``).
    name : str
        Human-readable display name.
    model : str
        Alias into the assembly's model library.
    mass : float
        Mass in kg.
    metadata : dict
        Arbitrary key-value data (material, tolerance, etc.).
    """
    part_id: str
    name: str
    model: str          # alias → models dict
    mass: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"name": self.name, "model": self.model}
        if self.mass != 0.0:
            d["mass"] = self.mass
        if self.metadata:
            d["metadata"] = copy.deepcopy(self.metadata)
        return d

    @classmethod
    def from_dict(cls, part_id: str, d: dict) -> "PartDef":
        return cls(
            part_id=part_id,
            name=d.get("name", part_id),
            model=d["model"],
            mass=float(d.get("mass", 0.0)),
            metadata=d.get("metadata", {}),
        )


@dataclass
class StepDef:
    """A single assembly step.

    Attributes
    ----------
    step_id : int
        Sequential index.
    part_id : str
        Which part is being assembled.
    parent_id : str
        What it attaches to (``"fixture"`` = workspace surface).
    rel_pos : np.ndarray
        Position relative to parent ``[x, y, z]``.
    rel_rotmat : np.ndarray
        Orientation relative to parent (3×3).
    insertion_axis : np.ndarray or None
        Optional EXPLICIT mating/insertion axis, expressed DIRECTLY in the
        WORLD frame (NOT the part local frame -- it is used verbatim, no
        ``R_goal`` transform). It is the world direction along which the part
        translates to SEAT into its parent (i.e. the place-approach direction).
        The place-depart direction is its negation, and the pick-depart (lift)
        direction is also its negation. Example: ``[0, 0, 1]`` means the part
        is seated by moving straight up (+Z world) and lifted/departed along
        ``-Z``. If ``None`` the motion layer infers the axis (child local +Z,
        then legacy world -Z fallback).
    insertion_axis_local : np.ndarray or None
        DEPRECATED legacy field: an insertion axis expressed in the part's
        LOCAL frame at goal (converted to world via ``R_goal @ axis``). Only
        used when ``insertion_axis`` is not given. Prefer ``insertion_axis``.
    deps : list of int
        Step IDs that must complete before this step.
    notes : str
        Free-text annotation.
    metadata : dict
        Arbitrary key-value data.
    """
    step_id: int
    part_id: str
    parent_id: str = "fixture"
    rel_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rel_rotmat: np.ndarray = field(default_factory=lambda: np.eye(3))
    insertion_axis: Optional[np.ndarray] = None
    insertion_axis_local: Optional[np.ndarray] = None
    deps: List[int] = field(default_factory=list)
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "step": self.step_id,
            "part": self.part_id,
            "parent": self.parent_id,
            "rel_pos": [float(v) for v in self.rel_pos],
            "rel_rotmat": [[float(v) for v in row]
                           for row in self.rel_rotmat],
            "deps": list(self.deps),
        }
        if self.insertion_axis is not None:
            d["insertion_axis"] = [float(v) for v in self.insertion_axis]
        if self.insertion_axis_local is not None:
            d["insertion_axis_local"] = [float(v)
                                         for v in self.insertion_axis_local]
        if self.notes:
            d["notes"] = self.notes
        if self.metadata:
            d["metadata"] = copy.deepcopy(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StepDef":
        _iax_world = d.get("insertion_axis", None)
        _iax_local = d.get("insertion_axis_local", None)
        return cls(
            step_id=int(d["step"]),
            part_id=d["part"],
            parent_id=d.get("parent", "fixture"),
            rel_pos=np.asarray(d.get("rel_pos", [0, 0, 0]), dtype=float),
            rel_rotmat=np.asarray(
                d.get("rel_rotmat", np.eye(3).tolist()), dtype=float),
            insertion_axis=(None if _iax_world is None
                            else np.asarray(_iax_world, dtype=float)),
            insertion_axis_local=(None if _iax_local is None
                                  else np.asarray(_iax_local, dtype=float)),
            deps=list(d.get("deps", [])),
            notes=d.get("notes", ""),
            metadata=d.get("metadata", {}),
        )

    def copy(self) -> "StepDef":
        return StepDef.from_dict(self.to_dict())


# ══════════════════════════════════════════════════════════════
#  AssemblyDef — top-level container
# ══════════════════════════════════════════════════════════════
class AssemblyDef:
    """Product-level assembly definition.

    Parameters
    ----------
    name : str
        Assembly name.
    description : str
        Long-form description.
    """

    def __init__(self, name: str = "unnamed",
                 description: str = ""):
        self.format_version: str = FORMAT_VERSION
        self.name: str = name
        self.description: str = description
        self.models: Dict[str, str] = {}          # alias → abs path
        self.symmetry_groups: Dict[str, List[str]] = {}
        self._parts: Dict[str, PartDef] = {}
        self._steps: List[StepDef] = []

    # ── Part management ──────────────────────────────────────
    def add_model(self, alias: str, path: str):
        """Register a model in the library."""
        self.models[alias] = path

    def add_part(self, part: PartDef):
        """Register a part.  Raises ``ValueError`` on duplicate ID."""
        if part.part_id in self._parts:
            raise ValueError(f"Duplicate part_id: {part.part_id!r}")
        if part.model not in self.models:
            raise ValueError(
                f"Part {part.part_id!r} references unknown model "
                f"alias {part.model!r}. Register it first with add_model().")
        self._parts[part.part_id] = part

    def get_part(self, part_id: str) -> PartDef:
        return self._parts[part_id]

    @property
    def parts(self) -> Dict[str, PartDef]:
        return dict(self._parts)

    @property
    def part_ids(self) -> List[str]:
        return list(self._parts.keys())

    def model_path(self, part_id: str) -> str:
        """Return the absolute model path for a part."""
        part = self._parts[part_id]
        return self.models[part.model]

    # ── Step management ──────────────────────────────────────
    def add_step(self, step: StepDef):
        """Append an assembly step."""
        existing = {s.step_id for s in self._steps}
        if step.step_id in existing:
            raise ValueError(f"Duplicate step_id: {step.step_id}")
        self._steps.append(step)

    @property
    def steps(self) -> List[StepDef]:
        return list(self._steps)

    @property
    def n_parts(self) -> int:
        return len(self._parts)

    @property
    def n_steps(self) -> int:
        return len(self._steps)

    # ── Symmetry ─────────────────────────────────────────────
    def add_symmetry_group(self, group_name: str,
                           part_ids: List[str]):
        """Declare a set of interchangeable parts."""
        self.symmetry_groups[group_name] = list(part_ids)

    def get_symmetry_group(self, part_id: str) -> Optional[str]:
        """Return the symmetry group name for a part, or None."""
        for gname, members in self.symmetry_groups.items():
            if part_id in members:
                return gname
        return None

    # ── Pose computation ─────────────────────────────────────
    def compute_world_poses(
        self, fixture_pos: np.ndarray = None,
        fixture_rotmat: np.ndarray = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute absolute world poses for all parts.

        Traverses the assembly graph from fixture (root) and
        accumulates relative transforms.

        Parameters
        ----------
        fixture_pos : np.ndarray, optional
            World position of the fixture (default: origin).
        fixture_rotmat : np.ndarray, optional
            World orientation of the fixture (default: identity).

        Returns
        -------
        dict
            ``{part_id: (world_pos, world_rotmat)}``
        """
        if fixture_pos is None:
            fixture_pos = np.zeros(3)
        if fixture_rotmat is None:
            fixture_rotmat = np.eye(3)

        # Build a map: part_id → step
        step_map: Dict[str, StepDef] = {}
        for s in self._steps:
            step_map[s.part_id] = s

        # BFS from fixture
        world_poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        world_poses["fixture"] = (fixture_pos.copy(), fixture_rotmat.copy())

        # Build adjacency: parent → children
        children: Dict[str, List[str]] = defaultdict(list)
        for s in self._steps:
            children[s.parent_id].append(s.part_id)

        queue = deque(["fixture"])
        while queue:
            parent = queue.popleft()
            p_pos, p_rot = world_poses[parent]
            for child_id in children.get(parent, []):
                step = step_map[child_id]
                # world = parent_rot @ rel_pos + parent_pos
                w_pos = p_rot @ step.rel_pos + p_pos
                w_rot = p_rot @ step.rel_rotmat
                world_poses[child_id] = (w_pos, w_rot)
                queue.append(child_id)

        return world_poses

    # ── Validation ───────────────────────────────────────────
    def validate(self, strict: bool = True) -> List[str]:
        """Validate the assembly definition.

        Checks:
        1. Every step references a known part_id.
        2. Every dependency step_id exists.
        3. The dependency graph is a DAG (no cycles).
        4. No part_id appears in more than one step.
        5. Every part's model alias exists in the model library.
        6. Symmetry group members exist as parts.
        """
        errors: List[str] = []
        step_ids = {s.step_id for s in self._steps}

        for step in self._steps:
            if step.part_id not in self._parts:
                errors.append(
                    f"Step {step.step_id}: part {step.part_id!r} not found.")
            for dep in step.deps:
                if dep not in step_ids:
                    errors.append(
                        f"Step {step.step_id}: dep {dep} not found.")

        if not self._is_dag():
            errors.append("Dependency graph has a cycle!")

        seen: Dict[str, int] = {}
        for step in self._steps:
            if step.part_id in seen:
                errors.append(
                    f"Part {step.part_id!r} in steps "
                    f"{seen[step.part_id]} and {step.step_id}.")
            seen[step.part_id] = step.step_id

        for pid, part in self._parts.items():
            if part.model not in self.models:
                errors.append(
                    f"Part {pid!r}: model alias {part.model!r} not found.")

        for gname, members in self.symmetry_groups.items():
            for m in members:
                if m not in self._parts:
                    errors.append(
                        f"Symmetry group {gname!r}: member {m!r} not found.")

        if strict and errors:
            raise ValueError(
                "Assembly validation failed:\n  • "
                + "\n  • ".join(errors))
        return errors

    def _is_dag(self) -> bool:
        graph: Dict[int, List[int]] = defaultdict(list)
        in_deg: Dict[int, int] = {}
        for s in self._steps:
            in_deg.setdefault(s.step_id, 0)
            for d in s.deps:
                graph[d].append(s.step_id)
                in_deg[s.step_id] = in_deg.get(s.step_id, 0) + 1
                in_deg.setdefault(d, 0)
        queue = deque(k for k, v in in_deg.items() if v == 0)
        visited = 0
        while queue:
            n = queue.popleft()
            visited += 1
            for c in graph[n]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
        return visited == len(in_deg)

    def get_execution_order(self) -> List[StepDef]:
        """Return steps in topological order."""
        graph: Dict[int, List[int]] = defaultdict(list)
        in_deg: Dict[int, int] = {}
        step_map = {s.step_id: s for s in self._steps}
        for s in self._steps:
            in_deg.setdefault(s.step_id, 0)
            for d in s.deps:
                graph[d].append(s.step_id)
                in_deg[s.step_id] = in_deg.get(s.step_id, 0) + 1
                in_deg.setdefault(d, 0)
        queue = deque(sorted(k for k, v in in_deg.items() if v == 0))
        order: List[StepDef] = []
        while queue:
            n = queue.popleft()
            if n in step_map:
                order.append(step_map[n])
            for c in sorted(graph[n]):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
        if len(order) != len(self._steps):
            raise ValueError("Cycle detected in dependency graph.")
        return order

    # ── Serialisation ────────────────────────────────────────
    def to_dict(self) -> dict:
        d = {
            "format_version": self.format_version,
            "name": self.name,
            "description": self.description,
            "models": {alias: {"path": path}
                       for alias, path in self.models.items()},
            "parts": {pid: p.to_dict()
                      for pid, p in self._parts.items()},
            "assembly": [s.to_dict() for s in self._steps],
        }
        if self.symmetry_groups:
            d["symmetry_groups"] = copy.deepcopy(self.symmetry_groups)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AssemblyDef":
        asm = cls(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
        )
        asm.format_version = d.get("format_version", FORMAT_VERSION)

        for alias, mdata in d.get("models", {}).items():
            path = mdata["path"] if isinstance(mdata, dict) else mdata
            asm.models[alias] = path

        # Parts must be added after models
        for pid, pdata in d.get("parts", {}).items():
            asm._parts[pid] = PartDef.from_dict(pid, pdata)

        for sdata in d.get("assembly", []):
            asm._steps.append(StepDef.from_dict(sdata))

        for gname, members in d.get("symmetry_groups", {}).items():
            asm.symmetry_groups[gname] = list(members)

        return asm

    # ── File I/O ─────────────────────────────────────────────
    def save(self, filepath: Union[str, Path]) -> None:
        """Save to a ``.asmdef`` file (YAML-backed)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(f"# SEALP Assembly Definition v{self.format_version}\n")
            fh.write(f"# {self.name}\n\n")
            yaml.dump(data, fh, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "AssemblyDef":
        """Load from a ``.asmdef`` file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)

    # ── Pretty printing ──────────────────────────────────────
    def summary(self) -> str:
        lines = [
            f"Assembly: {self.name}",
            f"Description: {self.description}",
            f"Format: v{self.format_version}",
            f"",
            f"Models ({len(self.models)}):",
        ]
        for alias, path in self.models.items():
            lines.append(f"  {alias}: {path}")
        if self.symmetry_groups:
            lines.append(f"Symmetry Groups:")
            for gname, members in self.symmetry_groups.items():
                lines.append(f"  {gname}: {members}")
        lines.append(f"")
        lines.append(f"Parts ({self.n_parts}):")
        for pid, p in self._parts.items():
            lines.append(f"  {pid}: {p.name} [model={p.model}, "
                         f"mass={p.mass:.2f}kg]")
        lines.append(f"")
        lines.append(f"Assembly Steps ({self.n_steps}):")
        for s in self._steps:
            deps = ",".join(str(d) for d in s.deps) or "-"
            lines.append(
                f"  {s.step_id}. {s.part_id} -> {s.parent_id} "
                f"rel_pos={s.rel_pos.tolist()} [deps: {deps}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"AssemblyDef(name={self.name!r}, "
                f"parts={self.n_parts}, steps={self.n_steps})")

    def __len__(self) -> int:
        return self.n_steps
