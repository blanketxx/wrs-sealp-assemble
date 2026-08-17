"""Safe (sound) pruning for the backward search.

All pruning here is applied BEFORE any RRT call and only removes options that are
provably incompatible with the current partial suffix:

  * ``hall_feasible`` -- a necessary condition from Hall's theorem: if the
    still-unassigned parts cannot even be matched to distinct free staging cells,
    the current suffix cannot be extended to a full layout -> prune.
  * ``propagate_domains`` -- forward-checking: occupy the newly fixed part's
    footprint cells; any earlier part whose domain becomes empty -> prune.
  * ``swept_segment_mask`` -- legacy XY transfer-corridor cells (optional via
    ``--transfer-corridor-prune``; NOT the paper ``W_req`` masks).

Paper required-motion masks (``W_req`` / ``F_{k,j}``) live in
``req_motion_masks.py`` and are applied inside ``StepOracle.certify``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching

from .domain import OccupancyBitmap, StagingGrid


def part_footprint_radius(searcher, pid: str) -> float:
    """Half-diagonal of the part's largest staging footprint (m)."""
    best = 0.03
    for c in searcher.rot_cands.get(pid, []) or []:
        fp = np.asarray(getattr(c, "footprint", [0.06, 0.06]), dtype=float)[:2]
        best = max(best, 0.5 * float(np.linalg.norm(fp)))
    return best


def hall_feasible(remaining: List[str], part_cells: Dict[str, List[int]],
                  occ: OccupancyBitmap) -> bool:
    """True iff every remaining part can be matched to a distinct FREE cell."""
    if not remaining:
        return True
    free = set(occ.free_cells())
    n_cells = occ.grid.n
    rows, cols = [], []
    for i, pid in enumerate(remaining):
        cells = [c for c in part_cells.get(pid, []) if c in free]
        if not cells:
            return False                     # a part has no free cell at all
        for c in cells:
            rows.append(i)
            cols.append(c)
    if not rows:
        return False
    data = np.ones(len(rows), dtype=np.int8)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(len(remaining), n_cells))
    match = maximum_bipartite_matching(graph, perm_type="column")
    matched = int(np.sum(match >= 0))
    return matched >= len(remaining)


def swept_segment_mask(grid: StagingGrid, sp_xy, gp_xy, radius: float) -> int:
    """Cells within ``radius`` of the staging->goal transfer segment (xy proj)."""
    a = np.asarray(sp_xy, dtype=float)[:2]
    b = np.asarray(gp_xy, dtype=float)[:2]
    ab = b - a
    denom = float(ab @ ab) or 1.0
    mask = 0
    for j, c in enumerate(grid.cell_xy):
        p = c[:2]
        t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        if float(np.linalg.norm(p - proj)) <= radius:
            mask |= (1 << j)
    return mask


def propagate_domains(remaining: List[str],
                      part_cells: Dict[str, List[int]],
                      occ: OccupancyBitmap,
                      foot_radius: Dict[str, float],
                      extra_block_mask: int = 0
                      ) -> Optional[Dict[str, List[int]]]:
    """Forward-check: drop cells that collide with occupancy / blocked corridor.

    Returns the shrunk per-part cell domains, or ``None`` if any remaining part's
    domain becomes empty (the suffix is a dead end).
    """
    blocked = occ.mask | int(extra_block_mask)
    out: Dict[str, List[int]] = {}
    for pid in remaining:
        r = foot_radius.get(pid, 0.03)
        kept = []
        for c in part_cells.get(pid, []):
            fp = occ.grid.footprint_mask(c, r)
            if (fp & blocked) == 0:
                kept.append(c)
        if not kept:
            return None
        out[pid] = kept
    return out
