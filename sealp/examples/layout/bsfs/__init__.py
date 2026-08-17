"""BSFS: Backward Suffix-factorization search for assembly staging layout.

This package implements the rigorous version of the backward suffix-feasibility
method described in the project task spec (``任务描述.txt``):

    backward suffix factorization
  + safe domain propagation
  + matching lower bound
  + branch-and-bound / A*  (exact, discrete-domain global optimum)
    or beam / weighted-A*  (anytime approximation)
  + lazy motion validation (L0 -> L1 -> L2 -> L3, RRT last)

Theoretical stance (see ``ASSUMPTIONS.md``):
  * The backward order is used ONLY as a constraint-closing variable ordering.
  * It does NOT assume locally optimal suffixes are globally optimal.
  * The provable structure is *suffix preservation / extension invariance*:
    once steps k..n are certified, adding an earlier part x_{k-1} cannot
    invalidate them, because part k-1 is already assembled (at goal) during every
    step j>=k, so x_{k-1} never enters O_j.
  * A feasible suffix is NOT guaranteed to extend to a full layout -- dead-end
    suffixes exist, which is exactly why we keep multiple nodes + matching /
    domain-propagation pruning rather than a single greedy suffix.

Modules
-------
- ``cost``    : RRT-free cost model (keypose polyline length) + admissible LB.
- ``oracle``  : layered per-step feasibility phi_k (L0-L3) wrapping the searcher.
- ``domain``  : candidate domains D_k (continuous + discrete grid) + occupancy.
- ``pruning`` : Hall-matching feasibility, domain propagation.
- ``req_motion_masks`` : paper ``W_req`` / ``F_{k,j}`` prescribed-motion masks.
- ``search``  : backward OPEN-list controller (exact A*/BB + anytime beam).
- ``run``     : CLI + per-site parallelism + final forward certification.
"""

from __future__ import annotations
