# BSFS assumptions and guarantees

This document states, for the paper, exactly what the backward suffix method
assumes and what it proves. The backward order is used **only as a
constraint-closing variable ordering**; it is not a claim that locally optimal
suffixes are globally optimal.

## Provable structure

Fix a direct-assembly order `pi = (pi_1, ..., pi_n)` and an assembly site `a`.
Let `x_k = (p_{pi_k}, g_{pi_k})` be part k's staging position + direct-assembly
grasp. When step `k` executes, parts `1..k-1` are already assembled (their final
geometry is in the partial assembly `A_{k-1}(a)`) and their *staging* positions
are gone, while parts `k+1..n` are still at staging. Hence the step-k obstacle
set is

    O_k = E  ∪  A_{k-1}(a)  ∪  ⋃_{j>k} M_{pi_j}(p_{pi_j})

so `phi_k = phi_k(a, x_k, x_{k+1}, ..., x_n)` does **not** depend on
`x_1, ..., x_{k-1}`. Full feasibility is the triangular conjunction
`Phi = AND_k phi_k(a, x_{k:n})`.

**Suffix preservation (extension invariance).** If `x_{k:n}` certifies steps
`k..n`, adding an earlier `x_{k-1}` cannot invalidate any step `j >= k` (its
parameters and partial-assembly geometry are unchanged). Only the new step
`phi_{k-1}` must be verified. This is what lets the exact search never
re-evaluate a certified suffix.

**A feasible suffix may NOT extend.** `proj_{k:n}(S_{k-1}) ⊆ S_k` but not
conversely; dead-end suffixes exist. That is why we keep multiple nodes plus Hall
matching + domain propagation, not a single greedy suffix.

## Required-motion collision masks (`W_req` / `F_{k,j}`)

Implemented in `req_motion_masks.py` and applied in `StepOracle.certify`
(default on; `--no-req-motion-masks` disables):

1. **post-grasp retreat** — attached-object mesh swept along pick depart (+Z)
2. **prescribed insertion** — attached-object mesh swept along mating approach
3. **post-release retreat** — under-approx EE capsule along mating depart
   (beam only; exact mode disables this over-approx so A* hard-prunes stay sound)

A hit against an already-assigned *later* staging part is a deterministic
`req_motion_mask` hard-prune. Forward search also shrinks undecided later
`(x,y)` domains by the same forbidden sets. Non-intersection does **not** prove
local feasibility; L1/L2/L3 still run.

## Assumptions (must hold for the guarantees below)

- **A1. Fixed assembly order.** `pi` is input and unchanged during search.
- **A2. Picked parts vacate their staging cell.** Once picked, a part's initial
  footprint is no longer an obstacle; it exists only as fixed geometry in
  `A_{k-1}(a)`. (Fixtures/pallets that remain must be added to `E`.)
  *In our chair setup the step-0 seat is preassembled at `a`: it lives in
  `A_0(a)` and its initial position never enters any `O_k`.*
- **A3. Final geometry depends only on `a`** and the product model, not on the
  staging position: `T = T_A(a) · ^A T_i`.
- **A4. Fixed robot interface state.** Each step starts and ends at a fixed
  home/standby config `q_H`, so steps are independently verifiable. (If the end
  config of step k feeds step k+1, add the interface config to the suffix state.)
- **A5. Fixed direct-assembly action mode** (pick → transfer → insert → release
  → retreat). Regrasp / handover / tool change would become extra search
  variables.
- **A6. Deterministic environment** (known CAD, known poses, deterministic
  collision). Perception/execution error is handled by inflated clearance
  (`tau_clear`), not automatically.
- **A7. Trusted motion oracle for pruning.** Deterministic geometry / IK
  failures are proofs of infeasibility and may hard-prune. An **RRT timeout is
  not** a proof; it returns `unproven` and never hard-prunes. Correctness is
  therefore stated *conditioned on an exact motion-feasibility oracle*; in
  practice we hard-prune only on L0/L1 deterministic failures.

## What each mode guarantees

| Mode | Returned layout valid | Complete | Global optimum |
|------|-----------------------|----------|----------------|
| Exact backward A* / branch-and-bound | yes (witnessed) | yes | yes, in the discretized domain, with an admissible lower bound |
| Anytime beam (width B) | yes (witnessed) | no | no |

The exact solver uses `h =` sum over remaining parts of the min straight-line
`||p_pick − p_place||` over their domain, an admissible lower bound on the
end-effector polyline cost, so the first complete node A* pops is optimal. Every
returned layout (either mode) is re-checked by the full forward
`evaluate_layout` witness, so there are **no false positives** — the beam's only
weakness is possibly missing a better layout, quantified by
`bench_optimality_gap.py`.

## Recommended paper phrasing

> The backward order is used solely as a constraint-closing variable ordering.
> It does not assume that locally optimal suffixes lead to globally optimal
> layouts. Under exhaustive branch-and-bound with an admissible lower bound, the
> method returns the globally optimal layout in the discretized search domain. A
> finite-width beam implementation is used only as an anytime approximation for
> larger instances, and its empirical optimality gap is reported.
