"""Paper experiment suite for the BSFS staging-layout planner.

This package contains ONLY experiment drivers / baselines. It never modifies the
frozen core (BSFS search, StepOracle, cost model, pick-depart gate, yaw
refinement, Hall/domain propagation, assembly-center search). Every experiment
reuses those components unchanged and varies exactly one controlled variable per
study:

    P1  search / variable-assignment ORDER   (backward BSFS vs forward beam)
    P2  obstacle STATE model                  (sequential / static-start / static-final)
    P3  solver                                (exact A* vs beam, reduced discrete)
    P4  cross-assembly task                   (chair / tower / ...)
    P5  pruning ablation                      (Hall / domain-propagation on/off)

All runs share the frozen final configuration: workers=4, BLAS threads pinned to
1 (see run.py), staging_aware witness, documented seeds. Each run is written to
its own JSON and appended to a single summary CSV; nothing is overwritten.
"""
