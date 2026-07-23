"""Deprecated entry point -- BSFS now lives in the ``bsfs`` package.

The single-file beam prototype has been superseded by the rigorous backward
suffix-factorization solver (exact A*/branch-and-bound + anytime beam,
admissible matching lower bound, safe domain propagation, lazy motion
validation, per-site parallelism). Use::

    python -m sealp.examples.layout.bsfs.run --mode exact ...
    python -m sealp.examples.layout.bsfs.run --mode beam  ...

This shim forwards to ``bsfs.run.main`` for backward compatibility. Note the
removed flags: ``--pick-first-part`` (step-0 is always preassembled at the site,
per assumptions A2/A3) and the ``--disable-order-x`` no-op.
"""

from __future__ import annotations

import sys

from sealp.examples.layout.bsfs.run import main


def _strip_removed(argv):
    out, skip = [], False
    for tok in argv:
        if skip:
            skip = False
            continue
        if tok in ("--pick-first-part", "--disable-order-x"):
            print(f"[bsfs] note: '{tok}' is removed and ignored.")
            continue
        out.append(tok)
    return out


if __name__ == "__main__":
    main(_strip_removed(sys.argv[1:]))
