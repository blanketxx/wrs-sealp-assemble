"""One-off repair for the summary CSV whose header predates four added columns.

The summary CSV is append-only, so its header was written by the first run ever
recorded and never revisited. Later runs started emitting complete_leaves,
l3_staging_aware, l3_fail_step, l3_fail_reason and center_source, which makes
every recent row four fields wider than the header claims -- so reading the file
by column name silently mis-maps them.

Rows are distinguished by field count, remapped onto the current CSV_COLUMNS and
rewritten in place after a timestamped backup. Values are never altered.

    python -m sealp.examples.layout.experiments.migrate_csv_header
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import shutil
import sys

from sealp.examples.layout.experiments.run_one import CSV_COLUMNS

# Columns introduced after the header was first written, newest batch last. Each
# historical schema is the current one minus every batch added since.
_ADDED_SINCE = [
    ("complete_leaves", "l3_staging_aware", "l3_fail_step", "l3_fail_reason",
     "center_source"),
    ("search_time_s", "yaw_time_s", "optimistic_certifications",
     "unsound_steps", "unsound_parts", "audit_certifications"),
]


def _historical_schemas():
    """Current schema plus every earlier one, keyed by field count."""
    schemas, dropped = {len(CSV_COLUMNS): CSV_COLUMNS}, set()
    for batch in reversed(_ADDED_SINCE):
        dropped |= set(batch)
        cols = [c for c in CSV_COLUMNS if c not in dropped]
        schemas[len(cols)] = cols
    return schemas

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_output", "experiments_summary.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    a = ap.parse_args(argv)

    if not os.path.isfile(a.csv):
        print(f"no such CSV: {a.csv}")
        return 1

    with open(a.csv, newline="", encoding="utf-8") as fh:
        raw = list(csv.reader(fh))
    if not raw:
        print("empty CSV, nothing to do")
        return 0

    header, body = raw[0], raw[1:]
    if header == CSV_COLUMNS:
        print(f"header already current ({len(CSV_COLUMNS)} columns), nothing to do")
        return 0

    schemas = _historical_schemas()
    out, counts = [], {}
    for row in body:
        schema = schemas.get(len(row))
        if schema is None:
            print(f"unrecognised row width {len(row)} (known: "
                  f"{sorted(schemas)}): {row[:5]} -- aborting, file untouched")
            return 2
        counts[len(row)] = counts.get(len(row), 0) + 1
        rec = dict(zip(schema, row))
        out.append([rec.get(c, "") for c in CSV_COLUMNS])

    backup = f"{a.csv}.bak_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(a.csv, backup)
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        w.writerows(out)

    print(f"backup     : {backup}")
    print(f"rewritten  : {a.csv}")
    print(f"rows       : {sum(counts.values())} " +
          ", ".join(f"{n} of width {k}" for k, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
