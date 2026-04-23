#!/usr/bin/env python3
"""
Deduplicate planner_data.csv
=============================
Removes duplicate rows (ignoring frame_id, which is just a counter and differs
even when all feature values are identical).  Prints a report and overwrites
the file in-place, with a .bak backup made first.

Usage
-----
    python dedup.py [path/to/planner_data.csv]
"""

import sys
import time
import shutil
import pandas as pd
from pathlib import Path

# ── Target file ───────────────────────────────────────────────────────────────
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 \
           else Path(__file__).resolve().parent / "data" / "planner_data.csv"

if not csv_path.exists():
    print(f"[ERROR] File not found: {csv_path}")
    sys.exit(1)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading  {csv_path} …")
df = pd.read_csv(csv_path)
n_before = len(df)

# frame_id is just a row counter — two rows recorded with identical features
# will have different frame_ids, so exclude it from the duplicate key.
feature_cols = [c for c in df.columns if c != "frame_id"]

# ── Find duplicates ───────────────────────────────────────────────────────────
dup_mask  = df.duplicated(subset=feature_cols, keep="first")
n_removed = dup_mask.sum()
n_after   = n_before - n_removed

# ── Per-scenario breakdown ────────────────────────────────────────────────────
from planner_model import SCENARIO_NAMES   # noqa: E402  (local import after path known)

sc_before = df.groupby("scenario").size()
sc_dups   = df[dup_mask].groupby("scenario").size().reindex(sc_before.index, fill_value=0)
sc_after  = sc_before - sc_dups

# ── Steering distribution before/after ───────────────────────────────────────
def steer_buckets(series):
    bins   = [-1.01, -0.5, -0.05, 0.05, 0.5, 1.01]
    labels = ["hard-left(≤-0.5)", "left(-0.5..-0.05)", "straight(±0.05)",
              "right(0.05..0.5)", "hard-right(≥0.5)"]
    return pd.cut(series, bins=bins, labels=labels).value_counts().reindex(labels)

steer_b = steer_buckets(df["target_steering"])
steer_a = steer_buckets(df[~dup_mask]["target_steering"])

# ── Report ────────────────────────────────────────────────────────────────────
sep = "─" * 52
print()
print(sep)
print("  Deduplication Report")
print(sep)
print(f"  File            : {csv_path.name}")
print(f"  Rows before     : {n_before:>6,}")
print(f"  Duplicates found: {n_removed:>6,}  ({100 * n_removed / n_before:.1f}%)")
print(f"  Rows after      : {n_after:>6,}")
print()

print("  Per-scenario breakdown:")
print(f"    {'Scenario':<22}  {'Before':>6}  {'Removed':>7}  {'After':>6}")
print(f"    {'─'*22}  {'─'*6}  {'─'*7}  {'─'*6}")
for sc in sc_before.index:
    name = SCENARIO_NAMES.get(int(sc), str(sc))
    print(f"    {name:<22}  {sc_before[sc]:>6,}  {sc_dups[sc]:>7,}  {sc_after[sc]:>6,}")
print()

print("  Steering distribution (target_steering):")
print(f"    {'Bucket':<22}  {'Before':>6}  {'After':>5}")
print(f"    {'─'*22}  {'─'*6}  {'─'*5}")
for label in steer_b.index:
    print(f"    {label:<22}  {steer_b[label]:>6,}  {steer_a[label]:>5,}")
print()

if n_removed == 0:
    print("  No duplicates — file unchanged.")
    print(sep)
    sys.exit(0)

# ── Backup + overwrite ────────────────────────────────────────────────────────
backup = csv_path.with_suffix(f".bak{int(time.time())}.csv")
shutil.copy2(csv_path, backup)
print(f"  Backup          : {backup.name}")

clean = df[~dup_mask].copy()
clean["frame_id"] = range(1, len(clean) + 1)   # re-number frame_ids sequentially
clean.to_csv(csv_path, index=False)
print(f"  Saved           : {csv_path.name}")
print(sep)
