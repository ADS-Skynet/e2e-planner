#!/usr/bin/env python3
"""
Planner Dataset Augmentation
=============================
Reads planner_data.csv (structured features) and generates a larger,
augmented dataset: augmented_data.csv.

Why augment structured data?
  Unlike images, every augmentation here has a clear physical meaning —
  we know *exactly* what the correct steering response should be after
  mirroring or adding sensor noise.  This gives us:
    • 8-16× more samples from the same driving session
    • Better generalisation to sensor noise and track mirroring
    • Coverage of rare scenarios without extra collection time

Augmentation strategies applied
---------------------------------
1. Identity          — original row kept as-is
2. Mirror            — lateral flip: negate lat_offsets, center_offset,
                       left/right x swap, negate steering target
3. Distance noise    — Gaussian noise on dist_norm  (σ = 0.03)
4. Lateral jitter    — small noise on lat_offset and lane features (σ = 0.04)
5. Confidence noise  — noise on object confidence values (σ = 0.05)
6. Object dropout    — randomly zero out 1 object slot (valid=0, all feats=0)
7. Distance scale    — scale all dist_norm values by U(0.85, 1.15)
8. Mirror + noise    — mirror followed by distance noise (combined)

Usage
-----
  python augment.py [--input data/planner_data.csv]
                    [--output data/augmented_data.csv]
                    [--seed 42]
"""

import argparse
import copy
import numpy as np
import pandas as pd
from pathlib import Path

from planner_model import (
    N_MAX_OBJECTS, OBJ_FEATURES, LANE_FEATURES, EGO_FEATURES,
    csv_columns,
)

# ─────────────────────────────────────────────────────────────────────────────
# Column index helpers  (build once from the schema)
# ─────────────────────────────────────────────────────────────────────────────

_COLS = csv_columns()

def _col(name: str) -> int:
    return _COLS.index(name)

# Object feature column indices per slot
def _obj_col(slot: int, feat: str) -> int:
    return _col(f"obj{slot}_{feat}")

OBJ_VALID_COLS     = [_obj_col(i, "valid")       for i in range(N_MAX_OBJECTS)]
OBJ_LAT_COLS       = [_obj_col(i, "lat_offset")  for i in range(N_MAX_OBJECTS)]
OBJ_DIST_COLS      = [_obj_col(i, "dist_norm")   for i in range(N_MAX_OBJECTS)]
OBJ_CONF_COLS      = [_obj_col(i, "conf")        for i in range(N_MAX_OBJECTS)]
OBJ_W_COLS         = [_obj_col(i, "width_norm")  for i in range(N_MAX_OBJECTS)]
OBJ_H_COLS         = [_obj_col(i, "height_norm") for i in range(N_MAX_OBJECTS)]
OBJ_OVERLAP_COLS   = [_obj_col(i, "lane_overlap")for i in range(N_MAX_OBJECTS)]

LANE_DETECTED_COL    = _col("lane_detected")
LANE_CENTER_COL      = _col("lane_center_offset")
LANE_WIDTH_COL       = _col("lane_width_norm")
LANE_LEFT_COL        = _col("lane_left_x_norm")
LANE_RIGHT_COL       = _col("lane_right_x_norm")

TARGET_STEER_COL  = _col("target_steering")
TARGET_THTL_COL   = _col("target_throttle")
EGO_STEER_COL     = _col("ego_steering")

# All object-feature column indices (for bulk operations)
ALL_OBJ_COLS = []
for i in range(N_MAX_OBJECTS):
    for f in ("valid", "class_norm", "conf", "dist_norm",
              "lat_offset", "width_norm", "height_norm", "lane_overlap"):
        ALL_OBJ_COLS.append(_obj_col(i, f))


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation functions  (all operate on a numpy row vector)
# ─────────────────────────────────────────────────────────────────────────────

def _clone(row: np.ndarray) -> np.ndarray:
    return row.copy()


def aug_identity(row: np.ndarray, rng) -> np.ndarray:
    """Return the original row unchanged."""
    return _clone(row)


def aug_mirror(row: np.ndarray, rng) -> np.ndarray:
    """
    Horizontal flip of the scene.
    - Negate all object lateral offsets
    - Negate lane center offset
    - Swap left/right lane x positions
    - Negate steering target and ego steering
    - Lane overlap stays the same (symmetric)
    """
    r = _clone(row)
    for c in OBJ_LAT_COLS:
        r[c] = -r[c]
    r[LANE_CENTER_COL] = -r[LANE_CENTER_COL]
    # Swap left/right x in normalised space  (left_x + right_x = 1 in many cases but not guaranteed)
    old_left  = r[LANE_LEFT_COL]
    old_right = r[LANE_RIGHT_COL]
    r[LANE_LEFT_COL]  = 1.0 - old_right
    r[LANE_RIGHT_COL] = 1.0 - old_left
    # Negate steering
    r[TARGET_STEER_COL] = -r[TARGET_STEER_COL]
    r[EGO_STEER_COL]    = -r[EGO_STEER_COL]
    return r


def aug_distance_noise(row: np.ndarray, rng, sigma: float = 0.03) -> np.ndarray:
    """Add Gaussian noise to all valid object distances."""
    r = _clone(row)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_DIST_COLS[i]] = float(np.clip(
                r[OBJ_DIST_COLS[i]] + rng.normal(0, sigma), 0.0, 1.0))
    return r


def aug_lateral_jitter(row: np.ndarray, rng, sigma: float = 0.04) -> np.ndarray:
    """Add Gaussian noise to lateral offsets and lane center offset."""
    r = _clone(row)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_LAT_COLS[i]] += rng.normal(0, sigma)
    r[LANE_CENTER_COL] += rng.normal(0, sigma)
    return r


def aug_confidence_noise(row: np.ndarray, rng, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to object confidence values."""
    r = _clone(row)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_CONF_COLS[i]] = float(np.clip(
                r[OBJ_CONF_COLS[i]] + rng.normal(0, sigma), 0.0, 1.0))
    return r


def aug_object_dropout(row: np.ndarray, rng, drop_prob: float = 0.25) -> np.ndarray:
    """Randomly zero out one valid object slot (simulates missed detection)."""
    r = _clone(row)
    valid_slots = [i for i in range(N_MAX_OBJECTS) if r[OBJ_VALID_COLS[i]] > 0.5]
    if not valid_slots:
        return r
    drop_slot = rng.choice(valid_slots)
    # Zero out all features for this slot
    start = ALL_OBJ_COLS[drop_slot * OBJ_FEATURES]
    for j in range(OBJ_FEATURES):
        r[ALL_OBJ_COLS[drop_slot * OBJ_FEATURES + j]] = 0.0
    return r


def aug_distance_scale(row: np.ndarray, rng,
                       low: float = 0.85, high: float = 1.15) -> np.ndarray:
    """Scale all valid object distances by a uniform random factor."""
    r     = _clone(row)
    scale = rng.uniform(low, high)
    for i in range(N_MAX_OBJECTS):
        if r[OBJ_VALID_COLS[i]] > 0.5:
            r[OBJ_DIST_COLS[i]] = float(np.clip(r[OBJ_DIST_COLS[i]] * scale, 0.0, 1.0))
    return r


def aug_mirror_and_noise(row: np.ndarray, rng) -> np.ndarray:
    """Mirror + distance noise (combined augmentation)."""
    return aug_distance_noise(aug_mirror(row, rng), rng)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (name, function, weight)
# Weight controls how many augmented copies are produced per original row.
# Identity has weight 1 (keeps the original).
AUGMENTATIONS = [
    ("identity",         aug_identity,         1),
    ("mirror",           aug_mirror,           1),
    ("distance_noise",   aug_distance_noise,   1),
    ("lateral_jitter",   aug_lateral_jitter,   1),
    ("confidence_noise", aug_confidence_noise, 1),
    ("object_dropout",   aug_object_dropout,   1),
    ("distance_scale",   aug_distance_scale,   1),
    ("mirror_noise",     aug_mirror_and_noise, 1),
]


def augment(input_csv: Path, output_csv: Path, seed: int = 42) -> None:
    print("=" * 58)
    print("  Planner Dataset Augmentation")
    print("=" * 58)
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_csv}")
    print(f"  Seed   : {seed}")
    print()

    if not input_csv.exists():
        print(f"[ERROR] Input CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    print(f"[AUG] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate schema
    expected = set(csv_columns())
    missing  = expected - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns in input CSV: {missing}")
        return

    # Show label distribution
    if "target_steering" in df.columns:
        print(f"[AUG] Steering distribution (original):")
        print(df["target_steering"].describe().to_string())
        print()

    rng = np.random.default_rng(seed)

    # Convert to numpy for fast row operations
    cols_ordered = csv_columns()
    col_idx      = {c: i for i, c in enumerate(cols_ordered)}
    data_np      = df[cols_ordered].to_numpy(dtype=float)

    aug_rows = []
    for orig_row in data_np:
        for name, fn, weight in AUGMENTATIONS:
            for _ in range(weight):
                aug_rows.append(fn(orig_row, rng))

    aug_np = np.stack(aug_rows, axis=0)

    # Convert back to DataFrame
    aug_df = pd.DataFrame(aug_np, columns=cols_ordered)

    # frame_id: reassign sequential integers
    aug_df["frame_id"] = np.arange(1, len(aug_df) + 1)

    # Clip all normalised features to their valid ranges
    for i in range(N_MAX_OBJECTS):
        aug_df[f"obj{i}_valid"]       = aug_df[f"obj{i}_valid"].clip(0, 1).round()
        aug_df[f"obj{i}_conf"]        = aug_df[f"obj{i}_conf"].clip(0, 1)
        aug_df[f"obj{i}_dist_norm"]   = aug_df[f"obj{i}_dist_norm"].clip(0, 1)
        aug_df[f"obj{i}_lane_overlap"]= aug_df[f"obj{i}_lane_overlap"].clip(0, 1)
        aug_df[f"obj{i}_width_norm"]  = aug_df[f"obj{i}_width_norm"].clip(0, 1)
        aug_df[f"obj{i}_height_norm"] = aug_df[f"obj{i}_height_norm"].clip(0, 1)
    aug_df["lane_left_x_norm"]  = aug_df["lane_left_x_norm"].clip(0, 1)
    aug_df["lane_right_x_norm"] = aug_df["lane_right_x_norm"].clip(0, 1)
    aug_df["target_steering"]   = aug_df["target_steering"].clip(-1, 1)
    aug_df["target_throttle"]   = aug_df["target_throttle"].clip(0, 1)

    # Scenario must stay integer
    aug_df["scenario"] = aug_df["scenario"].round().astype(int)

    aug_df.to_csv(output_csv, index=False)

    n_aug = len(aug_df)
    n_orig = len(df)
    print(f"[AUG] Augmented: {n_orig} → {n_aug} rows  (×{n_aug/n_orig:.1f})")
    print()
    print(f"[AUG] Augmented steering distribution:")
    print(aug_df["target_steering"].describe().to_string())
    print()
    print(f"[AUG] Scenario distribution:")
    print(aug_df["scenario"].value_counts().sort_index().to_string())
    print()
    print(f"[AUG] Saved → {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    data_dir   = script_dir / "data"

    parser = argparse.ArgumentParser(description="Augment the structured planner dataset")
    parser.add_argument('--input',  type=Path,
                        default=data_dir / "planner_data.csv",
                        help='Input CSV (default: data/planner_data.csv)')
    parser.add_argument('--output', type=Path,
                        default=data_dir / "augmented_data.csv",
                        help='Output CSV (default: data/augmented_data.csv)')
    parser.add_argument('--seed',   type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    augment(args.input, args.output, args.seed)
