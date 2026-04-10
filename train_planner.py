#!/usr/bin/env python3
"""
Planner Training Script
========================
Trains PlannerModel on the structured feature dataset.

  Input  : data/augmented_data.csv  (or planner_data.csv as fallback)
  Output : planner_model.pth

Architecture recap
------------------
  objects  (40-dim) → MLP → 64-dim  ─┐
  lane      (5-dim) → MLP → 32-dim  ─┤ concat (120-dim)
  ego       (2-dim) → MLP → 16-dim  ─┤   → trunk → steering_head  → steering ∈ [-1, 1]
  scenario  (int)   → Emb →  8-dim  ─┘           → throttle_head  → throttle ∈ [ 0, 1]

Loss: MSELoss on both outputs jointly (equal weight).
      Training is fast — no GPU memory pressure like image models.

Usage
-----
  python train_planner.py [--csv data/augmented_data.csv]
                          [--epochs 100]
                          [--lr 3e-4]
                          [--batch-size 64]
                          [--output planner_model.pth]
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from planner_model import (
    PlannerModel,
    csv_columns,
    row_to_tensors,
    N_MAX_OBJECTS, OBJ_FEATURES, LANE_FEATURES, EGO_FEATURES,
    GRID_ROWS, GRID_COLS,
    N_SCENARIOS, MAX_THROTTLE,
)

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
DEFAULT_CSV = DATA_DIR / "augmented_data.csv"
FALLBACK_CSV= DATA_DIR / "planner_data.csv"
SAVE_PATH   = SCRIPT_DIR / "planner_model.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PlannerDataset(Dataset):
    """
    Each sample: (objects, lane, ego, scenario, target)
      objects  : (40,)  float32
      lane     : ( 5,)  float32
      ego      : ( 2,)  float32
      scenario : ()     long
      target   : ( 2,)  float32  [steering, throttle_norm]
    """

    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)

        # Validate schema
        expected = set(csv_columns())
        missing  = expected - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        # Drop rows with NaN
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        if len(df) < before:
            print(f"[data] Dropped {before - len(df)} rows with NaN values")

        self.df = df
        print(f"[data] Loaded {len(df)} rows from {csv_path.name}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Object features: (N_MAX_OBJECTS * OBJ_FEATURES,)
        obj_vals = [float(row[f"obj{i}_{f}"])
                    for i in range(N_MAX_OBJECTS)
                    for f in ("valid", "class_norm", "conf", "dist_norm",
                              "lat_offset", "width_norm", "height_norm", "lane_overlap")]
        objects = torch.tensor(obj_vals, dtype=torch.float32)

        # Lane features: (LANE_FEATURES,) — 4×8 spatial grid
        lane_vals = [float(row[f"lane_r{r}c{c}"])
                     for r in range(GRID_ROWS) for c in range(GRID_COLS)]
        lane = torch.tensor(lane_vals, dtype=torch.float32)

        # Ego features: (EGO_FEATURES,)
        ego = torch.tensor([float(row["ego_steering"]),
                            float(row["ego_throttle"])], dtype=torch.float32)

        # Scenario token
        scenario = torch.tensor(int(row["scenario"]), dtype=torch.long)

        # Targets: [steering, throttle_norm]
        target = torch.tensor([float(row["target_steering"]),
                                float(row["target_throttle"])], dtype=torch.float32)

        return objects, lane, ego, scenario, target


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    csv_path:   Path  = DEFAULT_CSV,
    epochs:     int   = 100,
    lr:         float = 3e-4,
    batch_size: int   = 64,
    save_path:  Path  = SAVE_PATH,
) -> None:

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 58)
    print("  Planner Training")
    print("=" * 58)
    print(f"  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
        free, total = [x / 1024**2 for x in torch.cuda.mem_get_info(0)]
        print(f"  VRAM      : {free:.0f} MB free / {total:.0f} MB total")
    print(f"  CSV       : {csv_path}")
    print(f"  Epochs    : {epochs}")
    print(f"  LR        : {lr}")
    print(f"  Batch     : {batch_size}")
    print(f"  Save      : {save_path}")
    print()

    # ── Dataset ───────────────────────────────────────────────────────────────
    if not csv_path.exists():
        if FALLBACK_CSV.exists():
            print(f"[WARN] {csv_path.name} not found — using {FALLBACK_CSV.name}")
            csv_path = FALLBACK_CSV
        else:
            print(f"[ERROR] No dataset found. Run collect_data_planner.py first.")
            sys.exit(1)

    dataset = PlannerDataset(csv_path)

    if len(dataset) == 0:
        print("[ERROR] Dataset is empty.")
        sys.exit(1)

    # ── Dataset stats ─────────────────────────────────────────────────────────
    print("=" * 58)
    print("  Dataset Statistics")
    print("=" * 58)
    df = dataset.df
    print(f"  Total samples   : {len(df)}")
    print()
    print("  Scenario distribution:")
    from planner_model import SCENARIO_NAMES as scenario_map
    for sc, cnt in df["scenario"].value_counts().sort_index().items():
        print(f"    {scenario_map.get(sc, sc):20s}: {cnt:>6d}  ({100*cnt/len(df):.1f}%)")
    print()
    print("  Target steering:")
    print(f"    mean={df['target_steering'].mean():+.4f}  "
          f"std={df['target_steering'].std():.4f}  "
          f"min={df['target_steering'].min():+.4f}  "
          f"max={df['target_steering'].max():+.4f}")
    print("  Target throttle (normalised):")
    print(f"    mean={df['target_throttle'].mean():.4f}  "
          f"std={df['target_throttle'].std():.4f}  "
          f"min={df['target_throttle'].min():.4f}  "
          f"max={df['target_throttle'].max():.4f}")
    print()

    # ── Train / val split ─────────────────────────────────────────────────────
    n_total = len(dataset)
    n_train = int(0.85 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train / Val     : {n_train} / {n_val}")
    print()

    # Structured data is tiny — use larger batch & more workers than image models
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=(device.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PlannerModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params    : {n_params:,}")
    print()

    # ── Quick forward pass check ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        o_t = sample[0].unsqueeze(0).to(device)
        l_t = sample[1].unsqueeze(0).to(device)
        e_t = sample[2].unsqueeze(0).to(device)
        s_t = sample[3].unsqueeze(0).to(device)
        out = model(o_t, l_t, e_t, s_t)
    print(f"  Forward check   : input shapes obj={tuple(o_t.shape)} "
          f"lane={tuple(l_t.shape)} ego={tuple(e_t.shape)} sc={tuple(s_t.shape)}")
    print(f"  Output sample   : steer={out[0,0].item():+.4f}  "
          f"throttle={out[0,1].item():.4f}")
    print("  PASSED")
    print()

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    print("=" * 58)
    print(f"  Training  ({epochs} epochs, Adam+CosineAnnealing, MSELoss)")
    print("=" * 58)
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Steer MAE':>10}  {'Thtl MAE':>9}  LR")
    print("-" * 62)

    for epoch in range(1, epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for objects, lane, ego, scenario, target in train_loader:
            objects  = objects.to(device)
            lane     = lane.to(device)
            ego      = ego.to(device)
            scenario = scenario.to(device)
            target   = target.to(device)

            optimizer.zero_grad()
            pred = model(objects, lane, ego, scenario)   # (B, 2)
            loss = criterion(pred, target)
            loss.backward()
            # Gradient clipping — keeps training stable for tiny datasets
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(target)

        train_loss = running_loss / n_train
        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        running_loss = 0.0
        steer_abs    = 0.0
        thtl_abs     = 0.0
        with torch.no_grad():
            for objects, lane, ego, scenario, target in val_loader:
                objects  = objects.to(device)
                lane     = lane.to(device)
                ego      = ego.to(device)
                scenario = scenario.to(device)
                target   = target.to(device)
                pred     = model(objects, lane, ego, scenario)
                loss     = criterion(pred, target)
                running_loss += loss.item() * len(target)
                steer_abs    += (pred[:, 0] - target[:, 0]).abs().sum().item()
                thtl_abs     += (pred[:, 1] - target[:, 1]).abs().sum().item()

        val_loss  = running_loss / n_val
        steer_mae = steer_abs / n_val
        thtl_mae  = thtl_abs  / n_val
        cur_lr    = scheduler.get_last_lr()[0]

        # ── Save best ─────────────────────────────────────────────────────────
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            marker = "  ★"

        # Print every epoch up to 20, then every 10
        if epoch <= 20 or epoch % 10 == 0 or marker:
            print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{steer_mae:>10.4f}  {thtl_mae:>9.4f}  {cur_lr:.2e}{marker}")

    print()
    print(f"  Best val loss   : {best_val_loss:.6f}")
    print(f"  Model saved  →  {save_path}")
    print()
    print("  Next steps:")
    print("   1. Run evaluate.py to check per-scenario error")
    print("   2. Collect more data for under-performing scenarios")
    print("   3. Run planner_inference.py --motor to test on vehicle")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the structured planner model")
    parser.add_argument('--csv',        type=Path,  default=DEFAULT_CSV,
                        help=f'Dataset CSV (default: {DEFAULT_CSV})')
    parser.add_argument('--epochs',     type=int,   default=100,
                        help='Training epochs (default: 100)')
    parser.add_argument('--lr',         type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int,   default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--output',     type=Path,  default=SAVE_PATH,
                        help=f'Model output path (default: {SAVE_PATH})')
    args = parser.parse_args()

    train(
        csv_path   = args.csv,
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch_size,
        save_path  = args.output,
    )
