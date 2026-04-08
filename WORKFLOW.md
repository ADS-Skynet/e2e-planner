# Planner Workflow — Scenario 0 (Lane Follow)

End-to-end guide: data collection → augmentation → training → evaluation → inference.

---

## Prerequisites

### PyTorch (Jetson JetPack 6.x)

Standard pip torch does **not** have CUDA support on Jetson. Install the NVIDIA wheel:

```bash
pip uninstall torch -y
pip install https://developer.download.nvidia.com/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl

# Verify CUDA is available
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
# Expected: CUDA: True
```

### Other packages

```bash
pip install ultralytics pandas numpy matplotlib pyrealsense2
```

---

## Step 1 — Collect Data

**LKAS must start first** — it creates all shared memory (image, detection, control SHM).

**Terminal 1:**
```bash
lkas --broadcast
```
Wait until the status bar shows `image: ● CONNECTED`.

**Terminal 2:**
```bash
cd /home/peter/ads-skynet/planner/realsense_cam/v11
python collect_data_planner.py --scenario 0
```

Open the web viewer: `http://<jetson-ip>:8082`

### Controls

| Key | Action |
|-----|--------|
| `←` / `→` | Steer left / right (hold) |
| `↑` | Throttle +0.1 per press |
| `↓` | Throttle = 0 (stop) |
| `Space` | Toggle recording ON / OFF |

Red badge = recording active. Nothing is saved while badge is grey.

### What to collect for Scenario 0

Collect in multiple short sessions. Toggle recording OFF when repositioning the car.

**Session A — Normal lane following (~150 rows)**
- Press `↑` twice (throttle = 0.2), toggle recording ON
- Drive several laps staying centred in lane
- Steer smoothly — avoid sharp jerks
- Toggle OFF to reposition, ON to resume

**Session B — Recovery driving (~100 rows)**
- Deliberately drift 20–30% toward one lane boundary, then steer back to centre
- Do both sides (drift left → correct right, drift right → correct left)
- This is the most important data — without it the model can't recover from drift
- Record the full drift + correction arc, not just the endpoint

**Session C — Corners (~50 rows)**
- Slow down on corners (`↓` to reduce throttle), steer through, accelerate out
- Cover left and right corners equally

**Target: ~300 rows minimum before augmenting.**

The terminal shows a live counter:
```
[REC]  saved=  142  steer=+0.00 ↑  thr=0.20  objs=0  lane=YES
```

Output file: `data/planner_data.csv`

---

## Step 2 — Augment

Expands the dataset ~8× using physically meaningful transforms (mirror, noise, dropout, etc.).

```bash
python augment.py
```

Default paths:
- Input:  `data/planner_data.csv`
- Output: `data/augmented_data.csv`

Expected output:
```
[Augment] Loaded 300 rows from data/planner_data.csv
[Augment] Applying 8 transforms...
  identity          : 300 rows
  mirror            : 300 rows
  distance_noise    : 300 rows
  lateral_jitter    : 300 rows
  confidence_noise  : 300 rows
  object_dropout    : 300 rows
  distance_scale    : 300 rows
  mirror_noise      : 300 rows
[Augment] Total: 2400 rows → data/augmented_data.csv
```

> **Why mirror matters for recovery data:** mirroring a drift-left + correct-right sample
> automatically gives you a drift-right + correct-left sample. Your 100 recovery rows become
> 200 after augmentation.

---

## Step 3 — Train

```bash
python train_planner.py
```

This reads `data/augmented_data.csv` by default and saves `planner_model.pth`.

```bash
# Optional overrides:
python train_planner.py \
    --csv    data/augmented_data.csv \
    --epochs 100 \
    --lr     3e-4 \
    --batch-size 64 \
    --output planner_model.pth
```

Training output (one line per epoch after warmup):
```
  Training  (100 epochs, Adam+CosineAnnealing, MSELoss)
   Epoch       Train        Val   Steer MAE   Thtl MAE  LR
       1    0.234512   0.241234      0.3821     0.0512  3.00e-04
       2    0.198734   0.213456      0.3102     0.0481  3.00e-04  ★
      ...
      47    0.032145   0.041234      0.0821     0.0043  1.23e-04  ★
```

`★` marks epochs where the model improved — best checkpoint is saved automatically.

**What good convergence looks like:**
- Val loss drops steadily for ~40–60 epochs then plateaus
- Steering MAE < 0.10 is good for discrete {-0.9, 0, +0.9} targets
- If val loss rises while train loss drops → overfitting → collect more data

On Jetson Orin Nano: ~3–5 minutes for 100 epochs on 2400 rows.

---

## Step 4 — Evaluate

Before putting the model on the car, check its offline accuracy on the **raw** (non-augmented) data so you see real-world error, not augmented-set error.

```bash
python evaluate.py
```

```bash
# Optional overrides:
python evaluate.py \
    --csv     data/planner_data.csv \
    --model   planner_model.pth \
    --out-dir data/eval
```

Terminal output:
```
OVERALL RESULTS
  Samples          : 300
  Steering MAE     : 0.0821
  Steering RMSE    : 0.1134
  Throttle MAE     : 0.0043
  Throttle RMSE    : 0.0061

PER-SCENARIO RESULTS
  Scenario           N   Steer MAE   Steer RMSE   Thtl MAE
  LANE_FOLLOW      300      0.0821       0.1134     0.0043
```

Plots saved to `data/eval/`:

| File | What to look for |
|------|-----------------|
| `steering_scatter.png` | Points should cluster along the diagonal |
| `steering_error_hist.png` | Distribution centred at 0, no systematic bias |
| `per_scenario_mae.png` | Bar chart — all bars should be similar height |
| `timeseries.png` | Predicted vs actual steering over 200 frames |

**If results are poor:**
- Steering MAE > 0.15 → collect more data, especially recovery sessions
- Error histogram skewed left or right → model biased; check mirror augmentation worked
- Val loss was rising during training → reduce `--epochs` or collect more data

---

## Step 5 — Inference on Vehicle

**Terminal 1:**
```bash
lkas --broadcast
```

**Terminal 2 — simulation first (no motors):**
```bash
python planner_inference.py --scenario 0
```

Open `http://<jetson-ip>:8082` and watch the steering prediction in the web viewer. The predicted steering bar and throttle value should respond sensibly as you move the camera by hand:
- Point camera at lane centre → steering ≈ 0
- Tilt camera toward left boundary → steering should go positive (correct right)
- Tilt camera toward right boundary → steering should go negative (correct left)

**If it looks correct, enable motors:**
```bash
python planner_inference.py --scenario 0 --motor
```

Terminal shows:
```
[LANE_FOLLOW]  steer=+0.023  thr=0.200  objs=0  lane=YES  FPS=18.3
```

---

## Iteration Loop

If the vehicle drifts or behaves poorly on track:

```
1. Note what went wrong (drifts left? overshoots corners? freezes on obstacle?)
2. Collect more targeted data for that failure mode
3. Re-run augment → train → evaluate → deploy
```

Each iteration should add ~50–100 focused rows. Targeted data beats large random collection.

---

## Quick Reference

```
Terminal 1 (always):   lkas --broadcast
Terminal 2 (collect):  python collect_data_planner.py --scenario 0
Terminal 2 (train):    python augment.py && python train_planner.py
Terminal 2 (eval):     python evaluate.py
Terminal 2 (infer):    python planner_inference.py --scenario 0 [--motor]
```

### Key files

| File | Purpose |
|------|---------|
| `data/planner_data.csv` | Raw collected data |
| `data/augmented_data.csv` | Augmented dataset (input to training) |
| `planner_model.pth` | Trained model checkpoint |
| `data/eval/` | Evaluation plots and summary |

### Memory tips (Jetson Orin Nano)

- LKAS (BiSeNet on CPU) + YOLO (GPU) + RealSense share 7.4 GB unified memory
- If OOM: increase `YOLO_SKIP` in `collect_data_planner.py` (line ~119)
- Monitor with `tegrastats` in a separate terminal
