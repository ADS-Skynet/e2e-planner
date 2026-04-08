# v11 Structured Planner

Learned driving planner for the RC vehicle (Jetson Orin Nano + RealSense camera).

---

## Concept

### The Perception → Planner Split

Classic end-to-end driving feeds raw camera pixels directly into a neural network that outputs steering and throttle. That works but requires enormous amounts of data, is sensitive to lighting and visual domain, and is hard to debug.

This project takes a different approach — the same architecture Tesla used before FSD v12:

```
Camera ──► YOLO           ──► object list (class, distance, position…)  ─┐
       ──► Lane Seg (LKAS) ──► lane boundaries, centre offset…           ─┤──► PLANNER ──► steering
                                                                           │               throttle
                               ego state (prev steering/throttle)        ─┘
```

**Your colleague** owns the left side (perception — camera, YOLO, lane segmentation).
**This module** owns the right side (planner — structured numbers in, actuation out).

The planner never touches pixels. It only ever sees a fixed-size vector of normalised numbers describing the world, and it outputs two numbers: steering and throttle.

### Why this is better for this project

| Property | End-to-end (pixels → control) | This planner (features → control) |
|---|---|---|
| Data needed | Thousands of frames | Hundreds of rows |
| Sensitive to lighting | Yes | No |
| Augmentation | Hard (image transforms) | Easy (perturb numbers) |
| Model size | Millions of params | ~10 k params |
| Inference time on Jetson | 10–50 ms | < 1 ms |
| Debuggable | Hard | Read the CSV |

---

## Input / Output

### Planner Input (per frame, ~50 numbers total)

**Object block** — top 5 closest YOLO detections, padded with zeros if fewer:

| Feature | Description | Range |
|---|---|---|
| `valid` | 1 if slot has a real object, 0 if padding | {0, 1} |
| `class_norm` | YOLO class ID ÷ total classes | [0, 1] |
| `conf` | Detection confidence | [0, 1] |
| `dist_norm` | Distance ÷ 5 m | [0, 1] |
| `lat_offset` | Signed lateral offset from lane centre, normalised by lane width | (−∞, ∞) |
| `width_norm` | Bounding box width ÷ frame width | [0, 1] |
| `height_norm` | Bounding box height ÷ frame height | [0, 1] |
| `lane_overlap` | Fraction of lane width the object covers | [0, 1] |

5 objects × 8 features = **40 values**

**Lane block** — from LKAS lane segmentation:

| Feature | Description |
|---|---|
| `lane_detected` | 1 if LKAS found lanes, 0 if using fallback |
| `lane_center_offset` | (frame centre − lane centre) ÷ lane width — positive means car is left of centre |
| `lane_width_norm` | Lane width ÷ frame width |
| `lane_left_x_norm` | Left lane boundary ÷ frame width |
| `lane_right_x_norm` | Right lane boundary ÷ frame width |

5 values

**Ego state** — previous cycle's output:

| Feature | Description |
|---|---|
| `ego_steering` | Previous steering command |
| `ego_throttle` | Previous throttle ÷ MAX_THROTTLE |

2 values

**Scenario token** — integer that tells the planner what it is supposed to be doing:

| Value | Name | When to use |
|---|---|---|
| 0 | LANE_FOLLOW | Normal track driving |
| 1 | OBSTACLE_AVOID | Obstacles present |
| 2 | PARKING | Parking manoeuvre |
| 3 | STOP | Deliberate stop |

### Planner Output

| Output | Range | Notes |
|---|---|---|
| `steering` | [−1, 1] | Negative = left, positive = right |
| `throttle` | [0, 1] | Multiplied by `MAX_THROTTLE` before sending to JetRacer |

---

## Model Architecture

```
objects  (40) ── Linear(40→64) ── LayerNorm ── ReLU ── Linear(64→64) ── ReLU ──┐
lane      (5) ── Linear(5→32)  ── ReLU ────────────────────────────────────────┤
ego       (2) ── Linear(2→16)  ── ReLU ────────────────────────────────────────┤ concat (120)
scenario  (1) ── Embedding(4,8) ────────────────────────────────────────────────┘
                                        │
                              Linear(120→64) ── ReLU ── Dropout(0.2)
                              Linear(64→32)  ── ReLU
                                    ├── Linear(32→1) ── Tanh()    → steering ∈ [−1, 1]
                                    └── Linear(32→1) ── Sigmoid() → throttle ∈ [ 0, 1]
```

Total trainable parameters: ~10,000. Trains in minutes on the Jetson.

---

## File Map

```
v11/
├── planner_model.py          ← shared definitions — model, feature builders, CSV schema
│                               import from this in everything else
│
├── collect_data_planner.py   ← Step 1: drive manually and log structured features
├── augment.py                ← Step 2: synthetically expand the dataset
├── train_planner.py          ← Step 3: train the planner model
├── evaluate.py               ← Step 4: offline error metrics + plots
├── planner_inference.py      ← Step 5: run the trained model on the vehicle
│
│ ── legacy (camera-based approach, kept for reference) ──────────────────────
├── collect_data.py           ← old: saves raw .jpg + .npy + labels.csv
├── train.py                  ← old: MobileNetV3 + DepthCNN on camera input
├── yolo_depth_avoidance_ml.py← old: camera-based ML inference
├── yolo_depth_avoidance.py   ← old: rule-based avoidance
├── obstacle_avoidance.py     ← old: rule-based mode decision logic
└── yolo_web_viewer.py        ← web viewer (shared, still used)
```

---

## Step-by-Step Guide

### Prerequisites

```bash
pip install ultralytics torch torchvision pandas numpy matplotlib
# lkas and jetracer already installed as editable packages
```

---

### Step 1 — Collect Data

Run **two terminals**. LKAS must start first — it owns all shared memory.

**Terminal 1** — start LKAS first:
```bash
lkas --broadcast
```

Wait until you see `image: ● CONNECTED` in the LKAS status bar, then:

**Terminal 2** — start data collection:
```bash
cd /home/peter/ads-skynet/planner/realsense_cam/v11

# Normal track driving
python collect_data_planner.py --scenario 0

# With obstacles on the track
python collect_data_planner.py --scenario 1

# Parking manoeuvre
python collect_data_planner.py --scenario 2
```

Open the web viewer in a browser: `http://<jetson-ip>:8082`

**Controls in the browser:**
- `←` / `→` — steer left / right (hold the key)
- `↓` — stop (throttle = 0)
- `Space` — toggle recording ON/OFF (red badge = recording)
- `Ctrl+C` in Terminal 1 — quit and save

**Tips:**
- Collect at least ~300 rows per scenario before augmenting
- Cover edge cases: sharp corners, obstacle on left side, obstacle on right side, clear straight
- Check the live counter in the terminal to confirm rows are being saved
- LKAS not running is OK — `lane_detected=0` rows are still valid training data, the model learns to handle it

**Output:** `data/planner_data.csv` — one row per saved frame, appended across sessions.

**What each row contains:**
```
frame_id | obj0_valid … obj4_lane_overlap | lane_detected … lane_right_x_norm |
ego_steering | ego_throttle | scenario | target_steering | target_throttle
```

---

### Step 2 — Augment

Expands the dataset ~8× using physically meaningful transforms:

```bash
python augment.py
# or specify paths explicitly:
python augment.py --input data/planner_data.csv --output data/augmented_data.csv
```

**What augmentation does:**

| Transform | Physical meaning |
|---|---|
| Identity | Keep original |
| Mirror | Horizontal flip — negate lateral offsets and steering |
| Distance noise | Simulate RealSense depth noise (σ = 3 cm normalised) |
| Lateral jitter | Simulate YOLO box jitter and LKAS lane wobble |
| Confidence noise | Simulate varying detection confidence |
| Object dropout | Simulate a missed detection (one object randomly removed) |
| Distance scale | Simulate depth calibration drift (±15%) |
| Mirror + noise | Combination of mirror and distance noise |

**Output:** `data/augmented_data.csv`

```
Before: 300 rows  →  After: ~2400 rows  (×8)
```

---

### Step 3 — Train

```bash
python train_planner.py

# Optional flags:
python train_planner.py \
    --csv    data/augmented_data.csv \
    --epochs 100 \
    --lr     3e-4 \
    --batch-size 64 \
    --output planner_model.pth
```

**Training uses augmented_data.csv by default, falls back to planner_data.csv if augmentation was skipped.**

During training you will see:
```
Epoch   Train Loss    Val Loss   Steer MAE   Thtl MAE  LR
    1   0.123456    0.134567    0.2341      0.0412   3.00e-04
    2   0.098765    0.112345    0.1987      0.0381   3.00e-04  ★ (best saved)
  ...
```

`★` marks epochs where the model improved on validation — the best checkpoint is saved automatically.

**Output:** `planner_model.pth`

Training typically converges in 30–80 epochs on ~2000 rows. On the Jetson Orin Nano this takes 2–5 minutes.

---

### Step 4 — Evaluate (offline)

Before putting the model on the vehicle, check its offline accuracy:

```bash
python evaluate.py

# Optional flags:
python evaluate.py \
    --csv     data/planner_data.csv \
    --model   planner_model.pth \
    --out-dir data/eval
```

**Output — printed to terminal:**
```
OVERALL RESULTS
  Samples          : 300
  Steering MAE     : 0.0821
  Steering RMSE    : 0.1134
  Throttle MAE     : 0.0043
  Throttle RMSE    : 0.0061

PER-SCENARIO RESULTS
  Scenario               N   Steer MAE   Steer RMSE   Thtl MAE
  LANE_FOLLOW          120      0.0412       0.0634     0.0021
  OBSTACLE_AVOID       130      0.1204       0.1543     0.0061
  PARKING               50      0.0934       0.1123     0.0078
```

**Output — plots saved to `data/eval/`:**
- `steering_scatter.png` — predicted vs ground truth scatter
- `throttle_scatter.png` — same for throttle
- `steering_error_hist.png` — error distribution histogram
- `per_scenario_mae.png` — bar chart comparing scenarios
- `timeseries.png` — prediction tracking over 200 frames

**Reading the results:**
- Steering MAE < 0.10 is good for discrete {−0.9, 0, +0.9} targets
- If one scenario has much higher error than others → collect more data for that scenario
- A biased error histogram (not centred at 0) → the model is systematically off in one direction

---

### Step 5 — Inference on Vehicle

**Terminal 1:**
```bash
lkas --broadcast
```

**Terminal 2:**
```bash
# Simulation first (no motor output):
python planner_inference.py --scenario 0

# Enable motors once you've verified the steering looks correct in the web viewer:
python planner_inference.py --scenario 0 --motor

# With obstacles on the track:
python planner_inference.py --scenario 1 --motor

# Use a different model file:
python planner_inference.py --model planner_model.pth --scenario 0 --motor
```

**Web viewer:** `http://<jetson-ip>:8082`

The annotation overlay shows:
- Scenario name (colour-coded)
- Current predicted steering and throttle
- YOLO bounding boxes with distances
- Lane boundary lines

**Terminal output (updated every second):**
```
[LANE_FOLLOW]  steer=+0.023  thr=0.200  objs=2  lane=YES  FPS=18.3
```

---

## System Diagram

```
                    DATA COLLECTION
┌─────────────────────────────────────────────────────┐
│  Terminal 1: lkas --broadcast                        │
│    RealSense ──► image SHM ──► lane segmentation    │
│                              ──► detection SHM      │
└─────────────────────────────────────────────────────┘
                    │ (detection SHM)
┌─────────────────────────────────────────────────────┐
│  Terminal 2: collect_data_planner.py                 │
│    RealSense ──► YOLO ──► object features           │
│    detection SHM ──────► lane features              │
│    web viewer  ────────► human steering/throttle    │
│    all ────────────────► planner_data.csv           │
└─────────────────────────────────────────────────────┘

                    OFFLINE PIPELINE
  planner_data.csv
       │
       ▼
  augment.py ──► augmented_data.csv (×8)
       │
       ▼
  train_planner.py ──► planner_model.pth
       │
       ▼
  evaluate.py ──► data/eval/*.png + summary.txt

                    INFERENCE
┌─────────────────────────────────────────────────────┐
│  Terminal 1: lkas --broadcast                        │
└─────────────────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────────────────┐
│  Terminal 2: planner_inference.py                    │
│    image SHM ──► YOLO ──► object features          │
│    detection SHM ──────► lane features              │
│    ego state ──────────► ego features               │
│    --scenario flag ────► scenario token             │
│    all ────────────────► PlannerModel               │
│                              │                      │
│                    [steering, throttle]              │
│                         │          │                │
│                    JetRacer    control SHM          │
└─────────────────────────────────────────────────────┘
```

---

## Iterating — What to Do When Performance Is Poor

**High steering error on a specific scenario:**
1. `python evaluate.py` — confirm which scenario is worst in `per_scenario_mae.png`
2. Collect more data for that scenario: `python collect_data_planner.py --scenario <N>`
3. Re-run augment + train + evaluate

**Model steers in the wrong direction consistently:**
- Check the mirror augmentation is working: mirrored rows should have negated steering
- Verify the JetRacer hardware inversion (`car.steering = -final_steering`) is correct for your vehicle

**Throttle always too high or too low:**
- Check `MAX_THROTTLE` in `planner_model.py` matches the value used during collection
- Default is `0.30` — if `BASE_THROTTLE` in `collect_data_planner.py` was changed, update `MAX_THROTTLE` to match

**No lane detection (lane_detected always 0):**
- LKAS is not running or not detecting lanes
- The model still works but uses fallback lane positions — consider collecting dedicated data with LKAS running so the model learns both conditions

**FPS too low during inference:**
- Reduce `YOLO_SKIP` from 2 to 3 or 4 in `planner_inference.py`
- The planner forward pass itself is < 1 ms — YOLO is the bottleneck

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Constants Reference

All shared constants live in `planner_model.py`. Change them there and they propagate everywhere.

| Constant | Default | Meaning |
|---|---|---|
| `N_MAX_OBJECTS` | 5 | Max YOLO detections tracked per frame |
| `MAX_DIST_M` | 5.0 | Distance normalisation ceiling (metres) |
| `MAX_THROTTLE` | 0.30 | Physical throttle ceiling for JetRacer |
| `FRAME_W` | 768 | Camera resolution width |
| `FRAME_H` | 384 | Camera resolution height |
| `N_YOLO_CLASSES` | 80 | YOLO class count (COCO default) |
| `N_SCENARIOS` | 4 | Scenario token vocabulary size |
