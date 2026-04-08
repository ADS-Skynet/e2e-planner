# Planner Architecture — Lane Feature Roadmap

## Current state (v11)

The planner is a structured-feature MLP. It never sees pixels:

```
YOLO detections  →  object features  (40-dim) ─┐
LKAS BiSeNet     →  lane features     (5-dim)  ─┤→ PlannerMLP → [steer, thr]
Ego state        →  ego features      (2-dim)  ─┘
Scenario token   →  embedding         (8-dim)
```

The 5 lane features are scalar summaries derived from the BiSeNet mask inside LKAS:

```
[lane_detected, center_offset, width_norm, left_x_norm, right_x_norm]
```

### Why LKAS was used

The planner used LKAS purely for its BiSeNet lane segmentation result —
not for its decision/control logic. The image was sent to LKAS via SHM,
BiSeNet ran inside LKAS, and the planner read back the derived lane
geometry via `LKASClient.get_detection()`.

---

## Problem with 5 scalar lane features

| Scenario | Works? | Why not |
|----------|--------|---------|
| Lane follow (straight) | Barely | No curvature info; center_offset had near-zero variance during training |
| Lane follow (curve) | No | left_x/right_x at one depth is blind to curvature ahead |
| Intersection / turn | No | Cannot see road widening or turn lane markings ahead |
| Pull-over | Partial | Works if lane boundary is clearly tracked |
| Roundabout | No | Continuously changing radius is invisible to 1D features |
| Parking | No | Wrong abstraction entirely — parking needs 2D spatial reasoning |

---

## Step 1 — Spatial grid pooling  ← current target

Instead of deriving left/right x from the mask, feed a **coarse grid of
lane-pixel fractions** directly from the BiSeNet output mask.

```
Segmentation mask (480×848, uint8)
         ↓  resize to 64×112
  divide into R rows × C columns (e.g. 4×8 = 32 cells)
         ↓
  per-cell lane fraction  [0.0 – 1.0]
         →  32 lane features  (replaces the current 5)
```

Visual intuition:

```
Far   [0.0][0.0][0.3][0.8][0.8][0.3][0.0][0.0]   ← road curves right ahead
      [0.0][0.1][0.5][0.9][0.9][0.5][0.1][0.0]
      [0.0][0.2][0.7][1.0][1.0][0.7][0.2][0.0]
Near  [0.0][0.3][0.8][1.0][1.0][0.8][0.3][0.0]   ← nearly centred now
```

The model can now see curvature, lane width variation, and road geometry
at multiple depths — all without a CNN, without training a feature extractor.

### What changes

| File | Change |
|------|--------|
| `lane_seg.py` (new) | Thin wrapper: loads BiSeNet directly, runs inference, returns mask |
| `planner_model.py` | `LANE_FEATURES = 5` → `32`; `build_lane_features()` → `build_lane_grid()` |
| `collect_data_planner.py` | Drop LKAS dependency; call `lane_seg.py` directly |
| `planner_inference.py` | Same |
| `augment.py` | Grid features augment differently (mirror flips columns, jitter shifts grid) |
| `train_planner.py` | No change — reads CSV; lane columns just wider |

### Removing LKAS dependency

BiSeNet lives at:
- **Model code**: `ads-skynet/lane-detection-dl/model/bisenetv2.py`
- **Weights**: `ads-skynet/lane-detection-dl/inference/bisenet-0204.pth`

Both will be referenced (or copied) directly into v11 so the planner
runs standalone with no LKAS process required.

LKAS is still useful as a standalone lane-keeping controller for other
modules, but the planner no longer needs it as an intermediary.

### Effort

Low. No new model training. The changes are:
1. ~80 lines: `lane_seg.py` wrapper around the existing BiSeNet code
2. ~20 lines: replace `build_lane_features` with `build_lane_grid`
3. Update imports in collect + inference (remove `LKASClient`, add `LaneSeg`)
4. Re-collect data and retrain (necessary anyway after the data quality fix)

---

## Step 2 — CNN middleware  ← future improvement

For scenarios that need richer spatial reasoning (complex intersections,
roundabout entry/exit), a small learned CNN on the segmentation mask can
replace the grid-pooling step.

```
Segmentation mask (480×848, uint8)
         ↓  resize to 60×106
  Conv2d(1→8,  k=3, s=2)  →  30×53×8
  ReLU + MaxPool(2)        →  15×26×8
  Conv2d(8→16, k=3, s=2)  →   7×12×16
  ReLU + AdaptiveAvgPool  →  1× 1×16  (global avg)
  Flatten + Linear(16→32) →  32-dim lane embedding
```

This replaces the grid with a learned 32-dim embedding that captures
richer structure (curves, junctions, markings). The rest of the planner
is unchanged.

### Why it is more effort than grid pooling

| Concern | Grid pooling | CNN |
|---------|-------------|-----|
| Feature extractor training | None (fixed math) | CNN must be trained jointly with planner |
| Data pipeline | Features in CSV (same as now) | Must save segmentation masks during collection (PNG files alongside CSV) |
| Dataset size | Same as now | Needs more samples — CNN has ~5K extra params to learn |
| Debugging | Cells are interpretable | Embeddings are opaque |
| Inference cost | ~0.1 ms (numpy resize + sum) | ~2 ms extra (GPU) or ~20 ms (CPU) |

### When to move to CNN

- Grid pooling works for lane follow, pull-over, and most intersection scenarios
- Move to CNN if the model fails on roundabout or complex junctions where the
  coarse grid loses too much spatial detail
- At that point the data pipeline change (saving masks) is the main engineering cost

### What parking would need (out of scope)

Parking requires understanding 2D free-space geometry — the exact
position, angle, and size of an available slot. This is fundamentally
a 2D reasoning task that no scalar or 1D feature set handles well.
It is one of the core reasons the industry shifted toward end-to-end
(BEV perception → occupancy grid → planner). Out of scope for this
structured-feature planner.

---

## Target scenario coverage

| Scenario | Grid pooling | CNN |
|----------|-------------|-----|
| Lane follow | ✓ | ✓ |
| Intersection + turn L/R/straight | ✓ (with good data) | ✓ |
| Pull-over | ✓ | ✓ |
| Roundabout | Likely ✓ | ✓ |
| Parking | ✗ | ✗ (needs e2e) |
