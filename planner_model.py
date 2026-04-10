#!/usr/bin/env python3
"""
Planner Model — Structured Feature Planner
===========================================
Takes YOLO object detections + lane segmentation data + ego state
and outputs (steering, throttle) — no camera feed.

This is the Tesla-style perception→planner split:
  [YOLO + Lane Seg]  →  structured features  →  PlannerModel  →  [steering, throttle]

Feature layout
--------------
Objects   : N_MAX_OBJECTS rows × OBJ_FEATURES cols  (sorted by distance, padded with zeros)
  [valid, class_norm, confidence, dist_norm, lat_offset, width_norm, height_norm, lane_overlap]

Lane      : LANE_FEATURES cols  (6×12 spatial grid from BiSeNet mask)
  [lane_r0c0 … lane_r3c7]  per-cell lane fraction [0.0–1.0], row-major

Ego state : EGO_FEATURES cols
  [prev_steering, prev_throttle]

Scenario  : int  (task context token, embedded)
  0 = LANE_FOLLOW  1 = OBSTACLE_AVOID  2 = PARKING  3 = STOP

Output    : [steering, throttle]  both are raw floats (no activation clamp here)
  steering  ∈ [-1, 1]   (matches JetRacer convention)
  throttle  ∈ [ 0, 1]   (scaled by MAX_THROTTLE in the driver)
"""

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Feature dimensions  (must stay in sync across collect / augment / train / infer)
# ─────────────────────────────────────────────────────────────────────────────
N_MAX_OBJECTS = 5      # objects tracked per frame (padded to this length)
OBJ_FEATURES  = 8     # features per object slot
GRID_ROWS     = 6     # spatial grid rows (far → near)
GRID_COLS     = 12    # spatial grid columns (left → right)
LANE_FEATURES = GRID_ROWS * GRID_COLS   # 72 — spatial grid of lane fractions
EGO_FEATURES  = 2     # ego state features (prev_steering, prev_throttle)
N_SCENARIOS   = 4     # scenario vocabulary size
SCENARIO_DIM  = 8     # embedding dimension for scenario token

OBJECT_BLOCK_DIM = N_MAX_OBJECTS * OBJ_FEATURES   # 40
TOTAL_FLAT_DIM   = OBJECT_BLOCK_DIM + LANE_FEATURES + EGO_FEATURES  # 114

# Fallback lane boundary x-coordinates when the mask contains no lane pixels
FIXED_LEFT_LANE_X  = 255
FIXED_RIGHT_LANE_X = 485

# Scenario tokens
SCENARIO_LANE_FOLLOW    = 0
SCENARIO_OBSTACLE_AVOID = 1
SCENARIO_PARKING        = 2
SCENARIO_STOP           = 3

# Normalisation constants  (shared between collection and inference)
MAX_DIST_M     = 5.0    # clip distances beyond this to 1.0
MAX_THROTTLE   = 0.30   # physical max throttle used during collection
FRAME_W        = 848    # must match camera config — RealSense supported: 848x480, 640x480, 640x360
FRAME_H        = 480
N_YOLO_CLASSES = 80     # COCO classes; override if using custom model


# ─────────────────────────────────────────────────────────────────────────────
# Object feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_object_features(
    boxes:      list,      # list of [x1, y1, x2, y2] in pixels
    distances:  list,      # list of float metres (-1 = invalid)
    class_ids:  list,      # list of int
    confs:      list,      # list of float [0,1]
    left_lane_x:  float,
    right_lane_x: float,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
    n_classes: int = N_YOLO_CLASSES,
) -> list:
    """
    Convert raw YOLO output to a fixed-size normalised feature block.

    Returns a flat list of N_MAX_OBJECTS * OBJ_FEATURES floats.
    Objects are sorted closest-first; excess slots are zero-padded.

    Per-slot layout:
      [valid, class_norm, conf, dist_norm, lat_offset, w_norm, h_norm, lane_overlap]
    """
    lane_width  = max(right_lane_x - left_lane_x, 1.0)
    lane_center = (left_lane_x + right_lane_x) / 2.0

    # Build raw records and sort by distance (valid distances first, then invalids)
    records = []
    for box, dist, cid, conf in zip(boxes, distances, class_ids, confs):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        cx = (x1 + x2) / 2.0

        # Lateral offset: signed, normalised by lane width  (−ve = left of centre)
        lat_offset = (cx - lane_center) / lane_width

        # Lane overlap fraction [0, 1]
        overlap = max(0.0, min(x2, right_lane_x) - max(x1, left_lane_x))
        lane_overlap = float(overlap / lane_width)

        dist_norm  = float(min(max(dist, 0.0), MAX_DIST_M) / MAX_DIST_M) if dist > 0 else 1.0
        sort_key   = dist_norm if dist > 0 else 2.0   # invalid last

        records.append((sort_key, [
            1.0,                                      # valid
            float(cid) / max(n_classes - 1, 1),       # class_norm
            float(conf),                              # confidence
            dist_norm,                                # dist_norm
            float(lat_offset),                        # lat_offset
            float((x2 - x1) / frame_w),              # width_norm
            float((y2 - y1) / frame_h),              # height_norm
            lane_overlap,                             # lane_overlap
        ]))

    records.sort(key=lambda r: r[0])

    # Pad / truncate to N_MAX_OBJECTS
    out: list[float] = []
    for i in range(N_MAX_OBJECTS):
        if i < len(records):
            out.extend(records[i][1])
        else:
            out.extend([0.0] * OBJ_FEATURES)   # zero-pad
    return out


def build_lane_grid(mask: "np.ndarray") -> list:
    """
    Convert a BiSeNet segmentation mask to a coarse spatial grid of lane fractions.

    Resizes the binary lane mask to (_GRID_H × _GRID_W) using INTER_AREA
    (which averages pixels, giving per-pixel fractions), then returns the
    mean lane fraction for each of the (GRID_ROWS × GRID_COLS) cells.

    Row 0 = far (top of image), Row GRID_ROWS-1 = near (bottom).
    Column 0 = left edge, Column GRID_COLS-1 = right edge.

    Args:
        mask: (H, W) uint8 — 0 = background, 1+ = lane  (from LaneSeg.infer)

    Returns:
        list of LANE_FEATURES (32) floats — lane fraction per cell [0.0–1.0],
        in row-major order (r0c0, r0c1, …, r3c7).
    """
    import cv2
    import numpy as _np

    _GRID_H = GRID_ROWS * 16   # 64
    _GRID_W = GRID_COLS * 14   # 112  (≈ 848×480 aspect at coarse scale)

    binary = (mask > 0).astype(_np.float32)
    small  = cv2.resize(binary, (_GRID_W, _GRID_H), interpolation=cv2.INTER_AREA)

    cell_h = _GRID_H // GRID_ROWS   # 16
    cell_w = _GRID_W // GRID_COLS   # 14

    features = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = small[r * cell_h:(r + 1) * cell_h,
                         c * cell_w:(c + 1) * cell_w]
            features.append(float(cell.mean()))
    return features


def lane_boundaries_from_mask(
    mask: "np.ndarray",
    roi_fraction: float = 0.33,
) -> tuple:
    """
    Derive approximate left/right lane boundary x-coordinates from a
    segmentation mask.  Uses the bottom roi_fraction of the image
    (near-field), finds leftmost and rightmost lane columns.

    Returns:
        (left_x, right_x) in pixels — or (FIXED_LEFT_LANE_X, FIXED_RIGHT_LANE_X)
        if no lane pixels are found in the ROI.
    """
    import numpy as _np
    h    = mask.shape[0]
    roi  = mask[int(h * (1.0 - roi_fraction)):, :]
    cols = _np.where(roi.any(axis=0))[0]
    if len(cols) < 2:
        return float(FIXED_LEFT_LANE_X), float(FIXED_RIGHT_LANE_X)
    return float(cols[0]), float(cols[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Grid visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_lane_grid_overlay(
    frame: "np.ndarray",
    grid_features: list,
    alpha: float = 0.45,
) -> "np.ndarray":
    """
    Draw the 4×8 lane-fraction grid as a semi-transparent overlay on *frame*.

    Each cell is shaded green with intensity proportional to its lane fraction.
    Grid lines are drawn in translucent white.  The fraction value is printed
    inside every cell so you can read the exact model input while driving.

    Call this *after* draw_segmentation() so the grid appears on top.

    Args:
        frame:         BGR image, already annotated with lane segmentation
        grid_features: 32 floats from build_lane_grid() — row-major [r0c0 … r3c7]
        alpha:         blend weight of the grid layer (default 0.45)

    Returns:
        New BGR image with grid overlay blended in.
    """
    import cv2
    import numpy as _np

    h, w   = frame.shape[:2]
    cell_h = h // GRID_ROWS   # 120 for 480-px frames
    cell_w = w // GRID_COLS   # 106 for 848-px frames

    overlay = frame.copy()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            frac = float(grid_features[r * GRID_COLS + c])
            y1 = r * cell_h
            y2 = y1 + cell_h
            x1 = c * cell_w
            x2 = x1 + cell_w

            # Cell fill — bright green, intensity ∝ lane fraction
            if frac > 0.02:
                intensity = int(60 + 195 * frac)   # 60 (dim) … 255 (full)
                cv2.rectangle(overlay, (x1, y1), (x2, y2),
                              (0, intensity, 0), -1)  # BGR green

            # Fraction label in cell centre
            label = f"{frac:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            tx = x1 + (cell_w - tw) // 2
            ty = y1 + (cell_h + th) // 2
            cv2.putText(overlay, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # Blend cell fills with the incoming frame
    blended = cv2.addWeighted(frame, 1.0 - alpha, overlay, alpha, 0)

    # Draw grid lines on top (always fully opaque)
    for r in range(GRID_ROWS + 1):
        y = r * cell_h
        cv2.line(blended, (0, y), (w, y), (180, 180, 180), 1, cv2.LINE_AA)
    for c in range(GRID_COLS + 1):
        x = c * cell_w
        cv2.line(blended, (x, 0), (x, h), (180, 180, 180), 1, cv2.LINE_AA)

    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PlannerModel(nn.Module):
    """
    Structured-input planner.

    forward(objects, lane, ego, scenario) → (B, 2)  [steering, throttle]

    Inputs (all float32 except scenario which is long):
      objects  : (B, N_MAX_OBJECTS * OBJ_FEATURES)   40-dim
      lane     : (B, LANE_FEATURES)                   72-dim  (6×12 spatial grid)
      ego      : (B, EGO_FEATURES)                    2-dim
      scenario : (B,)                                 long
    """

    def __init__(self):
        super().__init__()

        # Object encoder
        self.obj_enc = nn.Sequential(
            nn.Linear(OBJECT_BLOCK_DIM, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Lane encoder — wider + deeper to exploit the 6×12 spatial grid
        self.lane_enc = nn.Sequential(
            nn.Linear(LANE_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Ego encoder
        self.ego_enc = nn.Sequential(
            nn.Linear(EGO_FEATURES, 32),
            nn.ReLU(inplace=True),
        )

        # Scenario embedding
        self.scenario_embed = nn.Embedding(N_SCENARIOS, SCENARIO_DIM)

        # Fusion + output
        fused_dim = 64 + 64 + 32 + SCENARIO_DIM  # 168
        self.shared_trunk = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Separate heads
        self.steering_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh(),                # steering  ∈ [-1, 1]
        )
        self.throttle_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),             # throttle  ∈ [0, 1]  (multiply by MAX_THROTTLE)
        )

    def forward(
        self,
        objects:  torch.Tensor,   # (B, 40)
        lane:     torch.Tensor,   # (B, LANE_FEATURES) = (B, 32)
        ego:      torch.Tensor,   # (B, 2)
        scenario: torch.Tensor,   # (B,) long
    ) -> torch.Tensor:            # (B, 2)  [steering, throttle_norm]

        o = self.obj_enc(objects)
        l = self.lane_enc(lane)
        e = self.ego_enc(ego)
        s = self.scenario_embed(scenario)

        fused  = torch.cat([o, l, e, s], dim=1)    # (B, 168)
        trunk  = self.shared_trunk(fused)            # (B, 64)

        steering = self.steering_head(trunk)         # (B, 1) ∈ [-1, 1]
        throttle = self.throttle_head(trunk)         # (B, 1) ∈ [ 0, 1]

        return torch.cat([steering, throttle], dim=1)  # (B, 2)


# ─────────────────────────────────────────────────────────────────────────────
# CSV column schema  (generated once, shared by all scripts)
# ─────────────────────────────────────────────────────────────────────────────

def csv_columns() -> list[str]:
    """Return ordered column names for the structured dataset CSV."""
    cols = ["frame_id"]
    for i in range(N_MAX_OBJECTS):
        cols += [
            f"obj{i}_valid",
            f"obj{i}_class_norm",
            f"obj{i}_conf",
            f"obj{i}_dist_norm",
            f"obj{i}_lat_offset",
            f"obj{i}_width_norm",
            f"obj{i}_height_norm",
            f"obj{i}_lane_overlap",
        ]
    cols += [f"lane_r{r}c{c}"
             for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    cols += ["ego_steering", "ego_throttle"]
    cols += ["scenario", "target_steering", "target_throttle"]
    return cols


def row_to_tensors(row, device=None):
    """
    Convert a single pandas Series / dict row from the structured CSV into
    model-ready tensors.

    Returns (objects, lane, ego, scenario) ready for PlannerModel.forward().
    """
    obj_vals  = [float(row[f"obj{i}_{f}"])
                 for i in range(N_MAX_OBJECTS)
                 for f in ("valid", "class_norm", "conf", "dist_norm",
                           "lat_offset", "width_norm", "height_norm", "lane_overlap")]
    lane_vals = [float(row[f"lane_r{r}c{c}"])
                 for r in range(GRID_ROWS) for c in range(GRID_COLS)]
    ego_vals  = [float(row["ego_steering"]), float(row["ego_throttle"])]
    scenario  = int(row["scenario"])

    kw = {"device": device} if device else {}

    objects_t  = torch.tensor(obj_vals,  dtype=torch.float32, **kw).unsqueeze(0)
    lane_t     = torch.tensor(lane_vals, dtype=torch.float32, **kw).unsqueeze(0)
    ego_t      = torch.tensor(ego_vals,  dtype=torch.float32, **kw).unsqueeze(0)
    scenario_t = torch.tensor([scenario], dtype=torch.long,   **kw)

    return objects_t, lane_t, ego_t, scenario_t
