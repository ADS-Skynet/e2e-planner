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

Lane      : LANE_FEATURES cols
  [lane_detected, center_offset, width_norm, left_x_norm, right_x_norm]

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
LANE_FEATURES = 5     # lane state features
EGO_FEATURES  = 2     # ego state features (prev_steering, prev_throttle)
N_SCENARIOS   = 4     # scenario vocabulary size
SCENARIO_DIM  = 8     # embedding dimension for scenario token

OBJECT_BLOCK_DIM = N_MAX_OBJECTS * OBJ_FEATURES   # 40
TOTAL_FLAT_DIM   = OBJECT_BLOCK_DIM + LANE_FEATURES + EGO_FEATURES  # 47

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


def build_lane_features(
    left_lane_x:  float,
    right_lane_x: float,
    lane_detected: bool,
    frame_w: int = FRAME_W,
) -> list:
    """
    Returns a flat list of LANE_FEATURES floats.

    Layout:
      [lane_detected, center_offset, width_norm, left_x_norm, right_x_norm]

    center_offset: (frame_centre − lane_centre) / lane_width
      positive  → vehicle is left of lane centre (needs right steering)
      negative  → vehicle is right of lane centre (needs left steering)
    """
    frame_cx    = frame_w / 2.0
    lane_width  = max(right_lane_x - left_lane_x, 1.0)
    lane_center = (left_lane_x + right_lane_x) / 2.0

    center_offset = (frame_cx - lane_center) / lane_width
    width_norm    = lane_width / frame_w
    left_x_norm   = left_lane_x  / frame_w
    right_x_norm  = right_lane_x / frame_w

    return [
        1.0 if lane_detected else 0.0,
        float(center_offset),
        float(width_norm),
        float(left_x_norm),
        float(right_x_norm),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PlannerModel(nn.Module):
    """
    Structured-input planner.

    forward(objects, lane, ego, scenario) → (B, 2)  [steering, throttle]

    Inputs (all float32 except scenario which is long):
      objects  : (B, N_MAX_OBJECTS * OBJ_FEATURES)   40-dim
      lane     : (B, LANE_FEATURES)                   5-dim
      ego      : (B, EGO_FEATURES)                    2-dim
      scenario : (B,)                                 long
    """

    def __init__(self):
        super().__init__()

        # Object encoder: sees all objects jointly so it can reason about relations
        self.obj_enc = nn.Sequential(
            nn.Linear(OBJECT_BLOCK_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # Lane encoder
        self.lane_enc = nn.Sequential(
            nn.Linear(LANE_FEATURES, 32),
            nn.ReLU(inplace=True),
        )

        # Ego encoder
        self.ego_enc = nn.Sequential(
            nn.Linear(EGO_FEATURES, 16),
            nn.ReLU(inplace=True),
        )

        # Scenario embedding
        self.scenario_embed = nn.Embedding(N_SCENARIOS, SCENARIO_DIM)

        # Fusion + output
        fused_dim = 64 + 32 + 16 + SCENARIO_DIM  # 120
        self.shared_trunk = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        # Separate heads: different activation + scale per output
        self.steering_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh(),                # steering  ∈ [-1, 1]
        )
        self.throttle_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),             # throttle  ∈ [0, 1]  (multiply by MAX_THROTTLE)
        )

    def forward(
        self,
        objects:  torch.Tensor,   # (B, 40)
        lane:     torch.Tensor,   # (B, 5)
        ego:      torch.Tensor,   # (B, 2)
        scenario: torch.Tensor,   # (B,) long
    ) -> torch.Tensor:            # (B, 2)  [steering, throttle_norm]

        o = self.obj_enc(objects)
        l = self.lane_enc(lane)
        e = self.ego_enc(ego)
        s = self.scenario_embed(scenario)

        fused  = torch.cat([o, l, e, s], dim=1)    # (B, 120)
        trunk  = self.shared_trunk(fused)            # (B, 32)

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
    cols += [
        "lane_detected",
        "lane_center_offset",
        "lane_width_norm",
        "lane_left_x_norm",
        "lane_right_x_norm",
    ]
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
    lane_vals = [float(row[k]) for k in (
        "lane_detected", "lane_center_offset",
        "lane_width_norm", "lane_left_x_norm", "lane_right_x_norm")]
    ego_vals  = [float(row["ego_steering"]), float(row["ego_throttle"])]
    scenario  = int(row["scenario"])

    kw = {"device": device} if device else {}

    objects_t  = torch.tensor(obj_vals,  dtype=torch.float32, **kw).unsqueeze(0)
    lane_t     = torch.tensor(lane_vals, dtype=torch.float32, **kw).unsqueeze(0)
    ego_t      = torch.tensor(ego_vals,  dtype=torch.float32, **kw).unsqueeze(0)
    scenario_t = torch.tensor([scenario], dtype=torch.long,   **kw)

    return objects_t, lane_t, ego_t, scenario_t
