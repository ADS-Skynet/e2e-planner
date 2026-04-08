#!/usr/bin/env python3
"""
Planner Inference  —  Structured Feature → Actuation
=====================================================
Real-time inference loop using the trained PlannerModel.

The planner receives NO camera pixels. It only sees:
  • YOLO object list    (class, distance, position, size, lane overlap)
  • LKAS lane data      (boundaries, centre offset, width)
  • Ego state           (previous steering / throttle)
  • Scenario token      (set via --scenario flag or web UI)

and outputs [steering, throttle] to the JetRacer and control SHM.

Pipeline per frame
------------------
  image SHM  →  YOLO          → object features  ─┐
  image SHM  →  LKAS client   → lane features    ─┤ → PlannerModel → [steer, thr]
  ego state  (from last cycle)                    ─┘
                                                        ↓
                                              control SHM + JetRacer

Run alongside
-------------
  lkas --broadcast    (creates image + detection SHM)
  vehicle.py          (reads control SHM)  ← OR use --motor for direct drive

Usage
-----
  python planner_inference.py [--web-port 8082] [--motor] [--scenario 0]
                              [--model planner_model.pth]
"""

import sys
import time
import argparse
import importlib.util
import numpy as np
from pathlib import Path

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent.parent / "vehicle" / "src"))

# ── PyTorch legacy weights fix ────────────────────────────────────────────────
_orig_torch_load = torch.load

def _torch_load_legacy(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_legacy

from ultralytics import YOLO as _YOLO

# ── JetRacer ──────────────────────────────────────────────────────────────────
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except ImportError:
    print("[WARN] JetRacer not available — simulation mode")
    JETRACER_AVAILABLE = False

# ── LKAS ──────────────────────────────────────────────────────────────────────
try:
    from lkas.integration.shared_memory import (
        SharedMemoryImageChannel,
        SharedMemoryControlChannel,
    )
    from lkas.integration.shared_memory.messages import ObstacleMessage, ObstacleAction
    from lkas import LKASClient
    LKAS_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] LKAS not available: {e}")
    sys.exit(1)

# ── Web viewer ────────────────────────────────────────────────────────────────
from planner_viewer import PlannerViewer

# ── YOLO config ───────────────────────────────────────────────────────────────
_cfg_path = script_dir.parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("yolo_config", _cfg_path)
_cfg  = importlib.util.module_from_spec(_spec)
_cfg.__file__ = str(_cfg_path)
_spec.loader.exec_module(_cfg)

MODEL_PATH           = _cfg.MODEL_PATH
CONFIDENCE_THRESHOLD = _cfg.CONFIDENCE_THRESHOLD
IOU_THRESHOLD        = _cfg.IOU_THRESHOLD
CLASS_NAMES          = _cfg.CLASS_NAMES
N_YOLO_CLASSES       = len(CLASS_NAMES)

# ── Planner ───────────────────────────────────────────────────────────────────
from planner_model import (
    PlannerModel,
    build_object_features,
    build_lane_features,
    MAX_THROTTLE,
    FRAME_W, FRAME_H,
    SCENARIO_LANE_FOLLOW, SCENARIO_OBSTACLE_AVOID, SCENARIO_PARKING, SCENARIO_STOP,
)

PLANNER_MODEL_PATH = script_dir / "planner_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
YOLO_SKIP = 2              # run YOLO every N frames

FIXED_LEFT_LANE_X  = 255
FIXED_RIGHT_LANE_X = 485

_SCENARIO_NAMES = {
    SCENARIO_LANE_FOLLOW:    "LANE_FOLLOW",
    SCENARIO_OBSTACLE_AVOID: "OBSTACLE_AVOID",
    SCENARIO_PARKING:        "PARKING",
    SCENARIO_STOP:           "STOP",
}

# ─────────────────────────────────────────────────────────────────────────────
# LKAS helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_y_at_distance(depth_array: np.ndarray, depth_scale: float,
                        target_m: float = 0.7) -> int:
    fallback = int(depth_array.shape[0] * 2 / 3)
    if depth_scale <= 0:
        return fallback
    cx   = depth_array.shape[1] // 2
    col  = depth_array[:, cx].astype(np.float32) * depth_scale
    valid = col > 0
    if not valid.any():
        return fallback
    diffs = np.abs(col - target_m)
    diffs[~valid] = np.inf
    return int(np.argmin(diffs))


def _interpolate_lane_x(lane, y: float) -> float:
    if hasattr(lane, 'x1'):
        x1, y1, x2, y2 = lane.x1, lane.y1, lane.x2, lane.y2
    else:
        x1, y1, x2, y2 = lane
    if y2 == y1:
        return float(x1)
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    boxes, distances, class_ids, confs,
    left_lane_x, right_lane_x, lane_detected,
    prev_steering, prev_throttle,
    device,
):
    """
    Build model-ready tensors from raw YOLO + lane data.
    Returns (objects, lane, ego) tensors, all batched with B=1.
    """
    obj_feats  = build_object_features(
        boxes=boxes, distances=distances,
        class_ids=class_ids, confs=confs,
        left_lane_x=left_lane_x, right_lane_x=right_lane_x,
        frame_w=FRAME_W, frame_h=FRAME_H, n_classes=N_YOLO_CLASSES,
    )
    lane_feats = build_lane_features(
        left_lane_x=left_lane_x, right_lane_x=right_lane_x,
        lane_detected=lane_detected, frame_w=FRAME_W,
    )
    ego_feats = [prev_steering, prev_throttle / MAX_THROTTLE]

    objects_t = torch.tensor(obj_feats,  dtype=torch.float32, device=device).unsqueeze(0)
    lane_t    = torch.tensor(lane_feats, dtype=torch.float32, device=device).unsqueeze(0)
    ego_t     = torch.tensor(ego_feats,  dtype=torch.float32, device=device).unsqueeze(0)

    return objects_t, lane_t, ego_t


# ─────────────────────────────────────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────────────────────────────────────
import cv2

_MODE_COLORS = {
    SCENARIO_LANE_FOLLOW:    (0, 255, 0),
    SCENARIO_OBSTACLE_AVOID: (0, 255, 255),
    SCENARIO_PARKING:        (255, 165, 0),
    SCENARIO_STOP:           (0, 0, 255),
}
_BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 255, 0), (0, 128, 255),
    (128, 128, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255), (0, 128, 0),
]


def _draw(frame, boxes, distances, class_ids, scenario, steering, throttle, fps,
          left_x, right_x):
    out = frame.copy()
    h   = out.shape[0]
    cv2.line(out, (int(left_x), 0), (int(left_x), h), (255, 200, 0), 1)
    cv2.line(out, (int(right_x), 0), (int(right_x), h), (255, 200, 0), 1)

    for box, dist, cid in zip(boxes, distances, class_ids):
        x1, y1, x2, y2 = map(int, box)
        c = _BOX_COLORS[cid % len(_BOX_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        lbl  = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        dtxt = f"{dist:.2f}m" if dist > 0 else "N/A"
        cv2.putText(out, f"{lbl} {dtxt}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    sc_name  = _SCENARIO_NAMES.get(scenario, str(scenario))
    sc_color = _MODE_COLORS.get(scenario, (255, 255, 255))
    cv2.putText(out, f"PLANNER [{sc_name}]", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"steer={steering:+.3f}  thr={throttle:.3f}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS={fps:.1f}", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    web_port:     int  = 8082,
    enable_motor: bool = False,
    scenario:     int  = SCENARIO_LANE_FOLLOW,
    model_path:   Path = PLANNER_MODEL_PATH,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sc_name = _SCENARIO_NAMES.get(scenario, str(scenario))

    print("=" * 62)
    print("  Planner Inference  (structured features → actuation)")
    print("=" * 62)
    print(f"  Device    : {device}")
    print(f"  Scenario  : {sc_name} ({scenario})")
    print(f"  Model     : {model_path}")
    print(f"  Motor     : {'ACTIVE' if enable_motor else 'SIMULATION'}")
    print(f"  Web       : {'port ' + str(web_port) if web_port > 0 else 'DISABLED'}")
    print()

    # ── Planner model ─────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"[ERROR] Planner model not found: {model_path}")
        print("        Run train_planner.py first.")
        sys.exit(1)
    planner = PlannerModel().to(device)
    state   = torch.load(str(model_path), map_location=device)
    planner.load_state_dict(state)
    planner.eval()
    n_params = sum(p.numel() for p in planner.parameters())
    print(f"[PLANNER] Loaded  ({n_params:,} params)")

    # ── YOLO model ────────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] YOLO model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"[YOLO] Loading: {MODEL_PATH}")
    yolo        = _YOLO(MODEL_PATH)
    yolo_device = 0 if device.type == 'cuda' else 'cpu'

    # ── Shared memory ─────────────────────────────────────────────────────────
    print("[SHM] Connecting to image SHM...")
    image_channel   = SharedMemoryImageChannel(name="image",   create=False,
                                               retry_count=60, retry_delay=1.0)
    control_channel = SharedMemoryControlChannel(name="control", create=False,
                                                 retry_count=30, retry_delay=1.0)

    # ── LKAS client ──────────────────────────────────────────────────────────
    lkas_client = None
    try:
        lkas_client = LKASClient(image_shm_name="image",
                                 detection_shm_name="detection",
                                 control_shm_name="control")
        print("[LKAS] Lane detection client connected")
    except Exception as e:
        print(f"[WARN] LKAS unavailable ({e}) — fixed lane fallback active")

    # ── JetRacer ─────────────────────────────────────────────────────────────
    car = None
    if enable_motor and JETRACER_AVAILABLE:
        car = NvidiaRacecar()
        car.steering_offset = 0.05
        car.steering = 0.0
        car.throttle = 0.0
        print("[CAR] NvidiaRacecar ready — MOTORS ACTIVE")
    else:
        print("[CAR] Simulation mode (motor control disabled)")

    # ── Web viewer ────────────────────────────────────────────────────────────
    web_viewer = None
    if web_port > 0:
        web_viewer = PlannerViewer(http_port=web_port, ws_port=web_port + 1)
        web_viewer.start()
        print(f"[WEB] Viewer at http://0.0.0.0:{web_port}")

    # ── State ─────────────────────────────────────────────────────────────────
    prev_steering    = 0.0
    prev_throttle    = 0.0
    yolo_frame_count = 0
    cached_boxes:     list = []
    cached_distances: list = []
    cached_class_ids: list = []
    cached_confs:     list = []

    fps       = 0.0
    fps_count = 0
    fps_start = time.time()

    scenario_t = torch.tensor([scenario], dtype=torch.long, device=device)

    # First frame for dimensions + depth_scale
    first_msg = image_channel.read_blocking(timeout=10.0)
    if first_msg is None:
        print("[ERROR] No frame from image SHM within 10 s")
        sys.exit(1)
    frame_h, frame_w = first_msg.image.shape[:2]
    depth_scale = first_msg.depth_scale if first_msg.depth_scale > 0 else 0.001
    print(f"[SHM] Frame: {frame_w}×{frame_h}  depth_scale={depth_scale}")
    print(f"\n[RUN] Running — Ctrl+C to stop\n")

    try:
        while True:
            msg = image_channel.read()
            if msg is None:
                time.sleep(0.001)
                continue

            color_bgr   = msg.image
            depth_array = msg.depth_image if msg.depth_image is not None else \
                          np.zeros((frame_h, frame_w), dtype=np.uint16)
            if msg.depth_scale > 0:
                depth_scale = msg.depth_scale

            # ── YOLO (every N frames) ─────────────────────────────────────────
            yolo_frame_count += 1
            if yolo_frame_count % YOLO_SKIP == 0:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                try:
                    results = yolo(color_bgr, conf=CONFIDENCE_THRESHOLD,
                                   iou=IOU_THRESHOLD, device=yolo_device, verbose=False)
                    boxes_r, dists_r, cids_r, confs_r = [], [], [], []
                    if len(results[0].boxes) > 0:
                        xyxy  = results[0].boxes.xyxy.cpu().numpy()
                        cids  = results[0].boxes.cls.cpu().numpy().astype(int)
                        confs = results[0].boxes.conf.cpu().numpy()
                        for box, cid, conf in zip(xyxy, cids, confs):
                            cx = int(max(0, min((box[0] + box[2]) / 2, frame_w - 1)))
                            cy = int(max(0, min((box[1] + box[3]) / 2, frame_h - 1)))
                            raw  = int(depth_array[cy, cx])
                            dist = raw * depth_scale if raw > 0 else -1.0
                            boxes_r.append(box)
                            dists_r.append(dist)
                            cids_r.append(int(cid))
                            confs_r.append(float(conf))
                    cached_boxes     = boxes_r
                    cached_distances = dists_r
                    cached_class_ids = cids_r
                    cached_confs     = confs_r
                except Exception as e:
                    print(f"[YOLO] Error: {e}")

            boxes     = cached_boxes
            distances = cached_distances
            class_ids = cached_class_ids
            confs     = cached_confs

            # ── Lane boundaries from LKAS ──────────────────────────────────────
            left_x        = float(FIXED_LEFT_LANE_X)
            right_x       = float(FIXED_RIGHT_LANE_X)
            lane_detected = False
            lkas_steering = 0.0
            if lkas_client is not None:
                try:
                    det = lkas_client.get_detection(timeout=0.01)
                    if det is not None and det.left_lane and det.right_lane:
                        y_ref = _find_y_at_distance(depth_array, depth_scale)
                        lx    = _interpolate_lane_x(det.left_lane, y_ref)
                        rx    = _interpolate_lane_x(det.right_lane, y_ref)
                        if 0 < lx < rx <= frame_w:
                            left_x        = lx
                            right_x       = rx
                            lane_detected = True
                    ctrl = lkas_client.get_control(timeout=0.01)
                    if ctrl is not None:
                        lkas_steering = ctrl.steering
                except Exception:
                    pass

            # ── Planner forward pass ───────────────────────────────────────────
            with torch.no_grad():
                objects_t, lane_t, ego_t = extract_features(
                    boxes=boxes, distances=distances,
                    class_ids=class_ids, confs=confs,
                    left_lane_x=left_x, right_lane_x=right_x,
                    lane_detected=lane_detected,
                    prev_steering=prev_steering, prev_throttle=prev_throttle,
                    device=device,
                )
                out = planner(objects_t, lane_t, ego_t, scenario_t)  # (1, 2)

            # Denormalise outputs
            final_steering = float(out[0, 0].item())               # tanh → [-1, 1]
            final_throttle = float(out[0, 1].item()) * MAX_THROTTLE # sigmoid × MAX_THROTTLE

            # Clamp for safety
            final_steering = float(np.clip(final_steering, -1.0,  1.0))
            final_throttle = float(np.clip(final_throttle,  0.0, MAX_THROTTLE))

            # ── Apply to vehicle ───────────────────────────────────────────────
            if car is not None:
                car.steering = -final_steering   # hardware inversion
                car.throttle = -final_throttle   # negative = forward

            # ── Write to control SHM ───────────────────────────────────────────
            nearest = min((d for d in distances if d > 0), default=-1.0)
            action  = (ObstacleAction.AVOID_LEFT  if final_steering < -0.1 else
                       ObstacleAction.AVOID_RIGHT if final_steering >  0.1 else
                       ObstacleAction.NORMAL)
            obstacle_msg = ObstacleMessage(
                active   = len(boxes) > 0,
                action   = action,
                distance = nearest,
                steering = final_steering,
                throttle = final_throttle,
                brake    = 1.0 if scenario == SCENARIO_STOP else 0.0,
                timestamp= time.time(),
                frame_id = msg.frame_id,
            )
            control_channel.write_obstacle(obstacle_msg)

            # Update ego state
            prev_steering = final_steering
            prev_throttle = final_throttle

            # ── FPS ───────────────────────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps       = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()
                sys.stdout.write(
                    f"\r[{sc_name}]  steer={final_steering:+.3f}  "
                    f"thr={final_throttle:.3f}  "
                    f"objs={len(boxes)}  "
                    f"lane={'YES' if lane_detected else 'NO '}  "
                    f"FPS={fps:.1f}   "
                )
                sys.stdout.flush()

            # ── Web viewer ────────────────────────────────────────────────────
            if web_viewer is not None:
                annotated = _draw(color_bgr, boxes, distances, class_ids,
                                  scenario, final_steering, final_throttle, fps,
                                  left_x, right_x)
                web_viewer.broadcast_frame(annotated)
                web_viewer.broadcast_status({
                    'fps':              fps,
                    'action':           sc_name,
                    'steering':         final_steering,
                    'throttle':         final_throttle,
                    'nearest_distance': nearest,
                    'overtaking_state': sc_name,
                    'lane_detected':    lane_detected,
                    'lkas_steering':    lkas_steering,
                    'left_lane_x':      left_x,
                    'right_lane_x':     right_x,
                })

    except KeyboardInterrupt:
        print("\n[RUN] Stopped by user")

    finally:
        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0
        if web_viewer is not None:
            try: web_viewer.stop()
            except Exception: pass
        if lkas_client is not None:
            try: lkas_client.close()
            except Exception: pass
        print("[RUN] Cleanup done")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured planner inference")
    parser.add_argument('--web-port', type=int,  default=8082,
                        help='Web viewer port (default 8082, 0 to disable)')
    parser.add_argument('--motor',    action='store_true',
                        help='Enable JetRacer motor output (default: simulation)')
    parser.add_argument('--scenario', type=int,  default=SCENARIO_LANE_FOLLOW,
                        choices=[0, 1, 2, 3],
                        help='0=LANE_FOLLOW 1=OBSTACLE_AVOID 2=PARKING 3=STOP')
    parser.add_argument('--model',    type=Path, default=PLANNER_MODEL_PATH,
                        help=f'Model .pth file (default: {PLANNER_MODEL_PATH})')
    args = parser.parse_args()

    main(web_port     = args.web_port,
         enable_motor = args.motor,
         scenario     = args.scenario,
         model_path   = args.model)
