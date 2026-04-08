#!/usr/bin/env python3
"""
Planner Data Collection — Structured Feature Logger
====================================================
Ghost-mode style data collection for the structured planner.
Camera + YOLO + LKAS run as usual, but instead of saving raw images
we log a single CSV row of normalised feature vectors per frame.

  Camera → YOLO → object features  ─┐
  Camera → LKAS → lane features    ─┤→ one CSV row  →  planner_data.csv
  web-viewer  → human steering/throttle ─┘

No .jpg / .npy files are created. The dataset is a plain CSV.

Usage
-----
  python collect_data_planner.py [--web-port 8082] [--scenario 0]

Scenarios (--scenario)
  0 = LANE_FOLLOW       normal track driving
  1 = OBSTACLE_AVOID    driving with obstacles present
  2 = PARKING           parking manoeuvre
  3 = STOP              deliberate stop / e-stop demo

Controls (web viewer browser)
  ← / →   steer left / right  (held key → ±STEER_VALUE)
  ↓        throttle = 0        (full stop)
  Ctrl+C   quit and save

Run alongside
  lkas --broadcast   (image + detection SHM)
  DO NOT run vehicle.py — this script controls JetRacer directly.

Data layout
  data/
  └── planner_data.csv   (one row per saved frame)
"""

import sys
import time
import signal
import csv
import argparse
import importlib.util
import numpy as np
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent.parent / "vehicle" / "src"))
sys.path.insert(0, str(script_dir.parent.parent.parent / "common" / "src"))

# ── PyTorch legacy weights fix ────────────────────────────────────────────────
import torch
_orig_torch_load = torch.load

def _torch_load_legacy(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _torch_load_legacy

from ultralytics import YOLO as _YOLO

# ── Camera ────────────────────────────────────────────────────────────────────
from camera import Camera
from visualization.visualizer import LKASVisualizer

# ── JetRacer ──────────────────────────────────────────────────────────────────
try:
    from jetracer.nvidia_racecar import NvidiaRacecar
    JETRACER_AVAILABLE = True
except ImportError:
    print("[WARN] JetRacer not available — simulation mode (no motor output)")
    JETRACER_AVAILABLE = False

# ── LKAS ──────────────────────────────────────────────────────────────────────
try:
    from lkas import LKASClient as _LKASClient
    LKAS_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] LKAS not available: {e}")
    sys.exit(1)

# ── Web viewer ────────────────────────────────────────────────────────────────
from planner_viewer import PlannerViewer

# ── YOLO config ───────────────────────────────────────────────────────────────
_config_path = script_dir.parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("yolo_config", _config_path)
_cfg  = importlib.util.module_from_spec(_spec)
_cfg.__file__ = str(_config_path)
_spec.loader.exec_module(_cfg)

MODEL_PATH           = _cfg.MODEL_PATH
CONFIDENCE_THRESHOLD = _cfg.CONFIDENCE_THRESHOLD
IOU_THRESHOLD        = _cfg.IOU_THRESHOLD
CLASS_NAMES          = _cfg.CLASS_NAMES
N_YOLO_CLASSES       = len(CLASS_NAMES)
YOLO_DEVICE          = 0 if torch.cuda.is_available() else 'cpu'

# ── Planner model shared definitions ─────────────────────────────────────────
from planner_model import (
    build_object_features,
    build_lane_features,
    csv_columns,
    N_MAX_OBJECTS,
    SCENARIO_LANE_FOLLOW, SCENARIO_OBSTACLE_AVOID, SCENARIO_PARKING, SCENARIO_STOP,
    MAX_THROTTLE,
    FRAME_W, FRAME_H,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_THROTTLE   = 0.20   # auto-forward throttle during collection
STEER_VALUE     = 0.9    # fixed steering magnitude for key presses
SAVE_FPS        = 10     # max rows written per second
YOLO_SKIP       = 3      # run YOLO every N frames (GPU memory constraint)

FIXED_LEFT_LANE_X  = 255
FIXED_RIGHT_LANE_X = 485

DATA_DIR      = script_dir / "data"
PLANNER_CSV   = DATA_DIR / "planner_data.csv"


# ─────────────────────────────────────────────────────────────────────────────
# LKAS helpers  (same as collect_data.py)
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
# Annotation  (broadcast only — nothing saved to disk)
# ─────────────────────────────────────────────────────────────────────────────
import cv2

_MODE_COLORS = {
    0: (0, 255, 0),    # LANE_FOLLOW  — green
    1: (0, 255, 255),  # OBSTACLE_AVOID — yellow
    2: (255, 165, 0),  # PARKING — orange
    3: (0, 0, 255),    # STOP — red
}
_SCENARIO_NAMES = {
    SCENARIO_LANE_FOLLOW:    "LANE_FOLLOW",
    SCENARIO_OBSTACLE_AVOID: "OBSTACLE_AVOID",
    SCENARIO_PARKING:        "PARKING",
    SCENARIO_STOP:           "STOP",
}
_BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 255, 0), (0, 128, 255),
    (128, 128, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255), (0, 128, 0),
]

_visualizer = LKASVisualizer(image_width=FRAME_W, image_height=FRAME_H)


def _annotate(frame, boxes, distances, class_ids, scenario, steering, throttle, fps,
              left_x, right_x, saved_count, det=None):
    out = frame.copy()

    # ── Lane overlay via shared LKASVisualizer ────────────────────────────────
    if det is not None and det.has_segmentation:
        out = _visualizer.draw_segmentation(out, det.segmentation_mask)
    if not (det is not None and det.has_segmentation):
        # fallback: dim vertical lines at fixed positions
        h = out.shape[0]
        cv2.line(out, (int(left_x),  0), (int(left_x),  h), (80, 80, 160), 1)
        cv2.line(out, (int(right_x), 0), (int(right_x), h), (80, 80, 160), 1)

    for box, dist, cid in zip(boxes, distances, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = _BOX_COLORS[cid % len(_BOX_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        lbl   = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        dist_txt = f"{dist:.2f}m" if dist > 0 else "N/A"
        cv2.putText(out, f"{lbl} {dist_txt}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    sc_name  = _SCENARIO_NAMES.get(scenario, str(scenario))
    sc_color = _MODE_COLORS.get(scenario, (255, 255, 255))
    cv2.putText(out, f"SCENARIO: {sc_name}",       (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"steer={steering:+.2f}  thr={throttle:.2f}",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS={fps:.1f}  saved={saved_count}", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_csv(csv_path: Path):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fh     = open(csv_path, "a", newline="")
    writer = csv.writer(fh)
    if not exists:
        writer.writerow(csv_columns())
        print(f"[CSV] Created new dataset: {csv_path}")
    else:
        print(f"[CSV] Appending to existing dataset: {csv_path}")
    return fh, writer


def _save_row(writer, fh, frame_id: int, obj_feats: list, lane_feats: list,
              ego_feats: list, scenario: int, steering: float, throttle: float):
    """Write one structured row to the CSV."""
    # target_throttle is normalised to [0, 1] for training
    throttle_norm = float(throttle) / MAX_THROTTLE

    row = [frame_id]
    row.extend(f"{v:.5f}" for v in obj_feats)
    row.extend(f"{v:.5f}" for v in lane_feats)
    row.extend([f"{ego_feats[0]:.5f}", f"{ego_feats[1]:.5f}"])
    row.append(scenario)
    row.append(f"{steering:.5f}")
    row.append(f"{throttle_norm:.5f}")
    writer.writerow(row)
    fh.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(web_port: int = 8082, scenario: int = SCENARIO_LANE_FOLLOW):
    sc_name = _SCENARIO_NAMES.get(scenario, str(scenario))
    print("=" * 62)
    print("  Planner Data Collection  (structured features, no images)")
    print("=" * 62)
    print(f"  Scenario      : {sc_name} ({scenario})")
    print(f"  Output CSV    : {PLANNER_CSV}")
    print(f"  Save rate cap : {SAVE_FPS} fps")
    print(f"  YOLO skip     : every {YOLO_SKIP} frames")
    print(f"  Base throttle : {BASE_THROTTLE}")
    print()
    print("  Controls: ← steer left  → steer right  ↓ stop  Ctrl+C quit")
    print("=" * 62)

    # ── YOLO ─────────────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] YOLO model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"\n[YOLO] Loading: {MODEL_PATH}")
    yolo = _YOLO(MODEL_PATH)
    if torch.cuda.is_available():
        free_mb, total_mb = [x / 1024**2 for x in torch.cuda.mem_get_info(0)]
        print(f"[YOLO] GPU {YOLO_DEVICE}  ({free_mb:.0f}/{total_mb:.0f} MB)")

    # ── Web viewer ────────────────────────────────────────────────────────────
    web_viewer = None
    if web_port > 0:
        web_viewer = PlannerViewer(http_port=web_port, ws_port=web_port + 1)
        web_viewer.start()
        print(f"[WEB] Viewer: http://0.0.0.0:{web_port}")

    # ── Camera ───────────────────────────────────────────────────────────────
    print("\n[CAM] Opening RealSense camera...")
    camera = Camera(width=FRAME_W, height=FRAME_H, enable_depth=True)
    depth_scale = camera.depth_scale if camera.depth_scale > 0 else 0.001

    # ── LKAS client (connects to LKAS-owned SHMs — same pattern as vehicle.py) ──
    lkas_client = None
    try:
        print("[LKAS] Connecting...")
        lkas_client = _LKASClient(
            image_shm_name="image",
            detection_shm_name="detection",
            control_shm_name="control",
        )
        print("[LKAS] Connected — send_image() will write frames to image SHM")
    except Exception as e:
        print(f"[WARN] LKAS unavailable ({e}) — fixed lane fallback, no frame broadcast")

    # ── JetRacer ─────────────────────────────────────────────────────────────
    car = None
    if JETRACER_AVAILABLE:
        print("\n[CAR] Initializing NvidiaRacecar...")
        car = NvidiaRacecar()
        car.steering_offset = 0.05
        car.steering = 0.0
        car.throttle = 0.0
        print("[CAR] Ready")
    else:
        print("\n[CAR] Simulation mode")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_fh, csv_writer = _init_csv(PLANNER_CSV)

    # ── Signal handling ───────────────────────────────────────────────────────
    running = True

    def _shutdown(sig, _f):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── State ─────────────────────────────────────────────────────────────────
    prev_steering    = 0.0
    prev_throttle    = 0.0
    frame_id         = 1
    saved_count      = 0
    yolo_frame_count = 0
    last_save_time   = 0.0
    save_interval    = 1.0 / SAVE_FPS
    fps              = 0.0
    fps_count        = 0
    fps_start        = time.time()

    cached_boxes:     list = []
    cached_distances: list = []
    cached_class_ids: list = []
    cached_confs:     list = []

    print(f"\n[COLLECT] Running — Ctrl+C to stop\n")

    try:
        while running:

            # ── Web viewer human input ────────────────────────────────────────
            raw_steer  = web_viewer.steering if web_viewer else 0.0
            left_held  = raw_steer < 0
            right_held = raw_steer > 0

            # ── Camera frame ─────────────────────────────────────────────────
            color_bgr, depth_raw = camera.read_frames()
            if color_bgr is None:
                continue

            depth_array = depth_raw if depth_raw is not None else \
                          np.zeros((FRAME_H, FRAME_W), dtype=np.uint16)

            # Send frame to LKAS via shared memory (same as vehicle.py)
            if lkas_client is not None:
                lkas_client.send_image(color_bgr, time.time(), frame_id)

            # ── YOLO (every YOLO_SKIP frames) ─────────────────────────────────
            yolo_frame_count += 1
            if yolo_frame_count % YOLO_SKIP == 0:
                try:
                    results = yolo(
                        color_bgr,
                        conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD,
                        device=YOLO_DEVICE,
                        half=torch.cuda.is_available(),  # FP16 halves VRAM usage
                        verbose=False,
                    )
                    boxes_raw, dists_raw, cids_raw, confs_raw = [], [], [], []
                    if len(results[0].boxes) > 0:
                        xyxy   = results[0].boxes.xyxy.cpu().numpy()
                        cids   = results[0].boxes.cls.cpu().numpy().astype(int)
                        confs  = results[0].boxes.conf.cpu().numpy()
                        for box, cid, conf in zip(xyxy, cids, confs):
                            cx = int((box[0] + box[2]) / 2)
                            cy = int((box[1] + box[3]) / 2)
                            cx = max(0, min(cx, FRAME_W - 1))
                            cy = max(0, min(cy, FRAME_H - 1))
                            raw  = int(depth_array[cy, cx])
                            dist = raw * depth_scale if raw > 0 else -1.0
                            boxes_raw.append(box)
                            dists_raw.append(dist)
                            cids_raw.append(int(cid))
                            confs_raw.append(float(conf))
                    cached_boxes     = boxes_raw
                    cached_distances = dists_raw
                    cached_class_ids = cids_raw
                    cached_confs     = confs_raw
                    del results  # free GPU tensors immediately
                except Exception as e:
                    print(f"\n[YOLO] Error: {e}")

            boxes     = cached_boxes
            distances = cached_distances
            class_ids = cached_class_ids
            confs     = cached_confs

            # ── Lane detection ───────────────────────────────────────────────
            left_lane_x   = float(FIXED_LEFT_LANE_X)
            right_lane_x  = float(FIXED_RIGHT_LANE_X)
            lane_detected = False
            det           = None
            if lkas_client is not None:
                try:
                    det = lkas_client.get_detection(timeout=0.01)
                    if det is not None and det.left_lane and det.right_lane:
                        y_ref = _find_y_at_distance(depth_array, depth_scale)
                        lx    = _interpolate_lane_x(det.left_lane, y_ref)
                        rx    = _interpolate_lane_x(det.right_lane, y_ref)
                        if 0 < lx < rx <= FRAME_W:
                            left_lane_x   = lx
                            right_lane_x  = rx
                            lane_detected = True
                except Exception:
                    pass

            # ── Human steering / throttle ─────────────────────────────────────
            if left_held and not right_held:
                input_steering = -STEER_VALUE
            elif right_held and not left_held:
                input_steering = STEER_VALUE
            else:
                input_steering = 0.0

            input_throttle = web_viewer.throttle if web_viewer else 0.0

            # Apply to vehicle
            if car is not None:
                car.steering = -float(input_steering)
                car.throttle = -float(input_throttle)

            # ── Build structured features ─────────────────────────────────────
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

            # ── Save row (rate-limited, only when recording is toggled ON) ─────
            is_recording = web_viewer.recording if web_viewer else True
            now = time.monotonic()
            if is_recording and now - last_save_time >= save_interval:
                last_save_time = now
                _save_row(
                    csv_writer, csv_fh,
                    frame_id   = frame_id,
                    obj_feats  = obj_feats,
                    lane_feats = lane_feats,
                    ego_feats  = ego_feats,
                    scenario   = scenario,
                    steering   = input_steering,
                    throttle   = input_throttle,
                )
                frame_id    += 1
                saved_count += 1

                steer_tag = " ←" if input_steering < 0 else " →" if input_steering > 0 else "  ↑"
                sys.stdout.write(
                    f"\r[{'REC' if is_recording else '---'}]  "
                    f"saved={saved_count:>5d}  "
                    f"steer={input_steering:+.2f}{steer_tag}  "
                    f"thr={input_throttle:.2f}  "
                    f"objs={len(boxes)}  "
                    f"lane={'YES' if lane_detected else 'NO '}   "
                )
                sys.stdout.flush()

            # Update ego state for next frame
            prev_steering = input_steering
            prev_throttle = input_throttle

            # ── FPS ──────────────────────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps       = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # ── Broadcast annotated frame ─────────────────────────────────────
            if web_viewer is not None:
                annotated = _annotate(
                    color_bgr, boxes, distances, class_ids,
                    scenario, input_steering, input_throttle, fps,
                    left_lane_x, right_lane_x, saved_count,
                    det=det if lkas_client is not None else None,
                )
                web_viewer.broadcast_frame(annotated)
                web_viewer.broadcast_status({
                    'scenario':      sc_name,
                    'steering':      input_steering,
                    'throttle':      input_throttle,
                    'fps':           fps,
                    'objects':       len(boxes),
                    'lane_detected': lane_detected,
                    'saved_rows':    saved_count,
                    'recording':     is_recording,
                })

            frame_id += 0   # already incremented inside save block

    finally:
        running = False

        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0

        csv_fh.close()
        sys.stdout.write("\n")

        try: camera.close()
        except Exception: pass
        if web_viewer is not None:
            try: web_viewer.stop()
            except Exception: pass
        if lkas_client is not None:
            try: lkas_client.close()
            except Exception: pass

        print(f"\n[COLLECT] Done — {saved_count} rows saved")
        print(f"[COLLECT] Dataset: {PLANNER_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planner data collection — structured features")
    parser.add_argument('--web-port', type=int, default=8082,
                        help='Web viewer HTTP port (default 8082, 0 to disable)')
    parser.add_argument('--scenario', type=int, default=SCENARIO_LANE_FOLLOW,
                        choices=[0, 1, 2, 3],
                        help='Scenario token: 0=LANE_FOLLOW 1=OBSTACLE_AVOID 2=PARKING 3=STOP')
    args = parser.parse_args()
    main(web_port=args.web_port, scenario=args.scenario)
