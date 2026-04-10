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
  Camera  →  YOLO    → object features  ─┐
  Camera  →  LaneSeg → lane grid (32)   ─┤ → PlannerModel → [steer, thr]
  ego state  (from last cycle)          ─┘
                                               ↓
                                       control SHM (optional) + JetRacer

Run standalone — no LKAS required.
  vehicle.py    (reads control SHM)  ← optional; OR use --motor for direct drive

Usage
-----
  python planner_inference.py [--web-port 8082] [--motor] [--scenario 0]
                              [--model planner_model.pth]
"""

import sys
import time
import argparse
import threading
import numpy as np
from pathlib import Path

import torch

# ── Path setup ────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent / "vehicle" / "src"))
sys.path.append(str(script_dir.parent / "common" / "src"))

# ── PyTorch legacy weights fix ────────────────────────────────────────────────
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
    print("[WARN] JetRacer not available — simulation mode")
    JETRACER_AVAILABLE = False

# ── Lane segmentation (direct BiSeNet — no LKAS required) ────────────────────
from lane_seg import LaneSeg

# ── Control SHM (optional — only needed when vehicle.py reads planner output) ─
try:
    from lkas.integration.shared_memory import SharedMemoryControlChannel
    from lkas.integration.shared_memory.messages import ControlMessage
    LKAS_SHM_AVAILABLE = True
except ImportError:
    LKAS_SHM_AVAILABLE = False
    print("[WARN] LKAS SHM not importable — control SHM disabled")

# ── Web viewer ────────────────────────────────────────────────────────────────
from planner_viewer import PlannerViewer

# ── YOLO config ───────────────────────────────────────────────────────────────
from yolo_config import MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, CLASS_NAMES
N_YOLO_CLASSES = len(CLASS_NAMES)

# ── Planner ───────────────────────────────────────────────────────────────────
from planner_model import (
    PlannerModel,
    build_object_features,
    build_lane_grid,
    lane_boundaries_from_mask,
    draw_lane_grid_overlay,
    MAX_THROTTLE,
    FRAME_W, FRAME_H,
    SCENARIO_LANE_FOLLOW, SCENARIO_LEFT_TURN, SCENARIO_RIGHT_TURN,
    SCENARIO_GO_STRAIGHT, SCENARIO_PULL_OVER, SCENARIO_PARKING,
    SCENARIO_NAMES,
)

PLANNER_MODEL_PATH = script_dir / "planner_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_SCENARIO_NAMES = SCENARIO_NAMES  # imported from planner_model

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    boxes, distances, class_ids, confs,
    mask,
    prev_steering, prev_throttle,
    device,
):
    """
    Build model-ready tensors from raw YOLO detections + BiSeNet lane mask.
    Returns (objects, lane, ego) tensors, all batched with B=1.
    """
    left_lane_x, right_lane_x = lane_boundaries_from_mask(mask)

    obj_feats  = build_object_features(
        boxes=boxes, distances=distances,
        class_ids=class_ids, confs=confs,
        left_lane_x=left_lane_x, right_lane_x=right_lane_x,
        frame_w=FRAME_W, frame_h=FRAME_H, n_classes=N_YOLO_CLASSES,
    )
    lane_feats = build_lane_grid(mask)
    ego_feats  = [prev_steering, prev_throttle / MAX_THROTTLE]

    objects_t = torch.tensor(obj_feats,  dtype=torch.float32, device=device).unsqueeze(0)
    lane_t    = torch.tensor(lane_feats, dtype=torch.float32, device=device).unsqueeze(0)
    ego_t     = torch.tensor(ego_feats,  dtype=torch.float32, device=device).unsqueeze(0)

    return objects_t, lane_t, ego_t


# ─────────────────────────────────────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────────────────────────────────────
import cv2

_visualizer = LKASVisualizer(image_width=FRAME_W, image_height=FRAME_H)

_MODE_COLORS = {
    SCENARIO_LANE_FOLLOW: (0, 255, 0),
    SCENARIO_LEFT_TURN:   (255, 255, 0),
    SCENARIO_RIGHT_TURN:  (0, 255, 255),
    SCENARIO_GO_STRAIGHT: (255, 165, 0),
    SCENARIO_PULL_OVER:   (255, 0, 255),
    SCENARIO_PARKING:     (0, 128, 255),
}
_BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 165, 0),
    (128, 0, 128), (0, 255, 255), (255, 255, 0), (0, 128, 255),
    (128, 128, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255), (0, 128, 0),
]


def _draw(frame, boxes, distances, class_ids, scenario, steering, throttle, fps,
          left_x, right_x, mask=None, lane_feats=None):
    out = frame.copy()

    # ── Lane segmentation overlay ─────────────────────────────────────────────
    if mask is not None:
        out = _visualizer.draw_segmentation(out, mask)
    else:
        h = out.shape[0]
        cv2.line(out, (int(left_x), 0), (int(left_x), h), (80, 80, 160), 1)
        cv2.line(out, (int(right_x), 0), (int(right_x), h), (80, 80, 160), 1)

    # ── Grid pooling overlay ──────────────────────────────────────────────────
    if lane_feats is not None:
        out = draw_lane_grid_overlay(out, lane_feats)

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
# YOLO background worker
# Runs YOLO on CPU in a separate thread so the main loop is never blocked.
# The main loop always reads the latest cached result.
# ─────────────────────────────────────────────────────────────────────────────

_yolo_lock    = threading.Lock()
_yolo_cache   = {'boxes': [], 'distances': [], 'class_ids': [], 'confs': []}
_yolo_running = False   # True while a YOLO inference is in progress


def _yolo_worker(yolo, frame, depth_array, depth_scale, frame_w, frame_h):
    global _yolo_running, _yolo_cache
    try:
        results = yolo(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
                       device='cpu', verbose=False)
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
        with _yolo_lock:
            _yolo_cache = {'boxes': boxes_r, 'distances': dists_r,
                           'class_ids': cids_r, 'confs': confs_r}
        del results
    except Exception as e:
        print(f"\n[YOLO] Error: {e}")
    finally:
        _yolo_running = False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    web_port:     int  = 8082,
    enable_motor: bool = False,
    scenario:     int  = SCENARIO_LANE_FOLLOW,
    model_path:   Path = PLANNER_MODEL_PATH,
    verbose:      bool = False,
    log_history:  bool = False,
):
    # Both YOLO and planner run on CPU.
    # LKAS BiSeNet (DL method, device="auto") claims the GPU; putting YOLO
    # on GPU too causes OOM on Jetson's 7.4 GB unified memory pool.
    yolo_device    = 'cpu'  # kept for reference; actual device set in _yolo_worker
    planner_device = torch.device('cpu')
    sc_name = _SCENARIO_NAMES.get(scenario, str(scenario))

    print("=" * 62)
    print("  Planner Inference  (structured features → actuation)")
    print("=" * 62)
    print(f"  YOLO device   : cpu  (GPU reserved for LaneSeg/BiSeNet)")
    print(f"  Planner device: cpu")
    print(f"  Scenario  : {sc_name} ({scenario})")
    print(f"  Model     : {model_path}")
    print(f"  Motor     : {'ACTIVE' if enable_motor else 'SIMULATION'}")
    print(f"  Web       : {'port ' + str(web_port) if web_port > 0 else 'DISABLED'}")
    print(f"  History   : {'ENABLED' if log_history else 'disabled'}")
    print()

    # ── Planner model ─────────────────────────────────────────────────────────
    if not model_path.exists():
        print(f"[ERROR] Planner model not found: {model_path}")
        print("        Run train_planner.py first.")
        sys.exit(1)
    planner = PlannerModel().to(planner_device)
    state   = torch.load(str(model_path), map_location=planner_device)
    planner.load_state_dict(state)
    planner.eval()
    n_params = sum(p.numel() for p in planner.parameters())
    print(f"[PLANNER] Loaded  ({n_params:,} params)")

    # ── YOLO model ────────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] YOLO model not found: {MODEL_PATH}")
        sys.exit(1)
    print(f"[YOLO] Loading: {MODEL_PATH}")
    yolo = _YOLO(MODEL_PATH)

    # ── LaneSeg (direct BiSeNet — GPU) ────────────────────────────────────────
    print("[LaneSeg] Loading BiSeNet...")
    lane_seg = LaneSeg(device="auto")

    # ── Camera ────────────────────────────────────────────────────────────────
    print("[CAM] Opening RealSense camera...")
    camera      = Camera(width=FRAME_W, height=FRAME_H, enable_depth=True)
    depth_scale = camera.depth_scale if camera.depth_scale > 0 else 0.001
    frame_w, frame_h = FRAME_W, FRAME_H
    print(f"[CAM] {frame_w}×{frame_h}  depth_scale={depth_scale}")

    # ── Control SHM (optional) ────────────────────────────────────────────────
    control_channel = None
    if LKAS_SHM_AVAILABLE:
        try:
            control_channel = SharedMemoryControlChannel(
                name="control", create=False, retry_count=3, retry_delay=0.5)
            print("[SHM] Control channel connected")
        except Exception as e:
            print(f"[WARN] Control SHM unavailable ({e}) — motor-only mode")

    # ── JetRacer ─────────────────────────────────────────────────────────────
    car = None
    if enable_motor and JETRACER_AVAILABLE:
        car = NvidiaRacecar()
        car.steering_offset = 0.035  # adjust if steering is not centred at 0.0
        car.throttle = 0.0
        car.steering = 0.0
        time.sleep(0.2)   # wait for servo to physically centre before starting
        print("[CAR] NvidiaRacecar ready — MOTORS ACTIVE")
    else:
        print("[CAR] Simulation mode (motor control disabled)")

    # ── Web viewer ────────────────────────────────────────────────────────────
    web_viewer = None
    if web_port > 0:
        web_viewer = PlannerViewer(http_port=web_port, ws_port=web_port + 1)
        web_viewer.start()
        print(f"[WEB] Viewer at http://0.0.0.0:{web_port}")

    # ── Output history logger ─────────────────────────────────────────────────
    _hist_fh = _hist_writer = None
    if log_history:
        import csv as _csv
        _hist_path = script_dir / f"inference_history_{int(time.time())}.csv"
        _hist_fh   = open(_hist_path, "w", newline="")
        _hist_writer = _csv.writer(_hist_fh)
        _hist_writer.writerow([
            "frame_id", "timestamp",
            "raw_steer", "raw_thr",          # raw sigmoid/tanh outputs
            "final_steer", "final_thr",       # after clamp
            "act_steer", "act_thr",           # actually sent to motor
            "lane_detected", "n_objects", "scenario",
        ])
        print(f"[HIST] Logging to {_hist_path}")

    # ── State ─────────────────────────────────────────────────────────────────
    prev_steering = 0.0
    prev_throttle = 0.0

    fps       = 0.0
    fps_count = 0
    fps_start = time.time()

    scenario_t = torch.tensor([scenario], dtype=torch.long, device=planner_device)

    frame_id = 1
    print(f"\n[RUN] Running — Ctrl+C to stop\n")

    try:
        while True:
            color_bgr, depth_raw = camera.read_frames()
            if color_bgr is None:
                continue

            depth_array = depth_raw if depth_raw is not None else \
                          np.zeros((frame_h, frame_w), dtype=np.uint16)
            frame_id += 1

            # ── Lane segmentation (BiSeNet, GPU, every frame) ──────────────────
            mask          = lane_seg.infer(color_bgr)
            lane_detected = bool(mask.any())
            left_x, right_x = lane_boundaries_from_mask(mask)

            # ── YOLO (background thread — never blocks the main loop) ─────────
            global _yolo_running
            if not _yolo_running:
                _yolo_running = True
                threading.Thread(
                    target=_yolo_worker,
                    args=(yolo, color_bgr.copy(), depth_array.copy(),
                          depth_scale, frame_w, frame_h),
                    daemon=True,
                ).start()

            with _yolo_lock:
                r         = _yolo_cache
            boxes     = r['boxes']
            distances = r['distances']
            class_ids = r['class_ids']
            confs     = r['confs']

            # ── Planner forward pass ───────────────────────────────────────────
            with torch.no_grad():
                objects_t, lane_t, ego_t = extract_features(
                    boxes=boxes, distances=distances,
                    class_ids=class_ids, confs=confs,
                    mask=mask,
                    prev_steering=prev_steering, prev_throttle=prev_throttle,
                    device=planner_device,
                )
                out = planner(objects_t, lane_t, ego_t, scenario_t)  # (1, 2)

            # Denormalise outputs
            final_steering = float(out[0, 0].item())               # tanh → [-1, 1]
            final_throttle = float(out[0, 1].item()) * MAX_THROTTLE # sigmoid [0,1] → [0, MAX_THROTTLE]

            # Clamp for safety
            final_steering = float(np.clip(final_steering, -1.0,       1.0))
            final_throttle = float(np.clip(final_throttle,  0.0, MAX_THROTTLE))

            # ── Verbose debug ──────────────────────────────────────────────────
            if verbose:
                lane_f = lane_t[0].tolist()
                ego_f  = ego_t[0].tolist()
                # Print 4×8 grid as rows (far → near)
                from planner_model import GRID_ROWS, GRID_COLS
                print(f"\n[DBG lane ] detected={'YES' if lane_detected else 'NO '}  "
                      f"left_x={left_x:.1f}  right_x={right_x:.1f}")
                for r in range(GRID_ROWS):
                    cells = "  ".join(f"{lane_f[r*GRID_COLS+c]:.2f}"
                                      for c in range(GRID_COLS))
                    depth = "far " if r == 0 else ("near" if r == GRID_ROWS-1 else f"r{r} ")
                    print(f"  [{depth}]  {cells}")
                print(f"[DBG ego  ] prev_steer={ego_f[0]:+.4f}  "
                      f"prev_thr_norm={ego_f[1]:.4f}")
                print(f"[DBG objs ] count={len(boxes)}", end="")
                for i, (box, dist, cid) in enumerate(zip(boxes, distances, class_ids)):
                    print(f"\n  [{i}] {CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else cid}"
                          f"  dist={dist:.2f}m  box={[int(v) for v in box]}", end="")
                print(f"\n[DBG out  ] steer={final_steering:+.4f}  thr={final_throttle:.4f}")

            # ── Apply to vehicle (suppressed when paused) ──────────────────────
            is_paused = web_viewer.paused if web_viewer is not None else False
            act_steering = 0.0 if is_paused else final_steering
            act_throttle = 0.0 if is_paused else final_throttle

            if car is not None:
                car.steering = -act_steering   # hardware inversion
                car.throttle = -act_throttle   # negative = forward

            # ── History log ───────────────────────────────────────────────────
            if _hist_writer is not None:
                _hist_writer.writerow([
                    frame_id, f"{time.time():.4f}",
                    f"{out[0,0].item():+.5f}", f"{out[0,1].item():.5f}",
                    f"{final_steering:+.5f}",  f"{final_throttle:.5f}",
                    f"{act_steering:+.5f}",    f"{act_throttle:.5f}",
                    int(lane_detected), len(boxes), scenario,
                ])

            # ── Write to control SHM (optional) ───────────────────────────────
            nearest = min((d for d in distances if d > 0), default=-1.0)
            if control_channel is not None:
                control_msg = ControlMessage(
                    steering = act_steering,
                    throttle = act_throttle,
                    brake    = 1.0 if is_paused else 0.0,
                )
                control_channel.write(control_msg, frame_id=frame_id,
                                      timestamp=time.time(), processing_time_ms=0.0)

            # Update ego state (always track model output, not suppressed values)
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
                                  left_x, right_x, mask=mask,
                                  lane_feats=lane_t[0].tolist())
                web_viewer.broadcast_frame(annotated)
                web_viewer.broadcast_status({
                    'fps':              fps,
                    'action':           sc_name,
                    'steering':         final_steering,
                    'throttle':         final_throttle,
                    'nearest_distance': nearest,
                    'lane_detected':    lane_detected,
                    'left_lane_x':      left_x,
                    'right_lane_x':     right_x,
                })

    except KeyboardInterrupt:
        print("\n[RUN] Stopped by user")

    finally:
        if car is not None:
            car.throttle = 0.0
            car.steering = 0.0
            time.sleep(0.3)   # hold neutral long enough for servo to physically reach centre
        if _hist_fh is not None:
            _hist_fh.flush()
            _hist_fh.close()
            print(f"[HIST] Saved → {_hist_path}")
        try: camera.close()
        except Exception: pass
        if web_viewer is not None:
            try: web_viewer.stop()
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
                        choices=[0, 1, 2, 3, 4, 5],
                        help='0=LANE_FOLLOW 1=LEFT_TURN 2=RIGHT_TURN 3=GO_STRAIGHT 4=PULL_OVER 5=PARKING')
    parser.add_argument('--model',    type=Path, default=PLANNER_MODEL_PATH,
                        help=f'Model .pth file (default: {PLANNER_MODEL_PATH})')
    parser.add_argument('--verbose',     action='store_true',
                        help='Print per-frame debug: object list, lane, model outputs')
    parser.add_argument('--log-history', action='store_true',
                        help='Write per-frame steering/throttle output to inference_history_<ts>.csv')
    args = parser.parse_args()

    main(web_port     = args.web_port,
         enable_motor = args.motor,
         scenario     = args.scenario,
         model_path   = args.model,
         verbose      = args.verbose,
         log_history  = args.log_history)
