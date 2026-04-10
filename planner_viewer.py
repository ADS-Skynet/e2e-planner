#!/usr/bin/env python3
"""
Planner Web Viewer
==================
Self-contained HTTP + WebSocket viewer for the planner scripts.
No dependency on yolo_web_viewer.py or the obstacle avoidance module.

Provides:
  - MJPEG-style frame streaming over WebSocket (binary)
  - JSON status panel (scenario, steering, throttle, fps, objects, lane)
  - Keyboard controls → steering / stop state readable by the main loop

Controls (browser keyboard)
  ← / →  steer — tap=0.3, hold 300ms=0.6, hold 600ms=0.9
  ↑      throttle +0.1
  ↓      throttle = 0
  Space  toggle recording
  P      pause/resume

Controls (gamepad — Xbox 360 via xboxdrv, no kernel modules needed)
  Left  stick X  throttle (push right = 0→0.45; release = 0)
  Right stick X  steer (continuous ±1.0, deadzone 0.05)
  LT             emergency throttle = 0
  Start          toggle recording
  Back           pause/resume

Usage
-----
    from planner_viewer import PlannerViewer

    viewer = PlannerViewer(http_port=8082, ws_port=8083)
    viewer.start()

    # in main loop:
    steering = viewer.steering      # float: -STEER_VALUE / 0 / +STEER_VALUE
    stop     = viewer.stop_held     # bool: True while ↓ is held

    viewer.broadcast_frame(annotated_bgr)
    viewer.broadcast_status({
        'scenario': 'LANE_FOLLOW',
        'steering': 0.12,
        'throttle': 0.20,
        'fps': 18.3,
        'objects': 2,
        'lane_detected': True,
    })

    viewer.stop()
"""

import asyncio
import json
import time
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread, Lock
from typing import Optional, Set

try:
    import websockets
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False
    print("[PlannerViewer] websockets not installed — WebSocket disabled")


# ─────────────────────────────────────────────────────────────────────────────
# HTML  (served at /)
# ─────────────────────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Planner Viewer</title>
<style>
  body  { margin:0; background:#111; color:#eee; font-family:monospace;
          display:flex; flex-direction:column; align-items:center; }
  h1    { font-size:1rem; margin:8px 0 4px; color:#adf; }
  #feed { max-width:100%; border:2px solid #333; margin-bottom:6px; }
  #status { display:grid; grid-template-columns:1fr 1fr; gap:4px 16px;
            font-size:0.85rem; padding:6px 16px; background:#1a1a1a;
            border:1px solid #333; border-radius:4px; min-width:320px; }
  .label { color:#888; }
  .val   { color:#ffe; text-align:right; }
  #ctrl  { margin-top:6px; font-size:0.75rem; color:#666; }
  #rec-badge { font-size:0.9rem; font-weight:bold; margin:6px 0 2px;
               padding:3px 14px; border-radius:4px; letter-spacing:1px; }
  #rec-badge.on  { background:#c00; color:#fff; }
  #rec-badge.off { background:#333; color:#666; }
  #pause-btn { font-size:0.9rem; font-weight:bold; margin:4px 0 2px;
               padding:4px 20px; border-radius:4px; letter-spacing:1px;
               border:none; cursor:pointer; }
  #pause-btn.paused   { background:#f80; color:#000; }
  #pause-btn.running  { background:#282; color:#fff; }
  .steer-bar { width:200px; height:14px; background:#222; border:1px solid #444;
               border-radius:3px; position:relative; margin:4px auto; }
  .steer-indicator { position:absolute; top:2px; width:6px; height:10px;
                     background:#4af; border-radius:2px; transform:translateX(-50%); }
  #scenario-bar { display:flex; flex-wrap:wrap; gap:4px; margin:6px 0 2px;
                  align-items:center; }
  #scenario-bar .label { color:#888; font-size:0.8rem; margin-right:4px; }
  .sc-btn { font-size:0.78rem; padding:3px 8px; border-radius:4px; border:1px solid #555;
            background:#222; color:#aaa; cursor:pointer; }
  .sc-btn.active { background:#246; color:#8cf; border-color:#48a; font-weight:bold; }
</style>
</head>
<body>
<h1>Planner Viewer</h1>
<canvas id="feed" width="848" height="480"></canvas>
<div id="rec-badge" class="off">⏺ REC OFF</div>
<button id="pause-btn" class="running" onclick="togglePause()">▶ RUNNING</button>
<div id="status">
  <span class="label">Scenario</span>  <span class="val" id="s-scenario">—</span>
  <span class="label">Steering</span>  <span class="val" id="s-steering">0.000</span>
  <span class="label">Throttle</span>  <span class="val" id="s-throttle">0.000</span>
  <span class="label">FPS</span>       <span class="val" id="s-fps">0.0</span>
  <span class="label">Objects</span>   <span class="val" id="s-objects">0</span>
  <span class="label">Lane</span>      <span class="val" id="s-lane">—</span>
  <span class="label">Saved rows</span><span class="val" id="s-saved">0</span>
</div>
<div class="steer-bar">
  <div class="steer-indicator" id="steer-indicator" style="left:50%"></div>
</div>
<div id="scenario-bar">
  <span class="label">Scenario:</span>
  <button id="sc-0" class="sc-btn active" onclick="setScenario(0)">0 Lane Follow</button>
  <button id="sc-1" class="sc-btn"        onclick="setScenario(1)">1 Left Turn</button>
  <button id="sc-2" class="sc-btn"        onclick="setScenario(2)">2 Right Turn</button>
  <button id="sc-3" class="sc-btn"        onclick="setScenario(3)">3 Go Straight</button>
  <button id="sc-4" class="sc-btn"        onclick="setScenario(4)">4 Pull Over</button>
  <button id="sc-5" class="sc-btn"        onclick="setScenario(5)">5 Parking</button>
</div>
<div id="ctrl">← / → steer (tap=0.3 · hold 300ms=0.6 · hold 600ms=0.9) &nbsp;|&nbsp; ↑ throttle +0.1 &nbsp; ↓ throttle = 0 &nbsp;|&nbsp; Space = record &nbsp;|&nbsp; P = pause &nbsp;|&nbsp; 0-5 = scenario</div>

<script>
const canvas = document.getElementById('feed');
const ctx    = canvas.getContext('2d');
const WS_PORT = """ + str(8083) + """;

let ws = null;
let reconnectTimer = null;

function connect() {
  ws = new WebSocket('ws://' + location.hostname + ':' + WS_PORT);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    console.log('WS connected');
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      // JPEG frame
      const blob = new Blob([e.data], {type:'image/jpeg'});
      const url  = URL.createObjectURL(blob);
      const img  = new Image();
      img.onload = () => {
        canvas.width  = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
      };
      img.src = url;
    } else {
      // JSON status
      try {
        const d = JSON.parse(e.data);
        if (d.type === 'status') {
          document.getElementById('s-scenario').textContent = d.scenario  ?? '—';
          document.getElementById('s-steering').textContent = (d.steering ?? 0).toFixed(3);
          document.getElementById('s-throttle').textContent = (d.throttle ?? 0).toFixed(3);
          document.getElementById('s-fps'     ).textContent = (d.fps      ?? 0).toFixed(1);
          document.getElementById('s-objects' ).textContent = d.objects   ?? 0;
          document.getElementById('s-lane'    ).textContent = d.lane_detected ? 'YES' : 'NO';
          document.getElementById('s-saved'   ).textContent = d.saved_rows ?? 0;
          // recording badge
          const badge = document.getElementById('rec-badge');
          if (d.recording) {
            badge.textContent = '⏺ REC ON';
            badge.className   = 'on';
          } else {
            badge.textContent = '⏺ REC OFF';
            badge.className   = 'off';
          }
          // steering indicator
          const pct = 50 + (d.steering ?? 0) * 50;
          document.getElementById('steer-indicator').style.left = pct + '%';
        }
      } catch(_) {}
    }
  };

  ws.onclose = () => {
    reconnectTimer = setTimeout(connect, 2000);
  };
}
connect();

// ── Controls ─────────────────────────────────────────────────────────────────
// Graduated steering: tap = 0.3, hold 300 ms = 0.6, hold 600 ms = 0.9
// Captures intermediate values in the CSV, not just ±0.9.
const STEER_LEVELS   = [0.3, 0.6, 0.9];
const STEER_HOLD_MS  = 300;
const THROTTLE_STEP  = 0.05;
const THROTTLE_MAX   = 0.4;

let steering      = 0.0;
let throttle      = 0.0;
let recording     = false;
let paused        = false;
let steerDir      = 0;
let steerLevel    = 0;
let steerTimer    = null;

function sendControl() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type:'control', steering, throttle}));
  }
}

function sendRecording() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type:'record_toggle'}));
  }
}

function togglePause() {
  paused = !paused;
  const btn = document.getElementById('pause-btn');
  if (paused) {
    btn.textContent = '⏸ PAUSED';
    btn.className   = 'paused';
  } else {
    btn.textContent = '▶ RUNNING';
    btn.className   = 'running';
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type:'pause_toggle'}));
  }
}

function setScenario(n) {
  document.querySelectorAll('.sc-btn').forEach((b, i) => {
    b.classList.toggle('active', i === n);
  });
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type: 'scenario_change', value: n}));
  }
}

function _stopSteerRamp() {
  if (steerTimer !== null) { clearInterval(steerTimer); steerTimer = null; }
  steerDir   = 0;
  steerLevel = 0;
  steering   = 0.0;
  sendControl();
}

function _startSteerRamp(dir) {
  if (steerDir === dir) return;   // suppress browser key-repeat
  _stopSteerRamp();
  steerDir   = dir;
  steerLevel = 0;
  steering   = dir * STEER_LEVELS[0];   // immediate 0.3 on first press
  sendControl();
  steerTimer = setInterval(() => {
    if (steerLevel < STEER_LEVELS.length - 1) {
      steerLevel++;
      steering = dir * STEER_LEVELS[steerLevel];
      sendControl();
    }
  }, STEER_HOLD_MS);
}

document.addEventListener('keydown', (e) => {
  if      (e.key === 'ArrowLeft')  { _startSteerRamp(-1); e.preventDefault(); return; }
  else if (e.key === 'ArrowRight') { _startSteerRamp( 1); e.preventDefault(); return; }
  let changed = false;
  if      (e.key === 'ArrowUp') {
    throttle = Math.min(THROTTLE_MAX, Math.round((throttle + THROTTLE_STEP) * 10) / 10);
    changed = true;
  }
  else if (e.key === 'ArrowDown')  { throttle = 0.0; changed = true; }
  else if (e.key === ' ')          { sendRecording(); e.preventDefault(); return; }
  else if (e.key === 'p' || e.key === 'P') { togglePause(); e.preventDefault(); return; }
  else if (e.key >= '0' && e.key <= '5')  { setScenario(parseInt(e.key)); e.preventDefault(); return; }
  if (changed) { e.preventDefault(); sendControl(); }
});

document.addEventListener('keyup', (e) => {
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    _stopSteerRamp();
    e.preventDefault();
  }
});
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler  (serves the HTML page)
# ─────────────────────────────────────────────────────────────────────────────
class _HTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/favicon.ico':
            self.send_response(204)   # no content — stops browser second request
            self.send_header('Connection', 'close')
            self.end_headers()
            return
        body = _HTML.encode()
        self.send_response(200)
        self.send_header('Content-Type',   'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection',     'close')   # close after response — stops browser loading spinner
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence access log


# ─────────────────────────────────────────────────────────────────────────────
# PlannerViewer
# ─────────────────────────────────────────────────────────────────────────────
class PlannerViewer:
    """
    Minimal web viewer for planner data collection and inference.

    Thread-safe properties:
      .steering   float  — current steering key state (-1 / 0 / +1 scaled)
      .stop_held  bool   — True while ↓ is held
    """

    STEER_VALUE    = 1.0   # JS and gamepad both output final values directly; no extra scaling
    FULL_THROTTLE  = 0.35  # maximum throttle value; right stick X and keyboard scale to this

    def __init__(self, http_port: int = 8082, ws_port: int = 8083):
        self._http_port = http_port
        self._ws_port   = ws_port

        self._lock      = Lock()
        self._steering  = 0.0   # raw from browser: -1/0/+1
        self._throttle  = 0.0   # 0–1; ↑ adds 0.1 per press, ↓ resets to 0
        self._recording = False  # toggled by Space
        self._paused    = False  # toggled by P / pause button
        self._scenario  = 0     # current scenario token (0–5)

        self._clients:  Set = set()
        self._loop:     Optional[asyncio.AbstractEventLoop] = None
        self._ws_ready  = False

        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[Thread] = None
        self._ws_thread:   Optional[Thread] = None
        self._gp_thread:   Optional[Thread] = None

    # ── Public state ──────────────────────────────────────────────────────────

    @property
    def steering(self) -> float:
        """Current steering command: −STEER_VALUE / 0 / +STEER_VALUE."""
        with self._lock:
            return self._steering * self.STEER_VALUE

    @property
    def throttle(self) -> float:
        """Current throttle level [0, 1]. ↑ adds 0.1 per press, ↓ resets to 0."""
        with self._lock:
            return self._throttle

    @property
    def recording(self) -> bool:
        """True when the Space toggle is ON — data should be saved."""
        with self._lock:
            return self._recording

    @property
    def paused(self) -> bool:
        """True when P / pause button is active — motor output should be suppressed."""
        with self._lock:
            return self._paused

    @property
    def scenario(self) -> int:
        """Current scenario token (0–5), settable from the web UI."""
        with self._lock:
            return self._scenario


    # ── Broadcast ─────────────────────────────────────────────────────────────

    def broadcast_frame(self, frame_bgr: np.ndarray, quality: int = 80):
        """Encode frame as JPEG and send to all connected browsers."""
        if not self._loop or not self._loop.is_running():
            return
        ok, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        data = buf.tobytes()
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                asyncio.run_coroutine_threadsafe(client.send(data), self._loop)
            except Exception:
                pass

    def broadcast_status(self, status: dict):
        """Send a JSON status dict to all connected browsers."""
        if not self._loop or not self._loop.is_running():
            return
        msg = json.dumps({'type': 'status', **status})
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                asyncio.run_coroutine_threadsafe(client.send(msg), self._loop)
            except Exception:
                pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start HTTP and WebSocket servers in background threads."""
        self._http_thread = Thread(target=self._run_http, daemon=True)
        self._http_thread.start()

        if _WS_AVAILABLE:
            self._ws_thread = Thread(target=self._run_ws, daemon=True)
            self._ws_thread.start()
            waited = 0.0
            while not self._ws_ready and waited < 5.0:
                time.sleep(0.05)
                waited += 0.05

        self._gp_thread = Thread(target=self._run_gamepad, daemon=True)
        self._gp_thread.start()

        print(f"[PlannerViewer] http://0.0.0.0:{self._http_port}   ws://0.0.0.0:{self._ws_port}")

    def stop(self):
        if self._http_server:
            try:
                self._http_server.shutdown()
            except Exception:
                pass
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ── Gamepad thread ────────────────────────────────────────────────────────

    def _run_gamepad(self):
        """
        Read Xbox 360 controller via `xboxdrv --no-uinput -v` subprocess.
        No kernel modules (xpad/uinput) required — xboxdrv accesses USB directly.

        Requires non-root USB access (run once):
          echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="028e", MODE="0666"' \\
            | sudo tee /etc/udev/rules.d/99-xbox.rules
          sudo udevadm control --reload-rules && sudo udevadm trigger

        xboxdrv stdout format (line per state change):
          X1:<int> Y1:<int>  X2:<int> Y2:<int>  ... back:<0|1> ... start:<0|1>  ... LT:<int> RT:<int>

        Mapping:
          X2 (right stick X)   → steering  (±1.0, deadzone 0.05)
          Y1 (left  stick Y)   → throttle (up = forward, down = reverse, ±FULL_THROTTLE)
          X2 (right stick X)   → steering (left = −1, right = +1)
          LT                  → emergency throttle = 0  (threshold 64/255)
          RB  (R1)            → start/resume recording  (on press)
          RT  (R2)            → pause recording         (on press, threshold 64/255)
        """
        import re
        import shutil
        import subprocess

        _DEADZONE  = 0.05
        _AXIS_MAX  = 32767.0
        _LT_THRESH = 64          # LT > this → emergency stop

        if not shutil.which('xboxdrv'):
            print("[PlannerViewer] xboxdrv not found — gamepad disabled")
            return

        _pat = re.compile(
            r'Y1:\s*(-?\d+).*?X2:\s*(-?\d+)'
            r'.*?RB:(\d)'
            r'.*?LT:\s*(\d+).*?RT:\s*(\d+)'
        )

        try:
            proc = subprocess.Popen(
                ['xboxdrv', '--no-uinput', '-v'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge stderr into stdout so we catch state lines
                text=True,
            )
        except Exception as e:
            print(f"[PlannerViewer] xboxdrv launch failed: {e}")
            return

        print("[PlannerViewer] Gamepad connected (Xbox 360 via xboxdrv)")

        prev_rb = 0
        prev_rt = 0

        for line in proc.stdout:
            m = _pat.search(line)
            if not m:
                continue

            y1  = int(m.group(1)) / _AXIS_MAX   # left  stick Y → throttle
            x2  = int(m.group(2)) / _AXIS_MAX   # right stick X → steering
            rb  = int(m.group(3))               # RB (R1) → record toggle
            lt  = int(m.group(4))               # LT      → emergency stop
            rt  = int(m.group(5))               # RT (R2) → pause toggle

            # ── Throttle: left stick Y (up = forward, down = reverse) ─────
            thr = 0.0 if abs(y1) < _DEADZONE or lt > _LT_THRESH \
                  else y1 * self.FULL_THROTTLE
            with self._lock:
                self._throttle = round(max(-self.FULL_THROTTLE, min(self.FULL_THROTTLE, thr)), 3)

            # ── Steering: right stick X ───────────────────────────────────
            steer = 0.0 if abs(x2) < _DEADZONE else round(x2, 3)
            with self._lock:
                self._steering = max(-1.0, min(1.0, steer))

            # ── Buttons: detect 0→1 press transition ─────────────────────
            rb_pressed = rb == 1 and prev_rb == 0
            rt_pressed = rt > _LT_THRESH and prev_rt <= _LT_THRESH

            if rb_pressed:
                with self._lock:
                    self._recording = True
                print("[Gamepad] Recording ON")

            if rt_pressed:
                with self._lock:
                    self._recording = False
                print("[Gamepad] Recording OFF")

            prev_rb = rb
            prev_rt = rt

        proc.wait()
        print("[PlannerViewer] Gamepad disconnected")

    # ── HTTP server ───────────────────────────────────────────────────────────

    def _run_http(self):
        # Patch the HTML with the actual WS port before serving
        global _HTML
        _HTML = _HTML.replace('WS_PORT = ' + str(8083),
                              'WS_PORT = ' + str(self._ws_port))
        self._http_server = ThreadingHTTPServer(('0.0.0.0', self._http_port), _HTTPHandler)
        print(f"[PlannerViewer-HTTP] Listening on port {self._http_port}")
        self._http_server.serve_forever()

    # ── WebSocket server ──────────────────────────────────────────────────────

    def _run_ws(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def _start():
            server = await websockets.serve(
                self._ws_handler, '0.0.0.0', self._ws_port,
                ping_interval=20, ping_timeout=20,
            )
            print(f"[PlannerViewer-WS] Listening on port {self._ws_port}")
            self._ws_ready = True
            return server

        try:
            self._loop.run_until_complete(_start())
            self._loop.run_forever()
        except Exception as e:
            print(f"[PlannerViewer-WS] Error: {e}")
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    async def _ws_handler(self, websocket):
        with self._lock:
            self._clients.add(websocket)
        try:
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                    if msg.get('type') == 'control':
                        raw_steer = float(msg.get('steering', 0.0))
                        raw_thr   = float(msg.get('throttle', 0.0))
                        with self._lock:
                            self._steering = max(-1.0, min(1.0, raw_steer))
                            self._throttle = max(-self.FULL_THROTTLE, min(self.FULL_THROTTLE, raw_thr))
                    elif msg.get('type') == 'record_toggle':
                        with self._lock:
                            self._recording = not self._recording
                        state = 'ON' if self._recording else 'OFF'
                        print(f"[PlannerViewer] Recording {state}")
                    elif msg.get('type') == 'pause_toggle':
                        with self._lock:
                            self._paused = not self._paused
                        state = 'PAUSED' if self._paused else 'RUNNING'
                        print(f"[PlannerViewer] {state}")
                    elif msg.get('type') == 'scenario_change':
                        sc = int(msg.get('value', 0))
                        if 0 <= sc <= 5:
                            with self._lock:
                                self._scenario = sc
                            from planner_model import SCENARIO_NAMES
                            print(f"[PlannerViewer] Scenario → {sc} ({SCENARIO_NAMES.get(sc, sc)})")
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.discard(websocket)
