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

Controls (browser)
  ←  steer left   (held)
  →  steer right  (held)
  ↓  stop         (held)
  releasing any key resets that axis to 0

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
import threading
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
  #ctrl  { margin-top:6px; font-size:0.75rem; color:#555; }
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
<div id="ctrl">← / → steer (hold) &nbsp;|&nbsp; ↑ throttle +0.1 &nbsp; ↓ throttle = 0 &nbsp;|&nbsp; Space = record &nbsp;|&nbsp; P = pause</div>

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

// ── Key controls ────────────────────────────────────────────────────────────
const STEER_VAL     = 1.0;
const THROTTLE_STEP = 0.1;
const THROTTLE_MAX  = 1.0;

let steering  = 0.0;
let throttle  = 0.0;  // default 0 — ↑ adds 0.1 per press, ↓ resets to 0
let recording = false;
let paused    = false;

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

document.addEventListener('keydown', (e) => {
  let changed = false;
  if      (e.key === 'ArrowLeft')  { steering = -STEER_VAL; changed = true; }
  else if (e.key === 'ArrowRight') { steering =  STEER_VAL; changed = true; }
  else if (e.key === 'ArrowUp') {
    throttle = Math.min(THROTTLE_MAX, Math.round((throttle + THROTTLE_STEP) * 10) / 10);
    changed = true;
  }
  else if (e.key === 'ArrowDown')  { throttle = 0.0; changed = true; }
  else if (e.key === ' ')          { sendRecording(); e.preventDefault(); return; }
  else if (e.key === 'p' || e.key === 'P') { togglePause(); e.preventDefault(); return; }
  if (changed) { e.preventDefault(); sendControl(); }
});

document.addEventListener('keyup', (e) => {
  let changed = false;
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { steering = 0.0; changed = true; }
  // ↑ and ↓ are one-shot (no reset on keyup) — throttle holds its value
  if (changed) { e.preventDefault(); sendControl(); }
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

    def log_message(self, fmt, *args):
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

    STEER_VALUE = 0.9   # magnitude applied when ← or → held

    def __init__(self, http_port: int = 8082, ws_port: int = 8083):
        self._http_port = http_port
        self._ws_port   = ws_port

        self._lock      = Lock()
        self._steering  = 0.0   # raw from browser: -1/0/+1
        self._throttle  = 0.0   # 0–1; ↑ adds 0.1 per press, ↓ resets to 0
        self._recording = False  # toggled by Space
        self._paused    = False  # toggled by P / pause button

        self._clients:  Set = set()
        self._loop:     Optional[asyncio.AbstractEventLoop] = None
        self._ws_ready  = False

        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[Thread] = None
        self._ws_thread:   Optional[Thread] = None

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

        print(f"[PlannerViewer] http://0.0.0.0:{self._http_port}   ws://0.0.0.0:{self._ws_port}")

    def stop(self):
        if self._http_server:
            try:
                self._http_server.shutdown()
            except Exception:
                pass
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

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
                            self._throttle = max(0.0,  min(1.0, raw_thr))
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
                except (json.JSONDecodeError, ValueError):
                    pass
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.discard(websocket)
