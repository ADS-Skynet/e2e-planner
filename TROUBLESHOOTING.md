# Troubleshooting

---

## Gamepad (Xbox 360 Controller)

### Background

The Jetson JetPack 6 kernel (`5.15.148-tegra`) is missing two modules required for standard gamepad support:

| Module | Purpose | Status |
|--------|---------|--------|
| `xpad` | Xbox 360 USB driver | **not in kernel** |
| `uinput` | Virtual input device creation | **not in kernel** |

Without these, plugging in the controller produces no `/dev/input/js*` device and `sudo modprobe xpad` fails:

```
modprobe: FATAL: Module xpad not found in directory /lib/modules/5.15.148-tegra
```

### Solution — xboxdrv (userspace driver)

`xboxdrv` is a userspace driver that accesses the controller directly via USB (libusb), bypassing the kernel driver entirely. It works on this kernel.

**Install:**
```bash
sudo apt install xboxdrv
```

**Verify the controller is detected** (requires sudo initially):
```bash
sudo xboxdrv --no-uinput -v
# Expected output:
# Controller:        Microsoft X-Box 360 pad
# Vendor/Product:    045e:028e
# X1:     0 Y1:     0  X2:     0 ...
```

If you see `LIBUSB_ERROR_ACCESS` instead of the state line, continue to the udev step below.

---

### Fix — USB Permission (LIBUSB_ERROR_ACCESS)

`xboxdrv` needs read+write access to the USB device. By default the device is owned by root (`crw-rw-r--`). The fix is a udev rule that sets `MODE="0666"` on plug-in.

**Run once:**
```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="045e", ATTRS{idProduct}=="028e", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/99-xbox.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**Unplug and replug** the controller (the rule only applies on new plug-in).

**Verify:**
```bash
# Find the device — number may be different each time
ls -la /dev/bus/usb/001/
# Look for the Xbox entry — should now show crw-rw-rw- (0666)

# Confirm xboxdrv works without sudo
xboxdrv --no-uinput -v
# Should print state line without LIBUSB_ERROR_ACCESS
```

---

### How the planner uses the gamepad

`planner_viewer.py` spawns `xboxdrv --no-uinput -v` as a background subprocess and parses its stdout. No `/dev/input/js*` or kernel modules required.

**Control mapping (Xbox 360):**

| Input | Action |
|-------|--------|
| Left stick Y (push up) | Throttle 0 → 0.45 |
| Right stick X | Steering −1.0 → +1.0 |
| LT | Emergency throttle = 0 |
| RB (R1) | Toggle recording ON/OFF |
| RT (R2) | Pause/resume |

Keyboard controls in the browser still work in parallel — use whichever is convenient.

**If the gamepad is not connected or xboxdrv is not installed**, the planner starts normally with keyboard-only control. No error is fatal.

---

## CUDA Out of Memory (OOM)

See [WORKFLOW.md](WORKFLOW.md) — Memory management section.

---

## Web Viewer Infinite Loading (browser tab spinner never stops)

**Symptom:** The browser tab spinner keeps spinning even though the page loads and the stream works.

**Cause:** HTTP responses without `Connection: close` keep the connection open; browsers treat this as "still loading".

**Fix** (already applied in `planner_viewer.py`):
```python
self.send_header('Connection', 'close')
```

Also add a `favicon.ico` handler returning 204 — browsers make a second request for the favicon which otherwise also hangs.

---

## Camera Import Error (enable_depth)

**Symptom:**
```
TypeError: Camera.__init__() got an unexpected keyword argument 'enable_depth'
```

**Cause:** `sys.path.insert(0, ...)` for `vehicle/src` puts it before the local directory, so Python imports the USB camera class from `vehicle/src/camera.py` instead of the local RealSense `camera.py`.

**Fix** (already applied): use `sys.path.append(...)` instead of `sys.path.insert(0, ...)` so local files take priority.

---

## Low FPS During Data Collection (~3 FPS)

**Symptom:** Terminal shows FPS around 3, frame is visibly laggy.

**Cause:** YOLO running synchronously on CPU takes ~300 ms per call, blocking the main loop.

**Fix** (already applied): YOLO runs in a daemon background thread (`_yolo_worker`). The main loop uses the last cached result and is never blocked by YOLO.
