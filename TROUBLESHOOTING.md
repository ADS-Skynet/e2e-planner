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

## PyTorch install on JetPack 6.x

**Do not** `pip install torch` from PyPI — those wheels have no CUDA support on Jetson aarch64.

**Step 1 — install from Jetson AI Lab:**
```bash
python3 -m pip install torch torchvision --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

**Step 2 — fix `ImportError: libcudss.so.0`**

torch ≥ 2.8 requires the CUDA Sparse Direct Solver library (`libcudss`), which JetPack 6.x does not install by default:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install libcudss0-cuda-12
```

The package installs the `.so` under a non-standard path, so `ldconfig` needs to be told about it:

```bash
echo "/usr/lib/aarch64-linux-gnu/libcudss/12" | sudo tee /etc/ld.so.conf.d/cudss.conf
sudo ldconfig
```

Verify everything works:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 2.10.0 True
```

Tested: torch 2.10.0 / torchvision 0.25.0 / JetPack 6.2 / CUDA 12.6

---

## RealSense Camera on JetPack 6.x

The pre-built `pyrealsense2` wheels on PyPI are compiled for x86 and do **not** work on Jetson aarch64. The Intel librealsense SDK must be built from source with the Python wheel flag enabled.

### Build librealsense from source

**Step 1 — install build dependencies:**
```bash
sudo apt-get install -y \
  git cmake build-essential \
  libusb-1.0-0-dev libssl-dev \
  libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
  python3-dev
```

**Step 2 — clone the repo:**
```bash
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
```

**Step 3 — build with Python bindings:**
```bash
mkdir build && cd build
cmake .. \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

The `-DBUILD_PYTHON_BINDINGS=ON` flag compiles `pyrealsense2` as a `.so` and installs it into the system Python path.

**Step 4 — verify:**
```bash
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

### udev rules (required for non-root access)

Without udev rules the camera is only accessible as root:
```bash
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Unplug and replug the camera after applying the rules.

### Verify the camera is detected

```bash
rs-enumerate-devices
# Should list your RealSense model, serial number, and supported streams
```

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
