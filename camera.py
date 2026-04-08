"""
RealSense Camera wrapper for v11 planner scripts.

Interface
---------
    cam = Camera(width=768, height=384, enable_depth=True)
    color_bgr, depth_uint16 = cam.read_frames()
    dist_m = depth_uint16[row, col] * cam.depth_scale

color_bgr    : np.ndarray (H, W, 3) uint8, BGR
depth_uint16 : np.ndarray (H, W)    uint16, raw sensor units
               None when enable_depth=False
cam.depth_scale : float  — multiply raw uint16 by this to get metres
                           (typically 0.001 for D435 / D455)
"""

import numpy as np
import pyrealsense2 as rs


class Camera:
    """RealSense D4xx camera with optional aligned depth stream."""

    def __init__(self, width: int = 640, height: int = 480,
                 enable_depth: bool = False, fps: int = 30):
        self._enable_depth = enable_depth

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = self.pipeline.start(cfg)

        if enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale: float = depth_sensor.get_depth_scale()
            self._align = rs.align(rs.stream.color)
        else:
            self.depth_scale: float = 0.001
            self._align = None

        # Warm up — discard the first few frames while exposure settles
        for _ in range(15):
            self.pipeline.wait_for_frames()

        print(f"[Camera] RealSense ready  {width}x{height}  "
              f"depth={'ON  scale=' + str(self.depth_scale) if enable_depth else 'OFF'}")

    def read_frames(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Return (color_bgr, depth_uint16).
        depth_uint16 is None when enable_depth=False.
        Blocks until a frame pair is available.
        """
        frames = self.pipeline.wait_for_frames()

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        color_bgr = np.asanyarray(color_frame.get_data())   # (H, W, 3) uint8 BGR

        depth_uint16 = None
        if self._enable_depth:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_uint16 = np.asanyarray(depth_frame.get_data())  # (H, W) uint16

        return color_bgr, depth_uint16

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def __del__(self):
        self.close()
