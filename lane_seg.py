#!/usr/bin/env python3
"""
LaneSeg — Thin BiSeNet V2 wrapper for direct lane segmentation.

Loads the BiSeNet model directly from ads-skynet/lane-detection-dl/ and
runs inference, returning the raw segmentation mask for use in build_lane_grid().

Replaces the LKAS dependency for lane segmentation in the planner pipeline.

Usage:
    from lane_seg import LaneSeg
    seg  = LaneSeg()                  # loads BiSeNet, auto-selects GPU/CPU
    mask = seg.infer(frame_bgr)       # (H, W) uint8 — 0=background, 1=lane
"""

import sys
import numpy as np
import cv2
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# v11/ is at  ads-skynet/planner/realsense_cam/v11/
# BiSeNet is at  ads-skynet/lane-detection-dl/
# ─────────────────────────────────────────────────────────────────────────────
_V11_DIR        = Path(__file__).resolve().parent
_BISENET_ROOT   = _V11_DIR.parent.parent.parent / "lane-detection-dl"
_MODEL_WEIGHTS  = _BISENET_ROOT / "inference" / "bisenet-0204.pth"
_MODEL_CODE_DIR = str(_BISENET_ROOT)

# Model input size — must match what bisenet-0204.pth was trained on
_INPUT_H = 512
_INPUT_W = 1024

# ImageNet normalisation
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LaneSeg:
    """
    Wraps BiSeNet V2 for lane segmentation.

    The model is loaded once and reused across frames. The returned mask has
    the same spatial size as the input frame.

    Args:
        device:  "auto" → cuda:0 if available, else cpu.
                 Pass "cpu" or "cuda:0" to override.
        weights: Path to .pth checkpoint.  Defaults to bisenet-0204.pth.
    """

    def __init__(self, device: str = "auto", weights: str | None = None):
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Make the BiSeNet model code importable
        if _MODEL_CODE_DIR not in sys.path:
            sys.path.insert(0, _MODEL_CODE_DIR)

        from model.bisenetv2 import BiSeNetV2  # noqa: PLC0415

        weights_path = Path(weights) if weights else _MODEL_WEIGHTS
        if not weights_path.exists():
            raise FileNotFoundError(
                f"[LaneSeg] BiSeNet weights not found: {weights_path}\n"
                f"          Expected: {_MODEL_WEIGHTS}"
            )

        print(f"[LaneSeg] Loading: {weights_path}")
        self.model = BiSeNetV2(n_classes=2, aux_mode='eval')

        ckpt = torch.load(str(weights_path), map_location=self.device,
                          weights_only=False)
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"[LaneSeg] Ready on {self.device}  (input {_INPUT_H}×{_INPUT_W})")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Run BiSeNet on one BGR frame.

        Args:
            frame_bgr: (H, W, 3) uint8  — OpenCV BGR format

        Returns:
            mask: (H, W) uint8  — 0 = background, 1 = lane
                  Same spatial size as the input frame.
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # BGR → RGB, resize to model input
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_in  = cv2.resize(img_rgb, (_INPUT_W, _INPUT_H))

        # Normalise
        img_f  = img_in.astype(np.float32) / 255.0
        img_f  = (img_f - _MEAN) / _STD
        tensor = (torch.from_numpy(img_f.transpose(2, 0, 1))
                  .unsqueeze(0)
                  .to(self.device))

        # Inference
        out    = self.model(tensor)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        pred   = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Resize mask back to original frame size
        if (orig_h, orig_w) != (_INPUT_H, _INPUT_W):
            pred = cv2.resize(pred, (orig_w, orig_h),
                              interpolation=cv2.INTER_NEAREST)
        return pred
