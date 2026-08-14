from typing import Any, List, Optional, Tuple
import cv2
import insightface
import logging
import threading
import numpy as np
import platform
import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import (
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid
from modules.gpu_processing import gpu_gaussian_blur, gpu_sharpen, gpu_add_weighted, gpu_resize
from modules.platform_info import OPENVINO_PROVIDER_CONFIG
from modules.model_manager import get_manager
import os
from collections import deque
import time

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

# --- START: Added for Interpolation ---
PREVIOUS_FRAME_RESULT = None # Stores the final processed frame from the previous step
# --- END: Added for Interpolation ---

# --- Poisson blend (ported from deep-live-cam-gumroad-edition) ---
# Root-cause fix for the "wobble": the blend mask is NOT built from the
# independently-detected 106-pt landmarks (they jitter sub-pixel every frame
# and seamlessClone is hyper-sensitive to its mask boundary). Instead it is
# derived from the swap's OWN affine transform (M) + the swapped pixels
# (bgr_fake), so the mask is locked exactly to where the swapped face was
# placed — no independent jitter source, no EMA, no lag. The mask is cached
# when the face is nearly still so an identical array is reused (zero wobble).
_ELLIPTICAL_MASK_CACHE: dict = {}
_poisson_cached_mask: Optional[np.ndarray] = None
_poisson_cached_key: Optional[tuple] = None


def _create_elliptical_mask(size: Tuple[int, int]) -> np.ndarray:
    """Fixed, heavily-blurred elliptical mask in aligned-face space.

    Geometry-based (not content-adaptive) and cached by size — identical
    every frame for the same model input size, so it contributes no jitter.
    """
    global _ELLIPTICAL_MASK_CACHE
    if size in _ELLIPTICAL_MASK_CACHE:
        return _ELLIPTICAL_MASK_CACHE[size]
    h, w = size
    center = (w // 2, h // 2)
    axes = (int(w * 0.44), int(h * 0.44))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    if h * w < 65536:
        mask = cv2.GaussianBlur(mask, (31, 31), 12)
    else:
        mask = gpu_gaussian_blur(mask, (31, 31), 12)
    _ELLIPTICAL_MASK_CACHE[size] = mask
    return mask


def _apply_poisson_blend(swapped_frame: Frame, original_frame: Frame,
                         target_face: Face, affine_matrix: np.ndarray = None,
                         bgr_fake: np.ndarray = None) -> Frame:
    """Poisson-blend the swapped face onto the original frame.

    Preferred path derives the blend mask from the swap's inverse affine so
    it tracks the swapped face exactly per-frame (no landmark jitter, no
    smoothing). Falls back to a cached bbox-ellipse if the affine is absent.
    Writes only the blended ellipse back so other faces are preserved.
    """
    global _poisson_cached_mask, _poisson_cached_key
    try:
        # ---- Preferred: blend ONLY the genuinely-swapped region ----
        # Use the exact paste-back mask (warped elliptical mask), eroded so
        # the Poisson seam sits on solidly-swapped pixels only.
        if affine_matrix is not None and bgr_fake is not None:
            try:
                h, w = swapped_frame.shape[:2]
                fh, fw = bgr_fake.shape[:2]
                inv = cv2.invertAffineTransform(affine_matrix)
                corners = np.array([[0, 0, 1], [fw, 0, 1], [fw, fh, 1], [0, fh, 1]],
                                   dtype=np.float32)
                t = corners @ inv.T
                px1 = max(0, int(np.floor(t[:, 0].min())))
                py1 = max(0, int(np.floor(t[:, 1].min())))
                px2 = min(w, int(np.ceil(t[:, 0].max())))
                py2 = min(h, int(np.ceil(t[:, 1].max())))
                rw, rh = px2 - px1, py2 - py1
                if rw > 8 and rh > 8:
                    roi_aff = inv.copy()
                    roi_aff[0, 2] -= px1
                    roi_aff[1, 2] -= py1
                    fm = _create_elliptical_mask((fh, fw))
                    mroi = cv2.warpAffine(fm, roi_aff, (rw, rh),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    bin_roi = np.where(mroi > 0.5, np.uint8(255), np.uint8(0))
                    k = max(3, (min(rw, rh) // 20) | 1)
                    bin_roi = cv2.erode(bin_roi,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                    bx, by, bw, bh = cv2.boundingRect(bin_roi)
                    if bw > 0 and bh > 0:
                        mx1, my1 = px1 + bx, py1 + by
                        mx2, my2 = mx1 + bw - 1, my1 + bh - 1
                        # seamlessClone needs the cloned region off the border
                        if mx1 > 0 and my1 > 0 and mx2 < w - 1 and my2 < h - 1:
                            mask = np.zeros((h, w), dtype=np.uint8)
                            mask[py1:py2, px1:px2] = bin_roi
                            center = (mx1 + bw // 2, my1 + bh // 2)
                            blended = cv2.seamlessClone(swapped_frame, original_frame,
                                                        mask, center, cv2.NORMAL_CLONE)
                            np.copyto(swapped_frame[my1:my2 + 1, mx1:mx2 + 1],
                                      blended[my1:my2 + 1, mx1:mx2 + 1],
                                      where=mask[my1:my2 + 1, mx1:mx2 + 1, None].astype(bool))
                            return swapped_frame
            except Exception:
                pass  # fall through to the robust bbox-ellipse path below
        # ---- Fallback: bbox-ellipse (defensive, cached when still) ----
        if not hasattr(target_face, 'bbox') or target_face.bbox is None:
            return swapped_frame
        x1, y1, x2, y2 = target_face.bbox.astype(int)
        h, w = swapped_frame.shape[:2]
        x1, y1 = (max(0, x1), max(0, y1))
        x2, y2 = (min(w, x2), min(h, y2))
        if x2 <= x1 or y2 <= y1 or x2 - x1 <= 10 or (y2 - y1 <= 10):
            return swapped_frame
        padding = int(min(x2 - x1, y2 - y1) * 0.1)
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(w, x2 + padding)
        y2_p = min(h, y2 + padding)
        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        radius_x = max(1, int(round((x2_p - x1_p) / 2.0)))
        radius_y = max(1, int(round((y2_p - y1_p) / 2.0)))
        if not (0 <= center_x < w and 0 <= center_y < h):
            return swapped_frame
        center = (center_x, center_y)
        if center_x - radius_x < 0 or center_x + radius_x >= w or center_y - radius_y < 0 or (center_y + radius_y >= h - 1):
            return swapped_frame
        # Reuse cached mask when center/radius unchanged frame-to-frame
        # (face nearly still) — saves the np.zeros + cv2.ellipse, and the
        # identical array means literally zero wobble while still.
        mask_key = (center_x, center_y, radius_x, radius_y, h, w)
        if _poisson_cached_key == mask_key and _poisson_cached_mask is not None:
            mask = _poisson_cached_mask
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 255, -1)
            if np.sum(mask) == 0:
                return swapped_frame
            _poisson_cached_mask = mask
            _poisson_cached_key = mask_key
        blended = cv2.seamlessClone(swapped_frame, original_frame, mask, center, cv2.NORMAL_CLONE)
        # Composite ONLY this face's ellipse back (ROI-bounded) so previously
        # blended faces in multi-face mode are preserved.
        rx0 = max(0, center_x - radius_x)
        rx1 = min(w, center_x + radius_x + 1)
        ry0 = max(0, center_y - radius_y)
        ry1 = min(h, center_y + radius_y + 1)
        roi_mask = mask[ry0:ry1, rx0:rx1]
        np.copyto(swapped_frame[ry0:ry1, rx0:rx1],
                  blended[ry0:ry1, rx0:rx1],
                  where=roi_mask[:, :, None].astype(bool))
        return swapped_frame
    except Exception:
        return swapped_frame

# --- START: Mac M1-M5 Optimizations ---
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'
FRAME_CACHE = deque(maxlen=3)  # Cache for frame reuse
FACE_DETECTION_CACHE = {}  # Cache face detections
LAST_DETECTION_TIME = 0
DETECTION_INTERVAL = 0.033  # ~30 FPS detection rate for live mode
FRAME_SKIP_COUNTER = 0
ADAPTIVE_QUALITY = True
# --- END: Mac M1-M5 Optimizations ---

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    try:
        os.makedirs(models_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {models_dir} due to permission error: {e}")
        update_status(f"Failed to prepare models directory: {models_dir}", NAME)
        return False

    manager = get_manager()
    status, message = manager.ensure_model('inswapper_128')
    if status in ('available', 'downloaded'):
        update_status(f"inswapper_128 model ready: {message}", NAME)
        return True

    update_status(f"inswapper_128 model unavailable: {message}", NAME)
    return False


def pre_start() -> bool:
    # Check for either model variant
    fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
    if not os.path.exists(fp16_path) and not os.path.exists(fp32_path):
        update_status(f"Model not found in {models_dir}. Please download inswapper_128.onnx.", NAME)
        return False

    # Try to get the face swapper to ensure it loads correctly
    if get_face_swapper() is None:
        # Error message already printed within get_face_swapper
        return False

    return True


_HAS_TORCH_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        _HAS_TORCH_CUDA = True
except ImportError:
    pass

# ... rest of file unchanged ...
