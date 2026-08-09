import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

# Advanced face-region controls utilities.
# This module provides functions to create region masks from facial landmarks
# and to combine per-region strength settings into a single alpha blending mask
# that can be applied to composite a swapped face into the target frame.

# NOTE: Landmark indices below are approximate groupings and may need
# adjustment to match the specific landmark layout of the face detector
# models used by Deep-Live-Cam (insightface / 2d106). They are provided
# as a practical starting point.

REGION_GROUPS = {
    "forehead": [19, 20, 21, 22, 23, 24, 25],
    "eyebrows": [17, 18, 19, 20, 21, 22, 23, 24],
    "eyes": [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    "upper_eyelids": [37, 38, 43, 44],
    "lower_eyelids": [40, 41, 46, 47],
    "nose": [27, 28, 29, 30, 31, 32, 33, 34, 35],
    "cheeks": [2, 3, 4, 48, 31, 35],
    "cheekbones": [1, 2, 14, 15],
    "mouth": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    "jaw": list(range(0, 17)),
    "chin": [8, 9, 10],
    "face_shape": list(range(0, 17)),
    "skin_texture": list(range(0, 68)),
    "color_matching": list(range(0, 68)),
    "hairline": [17, 18, 19, 20, 21],
}

DEFAULT_REGION_ORDER = [
    "forehead",
    "eyebrows",
    "eyes",
    "upper_eyelids",
    "lower_eyelids",
    "nose",
    "cheeks",
    "cheekbones",
    "mouth",
    "jaw",
    "chin",
    "face_shape",
    "skin_texture",
    "color_matching",
    "hairline",
]


def landmarks_to_points(landmarks: List[Tuple[float, float]]) -> np.ndarray:
    """Convert list of (x,y) landmarks to numpy array of ints (Nx2)."""
    pts = np.array(landmarks, dtype=np.float32)
    return pts.reshape(-1, 2)


def polygon_from_indices(landmarks: List[Tuple[float, float]], indices: List[int]) -> np.ndarray:
    pts = []
    for i in indices:
        if i < 0 or i >= len(landmarks):
            continue
        pts.append(landmarks[i])
    if not pts:
        return np.empty((0, 2), dtype=np.int32)
    poly = np.array(pts, dtype=np.int32)
    return poly


def create_region_mask(frame_shape: Tuple[int, int], polygons: List[np.ndarray], feather: int = 15) -> np.ndarray:
    """Create combined mask from a list of polygons with feathering.

    frame_shape: (h, w)
    polygons: list of int32 Nx2 arrays
    feather: gaussian blur radius (px)
    returns: float32 mask in [0,1]
    """
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        if poly.size == 0:
            continue
        try:
            cv2.fillPoly(mask, [poly], 255)
        except Exception:
            # fallback: ignore malformed polygon
            continue
    if feather > 0:
        k = max(1, feather // 2 * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=0)
    mask_f = mask.astype(np.float32) / 255.0
    return mask_f


def build_composite_alpha(
    frame_shape: Tuple[int, int],
    landmarks: List[Tuple[float, float]],
    strengths: Dict[str, float],
    mask_size: float = 1.0,
    mask_feather: int = 15,
) -> np.ndarray:
    """Build a composite alpha mask for blending based on per-region strengths.

    strengths: mapping region -> strength (0.0 - 1.0)
    mask_size: scaling for mask extents (not used heavily here, placeholder)
    mask_feather: blur radius in px
    """
    h, w = frame_shape
    polygons = []
    alpha_acc = np.zeros((h, w), dtype=np.float32)
    weight_acc = np.zeros((h, w), dtype=np.float32)

    # For each region, build a mask and add weighted contribution
    for region in DEFAULT_REGION_ORDER:
        indices = REGION_GROUPS.get(region, [])
        poly = polygon_from_indices(landmarks, indices)
        if poly.size == 0:
            continue
        region_mask = create_region_mask((h, w), [poly], feather=mask_feather)
        strength = float(strengths.get(region, 1.0))
        alpha_acc += region_mask * strength
        weight_acc += region_mask

    # Avoid division by zero
    mask = np.zeros_like(alpha_acc)
    nonzero = weight_acc > 0
    mask[nonzero] = alpha_acc[nonzero] / weight_acc[nonzero]

    # Clamp to [0,1]
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def apply_region_blend(swapped_face: np.ndarray, target_frame: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """Blend swapped_face onto target_frame using alpha_mask.

    swapped_face and target_frame must be same shape (h,w,3) and dtype uint8.
    alpha_mask is float32 in [0,1] shape (h,w).
    """
    if swapped_face.shape != target_frame.shape:
        # resize swapped_face to target_frame size
        swapped = cv2.resize(swapped_face, (target_frame.shape[1], target_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        swapped = swapped_face
    alpha = np.expand_dims(alpha_mask, axis=2)
    blended = (swapped.astype(np.float32) * alpha + target_frame.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return blended


class TemporalMaskCache:
    """Cache previous masks and provide exponential smoothing for temporal stability."""

    def __init__(self, smoothing: float = 0.6):
        # smoothing in [0,1], higher = more smoothing (keep past)
        self.smoothing = float(smoothing)
        self.prev_mask: np.ndarray | None = None

    def smooth(self, mask: np.ndarray) -> np.ndarray:
        if self.prev_mask is None:
            self.prev_mask = mask.copy()
            return mask
        smoothed = self.prev_mask * self.smoothing + mask * (1.0 - self.smoothing)
        self.prev_mask = smoothed
        return smoothed


# Utility: convert user-provided percentages (0-100) to 0-1 floats

def normalize_strengths(percentages: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in percentages.items():
        try:
            val = float(v) / 100.0
        except Exception:
            val = 0.0
        out[k] = float(np.clip(val, 0.0, 1.0))
    return out
