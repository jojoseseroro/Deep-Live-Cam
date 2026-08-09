"""Warp-based paste-back utilities.

Provides functions to warp a source patch to match target landmarks and blend
it back in. Uses an affine approximation via landmark triplets and cv2.estimateAffinePartial2D.
This is a practical, fast method suitable for live use; for higher quality you
could implement piecewise affine or TPS warping.
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple


def _get_mouth_landmark_indices():
    # Using 68-point landmark scheme: mouth outer corners are 48 and 54
    return [48, 54, 57]  # left corner, right corner, bottom center


def estimate_affine_from_landmarks(src_landmarks: List[Tuple[float, float]], dst_landmarks: List[Tuple[float, float]], indices: List[int]) -> np.ndarray | None:
    pts_src = []
    pts_dst = []
    for i in indices:
        if i < 0 or i >= len(src_landmarks) or i >= len(dst_landmarks):
            continue
        pts_src.append(src_landmarks[i])
        pts_dst.append(dst_landmarks[i])
    if len(pts_src) < 3:
        return None
    src = np.array(pts_src, dtype=np.float32)
    dst = np.array(pts_dst, dtype=np.float32)
    # estimate affine (partial) 2x3 matrix
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return M


def warp_patch_to_target(src_patch: np.ndarray, M: np.ndarray, output_shape: Tuple[int, int]) -> np.ndarray:
    h, w = output_shape
    warped = cv2.warpAffine(src_patch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def warp_paste_back(swapped_patch: np.ndarray, orig_patch: np.ndarray, src_landmarks: List[Tuple[float, float]], dst_landmarks: List[Tuple[float, float]], region_mask: np.ndarray, preserve_strength: float = 1.0) -> np.ndarray:
    """Warp orig_patch to swapped_patch geometry and blend according to region_mask.

    src_landmarks: landmarks on the original (target) face used to compute warp (full-frame coords)
    dst_landmarks: landmarks corresponding to the swapped face (full-frame coords)
    region_mask: mask cropped to patch coordinates, float32 [0,1]
    """
    try:
        if swapped_patch is None or orig_patch is None:
            return swapped_patch
        h, w = swapped_patch.shape[:2]
        indices = _get_mouth_landmark_indices()
        # Transform indices from full-frame coords to patch-local coords
        # Expect src_landmarks/dst_landmarks in full-frame coordinates; user should pass proper points
        # For simplicity, we use the absolute coords and compute affine directly
        M = estimate_affine_from_landmarks(src_landmarks, dst_landmarks, indices)
        if M is None:
            # fallback: no warp
            return pasted = _simple_paste_back(swapped_patch, orig_patch, region_mask, preserve_strength)
        # Warp orig_patch to swapped_patch coordinates
        warped = warp_patch_to_target(orig_patch, M, (h, w))
        # Blend
        alpha = region_mask.astype('float32') * float(preserve_strength)
        alpha3 = np.expand_dims(alpha, axis=2)
        out = (warped.astype('float32') * alpha3 + swapped_patch.astype('float32') * (1.0 - alpha3)).astype('uint8')
        return out
    except Exception:
        # fallback simple paste
        return _simple_paste_back(swapped_patch, orig_patch, region_mask, preserve_strength)


def _simple_paste_back(swapped_patch: np.ndarray, orig_patch: np.ndarray, region_mask: np.ndarray, preserve_strength: float = 1.0) -> np.ndarray:
    # Simple resize & blend
    try:
        if swapped_patch.shape != orig_patch.shape:
            import cv2
            orig_rs = cv2.resize(orig_patch, (swapped_patch.shape[1], swapped_patch.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            orig_rs = orig_patch
        import numpy as np
        alpha = region_mask.astype('float32') * float(preserve_strength)
        alpha3 = np.expand_dims(alpha, axis=2)
        blended = (orig_rs.astype('float32') * alpha3 + swapped_patch.astype('float32') * (1.0 - alpha3)).astype('uint8')
        return blended
    except Exception:
        return swapped_patch
