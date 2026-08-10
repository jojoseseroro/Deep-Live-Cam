"""Enhancer wrapper to apply GFPGAN / Real-ESRGAN with region-limited processing.

This version attempts to limit enhancer computation to the masked bounding box to
save compute. If enhancer package is not available or the bounding box is too
small, it falls back to whole-patch processing or a no-op.
"""
from __future__ import annotations

import time
from typing import Tuple


def _bbox_from_mask(mask):
    import numpy as np
    ys, xs = np.where(mask > 0.01)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 + 1, y1 + 1


def apply_enhancer(patch, model: str = 'gfpgan', strength: float = 0.5, region_mask=None):
    """Apply enhancer to a patch with optional region_mask.

    If region_mask is provided, try to crop to its bounding box and run enhancer
    on the crop. Then blend enhanced crop back into patch according to mask*strength.

    If enhancer model is not available, return original patch.
    """
    try:
        import numpy as np
        import cv2

        if strength <= 0.0:
            return patch

        # If no region mask or mask is trivial, fall back to full patch enhancement
        if region_mask is None:
            # full patch enhancement
            try:
                from gfpgan import GFPGANer  # type: ignore
                enhancer = GFPGANer()
                _, enhanced, _ = enhancer.enhance(patch, has_aligned=False, only_center_face=False, paste_back=False)
                out = (enhanced.astype('float32') * strength + patch.astype('float32') * (1.0 - strength)).astype('uint8')
                return out
            except Exception:
                return patch

        # Compute bounding box of the mask to limit enhancer scope
        bbox = _bbox_from_mask(region_mask)
        if bbox is None:
            return patch
        x0, y0, x1, y1 = bbox
        # Small region: if area too small, skip enhancer
        h = y1 - y0
        w = x1 - x0
        if h < 16 or w < 16:
            return patch

        # Crop the patch and the mask to bbox coordinates (patch coords)
        crop = patch[y0:y1, x0:x1]
        crop_mask = region_mask[y0:y1, x0:x1]

        # Try enhancer on crop
        try:
            from gfpgan import GFPGANer  # type: ignore
            enhancer_inst = GFPGANer()
            _, enhanced_crop, _ = enhancer_inst.enhance(crop, has_aligned=False, only_center_face=False, paste_back=False)
            # Blend enhanced_crop into original patch using crop_mask
            alpha = np.expand_dims(np.clip(crop_mask.astype('float32') * strength, 0.0, 1.0), axis=2)
            blended_crop = (enhanced_crop.astype('float32') * alpha + crop.astype('float32') * (1.0 - alpha)).astype('uint8')
            out = patch.copy()
            out[y0:y1, x0:x1] = blended_crop
            return out
        except Exception:
            # enhancer not available or failed on crop; fallback to full-patch attempt
            try:
                from gfpgan import GFPGANer  # type: ignore
                enhancer = GFPGANer()
                _, enhanced, _ = enhancer.enhance(patch, has_aligned=False, only_center_face=False, paste_back=False)
                out = (enhanced.astype('float32') * strength + patch.astype('float32') * (1.0 - strength)).astype('uint8')
                return out
            except Exception:
                return patch
    except Exception:
        return patch
