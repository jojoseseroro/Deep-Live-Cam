"""Enhancer wrapper to apply GFPGAN / Real-ESRGAN or fallback.

Provides an interface apply_enhancer(patch, model='gfpgan', strength=0.5, region_mask=None)
which returns an enhanced patch. If models are not installed, returns original.
This implementation supports optional region-limited enhancement: the enhancer
is applied to the full patch if required by the model, but only the masked
areas are blended into the original according to strength.
"""
from __future__ import annotations

import time
from typing import Tuple


def apply_enhancer(patch, model: str = 'gfpgan', strength: float = 0.5, region_mask=None):
    """Apply enhancer to a single patch. strength in [0,1] controls interpolation.

    If region_mask is provided (HxW float32 0..1), the enhancer's output is
    blended into the patch only in the masked region according to strength.

    If the requested model isn't available, this is a no-op.
    """
    try:
        import numpy as np
        # If no enhancement requested, return original
        if strength <= 0.0:
            return patch
        # Lazy imports for optional enhancers
        if model.lower() in ('gfpgan', 'gfpganv1'):
            try:
                # Attempt to import GFPGAN interface if present
                # This is conservative: if import fails, fallback to no-op
                from gfpgan import GFPGANer  # type: ignore
                enhancer = GFPGANer()
                _, enhanced, _ = enhancer.enhance(patch, has_aligned=False, only_center_face=False, paste_back=False)
                # If region_mask provided, blend only masked area
                if region_mask is not None:
                    # Resize mask to patch if needed
                    if region_mask.shape != patch.shape[:2]:
                        import cv2
                        mask_rs = cv2.resize(region_mask, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
                    else:
                        mask_rs = region_mask
                    alpha = np.expand_dims(np.clip(mask_rs.astype('float32') * strength, 0.0, 1.0), axis=2)
                    out = (enhanced.astype('float32') * alpha + patch.astype('float32') * (1.0 - alpha)).astype('uint8')
                    return out
                else:
                    out = (enhanced.astype('float32') * strength + patch.astype('float32') * (1.0 - strength)).astype('uint8')
                    return out
            except Exception:
                # GFPGAN not available or failed — fallback
                return patch
        elif model.lower().startswith('realesrgan') or model.lower() == 'realesrgan':
            try:
                # Real-ESRGAN integration placeholder — actual integration requires package
                # For safety, return patch
                return patch
            except Exception:
                return patch
        else:
            return patch
    except Exception:
        return patch
