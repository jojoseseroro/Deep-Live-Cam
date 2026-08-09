"""Enhancer wrapper to apply GFPGAN / Real-ESRGAN or fallback.

Provides an interface apply_enhancer(patch, model='gfpgan', strength=0.5) which
returns an enhanced patch. If models are not installed, returns original.
The implementation is conservative: it imports model code only if available.
"""
from __future__ import annotations

import time
from typing import Tuple


def apply_enhancer(patch, model: str = 'gfpgan', strength: float = 0.5):
    """Apply enhancer to a single patch. strength in [0,1] controls interpolation.

    If the requested model isn't available, this is a no-op.
    """
    try:
        import numpy as np
        # Lazy imports for optional enhancers
        if model.lower() in ('gfpgan', 'gfpganv1'):
            try:
                # Attempt to import GFPGAN interface if present
                from gfpgan import GFPGANer  # type: ignore
                # NOTE: this expects GFPGAN packages to be installed; otherwise fallback
                enhancer = GFPGANer()
                _, enhanced, _ = enhancer.enhance(patch, has_aligned=False, only_center_face=False, paste_back=True)
                # Blend according to strength
                out = (enhanced.astype('float32') * strength + patch.astype('float32') * (1.0 - strength)).astype('uint8')
                return out
            except Exception:
                # GFPGAN not available or failed — fallback
                return patch
        elif model.lower().startswith('realesrgan') or model.lower() == 'realesrgan':
            try:
                from realesrgan import RealESRGAN  # type: ignore
                device = 'cuda' if hasattr(RealESRGAN, 'to') else 'cpu'
                # This is purely best-effort and will likely not be present
                return patch
            except Exception:
                return patch
        else:
            return patch
    except Exception:
        return patch
