"""Attempt to monkey-patch the face swapper processor to inject advanced
mask-based blending at runtime.

This module is imported on startup (run.py) so that we can wrap existing
functions without changing the original source. The wrapper is conservative:
- it only patches functions if they exist (swap_face, apply_post_processing)
- it uses available landmarks from Face objects if present; otherwise the
  wrapper is a no-op.
"""
from __future__ import annotations

import sys
import traceback

try:
    import modules.advanced_face_controls as afc
    import modules.advanced_controls_config as acc
    import modules.processors.frame.face_swapper as fs
except Exception:
    # If import fails, we can't patch; just exit silently
    # Print traceback for debug purposes
    traceback.print_exc()
    fs = None


def _safe_get_landmarks(face):
    # face may be an insightface.app.common.Face with .kps or .landmark_2d_106
    try:
        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            return [(float(x), float(y)) for x, y in face.landmark_2d_106]
        if hasattr(face, "kps") and face.kps is not None:
            return [(float(x), float(y)) for x, y in face.kps]
        # older field
        if hasattr(face, "bbox") and face.bbox is not None:
            # fallback: build crude landmarks from bbox
            x0, y0, x1, y1 = face.bbox[:4]
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    except Exception:
        pass
    return []


if fs is not None:
    # Patch apply_post_processing if present
    if hasattr(fs, "apply_post_processing"):
        _orig_post = fs.apply_post_processing

        def _patched_apply_post_processing(frame, swapped_bboxes):
            # Call original first — in many implementations this may do color/overlay
            try:
                res = _orig_post(frame, swapped_bboxes)
            except Exception:
                res = frame
            try:
                cfg = acc.get_current()
                if not cfg.get("advanced_enabled", True):
                    return res
                # If there are swapped_bboxes, we need associated face objects.
                # The face_swapper context may have saved last detected faces on the frame.
                # Try to discover faces via face analyser as a fallback.
                from modules.face_analyser import get_many_faces, get_one_face
                faces = None
                try:
                    faces = get_many_faces(res)
                except Exception:
                    faces = None
                # For each detected face, attempt to build a mask and apply blending.
                if faces:
                    for face in faces:
                        lm = _safe_get_landmarks(face)
                        if not lm:
                            continue
                        h, w = res.shape[:2]
                        strengths = {}
                        # Build strengths dict from config
                        for k, v in cfg.items():
                            if isinstance(v, (int, float)):
                                strengths[k] = v / 100.0 if k not in ("blending_strength", "color_correction_strength", "enhancer_strength", "temporal_smoothing") else float(v)
                        # Compose mask
                        mask = afc.build_composite_alpha((h, w), lm, strengths, mask_size=cfg.get("mask_size", 100) / 100.0, mask_feather=int(cfg.get("mask_feathering", 16)))
                        # Create a swapped_face region — attempt to extract from frame using face.bbox
                        try:
                            x0, y0, x1, y1 = [int(round(p)) for p in face.bbox[:4]]
                            # clamp
                            x0 = max(0, min(x0, w - 1))
                            x1 = max(0, min(x1, w - 1))
                            y0 = max(0, min(y0, h - 1))
                            y1 = max(0, min(y1, h - 1))
                            target_face_region = res[y0:y1, x0:x1]
                            # For now we use the original frame as swapped_face placeholder —
                            # the actual swapped output may already be in res depending on processing order.
                            # If face_swapper provides a cached swapped patch we could use it.
                            swapped_face_region = target_face_region.copy()
                            # Apply blending for this region: expand mask to full image and blend
                            blended = afc.apply_region_blend(swapped_face_region, target_face_region, mask[y0:y1, x0:x1])
                            res[y0:y1, x0:x1] = blended
                        except Exception:
                            # fallback: skip region
                            continue
                return res
            except Exception:
                traceback.print_exc()
                return res

        fs.apply_post_processing = _patched_apply_post_processing

    # Patch swap_face if present — this is riskier; skip for now unless needed.

else:
    # face swapper module not available: nothing to patch
    pass
