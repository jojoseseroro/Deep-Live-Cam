from __future__ import annotations

import traceback

try:
    import numpy as np
    import modules.advanced_face_controls as afc
    import modules.advanced_controls_config as acc
    import modules.processors.frame.face_swapper as fs
    import modules.advanced_swap_cache as asc
    import modules.pasteback as pb
    import modules.blink_detector as bd
    import modules.face_state as fs_state
    from modules.face_analyser import get_many_faces, get_one_face
except Exception:
    traceback.print_exc()
    fs = None


def _safe_get_landmarks(face):
    try:
        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            return [(float(x), float(y)) for x, y in face.landmark_2d_106]
        if hasattr(face, "kps") and face.kps is not None:
            return [(float(x), float(y)) for x, y in face.kps]
        if hasattr(face, "bbox") and face.bbox is not None:
            x0, y0, x1, y1 = face.bbox[:4]
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    except Exception:
        pass
    return []


if fs is not None:
    # Wrap swap_face to capture swapped patch
    if hasattr(fs, "swap_face"):
        _orig_swap = fs.swap_face

        def _wrapped_swap_face(source_image, t_face, result_frame):
            try:
                h, w = result_frame.shape[:2]
                if hasattr(t_face, 'bbox') and t_face.bbox is not None:
                    x0, y0, x1, y1 = [int(round(p)) for p in t_face.bbox[:4]]
                    x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1))
                    y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1))
                    orig_patch = result_frame[y0:y1, x0:x1].copy()
                else:
                    orig_patch = None
            except Exception:
                orig_patch = None

            # Call original swapper
            try:
                new_res = _orig_swap(source_image, t_face, result_frame)
            except Exception:
                return result_frame

            # After swap, extract swapped patch and store in cache
            try:
                if hasattr(t_face, 'bbox') and t_face.bbox is not None:
                    x0, y0, x1, y1 = [int(round(p)) for p in t_face.bbox[:4]]
                    x0 = max(0, min(x0, new_res.shape[1]-1)); x1 = max(0, min(x1, new_res.shape[1]-1))
                    y0 = max(0, min(y0, new_res.shape[0]-1)); y1 = max(0, min(y1, new_res.shape[0]-1))
                    swapped_patch = new_res[y0:y1, x0:x1].copy()
                    if swapped_patch is not None and orig_patch is not None:
                        asc.set_patch((x0, y0, x1, y1), orig_patch, swapped_patch)
            except Exception:
                pass

            return new_res

        fs.swap_face = _wrapped_swap_face

    # Patch apply_post_processing to use cached swapped patches and advanced masks
    if hasattr(fs, "apply_post_processing"):
        _orig_post = fs.apply_post_processing

        def _patched_apply_post_processing(frame, swapped_bboxes):
            try:
                res = _orig_post(frame, swapped_bboxes)
            except Exception:
                res = frame
            try:
                cfg = acc.get_current()
                if not cfg.get('advanced_enabled', True):
                    return res

                # detect faces in the final frame to align operations
                faces = None
                try:
                    faces = get_many_faces(res)
                except Exception:
                    faces = None

                # publish face state for UI overlay
                try:
                    fs_state.set_last_faces(faces or [])
                except Exception:
                    pass

                if faces:
                    h, w = res.shape[:2]
                    for face in faces:
                        lm = _safe_get_landmarks(face)
                        if not lm:
                            continue

                        # Build strengths mapping from config
                        strengths = {}
                        for k, v in cfg.items():
                            if isinstance(v, (int, float)):
                                strengths[k] = v / 100.0 if k not in ("blending_strength", "color_correction_strength", "enhancer_strength", "temporal_smoothing") else float(v)

                        # Compose full-frame composite mask
                        mask = afc.build_composite_alpha((h, w), lm, strengths, mask_size=cfg.get('mask_size', 100)/100.0, mask_feather=int(cfg.get('mask_feathering', 16)))

                        # retrieve cached swapped patch by bbox
                        try:
                            x0, y0, x1, y1 = [int(round(p)) for p in face.bbox[:4]]
                            x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1))
                            y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1))
                            rec = asc.get_patch((x0, y0, x1, y1))
                        except Exception:
                            rec = None

                        if rec is None:
                            continue

                        orig_patch = rec['orig']
                        swapped_patch = rec['swapped']

                        # mouth preservation
                        try:
                            mouth_pres = cfg.get('mouth_movement_preservation', 75) / 100.0
                            mouth_mask_full = afc.build_composite_alpha((h, w), lm, {'mouth': 1.0}, mask_size=cfg.get('mask_size', 100)/100.0, mask_feather=int(cfg.get('mask_feathering', 16)))
                            mouth_mask = mouth_mask_full[y0:y1, x0:x1]
                            if mouth_pres > 0.0:
                                # Attempt to compute swapped-landmarks for better warp alignment
                                swapped_lm = None
                                try:
                                    sw_face = get_one_face(swapped_patch)
                                    if sw_face is not None:
                                        swapped_lm_raw = _safe_get_landmarks(sw_face)
                                        # transform swapped landmarks to full-frame coords
                                        swapped_lm = [(x + x0, y + y0) for x, y in swapped_lm_raw]
                                except Exception:
                                    swapped_lm = None

                                try:
                                    if swapped_lm:
                                        swapped_patch = pb.warp_paste_back(swapped_patch, orig_patch, src_landmarks=lm, dst_landmarks=swapped_lm, region_mask=mouth_mask, preserve_strength=mouth_pres)
                                    else:
                                        # fallback simple paste-back
                                        swapped_patch = afc.paste_back_region(swapped_patch, orig_patch, mouth_mask, mouth_pres)
                                except Exception:
                                    swapped_patch = afc.paste_back_region(swapped_patch, orig_patch, mouth_mask, mouth_pres)
                        except Exception:
                            pass

                        # eye/eyelid preservation with blink-awareness
                        try:
                            eye_pres = cfg.get('eye_blink_preservation', 80) / 100.0
                            is_blink = bd.is_blinking(lm)
                            if is_blink:
                                # If blinking, favor preserving the live eye pixels strongly
                                eye_pres = max(eye_pres, 0.95)

                            eye_mask_full = afc.build_composite_alpha((h, w), lm, {'upper_eyelids': 1.0, 'lower_eyelids': 1.0, 'eyes': 1.0}, mask_size=cfg.get('mask_size', 100)/100.0, mask_feather=int(cfg.get('mask_feathering', 12)))
                            eye_mask = eye_mask_full[y0:y1, x0:x1]

                            if eye_pres > 0.0:
                                # attempt to warp back if swapped landmarks available
                                swapped_lm = None
                                try:
                                    sw_face = get_one_face(swapped_patch)
                                    if sw_face is not None:
                                        swapped_lm_raw = _safe_get_landmarks(sw_face)
                                        swapped_lm = [(x + x0, y + y0) for x, y in swapped_lm_raw]
                                except Exception:
                                    swapped_lm = None

                                if swapped_lm:
                                    swapped_patch = pb.warp_paste_back(swapped_patch, orig_patch, src_landmarks=lm, dst_landmarks=swapped_lm, region_mask=eye_mask, preserve_strength=eye_pres)
                                else:
                                    swapped_patch = afc.paste_back_region(swapped_patch, orig_patch, eye_mask, eye_pres)
                        except Exception:
                            pass

                        # Now blend swapped_patch onto result using composite region mask
                        try:
                            region_mask = mask[y0:y1, x0:x1]
                            blended = afc.apply_region_blend(swapped_patch, orig_patch, region_mask)
                            res[y0:y1, x0:x1] = blended
                        except Exception:
                            try:
                                res[y0:y1, x0:x1] = swapped_patch
                            except Exception:
                                pass

                return res
            except Exception:
                traceback.print_exc()
                return res

        fs.apply_post_processing = _patched_apply_post_processing

else:
    pass
