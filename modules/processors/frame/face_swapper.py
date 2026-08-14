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
        if center_x - radius_x < 0 or center_x + radius_x >= w or center_y - radius_y < 0 or (center_y + radius_y >= h):
            return swapped_frame
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
FRAME_CACHE = deque(maxlen=3)
FACE_DETECTION_CACHE = {}
LAST_DETECTION_TIME = 0
DETECTION_INTERVAL = 0.033
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
    fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
    if not os.path.exists(fp16_path) and not os.path.exists(fp32_path):
        update_status(f"Model not found in {models_dir}. Please download inswapper_128.onnx.", NAME)
        return False

    if get_face_swapper() is None:
        return False

    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            fp32_path = os.path.join(models_dir, "inswapper_128.onnx")
            fp16_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
            use_fp16 = _HAS_TORCH_CUDA and os.path.exists(fp16_path)
            if use_fp16:
                model_path = fp16_path
            elif os.path.exists(fp32_path):
                model_path = fp32_path
            else:
                update_status(f"No inswapper model found in {models_dir}.", NAME)
                return None
            if IS_APPLE_SILICON:
                from modules.onnx_optimize import optimize_for_coreml
                model_path = optimize_for_coreml(model_path)

            update_status(f"Loading face swapper model from: {model_path}", NAME)
            try:
                providers_config = []
                for p in modules.globals.execution_providers:
                    if p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
                        providers_config.append((
                            "CoreMLExecutionProvider",
                            {
                                "ModelFormat": "MLProgram",
                                "MLComputeUnits": "ALL",
                                "SpecializationStrategy": "FastPrediction",
                                "AllowLowPrecisionAccumulationOnGPU": 1,
                                "EnableOnSubgraphs": 1,
                            }
                        ))
                    elif p == "CUDAExecutionProvider":
                        providers_config.append(p)
                    elif p == "OpenVINOExecutionProvider":
                        providers_config.append(OPENVINO_PROVIDER_CONFIG)
                    else:
                        providers_config.append(p)
                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path,
                    providers=providers_config,
                )
                if _HAS_TORCH_CUDA and any(
                    p == "CUDAExecutionProvider" or
                    (isinstance(p, tuple) and p[0] == "CUDAExecutionProvider")
                    for p in providers_config
                ):
                    _init_cuda_graph_session(model_path, FACE_SWAPPER)
                update_status("Face swapper model loaded successfully.", NAME)
            except Exception as e:
                update_status(f"Error loading face swapper model: {e}", NAME)
                FACE_SWAPPER = None
                return None
    return FACE_SWAPPER


_HAS_TORCH_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        _HAS_TORCH_CUDA = True
except ImportError:
    pass

_paste_cache = {
    'soft_alpha': None,
    'alpha_size': 0,
}


def _get_soft_alpha(size: int) -> np.ndarray:
    if _paste_cache['alpha_size'] != size:
        center = (size // 2, size // 2)
        axes = (int(size * 0.44), int(size * 0.44))
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 12)
        _paste_cache['soft_alpha'] = mask
        _paste_cache['alpha_size'] = size
    return _paste_cache['soft_alpha']

_cuda_graph_session = {
    'session': None,
    'io_binding': None,
    'ort_input': None,
    'ort_latent': None,
    'recorded': False,
}
_cuda_graph_lock = threading.Lock()


class _CudaGraphSessionAdapter:
    def __init__(self, underlying):
        object.__setattr__(self, "_underlying", underlying)

    def run(self, output_names, input_dict, **kwargs):
        if _cuda_graph_session['recorded']:
            try:
                keys = list(input_dict.keys())
                blob = input_dict[keys[0]]
                latent = input_dict[keys[1]]
                return [_cuda_graph_swap_inference(blob, latent)]
            except Exception:
                pass
        return self._underlying.run(output_names, input_dict, **kwargs)

    def __getattr__(self, name):
        return getattr(self._underlying, name)

    def __setattr__(self, name, value):
        setattr(self._underlying, name, value)


def _init_cuda_graph_session(model_path: str, swapper):
    import onnxruntime as ort
    try:
        providers = [('CUDAExecutionProvider', {'enable_cuda_graph': '1'})]
        sess = ort.InferenceSession(model_path, providers=providers)
        inp_shape = (1, 3, swapper.input_size[1], swapper.input_size[0])
        latent_shape = (1, 512)
        dummy_inp = np.zeros(inp_shape, dtype=np.float32)
        dummy_lat = np.zeros(latent_shape, dtype=np.float32)
        ort_input = ort.OrtValue.ortvalue_from_numpy(dummy_inp, 'cuda', 0)
        ort_latent = ort.OrtValue.ortvalue_from_numpy(dummy_lat, 'cuda', 0)
        io = sess.io_binding()
        io.bind_ortvalue_input(swapper.input_names[0], ort_input)
        io.bind_ortvalue_input(swapper.input_names[1], ort_latent)
        io.bind_output(swapper.output_names[0], 'cuda', 0)
        sess.run_with_iobinding(io)
        _cuda_graph_session['session'] = sess
        _cuda_graph_session['io_binding'] = io
        _cuda_graph_session['ort_input'] = ort_input
        _cuda_graph_session['ort_latent'] = ort_latent
        _cuda_graph_session['recorded'] = True
        if not isinstance(swapper.session, _CudaGraphSessionAdapter):
            swapper.session = _CudaGraphSessionAdapter(swapper.session)
        import sys
        print(f"[{NAME}] CUDA graph session initialized (swap model)")
        sys.stdout.flush()
    except Exception as e:
        print(f"[{NAME}] CUDA graph init failed, using standard session: {e}")
        _cuda_graph_session['recorded'] = False


def _cuda_graph_swap_inference(blob: np.ndarray, latent: np.ndarray) -> np.ndarray:
    cg = _cuda_graph_session
    with _cuda_graph_lock:
        cg['ort_input'].update_inplace(blob)
        cg['ort_latent'].update_inplace(latent)
        cg['session'].run_with_iobinding(cg['io_binding'])
        return cg['io_binding'].get_outputs()[0].numpy()


def _fast_paste_back(target_img: Frame, bgr_fake: np.ndarray, aimg: np.ndarray, M: np.ndarray) -> Frame:
    h, w = target_img.shape[:2]
    face_h, face_w = aimg.shape[:2]
    assert face_h == face_w, f"Expected square aligned face, got {face_h}x{face_w}"
    IM = cv2.invertAffineTransform(M)
    corners = np.array(
        [[0, 0], [face_w, 0], [face_w, face_h], [0, face_h]], dtype=np.float32
    )
    transformed = (IM[:, :2] @ corners.T).T + IM[:, 2]
    x1 = int(np.floor(transformed[:, 0].min()))
    x2 = int(np.ceil(transformed[:, 0].max()))
    y1 = int(np.floor(transformed[:, 1].min()))
    y2 = int(np.ceil(transformed[:, 1].max()))
    if x1 >= x2 or y1 >= y2:
        return target_img
    pad = 2
    y1p, y2p = max(0, y1 - pad), min(h, y2 + pad + 1)
    x1p, x2p = max(0, x1 - pad), min(w, x2 + pad + 1)
    IM_crop = IM.copy()
    IM_crop[0, 2] -= x1p
    IM_crop[1, 2] -= y1p
    crop_w, crop_h = x2p - x1p, y2p - y1p
    soft_alpha = _get_soft_alpha(face_h)
    bgr_fake_crop = cv2.warpAffine(bgr_fake, IM_crop, (crop_w, crop_h), borderMode=cv2.BORDER_REPLICATE)
    alpha_crop = cv2.warpAffine(soft_alpha, IM_crop, (crop_w, crop_h), borderValue=0)
    target_crop = target_img[y1p:y2p, x1p:x2p]
    if _HAS_TORCH_CUDA:
        mask_t = torch.from_numpy(alpha_crop).cuda().float().mul_(1.0 / 255.0).unsqueeze(2)
        fake_t = torch.from_numpy(bgr_fake_crop).float().cuda()
        tgt_t = torch.from_numpy(target_crop).float().cuda()
        blended = (mask_t * fake_t + (1.0 - mask_t) * tgt_t).to(torch.uint8).cpu().numpy()
        target_img[y1p:y2p, x1p:x2p] = blended
    else:
        alpha_3c = cv2.merge([alpha_crop, alpha_crop, alpha_crop])
        inv_alpha = 255 - alpha_3c
        a_fake = cv2.multiply(bgr_fake_crop, alpha_3c, scale=1.0 / 255.0)
        a_tgt = cv2.multiply(target_crop, inv_alpha, scale=1.0 / 255.0)
        target_img[y1p:y2p, x1p:x2p] = cv2.add(a_fake, a_tgt)
    return target_img


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()
    if face_swapper is None:
        update_status("Face swapper model not loaded or failed to load. Skipping swap.", NAME)
        return temp_frame
    if source_face is None or target_face is None:
        return temp_frame
    if not hasattr(source_face, 'normed_embedding') or source_face.normed_embedding is None:
        return temp_frame
    opacity = getattr(modules.globals, "opacity", 1.0)
    opacity = max(0.0, min(1.0, opacity))
    mouth_mask_enabled = getattr(modules.globals, "mouth_mask", False)
    poisson_blend_enabled = getattr(modules.globals, "poisson_blend", False)
    needs_original = opacity < 1.0 or mouth_mask_enabled or poisson_blend_enabled
    if needs_original:
        original_frame = temp_frame.copy()
    else:
        original_frame = temp_frame
    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)
    try:
        if not temp_frame.flags['C_CONTIGUOUS']:
            temp_frame = np.ascontiguousarray(temp_frame)
        if any("DmlExecutionProvider" in p for p in modules.globals.execution_providers):
            with modules.globals.dml_lock:
                bgr_fake, M = face_swapper.get(temp_frame, target_face, source_face, paste_back=False)
        else:
            bgr_fake, M = face_swapper.get(temp_frame, target_face, source_face, paste_back=False)
        if bgr_fake is None:
            return original_frame
        if not isinstance(bgr_fake, np.ndarray):
            return original_frame
        _face_size = face_swapper.input_size[0]
        _aimg_dummy = np.empty((_face_size, _face_size, 3), dtype=np.uint8)
        swapped_frame = _fast_paste_back(temp_frame, bgr_fake, _aimg_dummy, M)
    except Exception as e:
        print(f"Error during face swap: {e}")
        return original_frame
    if mouth_mask_enabled:
        face_mask = create_face_mask(target_face, original_frame)
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, original_frame)
        )
        if mouth_cutout is not None and mouth_box != (0,0,0,0):
            swapped_frame = apply_mouth_area(
                swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
            )
            if getattr(modules.globals, "show_mouth_mask_box", False):
                mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
                swapped_frame = draw_mouth_mask_visualization(
                    swapped_frame, target_face, mouth_mask_data
                )
    if getattr(modules.globals, "poisson_blend", False):
        swapped_frame = _apply_poisson_blend(
            swapped_frame, original_frame, target_face, M, bgr_fake
        )
    if opacity >= 1.0:
        return swapped_frame.astype(np.uint8)
    final_swapped_frame = gpu_add_weighted(original_frame.astype(np.uint8), 1 - opacity, swapped_frame.astype(np.uint8), opacity, 0)
    return final_swapped_frame.astype(np.uint8)


def get_faces_optimized(frame: Frame, use_cache: bool = True) -> Optional[List[Face]]:
    global LAST_DETECTION_TIME, FACE_DETECTION_CACHE
    if not use_cache or not IS_APPLE_SILICON:
        if modules.globals.many_faces:
            return get_many_faces(frame)
        else:
            face = get_one_face(frame)
            return [face] if face else None
    current_time = time.time()
    time_since_last = current_time - LAST_DETECTION_TIME
    if time_since_last < DETECTION_INTERVAL and FACE_DETECTION_CACHE:
        return FACE_DETECTION_CACHE.get('faces')
    LAST_DETECTION_TIME = current_time
    if modules.globals.many_faces:
        faces = get_many_faces(frame)
    else:
        face = get_one_face(frame)
        faces = [face] if face else None
    FACE_DETECTION_CACHE['faces'] = faces
    FACE_DETECTION_CACHE['timestamp'] = current_time
    return faces


def apply_post_processing(current_frame: Frame, swapped_face_bboxes: List[np.ndarray]) -> Frame:
    global PREVIOUS_FRAME_RESULT
    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)
    if sharpness_value <= 0.0 and not enable_interpolation:
        PREVIOUS_FRAME_RESULT = None
        return current_frame
    processed_frame = current_frame.copy()
    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    if sharpness_value > 0.0 and swapped_face_bboxes:
        height, width = processed_frame.shape[:2]
        for bbox in swapped_face_bboxes:
            if not hasattr(bbox, '__iter__') or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            try:
                 x1, y1 = max(0, int(x1)), max(0, int(y1))
                 x2, y2 = min(width, int(x2)), min(height, int(y2))
            except ValueError:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            face_region = processed_frame[y1:y2, x1:x2]
            if face_region.size == 0:
                continue
            try:
                sigma = 2 if IS_APPLE_SILICON else 3
                sharpened_region = gpu_sharpen(face_region, strength=sharpness_value, sigma=sigma)
                processed_frame[y1:y2, x1:x2] = sharpened_region
            except cv2.error:
                pass
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)
    interpolation_weight = getattr(modules.globals, "interpolation_weight", 0.2)
    final_frame = processed_frame
    if enable_interpolation and 0 < interpolation_weight < 1:
        if PREVIOUS_FRAME_RESULT is not None and PREVIOUS_FRAME_RESULT.shape == processed_frame.shape and PREVIOUS_FRAME_RESULT.dtype == processed_frame.dtype:
            try:
                 final_frame = gpu_add_weighted(
                    PREVIOUS_FRAME_RESULT, 1.0 - interpolation_weight,
                    processed_frame, interpolation_weight,
                    0
                 )
                 final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            except cv2.error:
                 final_frame = processed_frame
                 PREVIOUS_FRAME_RESULT = None
            PREVIOUS_FRAME_RESULT = final_frame.copy()
        else:
            PREVIOUS_FRAME_RESULT = processed_frame.copy()
    else:
         PREVIOUS_FRAME_RESULT = None
    return final_frame


def process_frame(source_face: Face, temp_frame: Frame, target_face: Face = None) -> Frame:
    if getattr(modules.globals, "opacity", 1.0) == 0:
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame
    processed_frame = temp_frame
    swapped_face_bboxes = []
    if modules.globals.many_faces:
        many_faces = get_many_faces(processed_frame)
        if many_faces:
            current_swap_target = processed_frame.copy()
            for face in many_faces:
                current_swap_target = swap_face(source_face, face, current_swap_target)
                if face is not None and hasattr(face, "bbox") and face.bbox is not None:
                    swapped_face_bboxes.append(face.bbox.astype(int))
            processed_frame = current_swap_target
    else:
        if target_face is None:
            target_face = get_one_face(processed_frame)
        if target_face:
            processed_frame = swap_face(source_face, target_face, processed_frame)
            if hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    return final_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    if getattr(modules.globals, "opacity", 1.0) == 0:
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame
    processed_frame = temp_frame
    swapped_face_bboxes = []
    source_target_pairs = []
    source_target_map = getattr(modules.globals, "source_target_map", None)
    simple_map = getattr(modules.globals, "simple_map", None)
    is_file_target = modules.globals.target_path and (is_image(modules.globals.target_path) or is_video(modules.globals.target_path))
    if is_file_target:
        if source_target_map:
            if modules.globals.many_faces:
                source_face = default_source_face()
                if source_face:
                    for map_data in source_target_map:
                        if is_image(modules.globals.target_path):
                            target_info = map_data.get("target", {})
                            if target_info:
                                target_face = target_info.get("face")
                                if target_face:
                                    source_target_pairs.append((source_face, target_face))
                        elif is_video(modules.globals.target_path):
                             target_frames_data = map_data.get("target_faces_in_frame", [])
                             if target_frames_data:
                                 target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                                 for frame_data in target_frames:
                                     faces_in_frame = frame_data.get("faces", [])
                                     if faces_in_frame:
                                         for target_face in faces_in_frame:
                                             source_target_pairs.append((source_face, target_face))
            else:
                 for map_data in source_target_map:
                    source_info = map_data.get("source", {})
                    if not source_info:
                        continue
                    source_face = source_info.get("face")
                    if not source_face:
                        continue
                    if is_image(modules.globals.target_path):
                        target_info = map_data.get("target", {})
                        if target_info:
                           target_face = target_info.get("face")
                           if target_face:
                              source_target_pairs.append((source_face, target_face))
                    elif is_video(modules.globals.target_path):
                        target_frames_data = map_data.get("target_faces_in_frame", [])
                        if target_frames_data:
                           target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                           for frame_data in target_frames:
                               faces_in_frame = frame_data.get("faces", [])
                               if faces_in_frame:
                                  for target_face in faces_in_frame:
                                      source_target_pairs.append((source_face, target_face))
    else:
        detected_faces = get_many_faces(processed_frame)
        if detected_faces:
            if modules.globals.many_faces:
                 source_face = default_source_face()
                 if source_face:
                     for target_face in detected_faces:
                        source_target_pairs.append((source_face, target_face))
            elif simple_map:
                source_faces = simple_map.get("source_faces", [])
                target_embeddings = simple_map.get("target_embeddings", [])
                if source_faces and target_embeddings and len(source_faces) == len(target_embeddings):
                     if len(detected_faces) <= len(target_embeddings):
                          for detected_face in detected_faces:
                              if detected_face.normed_embedding is None:
                                  continue
                              closest_idx, _ = find_closest_centroid(target_embeddings, detected_face.normed_embedding)
                              if 0 <= closest_idx < len(source_faces):
                                  source_target_pairs.append((source_faces[closest_idx], detected_face))
                     else:
                          detected_embeddings = [f.normed_embedding for f in detected_faces if f.normed_embedding is not None]
                          detected_faces_with_embedding = [f for f in detected_faces if f.normed_embedding is not None]
                          if not detected_embeddings:
                              return processed_frame
                          for i, target_embedding in enumerate(target_embeddings):
                              if 0 <= i < len(source_faces):
                                 closest_idx, _ = find_closest_centroid(detected_embeddings, target_embedding)
                                 if 0 <= closest_idx < len(detected_faces_with_embedding):
                                     source_target_pairs.append((source_faces[i], detected_faces_with_embedding[closest_idx]))
            else:
                source_face = default_source_face()
                target_face = get_one_face(processed_frame, detected_faces)
                if source_face and target_face:
                    source_target_pairs.append((source_face, target_face))
    current_swap_target = processed_frame.copy()
    for source_face, target_face in source_target_pairs:
        if source_face and target_face:
            current_swap_target = swap_face(source_face, target_face, current_swap_target)
            if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))
    processed_frame = current_swap_target
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)
    return final_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    use_v2 = getattr(modules.globals, "map_faces", False)
    source_face = None
    if not use_v2:
        if not source_path or not os.path.exists(source_path):
            update_status(f"Error: Source path invalid or not provided for simple mode: {source_path}", NAME)
        else:
            try:
                source_img = imread_unicode(source_path)
                if source_img is None:
                    update_status(f"Error reading source image file {source_path}. Please check the path and file integrity.", NAME)
                else:
                    source_face = get_one_face(source_img)
                    if source_face is None:
                        update_status(f"Warning: Successfully read source image {source_path}, but no face was detected. Swaps will be skipped.", NAME)
                    del source_img
            except Exception as e:
                import traceback
                print(f"{NAME}: Caught exception during source image processing for {source_path}:")
                traceback.print_exc()
                update_status(f"Error during source image reading or analysis {source_path}: {e}", NAME)
    total_frames = len(temp_frame_paths)
    if not use_v2 and source_face is None:
        update_status("Halting video processing: Invalid or no face detected in source image for simple mode.", NAME)
        if progress:
            remaining_updates = total_frames - progress.n if hasattr(progress, 'n') else total_frames
            if remaining_updates > 0:
                progress.update(remaining_updates)
        return
    for temp_frame_path in temp_frame_paths:
        temp_frame = None
        try:
            temp_frame = imread_unicode(temp_frame_path)
            if temp_frame is None:
                print(f"{NAME}: Error: Could not read frame: {temp_frame_path}, skipping.")
                if progress:
                    progress.update(1)
                continue
        except Exception as read_e:
            print(f"{NAME}: Error reading frame {temp_frame_path}: {read_e}, skipping.")
            if progress:
                progress.update(1)
            continue
        result_frame = None
        try:
            if use_v2:
                result_frame = process_frame_v2(temp_frame, temp_frame_path)
            else:
                result_frame = process_frame(source_face, temp_frame)
            if result_frame is None:
                 print(f"{NAME}: Warning: Processing returned None for frame {temp_frame_path}. Using original.")
                 result_frame = temp_frame
        except Exception as proc_e:
            print(f"{NAME}: Error processing frame {temp_frame_path}: {proc_e}")
            result_frame = temp_frame
        try:
            write_success = imwrite_unicode(temp_frame_path, result_frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not write_success:
                print(f"{NAME}: Error: Failed to write processed frame to {temp_frame_path}")
        except Exception as write_e:
            print(f"{NAME}: Error writing frame {temp_frame_path}: {write_e}")
        del temp_frame
        if result_frame is not None:
            del result_frame
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    use_v2 = getattr(modules.globals, "map_faces", False)
    try:
        target_frame = imread_unicode(target_path)
        if target_frame is None:
            update_status(f"Error: Could not read target image: {target_path}", NAME)
            return
    except Exception as read_e:
        update_status(f"Error reading target image {target_path}: {read_e}", NAME)
        return
    result = None
    try:
        if use_v2:
            if getattr(modules.globals, "many_faces", False):
                 update_status("Processing image with 'map_faces' and 'many_faces'. Using pre-analysis map.", NAME)
            result = process_frame_v2(target_frame, target_path)
        else:
            try:
                source_img = imread_unicode(source_path)
                if source_img is None:
                    update_status(f"Error: Could not read source image: {source_path}", NAME)
                    return
                source_face = get_one_face(source_img)
                if not source_face:
                    update_status(f"Error: No face found in source image: {source_path}", NAME)
                    return
            except Exception as src_e:
                 update_status(f"Error reading or analyzing source image {source_path}: {src_e}", NAME)
                 return
            result = process_frame(source_face, target_frame)
        if result is not None:
            write_success = imwrite_unicode(output_path, result)
            if write_success:
                update_status(f"Output image saved to: {output_path}", NAME)
            else:
                update_status(f"Error: Failed to write output image to {output_path}", NAME)
        else:
            update_status("Image processing failed (result was None).", NAME)
    except Exception as proc_e:
         update_status(f"Error during image processing: {proc_e}", NAME)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    mode_desc = "'map_faces'" if getattr(modules.globals, "map_faces", False) else "'simple'"
    if getattr(modules.globals, "map_faces", False) and getattr(modules.globals, "many_faces", False):
        mode_desc += " and 'many_faces'. Using pre-analysis map."
    update_status(f"Processing video with {mode_desc} mode.", NAME)
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )


def create_lower_mouth_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None
    mouth_box = (0,0,0,0)
    if face is None or not hasattr(face, 'landmark_2d_106'):
        return mask, mouth_cutout, mouth_box, lower_lip_polygon
    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        return mask, mouth_cutout, mouth_box, lower_lip_polygon
    try:
        lower_lip_order = list(range(52, 64))
        if max(lower_lip_order) >= landmarks.shape[0]:
            return mask, mouth_cutout, mouth_box, lower_lip_polygon
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)
        if not np.all(np.isfinite(lower_lip_landmarks)):
            return mask, mouth_cutout, mouth_box, lower_lip_polygon
        center = np.mean(lower_lip_landmarks, axis=0)
        if not np.all(np.isfinite(center)):
            return mask, mouth_cutout, mouth_box, lower_lip_polygon
        mouth_mask_size = getattr(modules.globals, "mouth_mask_size", 0.0)
        s = max(0.0, min(1.0, mouth_mask_size / 100.0))
        expansion_factor = 1.0 + s * 2.0
        chin_bias = 1.0 + s * 2.0
        offsets = lower_lip_landmarks - center
        scale_y = np.where(offsets[:, 1] > 0,
                           expansion_factor * chin_bias, expansion_factor)
        expanded_landmarks = lower_lip_landmarks.copy()
        expanded_landmarks[:, 0] = center[0] + offsets[:, 0] * expansion_factor
        expanded_landmarks[:, 1] = center[1] + offsets[:, 1] * scale_y
        if not np.all(np.isfinite(expanded_landmarks)):
            return mask, mouth_cutout, mouth_box, lower_lip_polygon
        expanded_landmarks = expanded_landmarks.astype(np.int32)
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)
        padding_ratio = 0.1
        padding_x = int((max_x - min_x) * padding_ratio)
        padding_y = int((max_y - min_y) * padding_ratio)
        frame_h, frame_w = frame.shape[:2]
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(frame_w, max_x + padding_x)
        max_y = min(frame_h, max_y + padding_y)
        if max_x > min_x and max_y > min_y:
            mask_roi_h = max_y - min_y
            mask_roi_w = max_x - min_x
            mask_roi = np.zeros((mask_roi_h, mask_roi_w), dtype=np.uint8)
            polygon_relative_to_roi = expanded_landmarks - [min_x, min_y]
            cv2.fillPoly(mask_roi, [polygon_relative_to_roi], 255)
            blur_k_size = getattr(modules.globals, "mask_blur_kernel", 15)
            blur_k_size = max(1, blur_k_size // 2 * 2 + 1)
            mask_roi = gpu_gaussian_blur(mask_roi, (blur_k_size, blur_k_size), 0)
            mask[min_y:max_y, min_x:max_x] = mask_roi
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
            lower_lip_polygon = expanded_landmarks
            mouth_box = (min_x, min_y, max_x, max_y)
    except Exception:
        pass
    return mask, mouth_cutout, mouth_box, lower_lip_polygon


def draw_mouth_mask_visualization(frame: Frame, face: Face, mouth_mask_data: tuple) -> Frame:
    if frame is None or face is None or mouth_mask_data is None or len(mouth_mask_data) != 4:
        return frame
    mask, mouth_cutout, box, lower_lip_polygon = mouth_mask_data
    (min_x, min_y, max_x, max_y) = box
    if lower_lip_polygon is None or not isinstance(lower_lip_polygon, np.ndarray) or len(lower_lip_polygon) < 3:
        return frame
    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]
    try:
        min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
        max_x, max_y = min(width, int(max_x)), min(height, int(max_y))
    except ValueError:
        return frame
    if max_x <= min_x or max_y <= min_y:
        return frame
    try:
         safe_polygon = lower_lip_polygon.copy()
         safe_polygon[:, 0] = np.clip(safe_polygon[:, 0], 0, width - 1)
         safe_polygon[:, 1] = np.clip(safe_polygon[:, 1], 0, height - 1)
         cv2.polylines(vis_frame, [safe_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    except Exception:
        pass
    cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    label_pos_y = min_y - 10 if min_y > 20 else max_y + 15
    label_pos_x = min_x
    try:
        cv2.putText(vis_frame, "Mouth Mask", (label_pos_x, label_pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception:
        pass
    return vis_frame


def apply_mouth_area(frame: np.ndarray, mouth_cutout: np.ndarray, mouth_box: tuple, face_mask: np.ndarray, mouth_polygon: np.ndarray) -> np.ndarray:
    if (frame is None or mouth_cutout is None or mouth_box is None or
        face_mask is None or mouth_polygon is None):
        return frame
    if (mouth_cutout.size == 0 or face_mask.size == 0 or len(mouth_polygon) < 3):
        return frame
    try:
        min_x, min_y, max_x, max_y = map(int, mouth_box)
        box_width = max_x - min_x
        box_height = max_y - min_y
        if box_width <= 0 or box_height <= 0:
            return frame
        frame_h, frame_w = frame.shape[:2]
        min_y, max_y = max(0, min_y), min(frame_h, max_y)
        min_x, max_x = max(0, min_x), min(frame_w, max_x)
        box_width = max_x - min_x
        box_height = max_y - min_y
        if box_width <= 0 or box_height <= 0:
            return frame
        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame
        resized_mouth_cutout = None
        if roi.shape[:2] != mouth_cutout.shape[:2]:
             if mouth_cutout.shape[0] > 0 and mouth_cutout.shape[1] > 0:
                  resized_mouth_cutout = gpu_resize(mouth_cutout, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
             else:
                 return frame
        else:
             resized_mouth_cutout = mouth_cutout
        if resized_mouth_cutout is None or resized_mouth_cutout.size == 0:
            return frame
        polygon_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask_roi, [adjusted_polygon.astype(np.int32)], 255)
        feather_amount = max(1, min(30, min(box_width, box_height) // 8))
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(polygon_mask_roi.astype(np.float32), (kernel_size, kernel_size), 0)
        max_val = feathered_mask.max()
        if max_val > 1e-6:
            feathered_mask = feathered_mask / max_val
        else:
            feathered_mask.fill(0.0)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            mask_3ch = feathered_mask[:, :, np.newaxis].astype(np.float32)
            inv_mask = 1.0 - mask_3ch
            blended_roi = (resized_mouth_cutout.astype(np.float32) * mask_3ch +
                           roi.astype(np.float32) * inv_mask)
            frame[min_y:max_y, min_x:max_x] = np.clip(blended_roi, 0, 255).astype(np.uint8)
    except Exception:
        pass
    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if face is None or not hasattr(face, 'landmark_2d_106'):
        return mask
    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        return mask
    try:
        if not np.all(np.isfinite(landmarks)):
            return mask
        landmarks_int = landmarks.astype(np.int32)
        face_outline = landmarks_int[0:33]
        eyebrows = landmarks_int[33:43]
        if eyebrows.shape[0] > 0:
            chin = landmarks_int[16]
            eyebrow_center = np.mean(eyebrows, axis=0)
            up_vector = eyebrow_center - chin
            norm = np.linalg.norm(up_vector)
            if norm > 0:
                up_vector /= norm
                forehead_offset = up_vector * (norm * 1.0)
                forehead_points = eyebrows + forehead_offset
                top_center = np.mean(forehead_points, axis=0)
                forehead_points = (forehead_points - top_center) * 1.2 + top_center
                face_outline = np.concatenate((face_outline, forehead_points.astype(np.int32)), axis=0)
        try:
             hull = cv2.convexHull(face_outline.astype(np.float32))
             if hull is None or len(hull) < 3:
                 return mask
             cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        except Exception:
             return mask
        blur_k_size = getattr(modules.globals, "face_mask_blur", 31)
        blur_k_size = max(1, blur_k_size // 2 * 2 + 1)
        mask = gpu_gaussian_blur(mask, (blur_k_size, blur_k_size), 0)
    except Exception:
        pass
    return mask


def apply_color_transfer(source, target):
    if source is None or target is None or source.size == 0 or target.size == 0:
        return source
    if len(source.shape) != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        try:
            if len(source.shape) == 2:
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            source = np.clip(source, 0, 255).astype(np.uint8)
            if len(source.shape) != 3 or source.shape[2] != 3:
                raise ValueError("Conversion failed")
        except Exception:
            return source
    if len(target.shape) != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        try:
            if len(target.shape) == 2:
                target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            target = np.clip(target, 0, 255).astype(np.uint8)
            if len(target.shape) != 3 or target.shape[2] != 3:
                raise ValueError("Conversion failed")
        except Exception:
             return source
    result_bgr = source
    try:
        source_float = source.astype(np.float32) / 255.0
        target_float = target.astype(np.float32) / 255.0
        source_lab = cv2.cvtColor(source_float, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_float, cv2.COLOR_BGR2LAB)
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)
        source_mean = source_mean.reshape((1, 1, 3))
        source_std = source_std.reshape((1, 1, 3))
        target_mean = target_mean.reshape((1, 1, 3))
        target_std = target_std.reshape((1, 1, 3))
        epsilon = 1e-6
        source_std = np.maximum(source_std, epsilon)
        result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
        result_bgr_float = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        result_bgr_float = np.clip(result_bgr_float, 0.0, 1.0)
        result_bgr = (result_bgr_float * 255.0).astype("uint8")
    except Exception:
         return source
    return result_bgr
