"""Hardware tuner and benchmark utilities.

Provides functions to detect GPU & VRAM, benchmark an available ONNX model if present,
and recommend an appropriate preset. This attempts to use nvidia-smi and onnxruntime
if available. Falls back to heuristic rules if tools are not present.
"""
from __future__ import annotations

import shutil
import subprocess
import json
import os
import time
from typing import Optional


def _query_nvidia_smi():
    nvsmi = shutil.which('nvidia-smi')
    if not nvsmi:
        return None
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], universal_newlines=True)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            return None
        name, mem = lines[0].split(',')
        return {'name': name.strip(), 'vram_mb': int(mem.strip())}
    except Exception:
        return None


def recommend_preset() -> str:
    info = _query_nvidia_smi()
    if info is None:
        return 'balanced'
    vram = info.get('vram_mb', 0)
    if vram >= 24576:
        return 'maximum'
    if vram >= 12288:
        return 'high_quality'
    if vram >= 6144:
        return 'balanced'
    return 'low_latency'


def benchmark_onnx_model(model_name: str, model_path: str, iterations: int = 3) -> Optional[float]:
    """Attempt to run a short ONNX Runtime benchmark on the provided model file.

    model_path: path to .onnx model
    Returns average inference time in ms or None on failure.
    """
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(model_path, providers=ort.get_available_providers())
        inp = sess.get_inputs()[0]
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
        # Create random input with expected shape
        data = np.random.randn(*shape).astype(np.float32)
        # Warm-up
        for _ in range(1):
            sess.run(None, {inp.name: data})
        times = []
        for _ in range(iterations):
            t0 = time.time()
            sess.run(None, {inp.name: data})
            t1 = time.time()
            times.append((t1 - t0) * 1000.0)
        return sum(times) / len(times)
    except Exception:
        return None


def auto_tune_using_model(model_name: str) -> str:
    """Auto-tune: if the model is available, attempt a benchmark and refine preset.

    Falls back to recommend_preset() if benchmarking isn't possible.
    """
    from modules.model_manager import get_model_path
    try:
        path = get_model_path(model_name)
    except Exception:
        path = None
    # Use VRAM heuristic first
    preset = recommend_preset()
    if path:
        ms = benchmark_onnx_model(model_name, path)
        if ms is not None:
            # crude thresholds: faster inference -> allow higher quality presets
            if ms < 20.0:
                return 'maximum'
            if ms < 50.0:
                return 'high_quality'
            if ms < 150.0:
                return 'balanced'
            return 'low_latency'
    return preset
