"""Apply standalone defaults and do a lightweight hardware probe at import time.

This module sets reasonable defaults in modules.globals based on the available
providers (CUDA/DirectML/CPU) and the presets defined in modules.standalone_flags.

Note: keep this lightweight — heavy benchmarks run separately.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

try:
    import modules.standalone_flags as sflags
    import modules.globals as globals
except Exception:
    # If modules aren't importable (e.g., during early dev), skip silently
    sflags = None
    globals = None


def _detect_onnx_providers():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return providers
    except Exception:
        return []


def _probe_hw():
    # Try to reuse the scripts/hardware_probe.py if present
    probe_script = os.path.join(os.path.dirname(__file__), "..", "scripts", "hardware_probe.py")
    if os.path.exists(probe_script):
        try:
            out = subprocess.check_output([sys.executable, probe_script], stderr=subprocess.STDOUT)
            # hardware_probe prints JSON to stdout
            try:
                return json.loads(out.decode('utf-8'))
            except Exception:
                return None
        except Exception:
            return None
    return None


def _apply_preset_to_globals(preset: dict):
    if globals is None or preset is None:
        return
    # Map known preset keys into globals
    res = preset.get("resolution")
    if res:
        globals.target_resolution = res
    fps = preset.get("fps")
    if fps:
        globals.target_fps = fps
    globals.use_enhancer = preset.get("use_enhancer", getattr(globals, 'use_enhancer', False))
    globals.poisson_blend = preset.get("poisson_blend", getattr(globals, 'poisson_blend', False))
    globals.fp16 = preset.get("fp16", getattr(globals, 'fp16', False))


def apply_defaults():
    if sflags is None or globals is None:
        return
    # Apply no-watermark default
    try:
        globals.no_watermark = getattr(sflags, 'no_watermark', True)
    except Exception:
        pass

    # Prefer providers by availability
    providers = _detect_onnx_providers()
    # Order preference: CUDA -> DirectML -> CPU
    if 'CUDAExecutionProvider' in providers:
        globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif 'DmlExecutionProvider' in providers or 'DirectML' in providers:
        globals.execution_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    else:
        globals.execution_providers = ['CPUExecutionProvider']

    # Apply default preset (balanced) unless user supplied another mechanism later
    preset = sflags.PRESETS.get('balanced')
    _apply_preset_to_globals(preset)

    # Store probe results if available
    hw = _probe_hw()
    if hw:
        try:
            globals.hw_probe = hw
        except Exception:
            pass


# Apply on import
try:
    apply_defaults()
except Exception:
    pass
