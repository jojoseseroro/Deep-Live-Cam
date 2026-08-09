"""Hardware tuner and preset recommender.

Detects GPU, VRAM and suggests a preset: low_latency, balanced, high_quality, maximum
based on available VRAM and presence of CUDA provider.
"""
from __future__ import annotations

import shutil
import subprocess
import json
import os


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
        # No NVIDIA: fallback to CPU/DirectML recommendations
        return 'balanced'
    vram = info.get('vram_mb', 0)
    # crude rules
    if vram >= 24576:
        return 'maximum'
    if vram >= 12288:
        return 'high_quality'
    if vram >= 6144:
        return 'balanced'
    return 'low_latency'


if __name__ == '__main__':
    print(json.dumps(_query_nvidia_smi(), indent=2))
    print('Recommended preset:', recommend_preset())
