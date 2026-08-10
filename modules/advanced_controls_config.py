"""Persistent JSON-backed config for Advanced Face Controls.

The live processing reads this config (if present) to apply per-region strengths
and other advanced parameters at runtime. The scripts/advanced_controls_gui.py
writes this JSON when the user adjusts sliders.

This module caches file modification time to avoid re-reading on every frame.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, Any

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".deeplivecam_advanced_controls.json")
_LOCK = threading.Lock()
_cached: Dict[str, Any] | None = None
_cached_mtime: float = 0.0

DEFAULTS = {
    # region strengths (0-100)
    "identity": 85,
    "forehead": 80,
    "eyebrows": 85,
    "eyes": 90,
    "upper_eyelids": 85,
    "lower_eyelids": 85,
    "nose": 85,
    "cheeks": 80,
    "cheekbones": 80,
    "mouth": 60,
    "jaw": 80,
    "chin": 80,
    "face_shape": 80,
    "skin_texture": 80,
    "color_matching": 85,
    "hairline": 70,
    # preservation / behavior (0-100)
    "expression_preservation": 70,
    "eye_blink_preservation": 80,
    "mouth_movement_preservation": 75,
    "teeth_preservation": 80,
    # mask & blending
    "mask_size": 100,
    "mask_feathering": 16,
    "blending_strength": 0.9,
    "color_correction_strength": 0.9,
    "enhancer_strength": 0.7,
    "temporal_smoothing": 0.8,
    # feature toggles
    "advanced_enabled": True,
}


def _read_config() -> Dict[str, Any]:
    global _cached, _cached_mtime
    try:
        if os.path.exists(CONFIG_PATH):
            mtime = os.path.getmtime(CONFIG_PATH)
            if _cached is None or mtime != _cached_mtime:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _cached = {**DEFAULTS, **data}
                _cached_mtime = mtime
        else:
            _cached = dict(DEFAULTS)
        return dict(_cached)
    except Exception:
        return dict(DEFAULTS)


def get_current() -> Dict[str, Any]:
    with _LOCK:
        return _read_config()


def write_config(data: Dict[str, Any]) -> None:
    """Write config to disk (atomic write).

    This is used by the advanced_controls GUI script.
    """
    try:
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
        # update cache
        global _cached, _cached_mtime
        _cached = {**DEFAULTS, **data}
        _cached_mtime = os.path.getmtime(CONFIG_PATH)
    except Exception:
        pass


if __name__ == "__main__":
    print("Advanced Controls config path:", CONFIG_PATH)
    print(json.dumps(get_current(), indent=2))
