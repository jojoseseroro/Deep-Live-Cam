"""Auto-tuner orchestration: picks a recommended preset and optionally applies it.

This module exposes run_auto_tune(apply=True) which will inspect hardware and
models and either return a recommended preset or apply it to the config.
"""
from __future__ import annotations

from modules import hw_tuner
from modules.preset_manager import apply_preset
from modules.advanced_controls_config import get_current, write_config


def run_auto_tune(apply: bool = True) -> str:
    # Use inswapper model name as the benchmark target if present
    preferred_model = 'inswapper_128.onnx'
    preset = hw_tuner.auto_tune_using_model(preferred_model)
    if apply:
        apply_preset(preset)
        cur = get_current()
        cur['_recommended_preset'] = preset
        write_config(cur)
    return preset
