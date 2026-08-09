"""Preset manager to expose PRESETS as named presets and apply them.

Wraps modules.standalone_flags.PRESETS so other UI code can call apply_preset.
"""
from __future__ import annotations

from modules import standalone_flags
from modules.advanced_controls_config import write_config, get_current


def apply_preset(name: str):
    preset = standalone_flags.PRESETS.get(name)
    if not preset:
        return False
    cur = get_current()
    for k, v in preset.items():
        if k == 'advanced_controls':
            for kk, vv in v.items():
                cur[kk] = vv
        else:
            cur[k] = v
    write_config(cur)
    return True
