"""Profile manager for saving and loading advanced control profiles.

Profiles are stored in ~/.deeplivecam_profiles/<profile_name>.json
Each profile contains the same structure as modules.advanced_controls_config.DEFAULTS
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, List

PROFILE_DIR = os.path.join(os.path.expanduser('~'), '.deeplivecam_profiles')

os.makedirs(PROFILE_DIR, exist_ok=True)


def save_profile(name: str, data: Dict[str, Any]) -> str:
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return path


def load_profile(name: str) -> Dict[str, Any] | None:
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_profiles() -> List[str]:
    files = [f for f in os.listdir(PROFILE_DIR) if f.endswith('.json')]
    return [os.path.splitext(f)[0] for f in files]
