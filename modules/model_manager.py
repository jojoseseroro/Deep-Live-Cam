"""Model manager: load models manifest and resolve local model paths.

Used by the application to locate models that were downloaded by the installer or CI.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any

_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "manifest.json")

_manifest_cache: Dict[str, Any] | None = None


def _load_manifest(path: str | None = None) -> Dict[str, Any]:
    global _manifest_cache
    if _manifest_cache is not None:
        return _manifest_cache
    p = path or _MANIFEST_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model manifest not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        _manifest_cache = json.load(f)
    return _manifest_cache


def get_model_entry(name: str) -> Dict[str, Any] | None:
    man = _load_manifest()
    for entry in man.get("models", []):
        if entry.get("name") == name or entry.get("filename") == name:
            return entry
    return None


def get_model_path(name: str, models_dir: str | None = None) -> str:
    entry = get_model_entry(name)
    if entry is None:
        raise KeyError(f"Model not found in manifest: {name}")
    filename = entry.get("filename") or os.path.basename(entry.get("url", ""))
    md = models_dir or os.path.join(os.path.dirname(__file__), "..", "models")
    path = os.path.abspath(os.path.join(md, filename))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at expected path: {path}")
    return path


if __name__ == "__main__":
    # Quick smoke test
    try:
        print("Manifest loaded. Models:")
        m = _load_manifest()
        for e in m.get("models", []):
            print(f" - {e.get('name')}: {e.get('filename')} -> {e.get('url')}")
    except Exception as exc:
        print(f"Model manager error: {exc}")
