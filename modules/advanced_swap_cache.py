"""Cache for swapped face patches produced by the face-swapper.

Store small per-face records (original target patch, swapped patch, timestamp)
keyed by a bbox tuple. Used by the injector to retrieve the exact swapped pixels
so advanced mask-based blending and paste-back can be applied.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Any, Tuple

_cache_lock = threading.Lock()
# key -> { 'orig': np.ndarray, 'swapped': np.ndarray, 'timestamp': float }
_cache: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
# max age seconds
_MAX_AGE = 2.0


def set_patch(bbox: Tuple[int, int, int, int], orig_patch, swapped_patch) -> None:
    with _cache_lock:
        _cache[bbox] = {
            'orig': orig_patch.copy() if hasattr(orig_patch, 'copy') else orig_patch,
            'swapped': swapped_patch.copy() if hasattr(swapped_patch, 'copy') else swapped_patch,
            'timestamp': time.time(),
        }


def get_patch(bbox: Tuple[int, int, int, int]):
    with _cache_lock:
        rec = _cache.get(bbox)
        if not rec:
            return None
        if time.time() - rec['timestamp'] > _MAX_AGE:
            # stale
            del _cache[bbox]
            return None
        return rec


def cleanup():
    now = time.time()
    with _cache_lock:
        to_delete = [k for k, v in _cache.items() if now - v['timestamp'] > _MAX_AGE]
        for k in to_delete:
            del _cache[k]
