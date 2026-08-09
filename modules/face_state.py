"""Shared face state container to expose last detected landmarks to the UI overlay.

The injector updates face_state.last_faces with the list of face objects or landmark lists.
The embedded UI can read this to render mask previews.
"""
from __future__ import annotations

from typing import List, Any
import threading

_lock = threading.Lock()
last_faces: List[Any] = []


def set_last_faces(faces: List[Any]):
    global last_faces
    with _lock:
        last_faces = faces.copy() if isinstance(faces, list) else list(faces)


def get_last_faces():
    with _lock:
        return list(last_faces)
