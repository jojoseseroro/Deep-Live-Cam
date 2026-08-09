"""Simple blink detector based on Eye Aspect Ratio (EAR).

This module computes EAR from 68-point landmarks and reports if eyes are
considered to be blinking. It's lightweight and well-suited for realtime.
"""
from typing import List, Tuple
import math


LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]


def eye_aspect_ratio(eye: List[Tuple[float, float]]) -> float:
    # eye: list of 6 (x,y) points
    import math
    A = math.hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = math.hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = math.hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear


def is_blinking(landmarks: List[Tuple[float, float]], threshold: float = 0.21) -> bool:
    try:
        left = [landmarks[i] for i in LEFT_EYE_IDX]
        right = [landmarks[i] for i in RIGHT_EYE_IDX]
        ear_l = eye_aspect_ratio(left)
        ear_r = eye_aspect_ratio(right)
        ear = (ear_l + ear_r) / 2.0
        return ear < threshold
    except Exception:
        return False
