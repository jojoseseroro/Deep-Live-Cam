"""ONNX runtime helper utilities for provider and FP16 checks.

Provides helper to select execution providers and to check FP16 support heuristically.
"""
from __future__ import annotations


def available_providers():
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return []


def prefer_cuda_provider():
    providers = available_providers()
    if 'CUDAExecutionProvider' in providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'DmlExecutionProvider' in providers:
        return ['DmlExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def fp16_possible():
    # Heuristic: if CUDA provider exists and GPU is >= compute capability 6.0 (not checked here)
    # For simplicity, return True if CUDA provider is available. Real check would inspect device.
    providers = available_providers()
    return 'CUDAExecutionProvider' in providers
