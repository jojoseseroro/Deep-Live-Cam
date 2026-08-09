#!/usr/bin/env python3
"""Hardware probe for Windows to detect GPU, VRAM, CUDA availability, NVENC and system specs.

This is intentionally lightweight and avoids heavy dependencies. It tries the following checks:
- nvidia-smi (for NVIDIA GPU and VRAM)
- onnxruntime providers (if installed) to check CUDA/DirectML availability
- ffmpeg -encoders to detect NVENC availability
- Windows WMI via "wmic" for CPU and RAM info (fallback)

Writes JSON summary to stdout and optionally to a file.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        return out.strip()
    except Exception:
        return ""


def probe_nvidia():
    info = {"present": False}
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return info
    out = run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
    if not out:
        return info
    try:
        name, mem = out.split(",")
        info["present"] = True
        info["name"] = name.strip()
        info["vram_mb"] = int(mem.strip())
    except Exception:
        info["present"] = True
        info["raw"] = out
    return info


def probe_ffmpeg_nvenc():
    ff = shutil.which("ffmpeg")
    if not ff:
        return {"nvenc": False}
    out = run_cmd("ffmpeg -hide_banner -encoders")
    return {"nvenc": "nvenc" in out or "h264_nvenc" in out or "hevc_nvenc" in out}


def probe_onnxruntime():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        return {"onnxruntime": True, "providers": providers}
    except Exception:
        return {"onnxruntime": False, "providers": []}


def probe_system():
    info = {}
    info["platform"] = platform.platform()
    try:
        # Windows: total physical memory via systeminfo or wmic
        if sys.platform == "win32":
            out = run_cmd("wmic computersystem get TotalPhysicalMemory")
            if out:
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                if len(lines) >= 2:
                    mem_bytes = int(lines[1])
                    info["ram_gb"] = round(mem_bytes / (1024**3))
    except Exception:
        pass
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        info["cpu_count"] = None
    return info


def main(out_file: str | None = None):
    report = {}
    report["nvidia"] = probe_nvidia()
    report["ffmpeg"] = probe_ffmpeg_nvenc()
    report["onnxruntime"] = probe_onnxruntime()
    report["system"] = probe_system()

    text = json.dumps(report, indent=2)
    print(text)
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Write JSON report to file", default=None)
    args = parser.parse_args()
    main(args.out)
