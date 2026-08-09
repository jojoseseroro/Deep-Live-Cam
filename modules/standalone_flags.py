# Standalone flags and presets for DeepLiveCam

# This module is intentionally small and non-invasive. It defines defaults used by the standalone build.

standalone_mode = True

# Model manifest path (relative to repository root)
model_manifest_path = "models/manifest.json"

# Installer behavior
bundle_models_in_installer = True  # CI will download and embed redistributable models

# Default UI flags
no_watermark = True

# Quality presets
PRESETS = {
    "low_latency": {
        "resolution": "720p",
        "fps": 30,
        "use_enhancer": False,
        "poisson_blend": False,
        "fp16": True,
    },
    "balanced": {
        "resolution": "1080p",
        "fps": 30,
        "use_enhancer": True,
        "poisson_blend": True,
        "fp16": True,
    },
    "high_quality": {
        "resolution": "1080p",
        "fps": 60,
        "use_enhancer": True,
        "poisson_blend": True,
        "fp16": True,
    },
    "maximum": {
        "resolution": "4k",
        "fps": 60,
        "use_enhancer": True,
        "poisson_blend": True,
        "fp16": True,
    }
}
