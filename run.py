#!/usr/bin/env python3

"""Entry point for Deep-Live-Cam.

Supports a --smoke-test mode for packaged-distribution validation. When run
without flags it starts the normal application (core.run()).
"""
from __future__ import annotations

import sys
import os
import json
import hashlib
import argparse
import traceback

# Import the tkinter fix to patch the ScreenChanged error (module patches Tk on import)
try:
    import tkinter_fix  # noqa: F401
except Exception:
    # In smoke-test mode we may not need GUI; continue
    pass

# Import standalone defaults and hardware probe (applies presets and sets globals defaults)
try:
    import modules.standalone_apply  # noqa: F401
except Exception:
    # standalone_apply should exist; log on smoke-test if requested
    pass

# Import the face_swapper injector so advanced blending is applied (monkey-patch)
try:
    import modules.face_swapper_inject  # noqa: F401
except Exception:
    # injector may be optional for smoke-test
    pass

# Import embedded advanced controls UI so it attaches to the main window
try:
    import modules.advanced_controls_embed  # noqa: F401
except Exception:
    pass

# Core application runtime
import core


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: str):
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def smoke_test():
    failures = []
    warnings = []

    print('Smoke test: starting packaged-environment validation')

    # 1) Imports of new/standalone/advanced modules
    modules_to_check = [
        'modules.advanced_face_controls',
        'modules.advanced_controls_config',
        'modules.face_swapper_inject',
        'modules.advanced_controls_embed',
        'modules.pasteback',
        'modules.enhancer',
        'modules.face_state',
        'modules.profile_manager',
        'modules.auto_tuner',
        'modules.hw_tuner',
    ]
    for m in modules_to_check:
        try:
            __import__(m)
            print(f'Import OK: {m}')
        except Exception:
            tb = traceback.format_exc()
            print(f'Import FAILED: {m}\n{tb}')
            failures.append(f'Import failed: {m}')

    # 2) Model/resource manifest checks
    repo_root = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(repo_root, 'models', 'manifest.json')
    manifest = load_manifest(manifest_path)
    if manifest is None:
        failures.append(f'Manifest not found or invalid JSON: {manifest_path}')
    else:
        print(f'Loaded manifest with {len(manifest.get("models", []))} entries')
        for entry in manifest.get('models', []):
            name = entry.get('name')
            url = entry.get('url', '')
            filename = entry.get('filename', '')
            sha256 = entry.get('sha256', '')
            license = entry.get('license', '')
            redistributable = entry.get('redistributable', False)

            # Basic placeholder detection
            if not url or 'example.com' in url or 'insert' in url or url.strip().endswith('/'):
                failures.append(f'Manifest entry {name} has placeholder or invalid URL: {url}')
            if not filename:
                failures.append(f'Manifest entry {name} missing filename')
            if not sha256:
                failures.append(f'Manifest entry {name} missing sha256')
            if not license or 'verify' in str(license).lower() or '?' in str(license):
                failures.append(f'Manifest entry {name} has unverified license: {license}')
            if not bool(redistributable):
                failures.append(f'Manifest entry {name} not marked redistributable')

            # If model file is present in models/, compute sha and compare
            model_file_path = os.path.join(repo_root, 'models', filename)
            if os.path.exists(model_file_path):
                try:
                    actual_sha = compute_file_sha256(model_file_path)
                    print(f'Model {name} found locally, sha256={actual_sha}')
                    if sha256 and actual_sha.lower() != sha256.lower():
                        failures.append(f'Model {name} local file sha256 mismatch: manifest={sha256} actual={actual_sha}')
                except Exception as e:
                    failures.append(f'Failed to compute sha256 for {model_file_path}: {e}')
            else:
                warnings.append(f'Model {name} not included in package (file {model_file_path} not found)')

    # 3) ONNX Runtime providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print('ONNX Runtime providers available:', providers)
        if not providers:
            warnings.append('ONNX Runtime is installed but no providers available')
    except Exception as e:
        warnings.append(f'ONNX Runtime not available: {e}')

    # 4) Optional enhancer availability
    try:
        import gfpgan  # type: ignore
        print('GFPGAN package available')
    except Exception:
        warnings.append('GFPGAN not available')

    # 5) Config/profile path creation
    try:
        home = os.path.expanduser('~')
        cfg_path = os.path.join(home, '.deeplivecam_advanced_controls.json')
        profiles_dir = os.path.join(home, '.deeplivecam_profiles')
        # Ensure profile dir exists
        os.makedirs(profiles_dir, exist_ok=True)
        # Touch config file if missing
        if not os.path.exists(cfg_path):
            with open(cfg_path, 'w', encoding='utf-8') as f:
                f.write('{}')
        print('Config and profile paths verified:', cfg_path, profiles_dir)
    except Exception as e:
        failures.append(f'Failed to create config/profile paths: {e}')

    # Summary
    print('\nSmoke test summary:')
    for w in warnings:
        print('WARNING:', w)
    for f in failures:
        print('FAIL:', f)

    if failures:
        print('Smoke test FAILED')
        return 2
    else:
        print('Smoke test PASSED')
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Run packaged-build smoke tests and exit')
    args = parser.parse_args()

    if args.smoke_test:
        rc = smoke_test()
        sys.exit(rc)

    # Normal application run
    core.run()


if __name__ == '__main__':
    main()
