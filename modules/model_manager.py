"""Model manager: handles model availability, downloads, and verification.

Behavior:
- Reads models/manifest.json to learn model entries.
- Determines application models directory (next to executable when frozen, otherwise repo/models).
- For each required model, ensures it is available: bundled (present in app models dir) or downloaded (if auto_download allowed).
- Downloads models to the app models dir when permitted. Verifies SHA-256 if manifest provides it.
- Exposes status codes for smoke tests and runtime checks.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
from typing import Dict, Any, Tuple

try:
    import requests
except Exception:
    requests = None  # will raise later if download needed


def _app_models_dir() -> str:
    # If frozen (PyInstaller), sys.executable points to the exe; otherwise use repo models dir
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        # modules/ is inside repo; move up to repo root
        base = os.path.abspath(os.path.join(base, '..'))
    models_dir = os.path.join(base, 'models')
    return models_dir


def _compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


class ModelManager:
    def __init__(self, manifest_path: str = None):
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if manifest_path is None:
            manifest_path = os.path.join(repo_dir, 'models', 'manifest.json')
        self.manifest_path = manifest_path
        self.models_dir = _app_models_dir()
        self._load_manifest()

    def _load_manifest(self):
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.entries = {m['name']: m for m in data.get('models', [])}
        except Exception:
            self.entries = {}

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        return self.entries

    def model_file_path(self, name: str) -> str:
        entry = self.entries.get(name)
        if not entry:
            return ''
        filename = entry.get('filename')
        return os.path.join(self.models_dir, filename)

    def is_model_present(self, name: str) -> bool:
        path = self.model_file_path(name)
        return os.path.exists(path)

    def verify_model_sha(self, name: str) -> Tuple[bool, str]:
        entry = self.entries.get(name)
        if not entry:
            return False, 'missing manifest entry'
        expected = entry.get('sha256')
        path = self.model_file_path(name)
        if not os.path.exists(path):
            return False, 'file missing'
        if not expected:
            return True, 'no sha provided'
        try:
            actual = _compute_sha256(path)
            if actual.lower() == expected.lower():
                return True, 'sha match'
            else:
                return False, f'sha mismatch: actual={actual} expected={expected}'
        except Exception as e:
            return False, f'sha compute error: {e}'

    def download_model(self, name: str) -> Tuple[bool, str]:
        entry = self.entries.get(name)
        if not entry:
            return False, 'missing manifest entry'
        url = entry.get('url')
        filename = entry.get('filename')
        if not url or not filename:
            return False, 'invalid manifest entry'
        if requests is None:
            return False, 'requests not installed in runtime'
        os.makedirs(self.models_dir, exist_ok=True)
        target = os.path.join(self.models_dir, filename)
        try:
            r = requests.get(url, stream=True, timeout=180)
            r.raise_for_status()
            with open(target + '.tmp', 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            # move into place
            if os.path.exists(target):
                os.remove(target)
            os.rename(target + '.tmp', target)
            # verify sha if available
            expected = entry.get('sha256')
            if expected:
                actual = _compute_sha256(target)
                if actual.lower() != expected.lower():
                    return False, f'sha mismatch after download: actual={actual} expected={expected}'
            return True, 'downloaded'
        except Exception as e:
            # cleanup tmp
            try:
                if os.path.exists(target + '.tmp'):
                    os.remove(target + '.tmp')
            except Exception:
                pass
            return False, f'download error: {e}'

    def ensure_model(self, name: str) -> Tuple[str, str]:
        """Ensure model is available. Returns (status, message).

        Status values:
        - 'available' : model file present and (if sha provided) verified
        - 'downloaded' : model successfully downloaded and verified
        - 'blocked' : not present and not allowed to auto-download
        - 'error' : attempted download failed
        """
        entry = self.entries.get(name)
        if not entry:
            return 'error', 'manifest missing entry'
        # Determine path
        path = self.model_file_path(name)
        # Present and verify
        if os.path.exists(path):
            ok, msg = self.verify_model_sha(name)
            if ok:
                return 'available', msg
            else:
                # attempt redownload if auto_download allowed
                if entry.get('auto_download'):
                    ok2, msg2 = self.download_model(name)
                    if ok2:
                        return 'downloaded', msg2
                    else:
                        return 'error', msg2
                else:
                    return 'blocked', f'present but sha invalid and auto_download not allowed: {msg}'
        else:
            # not present
            if entry.get('redistributable'):
                # developer intended to bundle; but missing -> error
                return 'blocked', 'expected bundled model missing'
            if entry.get('auto_download'):
                ok, msg = self.download_model(name)
                if ok:
                    return 'downloaded', msg
                else:
                    return 'error', msg
            else:
                return 'blocked', 'not bundled and auto_download not allowed'

    def ensure_required_models(self) -> Tuple[bool, Dict[str, Tuple[str, str]]]:
        """Ensure all models marked required are available. Returns (all_ok, details).

        details: name -> (status, message)
        """
        details = {}
        all_ok = True
        for name, entry in self.entries.items():
            if entry.get('required'):
                status, msg = self.ensure_model(name)
                details[name] = (status, msg)
                if status not in ('available', 'downloaded'):
                    all_ok = False
        return all_ok, details


# Provide a module-level default manager
_default_manager = None

def get_manager() -> ModelManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager()
    return _default_manager


def ensure_required_models() -> Tuple[bool, Dict[str, Tuple[str, str]]]:
    return get_manager().ensure_required_models()
