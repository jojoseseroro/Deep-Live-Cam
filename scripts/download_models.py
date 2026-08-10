#!/usr/bin/env python3
"""Model downloader and verifier for DeepLiveCam standalone builds.

Usage:
    python scripts/download_models.py --manifest models/manifest.json --out models/

This script reads a manifest JSON file listing model entries with fields:
  - name
  - filename
  - url
  - sha256
  - license

It downloads each model to the output directory, verifies SHA-256, and writes a models/installed_manifest.json
with the installed file paths and verification results.

This script is intended to be used during CI (GitHub Actions) and optionally by the installer.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

CHUNK_SIZE = 8192


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: str) -> None:
    req = Request(url, headers={"User-Agent": "DeepLiveCam-Downloader/1.0"})
    with urlopen(req) as resp, open(dest_path, "wb") as out:
        total = resp.getheader("Content-Length")
        if total:
            total = int(total)
        downloaded = 0
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"Downloading {os.path.basename(dest_path)}: {pct}%", end="\r")
    print()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main(manifest_path: str, out_dir: str, strict: bool = True) -> int:
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return 2
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ensure_dir(out_dir)
    installed = []
    for entry in manifest.get("models", []):
        name = entry.get("name")
        filename = entry.get("filename") or os.path.basename(entry.get("url", ""))
        url = entry.get("url")
        sha256 = entry.get("sha256")
        license = entry.get("license")

        if not url:
            print(f"Skipping {name}: no URL provided in manifest.")
            continue

        dest_path = os.path.join(out_dir, filename)
        print(f"Processing model: {name} -> {dest_path}")

        try:
            download_file(url, dest_path)
        except (HTTPError, URLError) as e:
            print(f"Failed to download {name} from {url}: {e}")
            if strict:
                return 3
            else:
                continue

        if sha256:
            actual = sha256_of_file(dest_path)
            if actual.lower() != sha256.lower():
                print(f"SHA256 mismatch for {name}: expected {sha256}, got {actual}")
                if strict:
                    return 4
                else:
                    print("Continuing despite mismatch (strict=False)")
        else:
            print(f"Warning: no sha256 provided for {name}; skipping verification.")

        installed.append({
            "name": name,
            "filename": filename,
            "path": dest_path,
            "sha256": sha256,
            "license": license,
        })

    installed_manifest = os.path.join(out_dir, "installed_manifest.json")
    with open(installed_manifest, "w", encoding="utf-8") as f:
        json.dump({"installed": installed}, f, indent=2)

    print(f"Installed {len(installed)} models. Manifest written to {installed_manifest}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to models/manifest.json")
    parser.add_argument("--out", required=True, help="Output directory for downloaded models")
    parser.add_argument("--no-strict", action="store_true", help="Do not fail build on download/verify errors")
    args = parser.parse_args()
    sys.exit(main(args.manifest, args.out, strict=not args.no_strict))
