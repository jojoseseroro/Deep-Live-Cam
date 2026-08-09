# INSTALL_WINDOWS.md

This document explains how to build the Windows installer and portable package for DeepLiveCam Standalone.

Overview
--------
We use PyInstaller to bundle the Python application, then NSIS (or Inno) to create a Windows Setup.exe. Models are downloaded during the CI build and packaged into the installer artifact.

Local build steps (developer machine)
------------------------------------
1. Ensure Python 3.10+ is installed and on PATH.
2. Create and activate a virtual environment and install requirements:
   python -m venv venv
   venv\\Scripts\\activate
   pip install -r requirements.txt
3. Download models for local testing:
   python scripts/download_models.py --manifest models/manifest.json --out models
4. Install PyInstaller and build:
   pip install pyinstaller
   pyinstaller windows_build/pyinstaller.spec
5. Build NSIS installer (requires NSIS installed):
   makensis windows_build/nsis/template.nsi

CI build
--------
The GitHub Actions workflow packaging/gh-actions-windows.yml will run on pushes to the standalone/2.7-equivalent branch and attempt to produce artifacts in the dist/ directory and upload them as build artifacts.

Reproducible builds
-------------------
- The CI will download models from official sources and verify SHA-256 checksums; make sure manifest entries are accurate.
- Pin dependency versions in requirements.txt to ensure deterministic builds.

Licenses
--------
- The source code is AGPL-3.0: any distributed binaries must be accompanied by source or an offer for source per AGPL.
- Verify each model license before bundling. If a model is not redistributable, the installer will download it at install time and present license info.
