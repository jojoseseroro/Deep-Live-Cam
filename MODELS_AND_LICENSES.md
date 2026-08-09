MODELS AND LICENSES

This file lists the models used by the DeepLiveCam standalone build and their license and source information.

When building installers, the CI will consult models/manifest.json and will download redistributable models and verify SHA-256 checksums. If a model is not redistributable by license, the installer will download it during installation after presenting the user with license information.

Please update this file with authoritative license text or links for each model included in the manifest.

Example:
- inswapper_128.onnx
  - Source: https://github.com/insightface/inswapper
  - License: (insert)
  - SHA256: (insert)

- GFPGANv1.3.pth
  - Source: https://github.com/TencentARC/GFPGAN
  - License: (insert)
  - SHA256: (insert)

Make sure to verify each model's license before bundling.
