# Real Estate DA3 Worker

Minimal RunPod Serverless worker for deterministic monocular depth inference with
Depth Anything 3 Mono Large. This repository contains only the deployable worker;
it contains no product source, customer media, credentials, or environment files.

## Reproducibility and safety

- The model revision, model weight SHA-256, upstream code revision, preprocessing
  revision, runtime, and Python dependencies are pinned.
- Model weights are embedded while the image is built. Runtime inference is offline.
- Every request must supply the exact reviewed pins, a canonical cache identity, and
  the source image checksum before GPU work begins.
- The container runs as an unprivileged user and returns a bounded compressed depth
  artifact.

## Test

```bash
python -m unittest python/test_runpod_definition.py
```

The published image is intended for a RunPod queue endpoint with one L4 worker at
most, zero active workers, and scale-to-zero enabled.

Depth Anything 3 is developed by ByteDance Seed and distributed under Apache-2.0.
