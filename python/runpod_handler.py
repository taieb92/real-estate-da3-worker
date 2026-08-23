"""RunPod Serverless handler for the pinned DA3 monocular depth model.

The image build embeds the reviewed source and weights. Runtime networking is
not required for model loading, and one queue job processes one bounded image.
"""

import base64
import binascii
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path


MODEL_REPOSITORY = "depth-anything/DA3MONO-LARGE"
MODEL_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
MODEL_WEIGHTS_SHA256 = "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
PREPROCESSING_REVISION = "da3-preprocess-v1"
MODEL_DIR = Path(os.environ.get("DA3_MODEL_DIR", "/models/DA3MONO-LARGE"))
MAX_IMAGE_BYTES = 15_000_000
ALLOWED_RESOLUTIONS = (504, 756, 1008)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_cache_identity(request: dict) -> str:
    identity = {key: value for key, value in request.items()
                if key not in ("schemaVersion", "id", "cacheIdentity")}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalized_confidence(confidence, np):
    finite = np.isfinite(confidence)
    if not finite.any():
        return np.zeros_like(confidence, dtype=np.float32)
    scale = float(np.percentile(confidence[finite], 99))
    return np.clip(confidence / scale, 0, 1).astype(np.float32) if scale > 1e-8 else np.zeros_like(confidence)


def load_model():
    import torch
    from depth_anything_3.api import DepthAnything3

    weights = MODEL_DIR / "model.safetensors"
    revision = MODEL_DIR / "REVISION"
    if (not weights.is_file() or not revision.is_file()
            or revision.read_text(encoding="utf-8").strip() != MODEL_REVISION
            or file_sha256(weights) != MODEL_WEIGHTS_SHA256):
        raise RuntimeError("MODEL_UNAVAILABLE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return DepthAnything3.from_pretrained(str(MODEL_DIR), local_files_only=True).to("cuda")


def validate_request(payload: object) -> tuple[dict, bytes]:
    if not isinstance(payload, dict):
        raise ValueError("INVALID_REQUEST")
    request = payload.get("request")
    encoded = payload.get("imageBase64")
    if not isinstance(request, dict) or not isinstance(encoded, str):
        raise ValueError("INVALID_REQUEST")
    expected = {
        "model": "da3mono-large",
        "modelRepository": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "modelWeightsSha256": MODEL_WEIGHTS_SHA256,
        "modelLicense": "Apache-2.0",
        "codeRepository": "ByteDance-Seed/Depth-Anything-3",
        "codeRevision": CODE_REVISION,
        "preprocessingRevision": PREPROCESSING_REVISION,
        "encoding": "float16_npz",
    }
    if any(request.get(key) != value for key, value in expected.items()):
        raise ValueError("PIN_MISMATCH")
    resolution = request.get("inferenceResolution")
    inputs = request.get("inputs")
    if resolution not in ALLOWED_RESOLUTIONS or not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("INVALID_INFERENCE_SHAPE")
    if (not isinstance(inputs[0], dict)
            or request.get("referenceAssetId") != inputs[0].get("assetId")
            or request.get("cacheIdentity") != request_cache_identity(request)):
        raise ValueError("REQUEST_IDENTITY_MISMATCH")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("INVALID_IMAGE_ENCODING") from error
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("IMAGE_SIZE_EXCEEDED")
    if inputs[0].get("sourceSha256") != hashlib.sha256(image_bytes).hexdigest():
        raise ValueError("SOURCE_CHECKSUM_MISMATCH")
    return request, image_bytes


def infer(model, request: dict, image_bytes: bytes) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(image_bytes)) as opened:
            width, height = opened.size
            if width < 360 or height < 360 or width * height > 40_000_000:
                raise ValueError("INVALID_IMAGE_DIMENSIONS")
            suffix = ".png" if opened.format == "PNG" else ".jpg"
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("INVALID_IMAGE") from error
    with tempfile.TemporaryDirectory(prefix="da3-request-") as directory:
        source_path = Path(directory, "source" + suffix)
        source_path.write_bytes(image_bytes)
        with torch.inference_mode():
            prediction = model.inference(
                image=[str(source_path)], process_res=request["inferenceResolution"],
                process_res_method="upper_bound_resize", infer_gs=False,
                export_dir=None, export_format="mini_npz")
    depth = np.asarray(prediction.depth, dtype=np.float32)
    confidence = np.asarray(prediction.conf, dtype=np.float32)
    if (depth.ndim != 3 or depth.shape[0] != 1 or confidence.shape != depth.shape
            or not np.isfinite(depth).all() or np.any(depth <= 0)):
        raise RuntimeError("INVALID_DEPTH_OUTPUT")
    normalized = normalized_confidence(confidence, np)
    valid = np.isfinite(depth) & (depth > 0)
    gradient_y, gradient_x = np.gradient(depth[0])
    relative_gradient = np.hypot(gradient_x, gradient_y) / np.maximum(depth[0], 1e-6)
    reference = depth[0][valid[0]]
    metrics = {
        "meanConfidence": float(normalized[valid].mean()),
        "lowConfidenceFraction": float((normalized[valid] < 0.25).mean()),
        "validPixelFraction": float(valid.mean()),
        "edgeDiscontinuityScore": float(np.clip(1 - np.percentile(relative_gradient[valid[0]], 95), 0, 1)),
        "depthP05": float(np.percentile(reference, 5)),
        "depthP95": float(np.percentile(reference, 95)),
    }
    output = io.BytesIO()
    np.savez_compressed(output, depth=depth.astype(np.float16), confidence=normalized.astype(np.float16))
    return {"schemaVersion": 1, "requestId": request.get("id"), "width": int(depth.shape[2]),
            "height": int(depth.shape[1]), "metrics": metrics,
            "npzBase64": base64.b64encode(output.getvalue()).decode("ascii")}


_model = None


def handler(job: dict) -> dict:
    global _model
    request, image_bytes = validate_request(job.get("input") if isinstance(job, dict) else None)
    if _model is None:
        _model = load_model()
    return infer(_model, request, image_bytes)


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
