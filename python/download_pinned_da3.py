"""Download and verify the exact DA3 model revision during the image build."""

from pathlib import Path

from huggingface_hub import snapshot_download

from runpod_handler import MODEL_REPOSITORY, MODEL_REVISION, MODEL_WEIGHTS_SHA256, file_sha256


destination = Path("/models/DA3MONO-LARGE")
snapshot_download(repo_id=MODEL_REPOSITORY, revision=MODEL_REVISION, local_dir=destination)
weights = destination / "model.safetensors"
if not weights.is_file() or file_sha256(weights) != MODEL_WEIGHTS_SHA256:
    raise RuntimeError("MODEL_CHECKSUM_MISMATCH")
(destination / "REVISION").write_text(MODEL_REVISION + "\n", encoding="utf-8")
