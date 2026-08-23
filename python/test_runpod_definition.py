import base64
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "runpod_handler.py"
SPEC = importlib.util.spec_from_file_location("runpod_handler", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def valid_request(image_bytes=b"verified-image"):
    request = {
        "schemaVersion": 1,
        "id": "11111111-1111-4111-8111-111111111111",
        "organizationId": "22222222-2222-4222-8222-222222222222",
        "listingId": "33333333-3333-4333-8333-333333333333",
        "referenceAssetId": "44444444-4444-4444-8444-444444444444",
        "inputs": [{"assetId": "44444444-4444-4444-8444-444444444444",
                    "sourceSha256": hashlib.sha256(image_bytes).hexdigest()}],
        "model": "da3mono-large",
        "modelRepository": MODULE.MODEL_REPOSITORY,
        "modelRevision": MODULE.MODEL_REVISION,
        "modelWeightsSha256": MODULE.MODEL_WEIGHTS_SHA256,
        "modelLicense": "Apache-2.0",
        "codeRepository": "ByteDance-Seed/Depth-Anything-3",
        "codeRevision": MODULE.CODE_REVISION,
        "preprocessingRevision": MODULE.PREPROCESSING_REVISION,
        "inferenceResolution": 504,
        "encoding": "float16_npz",
    }
    identity = {key: value for key, value in request.items() if key not in ("schemaVersion", "id", "cacheIdentity")}
    request["cacheIdentity"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                                           ensure_ascii=False).encode("utf-8")).hexdigest()
    return request


class RunPodDefinitionTest(unittest.TestCase):
    def test_pins_the_reviewed_commercial_model_code_and_runtime(self):
        self.assertEqual(MODULE.MODEL_REPOSITORY, "depth-anything/DA3MONO-LARGE")
        self.assertEqual(MODULE.MODEL_REVISION, "f465978e618db8cc79c83b8bbf24964857db1875")
        self.assertEqual(MODULE.MODEL_WEIGHTS_SHA256,
                         "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960")
        self.assertEqual(MODULE.CODE_REVISION, "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4")
        source = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime", source)
        self.assertIn("runpod==1.8.1", source)
        self.assertIn("--no-deps -e /opt/depth-anything-3", source)
        self.assertIn("git -C /opt/depth-anything-3 apply --check", source)
        self.assertIn("USER 1000:1000", source)

    def test_validates_identity_pin_checksum_and_bounds_before_gpu_use(self):
        image = b"verified-image"
        request = valid_request(image)
        parsed, decoded = MODULE.validate_request({"request": request,
                                                    "imageBase64": base64.b64encode(image).decode("ascii")})
        self.assertEqual(parsed["id"], request["id"])
        self.assertEqual(decoded, image)
        with self.assertRaisesRegex(ValueError, "PIN_MISMATCH"):
            MODULE.validate_request({"request": {**request, "codeRevision": "unreviewed"},
                                     "imageBase64": base64.b64encode(image).decode("ascii")})
        with self.assertRaisesRegex(ValueError, "SOURCE_CHECKSUM_MISMATCH"):
            MODULE.validate_request({"request": request,
                                     "imageBase64": base64.b64encode(b"different").decode("ascii")})

    def test_checksum_helper_reads_the_complete_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory, "artifact")
            artifact.write_bytes(b"verified-model-fixture")
            self.assertEqual(MODULE.file_sha256(artifact), hashlib.sha256(artifact.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
