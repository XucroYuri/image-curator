import json
import sqlite3

from PIL import Image, PngImagePlugin

from image_curator.checkpoint import CheckpointStore
from image_curator.extract import extract_pending
from image_curator.inference import CallableInferenceAdapter
from image_curator.scan import scan_and_enqueue


def test_extract_records_pillow_features_metadata_digest_and_optional_vector(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    image_path = library / "image.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", "private embedded prompt")
    Image.new("RGB", (3, 2), "red").save(image_path, pnginfo=metadata)
    database_path = tmp_path / "output" / "checkpoint.sqlite"
    adapter = CallableInferenceAdapter("test-vector", lambda image_bytes, source: [3.0, 4.0])

    with CheckpointStore(database_path) as store:
        scan_and_enqueue(store, [library])
        result = extract_pending(store, adapter=adapter, limit=1)
        assert result.as_dict() == {"attempted": 1, "completed": 1, "failed": 0}
        assert store.status() == {"COMPLETE": 1}

    connection = sqlite3.connect(database_path)
    features, evidence, blob, dimensions = connection.execute(
        "SELECT features_json,metadata_evidence_json,embedding_f16,embedding_dim FROM work_items"
    ).fetchone()
    assert json.loads(features)["width"] == 3
    assert json.loads(features)["inference"] == {"adapter": "test-vector", "state": "complete"}
    assert "private embedded prompt" not in evidence
    assert json.loads(evidence)["metadata_keys"] == ["prompt"]
    assert len(blob) == 4
    assert dimensions == 2


def test_extract_marks_bad_image_retryable(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    (library / "not-image.jpg").write_bytes(b"not a real image")

    with CheckpointStore(tmp_path / "checkpoint.sqlite") as store:
        scan_and_enqueue(store, [library])
        result = extract_pending(store)
        assert result.failed == 1
        assert store.status() == {"RETRYABLE_FAILED": 1}
