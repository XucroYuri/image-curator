import json
import sqlite3

from PIL import Image, PngImagePlugin

from image_curator.checkpoint import CheckpointStore
from image_curator.extract import extract_pending
from image_curator.inference import (
    AnalysisResult,
    CallableInferenceAdapter,
    CombinedAnalysisAdapter,
)
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


def test_extract_persists_structured_analysis_adapter_result(tmp_path):
    class StructuredAdapter:
        name = "structured"

        def analyze(self, image, image_bytes, source):
            assert image.size == (2, 2)
            assert image_bytes
            return AnalysisResult(embedding=[3.0, 4.0], features={"score": 0.7}, evidence={"model": "local"})

    library = tmp_path / "library"
    library.mkdir()
    Image.new("RGB", (2, 2), "blue").save(library / "image.png")
    database_path = tmp_path / "checkpoint.sqlite"
    with CheckpointStore(database_path) as store:
        scan_and_enqueue(store, [library])
        assert extract_pending(store, adapter=StructuredAdapter()).completed == 1

    connection = sqlite3.connect(database_path)
    features, evidence, dimensions = connection.execute(
        "SELECT features_json,metadata_evidence_json,embedding_dim FROM work_items"
    ).fetchone()
    assert json.loads(features)["analysis"] == {"score": 0.7}
    assert json.loads(evidence)["analysis"] == {"model": "local"}
    assert dimensions == 2


def test_extract_reads_and_decodes_once_for_combined_adapters(tmp_path, monkeypatch):
    import image_curator.extract as extract_module

    class Structured:
        def __init__(self, name):
            self.name = name

        def analyze(self, image, image_bytes, source):
            assert image.size == (4, 3)
            assert image_bytes
            return AnalysisResult(features={"checked": True})

    library = tmp_path / "library"
    library.mkdir()
    Image.new("RGB", (4, 3), "green").save(library / "image.png")
    original_read = extract_module.read_verified
    original_open = extract_module.Image.open
    calls = {"read": 0, "open": 0}

    def counted_read(*args, **kwargs):
        calls["read"] += 1
        return original_read(*args, **kwargs)

    def counted_open(*args, **kwargs):
        calls["open"] += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(extract_module, "read_verified", counted_read)
    monkeypatch.setattr(extract_module.Image, "open", counted_open)
    adapter = CombinedAnalysisAdapter(Structured("first"), Structured("second"))
    with CheckpointStore(tmp_path / "checkpoint.sqlite") as store:
        scan_and_enqueue(store, [library])
        assert extract_pending(store, adapter=adapter).completed == 1

    assert calls == {"read": 1, "open": 1}
