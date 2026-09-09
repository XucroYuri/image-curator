import numpy as np
import pytest
from PIL import Image

from image_curator.cli import main
from image_curator.inference import (
    AnalysisResult,
    CallableInferenceAdapter,
    CombinedAnalysisAdapter,
)
from image_curator.onnx_adapters import (
    OnnxDependencyError,
    WD14MoatNudeNetAdapter,
    _create_session,
    prepare_nudenet,
    prepare_wd14_moat,
)


class Input:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, shape, outputs):
        self.input = Input("input", shape)
        self.outputs = outputs
        self.received = None

    def get_inputs(self):
        return [self.input]

    def run(self, _, values):
        self.received = values["input"]
        return self.outputs


def _nude_row(class_index, confidence):
    row = [100, 100, 40, 40] + [0.0] * 18
    row[4 + class_index] = confidence
    return row


def test_wd14_and_nudenet_preprocessing_layouts_are_model_compatible():
    image = Image.new("RGB", (2, 1), "red")

    moat = prepare_wd14_moat(image, np)
    nude_nchw = prepare_nudenet(image, size=320, nhwc=False, numpy=np)
    nude_nhwc = prepare_nudenet(image, size=320, nhwc=True, numpy=np)

    assert moat.shape == (1, 448, 448, 3)
    assert moat[0, 0, 0].tolist() == [0.0, 0.0, 255.0]
    assert moat[0, -1, 0].tolist() == [255.0, 255.0, 255.0]
    assert nude_nchw.shape == (1, 3, 320, 320)
    assert nude_nchw[0, :, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert nude_nhwc.shape == (1, 320, 320, 3)


def test_adapter_maps_wd14_tags_embedding_and_classwise_nudenet_safety():
    tags = [
        {"name": "general", "category": "rating"},
        {"name": "sensitive", "category": "rating"},
        {"name": "questionable", "category": "rating"},
        {"name": "explicit", "category": "rating"},
        {"name": "example_tag", "category": "general"},
    ]
    moat = FakeSession([1, 448, 448, 3], [np.array([[0.9, 0.1, 0.2, 0.3, 0.8]]), np.array([[3.0, 4.0]])])
    nude = FakeSession([1, 3, 320, 320], [np.array([_nude_row(16, 0.9), _nude_row(3, 0.8)], dtype=np.float32)])
    adapter = WD14MoatNudeNetAdapter(moat, tags, nudenet_session=nude, embedding_dimensions=2)

    result = adapter.analyze(Image.new("RGB", (4, 2), "red"), b"already read", __file__)

    assert result.embedding == pytest.approx((0.6, 0.8))
    assert result.features["wd14_moat"]["rating_explicit"] == pytest.approx(0.3)
    assert result.features["wd14_moat"]["top_tags"] == [
        {"tag": "example_tag", "confidence": 0.8, "category": "general"}
    ]
    safety = result.features["nudenet"]
    assert safety["explicit_score"] == 0.8
    assert safety["intimate_covered_score"] == 0.9
    assert moat.received.shape == (1, 448, 448, 3)
    assert nude.received.shape == (1, 3, 320, 320)


def test_combined_adapter_keeps_legacy_embed_adapter_compatible(tmp_path):
    class Structured:
        name = "structured"

        def analyze(self, image, image_bytes, source):
            return AnalysisResult(features={"width": image.width}, evidence={"checked": True})

    combined = CombinedAnalysisAdapter(
        Structured(), CallableInferenceAdapter("legacy", lambda image_bytes, source: [3.0, 4.0])
    )

    result = combined.analyze(Image.new("RGB", (2, 1)), b"bytes", tmp_path / "image.png")

    assert result.embedding == [3.0, 4.0]
    assert result.features["structured"] == {"width": 2}
    assert result.evidence["structured"] == {"checked": True}


def test_cli_reports_clear_error_for_incomplete_onnx_arguments(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["extract", str(tmp_path / "checkpoint.sqlite"), "--moat-model", "model.onnx"])

    assert error.value.code == 2
    assert "--moat-model and --wd14-tags must be supplied together" in capsys.readouterr().err


def test_cli_reports_missing_onnx_runtime_without_downloading(tmp_path, capsys, monkeypatch):
    model = tmp_path / "model.onnx"
    tags = tmp_path / "tags.csv"
    model.write_bytes(b"caller supplied")
    tags.write_text("name\ngeneral\n", encoding="utf-8")

    def unavailable(*args, **kwargs):
        raise OnnxDependencyError("local ONNX inference requires onnxruntime")

    monkeypatch.setattr(WD14MoatNudeNetAdapter, "from_paths", unavailable)
    with pytest.raises(SystemExit) as error:
        main(["extract", str(tmp_path / "checkpoint.sqlite"), "--moat-model", str(model), "--wd14-tags", str(tags)])

    assert error.value.code == 2
    assert "requires onnxruntime" in capsys.readouterr().err


def test_session_accepts_explicit_cpu_and_bounds_runtime_threads(tmp_path):
    class Runtime:
        class SessionOptions:
            intra_op_num_threads = 0
            inter_op_num_threads = 0

        @staticmethod
        def InferenceSession(path, *, sess_options, providers):
            assert path.endswith("model.onnx")
            assert providers == ["CPUExecutionProvider"]
            assert sess_options.intra_op_num_threads == 1
            assert sess_options.inter_op_num_threads == 1
            return type("Session", (), {"get_providers": lambda self: ["CPUExecutionProvider"]})()

    session = _create_session(Runtime, tmp_path / "model.onnx", ["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_session_refuses_silent_cuda_fallback():
    class Runtime:
        class SessionOptions:
            intra_op_num_threads = 0
            inter_op_num_threads = 0

        @staticmethod
        def InferenceSession(path, *, sess_options, providers):
            return type("Session", (), {"get_providers": lambda self: ["CPUExecutionProvider"]})()

    with pytest.raises(ValueError, match="requested ONNX provider is unavailable"):
        _create_session(Runtime, __file__, ["CUDAExecutionProvider", "CPUExecutionProvider"])
