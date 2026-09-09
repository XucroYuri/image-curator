"""Explicit, user-weighted local ONNX adapters for WD14 MoAT and NudeNet.

This module never downloads or bundles models, tag CSV files, or model labels.
Its ONNX Runtime and NumPy imports are deliberately lazy so normal scanning and
metadata extraction remain lightweight.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .detections import classwise_nms, detection_summary
from .embeddings import normalise_moat_vector
from .inference import AnalysisResult

RATING_NAMES = frozenset({"general", "sensitive", "questionable", "explicit"})


class OnnxDependencyError(RuntimeError):
    """Raised when a caller asked for local ONNX inference without its runtime."""


def _numpy_module() -> Any:
    try:
        import numpy as numpy
    except ImportError as error:
        raise OnnxDependencyError(
            "local ONNX inference requires numpy; install the 'onnx' or 'onnx-gpu' extra"
        ) from error
    return numpy


def _optional_modules() -> tuple[Any, Any]:
    numpy = _numpy_module()
    try:
        import onnxruntime as onnxruntime
    except ImportError as error:
        raise OnnxDependencyError(
            "local ONNX inference requires onnxruntime; install the 'onnx' or 'onnx-gpu' extra for your provider"
        ) from error
    return numpy, onnxruntime


def _resize_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def prepare_wd14_moat(image: Image.Image, numpy: Any) -> Any:
    """Match WD14-style square padding and 448-pixel float32 batch layout.

    Existing WD14 ONNX exports commonly expect BGR channel order and 0..255
    float values.  The source Pillow image is already decoded by ``extract``.
    """
    rgb = numpy.asarray(image.convert("RGB"), dtype=numpy.uint8)
    bgr = rgb[:, :, ::-1]
    height, width = bgr.shape[:2]
    side = max(height, width)
    canvas = numpy.full((side, side, 3), 255, dtype=numpy.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top:top + height, left:left + width] = bgr
    resized = numpy.asarray(Image.fromarray(canvas[:, :, ::-1], "RGB").resize(
        (448, 448), Image.Resampling.LANCZOS), dtype=numpy.uint8)[:, :, ::-1]
    return resized.astype(numpy.float32)[None]


def prepare_nudenet(image: Image.Image, *, size: int, nhwc: bool, numpy: Any) -> Any:
    """Letterbox RGB pixels into NudeNet's expected normalized input layout."""
    source = image.convert("RGB")
    scale = size / max(source.width, source.height)
    width, height = max(1, round(source.width * scale)), max(1, round(source.height * scale))
    resized = _resize_rgb(source, (width, height))
    canvas = numpy.zeros((size, size, 3), dtype=numpy.float32)
    canvas[:height, :width] = numpy.asarray(resized, dtype=numpy.float32)
    normalized = canvas / 255.0
    return normalized[None] if nhwc else numpy.transpose(normalized, (2, 0, 1))[None]


def load_wd14_tags(path: Path) -> list[dict[str, str]]:
    """Load a caller-provided WD14 tag CSV; raw tag files are never persisted."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        tags = list(csv.DictReader(handle))
    if not tags or any(not row.get("name") for row in tags):
        raise ValueError("WD14 tag CSV must include a non-empty 'name' column")
    return tags


def _create_session(onnxruntime: Any, model_path: Path, providers: Sequence[str]) -> Any:
    """Create a bounded ONNX session and refuse silent provider fallback."""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        str(model_path), sess_options=options, providers=list(providers)
    )
    actual = list(session.get_providers())
    if providers and (not actual or actual[0] != providers[0]):
        raise ValueError(
            f"requested ONNX provider is unavailable: {providers[0]}; available: {actual}"
        )
    return session


class WD14MoatNudeNetAdapter:
    """Run caller-supplied WD14 MoAT and optional NudeNet ONNX sessions locally."""

    name = "wd14-moat+nudenet"

    def __init__(self, moat_session: Any, tags: Sequence[dict[str, str]], *, nudenet_session: Any | None = None,
                 embedding_dimensions: int | None = 1024, score_threshold: float = 0.15,
                 nms_threshold: float = 0.45):
        self.moat_session = moat_session
        self.tags = list(tags)
        self.nudenet_session = nudenet_session
        self.embedding_dimensions = embedding_dimensions
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._numpy = _numpy_module()

    @classmethod
    def from_paths(cls, moat_model: Path, wd14_tags: Path, *, nudenet_model: Path | None = None,
                   providers: Sequence[str] = ("CPUExecutionProvider",)) -> "WD14MoatNudeNetAdapter":
        """Open explicit local files only; no registry, cache, or network lookup is used."""
        _, onnxruntime = _optional_modules()
        for path in (moat_model, wd14_tags, nudenet_model):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"required local inference file does not exist: {path}")
        tags = load_wd14_tags(wd14_tags)
        try:
            moat_session = _create_session(onnxruntime, moat_model, providers)
            nudenet_session = (
                _create_session(onnxruntime, nudenet_model, providers) if nudenet_model else None
            )
        except Exception as error:
            raise ValueError(f"unable to initialize caller-supplied ONNX model: {error}") from error
        return cls(moat_session, tags, nudenet_session=nudenet_session)

    def _wd14(self, image: Image.Image) -> tuple[dict[str, Any], tuple[float, ...]]:
        numpy = self._numpy
        input_name = self.moat_session.get_inputs()[0].name
        outputs = self.moat_session.run(None, {input_name: prepare_wd14_moat(image, numpy)})
        if len(outputs) < 2:
            raise ValueError("WD14 MoAT session must return probabilities and an embedding")
        probabilities = numpy.asarray(outputs[0], dtype=numpy.float32).reshape(-1)
        if len(probabilities) != len(self.tags):
            raise ValueError(f"WD14 output/tag count mismatch: {len(probabilities)} and {len(self.tags)}")
        vector = tuple(float(value) for value in numpy.asarray(outputs[1], dtype=numpy.float32).reshape(-1))
        if self.embedding_dimensions is not None and len(vector) != self.embedding_dimensions:
            raise ValueError(f"MoAT embedding dimension mismatch: expected {self.embedding_dimensions}, got {len(vector)}")
        pairs = [(tag["name"], float(score), tag.get("category", ""))
                 for tag, score in zip(self.tags, probabilities, strict=True)]
        ratings = {name: score for name, score, _ in pairs if name in RATING_NAMES}
        top_tags = sorted((pair for pair in pairs if pair[0] not in RATING_NAMES), key=lambda item: item[1], reverse=True)
        scores = {
            "rating_general": ratings.get("general", 0.0),
            "rating_sensitive": ratings.get("sensitive", 0.0),
            "rating_questionable": ratings.get("questionable", 0.0),
            "rating_explicit": ratings.get("explicit", 0.0),
            "top_tags": [{"tag": name, "confidence": round(score, 6), "category": category}
                         for name, score, category in top_tags[:30] if score >= 0.25],
        }
        return scores, normalise_moat_vector(vector, dimensions=len(vector))

    def _nudenet(self, image: Image.Image) -> dict[str, Any]:
        if self.nudenet_session is None:
            return {}
        input_descriptor = self.nudenet_session.get_inputs()[0]
        shape = input_descriptor.shape
        nhwc = len(shape) == 4 and shape[-1] in (3, "3")
        size = next((dimension for dimension in shape if isinstance(dimension, int) and dimension in (320, 640)), 320)
        outputs = self.nudenet_session.run(None, {
            input_descriptor.name: prepare_nudenet(image, size=size, nhwc=nhwc, numpy=self._numpy)
        })
        detections = []
        for output in outputs:
            detections.extend(classwise_nms(output, score_threshold=self.score_threshold,
                                             iou_threshold=self.nms_threshold, canvas_size=size))
        return detection_summary(detections)

    def analyze(self, image: Image.Image, image_bytes: bytes, source: Path) -> AnalysisResult:
        """Analyze the already-decoded image; bytes/path are intentionally not read again."""
        del image_bytes, source
        wd14_scores, embedding = self._wd14(image)
        nudenet_scores = self._nudenet(image)
        features: dict[str, Any] = {"wd14_moat": wd14_scores}
        if nudenet_scores:
            features["nudenet"] = nudenet_scores
        moat_providers = getattr(self.moat_session, "get_providers", lambda: ["test_or_unknown"])()
        nude_providers = (
            getattr(self.nudenet_session, "get_providers", lambda: ["test_or_unknown"])()
            if self.nudenet_session else []
        )
        return AnalysisResult(
            embedding=embedding,
            features=features,
            evidence={
                "models": {
                    "wd14_moat": {"source": "user_supplied", "providers": list(moat_providers)},
                    "nudenet": (
                        {"source": "user_supplied", "providers": list(nude_providers)}
                        if self.nudenet_session else None
                    ),
                }
            },
        )
