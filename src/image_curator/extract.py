"""Resumable metadata and optional-vector extraction from a local checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .checkpoint import CheckpointStore
from .embeddings import moat_float16_blob
from .evidence import metadata_evidence
from .inference import InferenceAdapter
from .readonly import read_verified


@dataclass(frozen=True)
class ExtractResult:
    """Per-run counts; failures remain retryable in the checkpoint."""

    attempted: int = 0
    completed: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _image_features(image_bytes: bytes) -> tuple[dict[str, object], dict[str, object]]:
    """Extract basic Pillow metadata in memory without retaining raw embedded values."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            metadata = dict(image.info)
            features: dict[str, object] = {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
            }
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"image decode failed: {error}") from error
    return features, metadata_evidence(metadata)


def extract_pending(store: CheckpointStore, *, adapter: InferenceAdapter | None = None,
                    limit: int | None = None) -> ExtractResult:
    """Extract pending unique assets; each item becomes COMPLETE or RETRYABLE_FAILED atomically."""
    result = ExtractResult()
    for item in store.pending(limit):
        result = ExtractResult(result.attempted + 1, result.completed, result.failed)
        try:
            image_bytes = read_verified(item.source.path, item.source)
            features, evidence = _image_features(image_bytes)
            embedding_blob: bytes | None = None
            dimensions: int | None = None
            if adapter is None:
                features["inference"] = {"state": "not_requested"}
            else:
                vector = tuple(adapter.embed(image_bytes, item.source.path))
                dimensions = len(vector)
                embedding_blob = moat_float16_blob(vector, dimensions=dimensions)
                features["inference"] = {"adapter": adapter.name, "state": "complete"}
            store.complete(item.asset_id, metadata_evidence=evidence, features=features,
                           embedding_f16=embedding_blob, embedding_dim=dimensions)
            result = ExtractResult(result.attempted, result.completed + 1, result.failed)
        except Exception as error:  # Persist every image-specific decode, I/O, or adapter failure for retry.
            store.fail_retryable(item.asset_id, error)
            result = ExtractResult(result.attempted, result.completed, result.failed + 1)
    return result
