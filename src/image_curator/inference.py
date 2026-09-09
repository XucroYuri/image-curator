"""Interfaces for user-supplied inference backends; no weights are bundled."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class AnalysisResult:
    """Structured optional inference output merged into one checkpoint item."""

    embedding: Sequence[float] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InferenceAdapter(Protocol):
    """An optional local or remote embedding provider supplied by the user."""

    name: str

    def embed(self, image_bytes: bytes, source: Path) -> Sequence[float]:
        """Return one embedding for a read-only source; never modify `source`."""


@runtime_checkable
class AnalysisAdapter(Protocol):
    """Adapter using the already-decoded Pillow image and optional source bytes."""

    name: str

    def analyze(self, image: Image.Image, image_bytes: bytes, source: Path) -> AnalysisResult:
        """Return structured analysis without reading, writing, or decoding `source` again."""


@runtime_checkable
class ManifestedAdapter(Protocol):
    """Adapter that can participate in a reproducible versioned run."""

    name: str

    def manifest(self) -> Mapping[str, Any]:
        """Return a path-free manifest that fingerprints every inference input."""


def require_adapter_manifest(adapter: object) -> dict[str, Any]:
    """Validate a reproducible adapter manifest without accepting local paths."""
    if not isinstance(adapter, ManifestedAdapter):
        raise TypeError("versioned reprocessing requires an adapter manifest")
    value = dict(adapter.manifest())
    if not value.get("name") or not value.get("version"):
        raise ValueError("adapter manifest requires non-empty name and version")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("adapter manifest requires at least one fingerprinted artifact")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not artifact.get("role") or not artifact.get("sha256"):
            raise ValueError("each adapter artifact requires role and sha256")
        if "path" in artifact:
            raise ValueError("adapter manifests must not persist local paths")
    return value


class CallableInferenceAdapter:
    """Wrap a caller-owned embedding function without importing a model runtime."""

    def __init__(self, name: str, embedder: Callable[[bytes, Path], Sequence[float]]):
        self.name = name
        self._embedder = embedder

    def embed(self, image_bytes: bytes, source: Path) -> Sequence[float]:
        return self._embedder(image_bytes, source)


class CombinedAnalysisAdapter:
    """Combine structured adapters and legacy embedding adapters after one image decode."""

    def __init__(self, *adapters: AnalysisAdapter | InferenceAdapter):
        if not adapters:
            raise ValueError("at least one adapter is required")
        self.adapters = adapters
        self.name = "+".join(adapter.name for adapter in adapters)

    def analyze(self, image: Image.Image, image_bytes: bytes, source: Path) -> AnalysisResult:
        embedding: Sequence[float] | None = None
        features: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        for adapter in self.adapters:
            if isinstance(adapter, AnalysisAdapter):
                result = adapter.analyze(image, image_bytes, source)
            else:
                result = AnalysisResult(embedding=adapter.embed(image_bytes, source),
                                        features={"adapter": adapter.name})
            if result.embedding is not None:
                if embedding is not None:
                    raise ValueError("combined adapters produced more than one embedding")
                embedding = result.embedding
            features[adapter.name] = result.features
            evidence[adapter.name] = result.evidence
        return AnalysisResult(embedding=embedding, features=features, evidence=evidence)


def load_adapter(specification: str) -> InferenceAdapter:
    """Load a caller-provided ``module:factory`` adapter without fetching weights.

    The factory is called with no arguments and must return an
    :class:`InferenceAdapter`.  Model files, tags, credentials, and runtime
    configuration remain entirely the caller's responsibility.
    """
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("adapter must use 'module:factory' syntax")
    factory = getattr(import_module(module_name), factory_name, None)
    if not callable(factory):
        raise ValueError(f"adapter factory not found: {specification}")
    adapter = factory()
    if not isinstance(adapter, InferenceAdapter):
        raise TypeError("adapter factory must return an InferenceAdapter")
    return adapter
