"""Interfaces for user-supplied inference backends; no weights are bundled."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class InferenceAdapter(Protocol):
    """An optional local or remote embedding provider supplied by the user."""

    name: str

    def embed(self, image_bytes: bytes, source: Path) -> Sequence[float]:
        """Return one embedding for a read-only source; never modify `source`."""


class CallableInferenceAdapter:
    """Wrap a caller-owned embedding function without importing a model runtime."""

    def __init__(self, name: str, embedder: Callable[[bytes, Path], Sequence[float]]):
        self.name = name
        self._embedder = embedder

    def embed(self, image_bytes: bytes, source: Path) -> Sequence[float]:
        return self._embedder(image_bytes, source)


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
