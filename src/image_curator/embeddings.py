"""MoAT embedding validation and compact storage helpers."""

from __future__ import annotations

import struct
from collections.abc import Sequence


def normalise_moat_vector(values: Sequence[float], *, dimensions: int = 1024) -> tuple[float, ...]:
    """Return a unit-length MoAT vector after checking its declared dimension."""
    vector = tuple(float(value) for value in values)
    if len(vector) != dimensions:
        raise ValueError(f"expected {dimensions} embedding values, got {len(vector)}")
    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        raise ValueError("embedding must not be the zero vector")
    return tuple(value / magnitude for value in vector)


def moat_float16_blob(values: Sequence[float], *, dimensions: int = 1024) -> bytes:
    """Validate, normalise, and encode an embedding as an IEEE-754 float16 BLOB."""
    vector = normalise_moat_vector(values, dimensions=dimensions)
    return struct.pack(f"<{dimensions}e", *vector)


def moat_vector_from_blob(blob: bytes, *, dimensions: int = 1024) -> tuple[float, ...]:
    """Decode a stored float16 embedding and re-normalise numerical rounding error."""
    expected_size = dimensions * 2
    if len(blob) != expected_size:
        raise ValueError(f"expected {expected_size} bytes, got {len(blob)}")
    return normalise_moat_vector(struct.unpack(f"<{dimensions}e", blob), dimensions=dimensions)
