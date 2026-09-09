"""Source-asset guardrails for curation pipelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class SourceChangedError(RuntimeError):
    """Raised when an input changed while it was being read."""


@dataclass(frozen=True)
class SourceSnapshot:
    """Identity fields used to detect source changes without writing to the source."""

    path: Path
    size: int
    mtime_ns: int
    sha256: str


def snapshot_source(path: Path) -> SourceSnapshot:
    """Read and fingerprint a regular source file; never mutate it."""
    _, snapshot = read_source(path)
    return snapshot


def read_source(path: Path) -> tuple[bytes, SourceSnapshot]:
    """Read a source exactly once and return its bytes with a stable snapshot."""
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"source is not a regular file: {path}")
    before = resolved.stat()
    data = resolved.read_bytes()
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SourceChangedError(f"source changed during read: {path}")
    return data, SourceSnapshot(resolved, after.st_size, after.st_mtime_ns, hashlib.sha256(data).hexdigest())


def read_verified(path: Path, expected: SourceSnapshot) -> bytes:
    """Return bytes only when the source still matches a prior snapshot."""
    data, current = read_source(path)
    if (current.size, current.mtime_ns, current.sha256) != (expected.size, expected.mtime_ns, expected.sha256):
        raise SourceChangedError(f"source no longer matches checkpoint: {path}")
    return data
