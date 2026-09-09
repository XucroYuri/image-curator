"""Read-only recursive discovery and checkpoint enqueueing."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .checkpoint import CheckpointStore, WorkItem
from .readonly import SourceChangedError, read_source

IMAGE_EXTENSIONS = frozenset({".avif", ".bmp", ".gif", ".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})


@dataclass(frozen=True)
class ScanResult:
    """Counts from a scan; errors refer to files that were skipped safely."""

    discovered: int = 0
    enqueued: int = 0
    duplicates: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def iter_image_files(roots: Iterable[Path], *, extensions: Iterable[str] = IMAGE_EXTENSIONS,
                     follow_symlinks: bool = False) -> Iterable[Path]:
    """Yield image-like files in deterministic order without following links by default."""
    suffixes = {extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
                for extension in extensions}
    for root in sorted((Path(root) for root in roots), key=lambda path: str(path).casefold()):
        if not root.is_dir():
            raise NotADirectoryError(f"scan root is not a directory: {root}")
        for directory, subdirectories, files in os.walk(root, followlinks=follow_symlinks):
            subdirectories.sort(key=str.casefold)
            files.sort(key=str.casefold)
            for filename in files:
                path = Path(directory, filename)
                if path.suffix.casefold() in suffixes and (follow_symlinks or not path.is_symlink()):
                    yield path


def scan_and_enqueue(store: CheckpointStore, roots: Iterable[Path], *,
                     extensions: Iterable[str] = IMAGE_EXTENSIONS, follow_symlinks: bool = False) -> ScanResult:
    """Fingerprint each eligible file once and add unique content to the local checkpoint.

    `asset_id` is the SHA-256 of the bytes, so copies encountered in one scan or
    a later resume are deduplicated.  This function never modifies a scan root.
    """
    result = ScanResult()
    for path in iter_image_files(roots, extensions=extensions, follow_symlinks=follow_symlinks):
        result = ScanResult(result.discovered + 1, result.enqueued, result.duplicates, result.errors)
        try:
            _, snapshot = read_source(path)
            if store.enqueue(WorkItem(snapshot.sha256, snapshot)):
                result = ScanResult(result.discovered, result.enqueued + 1, result.duplicates, result.errors)
            else:
                result = ScanResult(result.discovered, result.enqueued, result.duplicates + 1, result.errors)
        except (OSError, SourceChangedError, ValueError):
            result = ScanResult(result.discovered, result.enqueued, result.duplicates, result.errors + 1)
    return result
