"""Metadata evidence that stays separate from file-system hints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def metadata_evidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Create a compact evidence record without retaining raw metadata values.

    The digest lets callers prove which metadata was examined while avoiding
    prompt, author, and other potentially sensitive text in the checkpoint.
    """
    canonical = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True, default=str,
                           separators=(",", ":"))
    keys = sorted(str(key) for key in metadata)
    return {
        "metadata_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "metadata_keys": keys,
        "metadata_field_count": len(keys),
    }
