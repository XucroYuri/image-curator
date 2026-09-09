"""Pure, read-only helpers for versioned historical image reprocessing.

This module deliberately has no SQLite or CLI dependency.  It turns a legacy
migration CSV into immutable occurrence records and produces deterministic,
machine-readable candidate decisions.  Callers are responsible for persisting
those records in a run-specific database; none of the helpers writes to a
source image or its directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PureWindowsPath
from typing import Any

from .scan import IMAGE_EXTENSIONS

DEFAULT_EXCLUDED_COMPONENTS = frozenset(
    {".seekmeta", ".seektrash", "_extracted_workflows", "_test_20260728", "_organized_v2", "_系统",
     "test", "tests"}
)
TEMPORARY_SUFFIXES = frozenset({".tmp", ".part", ".partial", ".crdownload", ".download", ".incomplete"})
POSTED_BUCKETS = frozenset({"已发帖", "posted", "published"})


def sha256_file(path: Path) -> str:
    """Return a content digest while opening ``path`` only for reading."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_name(value: object) -> str:
    """Return a basename for POSIX or Windows path text without exposing roots."""
    text = str(value).replace("\\", "/")
    return PureWindowsPath(text).name or Path(text).name


def _normalise_config(value: Any, *, key: str = "") -> Any:
    """Canonicalise JSON-like config while replacing machine-local path values."""
    path_key = key.casefold().endswith(("path", "file", "model", "tags", "directory", "root"))
    if isinstance(value, Mapping):
        return {str(item_key): _normalise_config(item_value, key=str(item_key))
                for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise_config(item, key=key) for item in value]
    if isinstance(value, Path):
        return _portable_name(value)
    if path_key and isinstance(value, str):
        return _portable_name(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("configuration must not contain non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"configuration contains unsupported value: {type(value).__name__}")


def config_fingerprint(mapping: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint without incorporating absolute paths."""
    canonical = json.dumps(_normalise_config(mapping), ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _manifest_file(path: Path) -> dict[str, str]:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"required model artifact is not a file: {_portable_name(candidate)}")
    return {"name": _portable_name(candidate), "sha256": sha256_file(candidate)}


def build_model_manifest(moat_model: Path, wd14_tags: Path, nudenet_model: Path | None,
                         providers: Sequence[str]) -> dict[str, Any]:
    """Build a portable, verifiable local-model manifest.

    Model paths are intentionally reduced to file names before persistence.
    The caller must resolve local paths again when a run is resumed.
    """
    provider_list = [str(provider) for provider in providers]
    if not provider_list or any(not provider.strip() for provider in provider_list):
        raise ValueError("at least one non-empty ONNX provider is required")
    if len(set(provider_list)) != len(provider_list):
        raise ValueError("ONNX providers must not be duplicated")
    models: dict[str, dict[str, str]] = {
        "moat": _manifest_file(moat_model),
        "wd14_tags": _manifest_file(wd14_tags),
    }
    if nudenet_model is not None:
        models["nudenet"] = _manifest_file(nudenet_model)
    return {
        "adapter": "wd14-moat-nudenet",
        "adapter_version": "wd14-moat-nudenet-v1",
        "embedding_dimensions": 1024,
        "models": models,
        "postprocessing": {"nudenet_score_threshold": 0.15, "nudenet_nms_iou": 0.45},
        "preprocessing": {"wd14": "bgr-square-448-v1", "nudenet": "rgb-letterbox-v1"},
        "providers": provider_list,
    }


def is_excluded_path(path: Path | str, *, excluded_components: Iterable[str] = DEFAULT_EXCLUDED_COMPONENTS) -> bool:
    """Return whether a historical path is outside the reprocessing contract."""
    text = str(path).replace("/", "\\")
    components = [part.casefold() for part in PureWindowsPath(text).parts if part not in ("\\", "/")]
    excluded = {component.casefold() for component in excluded_components}
    if any(component in excluded for component in components):
        return True
    name = components[-1] if components else ""
    return name.startswith("~$") or any(name.endswith(suffix) for suffix in TEMPORARY_SUFFIXES)


def _first(record: Mapping[str, str], *names: str) -> str:
    casefolded = {str(key).casefold(): value for key, value in record.items()}
    for name in names:
        value = casefolded.get(name.casefold())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _integer(value: str) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


@dataclass(frozen=True)
class BaselineOccurrence:
    """One historical path and its old bucket, never a classification truth label."""

    source_path: str
    old_bucket: str
    destination_path: str | None = None
    source_engine: str | None = None
    entity: str | None = None
    subject: str | None = None
    source_size: int | None = None
    baseline_sha256: str | None = None
    audit_locked: bool = False
    baseline_identity_unverifiable: bool = False

    def __post_init__(self) -> None:
        # CSV exports before hashing cannot prove that the historical byte
        # sequence and today's file are identical.  Published rows are a
        # path-level audit constraint even when an older CSV omitted a flag.
        if self.baseline_sha256 is None and not self.baseline_identity_unverifiable:
            object.__setattr__(self, "baseline_identity_unverifiable", True)
        if self.old_bucket.casefold() in POSTED_BUCKETS and not self.audit_locked:
            object.__setattr__(self, "audit_locked", True)

    @property
    def occurrence_id(self) -> str:
        return hashlib.sha256(self.source_path.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"occurrence_id": self.occurrence_id}


def load_migration_csv(path: Path, *, excluded_components: Iterable[str] = DEFAULT_EXCLUDED_COMPONENTS) -> list[BaselineOccurrence]:
    """Load only image rows from a legacy migration plan in deterministic order.

    ``source``/``source_path``/``path`` and common old-bucket aliases are
    accepted so exports from older tools remain usable.  The CSV itself is only
    a baseline audit source; folder names and notes are not labels.
    """
    occurrences: list[BaselineOccurrence] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("migration CSV must have a header row")
        for row in reader:
            kind = _first(row, "kind", "type", "asset_kind").casefold()
            if kind != "image":
                continue
            source = _first(row, "source", "source_path", "path", "source_file")
            if not source or is_excluded_path(source, excluded_components=excluded_components):
                continue
            suffix = PureWindowsPath(source).suffix.casefold()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            old_bucket = _first(row, "bucket", "old_bucket", "status", "state") or "待复核"
            baseline_sha = _first(row, "sha256", "source_sha256", "hash") or None
            if baseline_sha and (len(baseline_sha) != 64 or any(char not in "0123456789abcdefABCDEF" for char in baseline_sha)):
                baseline_sha = None
            occurrences.append(BaselineOccurrence(
                source_path=source,
                old_bucket=old_bucket,
                destination_path=_first(row, "destination", "destination_path", "target") or None,
                source_engine=_first(row, "source_engine", "engine", "model") or None,
                entity=_first(row, "entity", "ip", "project") or None,
                subject=_first(row, "subject", "character", "name") or None,
                source_size=_integer(_first(row, "size", "source_size", "bytes")),
                baseline_sha256=baseline_sha.lower() if baseline_sha else None,
                audit_locked=old_bucket.casefold() in POSTED_BUCKETS,
                baseline_identity_unverifiable=baseline_sha is None,
            ))
    return sorted(occurrences, key=lambda item: item.source_path.casefold())


@dataclass(frozen=True)
class FrozenOccurrence:
    """Read-only frozen path evidence used before an inference run."""

    occurrence: BaselineOccurrence
    status: str
    size: int | None = None
    mtime_ns: int | None = None
    sha256: str | None = None
    reason: str | None = None

    @property
    def asset_id(self) -> str | None:
        return self.sha256

    def as_dict(self) -> dict[str, Any]:
        return self.occurrence.as_dict() | {
            "status": self.status, "size": self.size, "mtime_ns": self.mtime_ns,
            "sha256": self.sha256, "reason": self.reason,
        }


def _to_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def freeze_occurrence(occurrence: BaselineOccurrence, *, cutoff: datetime,
                      now: datetime | None = None, stable_seconds: int = 120,
                      digest_file: Callable[[Path], str] = sha256_file) -> FrozenOccurrence:
    """Freeze one regular, stable file without modifying it.

    A change after the fixed cutoff is deferred.  A file too recently modified
    is considered changed and is not read for hashing.  Missing, symlinked, and
    non-regular objects are also never sent to inference.
    """
    if stable_seconds < 0:
        raise ValueError("stable_seconds must be non-negative")
    candidate = Path(occurrence.source_path)
    now_value = _to_utc(now or datetime.now(UTC))
    cutoff_value = _to_utc(cutoff)
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return FrozenOccurrence(occurrence, "missing", reason="not_a_regular_file")
        info = candidate.stat()
    except OSError:
        return FrozenOccurrence(occurrence, "missing", reason="missing_or_unreadable")
    modified = datetime.fromtimestamp(info.st_mtime, UTC)
    if modified > cutoff_value:
        return FrozenOccurrence(occurrence, "added_deferred", size=info.st_size, mtime_ns=info.st_mtime_ns,
                                reason="modified_after_cutoff")
    if modified > now_value - timedelta(seconds=stable_seconds):
        return FrozenOccurrence(occurrence, "source_changed", size=info.st_size, mtime_ns=info.st_mtime_ns,
                                reason="not_stable_for_required_interval")
    try:
        digest = digest_file(candidate)
        after = candidate.stat()
    except OSError:
        return FrozenOccurrence(occurrence, "source_changed", reason="changed_or_unreadable_while_hashing")
    if (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
        return FrozenOccurrence(occurrence, "source_changed", size=after.st_size, mtime_ns=after.st_mtime_ns,
                                reason="changed_while_hashing")
    if occurrence.baseline_sha256 and digest != occurrence.baseline_sha256:
        return FrozenOccurrence(occurrence, "source_changed", size=after.st_size, mtime_ns=after.st_mtime_ns,
                                sha256=digest, reason="baseline_sha256_mismatch")
    state = "baseline_identity_unverifiable" if occurrence.baseline_identity_unverifiable else "processed"
    return FrozenOccurrence(occurrence, state, size=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest)


def freeze_occurrences(occurrences: Iterable[BaselineOccurrence], *, cutoff: datetime,
                       now: datetime | None = None, stable_seconds: int = 120,
                       digest_file: Callable[[Path], str] = sha256_file) -> list[FrozenOccurrence]:
    """Freeze every baseline path in deterministic order."""
    return [freeze_occurrence(item, cutoff=cutoff, now=now, stable_seconds=stable_seconds,
                              digest_file=digest_file)
            for item in sorted(occurrences, key=lambda item: item.source_path.casefold())]


def added_deferred_rows(discovered_paths: Iterable[Path | str], occurrences: Iterable[BaselineOccurrence]) -> list[dict[str, Any]]:
    """Return paths discovered after baseline creation without adding them to this run."""
    known = {item.source_path.casefold() for item in occurrences}
    rows = []
    for source in sorted({str(path) for path in discovered_paths}, key=str.casefold):
        if source.casefold() not in known and not is_excluded_path(source):
            rows.append({"source_path": source, "status": "added_deferred", "reason": "not_in_baseline"})
    return rows


def unique_assets(frozen: Iterable[FrozenOccurrence]) -> dict[str, tuple[str, ...]]:
    """Group stable baseline paths by content digest for one-inference-per-content scheduling."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in frozen:
        if item.sha256 and item.status in {"processed", "baseline_identity_unverifiable"}:
            grouped[item.sha256].append(item.occurrence.source_path)
    return {digest: tuple(sorted(paths, key=str.casefold)) for digest, paths in sorted(grouped.items())}


@dataclass(frozen=True)
class ProposedDecision:
    """A model/decision result attached to one content digest, never a filesystem action."""

    asset_id: str
    proposed_bucket: str | None
    identity: str | None = None
    confidence: float | None = None
    review_reasons: tuple[str, ...] = ()
    model_fingerprint: str | None = None
    decision_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


def build_diff_rows(frozen: Iterable[FrozenOccurrence], decisions: Mapping[str, ProposedDecision]) -> list[dict[str, Any]]:
    """Create one audit row per path, preserving old posted state as effective state."""
    rows: list[dict[str, Any]] = []
    for item in sorted(frozen, key=lambda value: value.occurrence.source_path.casefold()):
        decision = decisions.get(item.asset_id or "")
        proposed = decision.proposed_bucket if decision else None
        locked = item.occurrence.audit_locked
        effective = item.occurrence.old_bucket if locked else proposed
        processing_state = "processed" if item.status in {"processed", "baseline_identity_unverifiable"} else item.status
        review_reasons = list(decision.review_reasons if decision else ())
        if item.status not in {"processed", "baseline_identity_unverifiable"}:
            review_reasons.append(item.status)
        if decision is None and processing_state == "processed":
            review_reasons.append("decision_missing")
        rows.append({
            "occurrence_id": item.occurrence.occurrence_id,
            "source_path": item.occurrence.source_path,
            "asset_id": item.asset_id,
            "status": item.status,
            "processing_state": processing_state,
            "old_bucket": item.occurrence.old_bucket,
            "proposed_bucket": proposed,
            "effective_bucket": effective,
            "identity": decision.identity if decision else None,
            "confidence": decision.confidence if decision else None,
            "audit_locked": locked,
            "action": None if locked else "candidate_only",
            "review_reasons": sorted(set(review_reasons)),
            "reason": item.reason,
            "model_fingerprint": decision.model_fingerprint if decision else None,
            "decision_fingerprint": decision.decision_fingerprint if decision else None,
        })
    return rows


def summarize_diff(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce compact JSON-ready counts without inspecting any source file."""
    materialized = list(rows)
    statuses = Counter(str(row.get("status", "unknown")) for row in materialized)
    effective = Counter(str(row.get("effective_bucket")) for row in materialized if row.get("effective_bucket") is not None)
    review = sum(bool(row.get("review_reasons")) for row in materialized)
    return {
        "occurrences": len(materialized),
        "audit_locked": sum(bool(row.get("audit_locked")) for row in materialized),
        "review_required": review,
        "by_status": dict(sorted(statuses.items())),
        "by_effective_bucket": dict(sorted(effective.items())),
    }


def _row_strata(row: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = row.get("review_strata")
    if isinstance(explicit, str) and explicit.strip():
        return (explicit.strip(),)
    if isinstance(explicit, Sequence) and not isinstance(explicit, (bytes, bytearray)):
        values = tuple(sorted({str(item).strip() for item in explicit if str(item).strip()}))
        if values:
            return values
    reasons = [str(reason) for reason in row.get("review_reasons", ())]
    categories = []
    for prefix, category in (("safety", "safety"), ("identity", "identity"), ("quality", "quality"),
                             ("source", "source"), ("baseline", "baseline")):
        if any(reason.casefold().startswith(prefix) for reason in reasons):
            categories.append(category)
    return tuple(categories) or ("general",)


def select_review_queue(rows: Iterable[Mapping[str, Any]], *, limit: int = 500,
                        seed: str = "image-curator-review-v1") -> list[dict[str, Any]]:
    """Select a deterministic balanced review sample from evidence, never path hints.

    Callers may supply explicit ``review_strata`` derived from model evidence.
    Directory names, prompt text, old buckets, and destination paths are not
    read to determine strata or ranking.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for stratum in _row_strata(row):
            grouped[stratum].append(row)
    if not grouped:
        return []
    def rank(row: Mapping[str, Any]) -> tuple[str, str]:
        stable_id = str(row.get("occurrence_id") or row.get("asset_id") or "")
        token = hashlib.sha256(f"{seed}:{stable_id}".encode()).hexdigest()
        return token, stable_id
    ordered = {key: sorted(values, key=rank) for key, values in grouped.items()}
    keys = sorted(ordered)
    selected: dict[str, tuple[Mapping[str, Any], str]] = {}
    positions = {key: 0 for key in keys}
    while len(selected) < limit:
        progressed = False
        for key in keys:
            candidates = ordered[key]
            while positions[key] < len(candidates):
                candidate = candidates[positions[key]]
                positions[key] += 1
                identifier = str(candidate.get("occurrence_id") or candidate.get("asset_id") or rank(candidate)[1])
                if identifier not in selected:
                    selected[identifier] = (candidate, key)
                    progressed = True
                    break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    queue = []
    for identifier, (row, stratum) in sorted(selected.items(), key=lambda item: rank(item[1][0])):
        queue.append({
            "review_id": hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest(),
            "occurrence_id": row.get("occurrence_id"),
            "asset_id": row.get("asset_id"),
            "stratum": stratum,
            "status": row.get("status"),
            "proposed_bucket": row.get("proposed_bucket"),
            "identity": row.get("identity"),
            "confidence": row.get("confidence"),
            "review_reasons": row.get("review_reasons", []),
            "evidence": row.get("evidence", {}),
        })
    return queue


def build_machine_outputs(diff_rows: Iterable[Mapping[str, Any]], *, review_limit: int = 500,
                          review_seed: str = "image-curator-review-v1") -> dict[str, Any]:
    """Return the three JSON-serialisable report payloads expected by a caller."""
    rows = [dict(row) for row in diff_rows]
    review_rows = select_review_queue(rows, limit=review_limit, seed=review_seed)
    return {"summary": summarize_diff(rows), "diff_rows": rows, "review_rows": review_rows}
