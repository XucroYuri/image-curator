"""Restart-safe, same-share image-library migration.

This module intentionally performs only file renames.  It never overwrites a
destination and never removes a source until the destination identity is
verified.  Its SQLite journal lives on the workstation, not on the NAS.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .reprocess import DEFAULT_EXCLUDED_COMPONENTS, is_excluded_path, sha256_file

SCHEMA_VERSION = 1
POSTED = frozenset({"已发帖", "posted", "published"})


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def safe_component(value: str | None, fallback: str) -> str:
    """Make one Windows-safe component while retaining readable names."""
    text = (value or "").strip() or fallback
    text = "".join("_" if char in '<>:"/\\|?*' or ord(char) < 32 else char for char in text)
    text = text.rstrip(". ")[:120]
    return text or fallback


def source_directory(source: Path, root: Path) -> str:
    """Return the first directory below ``output`` used for batching."""
    try:
        relative = source.relative_to(root)
    except ValueError:
        return "(outside-root)"
    return relative.parts[0] if relative.parts else "(root)"


def destination_for(source: Path, *, root: Path, target_root: Path, entity: str | None,
                    subject: str | None, engine: str | None, old_bucket: str | None,
                    audit_locked: bool, sha256: str, mtime_ns: int) -> tuple[Path, str]:
    """Return the v3 target and a conservative effective state.

    There is deliberately no unsafe automatic SFW/NSFW assertion here.  A
    calibrated decision policy may update this state in a later run; files
    without that policy remain in ``待复核``.  Published paths stay locked.
    """
    if audit_locked or (old_bucket or "").casefold() in POSTED:
        state = "已发帖"
    else:
        state = "待复核"
    moment = datetime.fromtimestamp(mtime_ns / 1_000_000_000)
    engine_part = safe_component(engine, "未知模型")
    stem = safe_component(source.stem, "image")
    # Model/version remains in every new filename, even after a user later
    # moves the file out of its engine directory.
    prefix = engine_part.casefold()
    name = source.name if stem.casefold().startswith(prefix) else f"{engine_part}__{stem}{source.suffix.lower()}"
    unclassified = (not entity or entity in {"_未归档", "未识别"} or
                    not subject or subject in {"未识别", "待分类"})
    if unclassified:
        path = (target_root / "待分拣" / state / f"{moment:%Y}" / f"{moment:%m}" / name)
    else:
        path = (target_root / safe_component(entity, "_未归档") / safe_component(subject, "未识别") /
                state / f"{moment:%Y}" / f"{moment:%m}" / name)
    # When the final tree is rooted directly at ``output``, never place a
    # result back inside the source model directory being drained.
    if target_root == root:
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0].casefold() == source_directory(source, root).casefold():
            path = target_root / "待分拣" / state / f"{moment:%Y}" / f"{moment:%m}" / name
    return path, state


def inherit_nested_context(source: Path, root: Path, *, entity: str | None,
                           subject: str | None, engine: str | None) -> tuple[str | None, str | None, str | None]:
    """Recover semantic folders from Krea/Grok trees when CSV fields are sparse.

    Their export layouts contain helper folders (``_json``, ``_txt``,
    ``_hitl``) and sometimes pre-sharded year/month folders.  Those are
    transport metadata, never an IP/project/character label.  The first two
    remaining components are safe fallbacks only when the audited CSV did not
    already provide a value.
    """
    try:
        parts = list(source.relative_to(root).parts)
    except ValueError:
        return entity, subject, engine
    if not parts:
        return entity, subject, engine
    model = parts[0]
    if model.casefold() not in {"krea2_turbo", "grok imagine"}:
        return entity, subject, engine
    semantic: list[str] = []
    for part in parts[1:-1]:
        folded = part.casefold()
        if folded in {"_json", "_txt", "_hitl", "json", "txt"} or folded.startswith("p1-"):
            continue
        if len(part) == 4 and part.isdigit():
            continue
        if len(part) == 2 and part.isdigit():
            continue
        semantic.append(part)
    clean_entity = entity if entity and entity not in {"_未归档", "未识别"} else (semantic[0] if semantic else entity)
    clean_subject = subject if subject and subject not in {"未识别", "待分类"} else (semantic[1] if len(semantic) > 1 else subject)
    return clean_entity, clean_subject, model if not engine or engine in {"other", "未知模型"} else engine


@dataclass(frozen=True)
class MigrationRow:
    source: Path
    destination: Path
    sha256: str
    size: int
    mtime_ns: int
    state: str


class MigrationStore:
    """Durable local journal for a single migration plan."""
    def __init__(self, database: Path):
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def __enter__(self): return self
    def __exit__(self, *_): self.connection.close()

    def _migrate(self) -> None:
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError("migration database is newer than this program")
        if version:
            return
        self.connection.executescript("""
          CREATE TABLE migration_runs(
            run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source_root TEXT NOT NULL,
            target_root TEXT NOT NULL, status TEXT NOT NULL,
            CHECK(status IN ('PLANNED','RUNNING','PAUSED','COMPLETE','FAILED')));
          CREATE TABLE migration_items(
            run_id TEXT NOT NULL REFERENCES migration_runs(run_id) ON DELETE CASCADE,
            source_path TEXT NOT NULL, destination_path TEXT NOT NULL, kind TEXT NOT NULL,
            source_size INTEGER NOT NULL, source_mtime_ns INTEGER NOT NULL, source_sha256 TEXT NOT NULL,
            effective_state TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'PLANNED', attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY(run_id, source_path), UNIQUE(run_id, destination_path),
            CHECK(state IN ('PLANNED','MOVED','SOURCE_CHANGED','COLLISION','FAILED','MISSING')));
          CREATE INDEX migration_items_state ON migration_items(run_id, state, source_path);
          PRAGMA user_version=1;
        """)
        self.connection.commit()

    def create(self, run_id: str, root: Path, target: Path, rows: Iterable[MigrationRow]) -> int:
        if self.connection.execute("SELECT 1 FROM migration_runs WHERE run_id=?", (run_id,)).fetchone():
            return self.add_rows(run_id, rows)
        with self.connection:
            self.connection.execute("INSERT INTO migration_runs VALUES (?,?,?,?,?)",
                                    (run_id, _now(), str(root), str(target), "PLANNED"))
        return self.add_rows(run_id, rows)

    def add_rows(self, run_id: str, rows: Iterable[MigrationRow]) -> int:
        """Append newly settled directories to an existing plan idempotently."""
        if not self.connection.execute("SELECT 1 FROM migration_runs WHERE run_id=?", (run_id,)).fetchone():
            raise KeyError(f"unknown migration run: {run_id}")
        inserted = 0
        for row in rows:
            existing = self.connection.execute(
                "SELECT source_path FROM migration_items WHERE run_id=? AND destination_path=?",
                (run_id, str(row.destination))).fetchone()
            if existing and str(existing[0]) != str(row.source):
                raise ValueError(f"destination collision: {row.destination}")
            with self.connection:
                cursor = self.connection.execute("""
                  INSERT OR IGNORE INTO migration_items(run_id,source_path,destination_path,kind,source_size,
                    source_mtime_ns,source_sha256,effective_state,updated_at)
                  VALUES (?,?,?,?,?,?,?,?,?)
                """, (run_id, str(row.source), str(row.destination), row.source.suffix.casefold(), row.size,
                       row.mtime_ns, row.sha256, row.state, _now()))
            inserted += cursor.rowcount
        return inserted

    def run_status(self, run_id: str) -> str:
        row = self.connection.execute("SELECT status FROM migration_runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(f"unknown migration run: {run_id}")
        return str(row[0])

    def rows(self, run_id: str) -> list[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM migration_items WHERE run_id=? ORDER BY source_path",
                                       (run_id,)).fetchall()

    def source_paths(self, run_id: str) -> set[str]:
        return {str(row[0]).casefold() for row in self.connection.execute(
            "SELECT source_path FROM migration_items WHERE run_id=?", (run_id,))}

    def mark(self, run_id: str, source: str, state: str, error: str | None = None) -> None:
        with self.connection:
            self.connection.execute("""UPDATE migration_items SET state=?, attempts=attempts+1,
              last_error=?, updated_at=? WHERE run_id=? AND source_path=?""",
                                    (state, error, _now(), run_id, source))

    def summary(self, run_id: str) -> dict[str, int]:
        return dict(self.connection.execute("SELECT state,count(*) FROM migration_items WHERE run_id=? GROUP BY state",
                                            (run_id,)).fetchall())


def ready_directories(feature_database: Path, feature_run_id: str, root: Path | None = None) -> set[str]:
    """Return source top-level directories with no non-terminal occurrences."""
    with sqlite3.connect(feature_database) as connection:
        rows = connection.execute("""
          SELECT source_path, occurrence_state FROM run_occurrences WHERE run_id=?
        """, (feature_run_id,)).fetchall()
    grouped: dict[str, list[str]] = {}
    base = Path(root) if root is not None else None
    for source, state in rows:
        source_path = Path(source)
        grouped.setdefault(source_directory(source_path, base or Path(_feature_root(source))), []).append(state)
    terminal = {"PROCESSED", "MISSING", "SOURCE_CHANGED", "ADDED_DEFERRED"}
    return {directory for directory, states in grouped.items() if states and set(states) <= terminal}


def _feature_root(source: str) -> str:
    """Extract the output directory from a stored path without host-specific assumptions."""
    path = Path(source)
    parts = list(path.parts)
    folded = [part.casefold() for part in parts]
    if "output" in folded:
        return str(Path(*parts[:folded.index("output") + 1]))
    return str(path.parent)


def validate_roots(root: Path, target: Path) -> tuple[Path, Path]:
    root = Path(os.path.abspath(root))
    target = Path(os.path.abspath(target))
    if root.is_symlink() or target.is_symlink():
        raise ValueError("source and target roots must not be symbolic links")
    if target != root and root not in target.parents:
        raise ValueError("target must be the authorized source root or its child")
    return root, target


def execute_row(store: MigrationStore, run_id: str, row: sqlite3.Row) -> str:
    """Move exactly one prehashed item, handling interruption reconciliation."""
    source, destination = Path(row["source_path"]), Path(row["destination_path"])
    expected = row["source_sha256"]
    # An interrupted rename is safe to acknowledge only after validating the
    # destination bytes.  This makes reruns idempotent.
    if destination.is_file() and sha256_file(destination) == expected:
        if not source.exists():
            store.mark(run_id, str(source), "MOVED")
            return "MOVED"
        if sha256_file(source) == expected:
            store.mark(run_id, str(source), "COLLISION", "both_source_and_destination_exist")
            return "COLLISION"
    if not source.is_file():
        store.mark(run_id, str(source), "MISSING", "source_missing")
        return "MISSING"
    info = source.stat()
    if info.st_size != row["source_size"] or info.st_mtime_ns != row["source_mtime_ns"] or sha256_file(source) != expected:
        store.mark(run_id, str(source), "SOURCE_CHANGED", "source_identity_changed")
        return "SOURCE_CHANGED"
    if destination.exists():
        store.mark(run_id, str(source), "COLLISION", "destination_exists_with_different_content")
        return "COLLISION"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(source, destination)  # same SMB share: atomic and no duplicate allocation
    except OSError as error:
        store.mark(run_id, str(source), "FAILED", type(error).__name__)
        return "FAILED"
    if destination.is_file() and sha256_file(destination) == expected:
        store.mark(run_id, str(source), "MOVED")
        return "MOVED"
    store.mark(run_id, str(source), "FAILED", "post_rename_hash_mismatch")
    return "FAILED"
