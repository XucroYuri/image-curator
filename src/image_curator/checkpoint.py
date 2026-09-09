"""SQLite-backed, resumable feature checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .readonly import SourceSnapshot


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class WorkItem:
    """A stable, read-only source reference stored before extraction."""

    asset_id: str
    source: SourceSnapshot


class CheckpointStore:
    """Owns a local SQLite checkpoint; it never opens a source for writing."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS work_items(
              asset_id TEXT PRIMARY KEY,
              source_path TEXT NOT NULL,
              source_size INTEGER NOT NULL,
              source_mtime_ns INTEGER NOT NULL,
              source_sha256 TEXT NOT NULL,
              state TEXT NOT NULL DEFAULT 'PENDING',
              attempts INTEGER NOT NULL DEFAULT 0,
              metadata_evidence_json TEXT NOT NULL DEFAULT '{}',
              features_json TEXT NOT NULL DEFAULT '{}',
              embedding_f16 BLOB,
              embedding_dim INTEGER,
              last_error TEXT,
              updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS work_items_state ON work_items(state);
            CREATE TABLE IF NOT EXISTS occurrences(
              source_path TEXT PRIMARY KEY,
              asset_id TEXT NOT NULL REFERENCES work_items(asset_id),
              source_size INTEGER NOT NULL,
              source_mtime_ns INTEGER NOT NULL,
              source_sha256 TEXT NOT NULL,
              discovered_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS occurrences_asset_id ON occurrences(asset_id);
        """)
        # Migrate checkpoints created before occurrence tracking without
        # altering any source asset: their canonical paths become occurrences.
        self.connection.execute(
            "INSERT OR IGNORE INTO occurrences(source_path,asset_id,source_size,source_mtime_ns,source_sha256,discovered_at) "
            "SELECT source_path,asset_id,source_size,source_mtime_ns,source_sha256,updated_at FROM work_items"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def enqueue(self, item: WorkItem) -> bool:
        """Record one content asset and every observed source path atomically."""
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO work_items(asset_id,source_path,source_size,source_mtime_ns,source_sha256,updated_at) "
            "VALUES(?,?,?,?,?,?)", (item.asset_id, str(item.source.path), item.source.size,
                                     item.source.mtime_ns, item.source.sha256, utc_now()))
        self.connection.execute(
            "INSERT INTO occurrences(source_path,asset_id,source_size,source_mtime_ns,source_sha256,discovered_at) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(source_path) DO UPDATE SET asset_id=excluded.asset_id,"
            "source_size=excluded.source_size,source_mtime_ns=excluded.source_mtime_ns,"
            "source_sha256=excluded.source_sha256,discovered_at=excluded.discovered_at",
            (str(item.source.path), item.asset_id, item.source.size, item.source.mtime_ns,
             item.source.sha256, utc_now()),
        )
        self.connection.commit()
        return cursor.rowcount == 1

    def pending(self, limit: int | None = None) -> list[WorkItem]:
        """Return unfinished work in deterministic order for safe resume."""
        query = "SELECT * FROM work_items WHERE state IN ('PENDING','RETRYABLE_FAILED') ORDER BY asset_id"
        parameters: tuple[Any, ...] = () if limit is None else (limit,)
        if limit is not None:
            query += " LIMIT ?"
        rows = self.connection.execute(query, parameters).fetchall()
        return [WorkItem(row["asset_id"], SourceSnapshot(Path(row["source_path"]), row["source_size"],
                                                           row["source_mtime_ns"], row["source_sha256"]))
                for row in rows]

    def complete(self, asset_id: str, *, metadata_evidence: dict[str, Any], features: dict[str, Any],
                 embedding_f16: bytes | None = None, embedding_dim: int | None = None) -> None:
        """Persist compact results after successful read-only extraction."""
        if (embedding_f16 is None) != (embedding_dim is None):
            raise ValueError("embedding bytes and dimensions must be provided together")
        cursor = self.connection.execute(
            "UPDATE work_items SET state='COMPLETE',attempts=attempts+1,metadata_evidence_json=?,features_json=?,"
            "embedding_f16=?,embedding_dim=?,last_error=NULL,updated_at=? WHERE asset_id=?",
            (json.dumps(metadata_evidence, sort_keys=True, separators=(",", ":")),
             json.dumps(features, sort_keys=True, separators=(",", ":")), embedding_f16, embedding_dim,
             utc_now(), asset_id))
        if cursor.rowcount != 1:
            raise KeyError(f"unknown asset_id: {asset_id}")
        self.connection.commit()

    def fail_retryable(self, asset_id: str, error: Exception) -> None:
        """Preserve a retryable failure without storing adapter or source secrets."""
        message_digest = hashlib.sha256(str(error).encode("utf-8", errors="replace")).hexdigest()[:16]
        safe_error = f"{type(error).__name__}:message_sha256={message_digest}"
        cursor = self.connection.execute(
            "UPDATE work_items SET state='RETRYABLE_FAILED',attempts=attempts+1,last_error=?,updated_at=? WHERE asset_id=?",
            (safe_error, utc_now(), asset_id))
        if cursor.rowcount != 1:
            raise KeyError(f"unknown asset_id: {asset_id}")
        self.connection.commit()

    def status(self) -> dict[str, int]:
        return {str(state): int(count) for state, count in self.connection.execute(
            "SELECT state, count(*) FROM work_items GROUP BY state ORDER BY state")}

    def inventory_counts(self) -> dict[str, int]:
        """Return unique-content and source-location counts for scan/status reporting."""
        unique_assets = self.connection.execute("SELECT count(*) FROM work_items").fetchone()[0]
        occurrences = self.connection.execute("SELECT count(*) FROM occurrences").fetchone()[0]
        return {"unique_assets": int(unique_assets), "occurrences": int(occurrences)}

    def occurrences_for(self, asset_id: str) -> list[SourceSnapshot]:
        """Return all known locations for content, in stable path order."""
        rows = self.connection.execute(
            "SELECT source_path,source_size,source_mtime_ns,source_sha256 FROM occurrences "
            "WHERE asset_id=? ORDER BY source_path", (asset_id,)
        ).fetchall()
        return [SourceSnapshot(Path(row["source_path"]), row["source_size"], row["source_mtime_ns"],
                               row["source_sha256"]) for row in rows]
