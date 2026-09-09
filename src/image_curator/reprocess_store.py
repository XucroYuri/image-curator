"""Versioned, read-only reprocessing runs backed by a local SQLite database.

The store deliberately separates expensive feature extraction from later decision
versions.  It stores source snapshots and compact evidence only; callers remain
responsible for reading a source and must never use this module to mutate one.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from .readonly import SourceSnapshot

SCHEMA_VERSION = 4


def utc_now() -> str:
    """Return a lexicographically comparable UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _canonical_json(value: dict[str, Any] | list[Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _safe_error(error: Exception) -> str:
    digest = hashlib.sha256(str(error).encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{type(error).__name__}:message_sha256={digest}"


@dataclass(frozen=True)
class ReprocessItem:
    """A claimed unique-content item scoped to exactly one analysis run."""

    run_id: str
    asset_id: str
    source: SourceSnapshot
    attempts: int
    lease_owner: str
    lease_token: str
    lease_expires_at: str


@dataclass(frozen=True)
class ReprocessOccurrence:
    """An immutable, path-level baseline row for audit and diff reporting."""

    run_id: str
    source_path: Path
    source: SourceSnapshot | None
    asset_id: str | None
    old_bucket: str | None
    audit_locked: bool
    baseline_identity_unverifiable: bool
    occurrence_state: str


class ReprocessStore:
    """Own a local, restart-safe store for versioned image analysis runs."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ReprocessStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _migrate(self) -> None:
        """Create missing schema in place, without rewriting existing run data."""
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema version {version} is newer than supported {SCHEMA_VERSION}"
            )
        if version < 1:
            with self._transaction():
                self._execute_schema(
                    """
                CREATE TABLE IF NOT EXISTS analysis_runs(
                  run_id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL,
                  cutoff_at TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'CREATED',
                  config_fingerprint TEXT NOT NULL,
                  code_version TEXT NOT NULL,
                  model_manifest_json TEXT NOT NULL,
                  resource_snapshot_json TEXT NOT NULL DEFAULT '{}',
                  statistics_json TEXT NOT NULL DEFAULT '{}',
                  CHECK(status IN ('CREATED','RUNNING','PAUSED','COMPLETE','FAILED'))
                );

                CREATE TABLE IF NOT EXISTS run_items(
                  run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                  asset_id TEXT NOT NULL,
                  source_path TEXT NOT NULL,
                  source_size INTEGER NOT NULL,
                  source_mtime_ns INTEGER NOT NULL,
                  source_sha256 TEXT NOT NULL,
                  state TEXT NOT NULL DEFAULT 'PENDING',
                  attempts INTEGER NOT NULL DEFAULT 0,
                  lease_owner TEXT,
                  lease_token TEXT,
                  lease_expires_at TEXT,
                  metadata_evidence_json TEXT NOT NULL DEFAULT '{}',
                  features_json TEXT NOT NULL DEFAULT '{}',
                  embedding_f16 BLOB,
                  embedding_dim INTEGER,
                  last_error TEXT,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(run_id, asset_id),
                  CHECK(state IN ('PENDING','LEASED','RETRYABLE_FAILED','COMPLETE','SOURCE_CHANGED','FAILED')),
                  CHECK((embedding_f16 IS NULL) = (embedding_dim IS NULL))
                );
                CREATE INDEX IF NOT EXISTS run_items_claim ON run_items(run_id, state, asset_id);
                CREATE INDEX IF NOT EXISTS run_items_lease ON run_items(run_id, lease_expires_at);

                CREATE TABLE IF NOT EXISTS run_occurrences(
                  run_id TEXT NOT NULL,
                  source_path TEXT NOT NULL,
                  asset_id TEXT,
                  source_size INTEGER,
                  source_mtime_ns INTEGER,
                  source_sha256 TEXT,
                  old_bucket TEXT,
                  audit_locked INTEGER NOT NULL DEFAULT 0 CHECK(audit_locked IN (0,1)),
                  baseline_identity_unverifiable INTEGER NOT NULL DEFAULT 0
                    CHECK(baseline_identity_unverifiable IN (0,1)),
                  occurrence_state TEXT NOT NULL DEFAULT 'FROZEN',
                  discovered_at TEXT NOT NULL,
                  PRIMARY KEY(run_id, source_path),
                  FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                  FOREIGN KEY(run_id, asset_id) REFERENCES run_items(run_id, asset_id) ON DELETE CASCADE,
                  CHECK(occurrence_state IN ('FROZEN','PROCESSED','MISSING','SOURCE_CHANGED','ADDED_DEFERRED'))
                );
                CREATE INDEX IF NOT EXISTS run_occurrences_asset ON run_occurrences(run_id, asset_id);
                CREATE INDEX IF NOT EXISTS run_occurrences_state ON run_occurrences(run_id, occurrence_state);

                CREATE TABLE IF NOT EXISTS decision_runs(
                  decision_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL REFERENCES analysis_runs(run_id) ON DELETE RESTRICT,
                  created_at TEXT NOT NULL,
                  config_fingerprint TEXT NOT NULL,
                  reference_fingerprint TEXT NOT NULL,
                  human_labels_fingerprint TEXT NOT NULL,
                  thresholds_json TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'CREATED',
                  CHECK(status IN ('CREATED','COMPLETE','FAILED'))
                );
                CREATE INDEX IF NOT EXISTS decision_runs_analysis ON decision_runs(run_id, decision_id);

                CREATE TABLE IF NOT EXISTS decision_results(
                  decision_id TEXT NOT NULL REFERENCES decision_runs(decision_id) ON DELETE CASCADE,
                  asset_id TEXT NOT NULL,
                  proposed_bucket TEXT,
                  candidate_identity TEXT,
                  confidence REAL,
                  decision_state TEXT NOT NULL DEFAULT 'REVIEW',
                  reasons_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(decision_id, asset_id),
                  CHECK(decision_state IN ('ACCEPTED','REVIEW','UNKNOWN','REJECTED'))
                );
                CREATE INDEX IF NOT EXISTS decision_results_state ON decision_results(decision_id, decision_state);
                """
                )
                self.connection.execute("PRAGMA user_version=1")
            version = 1
        # Version 2 added an opaque claim token.  Keep a genuine v1 database
        # in place and only append this nullable column.
        if version < 2:
            with self._transaction():
                columns = {
                    str(row["name"])
                    for row in self.connection.execute("PRAGMA table_info(run_items)")
                }
                if "lease_token" not in columns:
                    self.connection.execute("ALTER TABLE run_items ADD COLUMN lease_token TEXT")
                self.connection.execute("PRAGMA user_version=2")
            version = 2
        if version < 3:
            self._migrate_occurrences_v3()
            version = 3
        if version < 4:
            self._migrate_occurrences_v4()

    def _migrate_occurrences_v3(self) -> None:
        """Allow baseline paths without a stable asset, preserving every v2 row."""
        self.connection.execute("PRAGMA foreign_keys=OFF")
        try:
            with self._transaction():
                self._execute_schema(
                    """
                    CREATE TABLE run_occurrences_v3(
                      run_id TEXT NOT NULL,
                      source_path TEXT NOT NULL,
                      asset_id TEXT,
                      source_size INTEGER,
                      source_mtime_ns INTEGER,
                      source_sha256 TEXT,
                      old_bucket TEXT,
                      audit_locked INTEGER NOT NULL DEFAULT 0 CHECK(audit_locked IN (0,1)),
                      baseline_identity_unverifiable INTEGER NOT NULL DEFAULT 0
                        CHECK(baseline_identity_unverifiable IN (0,1)),
                      occurrence_state TEXT NOT NULL DEFAULT 'FROZEN',
                      discovered_at TEXT NOT NULL,
                      PRIMARY KEY(run_id, source_path),
                      FOREIGN KEY(run_id, asset_id) REFERENCES run_items(run_id, asset_id) ON DELETE CASCADE,
                      CHECK(occurrence_state IN ('FROZEN','PROCESSED','MISSING','SOURCE_CHANGED','ADDED_DEFERRED'))
                    );
                    INSERT INTO run_occurrences_v3(
                      run_id,source_path,asset_id,source_size,source_mtime_ns,source_sha256,old_bucket,
                      audit_locked,baseline_identity_unverifiable,occurrence_state,discovered_at
                    ) SELECT
                      run_id,source_path,asset_id,source_size,source_mtime_ns,source_sha256,old_bucket,
                      audit_locked,baseline_identity_unverifiable,occurrence_state,discovered_at
                    FROM run_occurrences;
                    DROP TABLE run_occurrences;
                    ALTER TABLE run_occurrences_v3 RENAME TO run_occurrences;
                    CREATE INDEX run_occurrences_asset ON run_occurrences(run_id, asset_id);
                    CREATE INDEX run_occurrences_state ON run_occurrences(run_id, occurrence_state);
                    """
                )
                self.connection.execute("PRAGMA user_version=3")
        finally:
            self.connection.execute("PRAGMA foreign_keys=ON")

    def _migrate_occurrences_v4(self) -> None:
        """Bind hashless path outcomes to an analysis run with an independent FK."""
        self.connection.execute("PRAGMA foreign_keys=OFF")
        try:
            with self._transaction():
                self._execute_schema(
                    """
                    CREATE TABLE run_occurrences_v4(
                      run_id TEXT NOT NULL,
                      source_path TEXT NOT NULL,
                      asset_id TEXT,
                      source_size INTEGER,
                      source_mtime_ns INTEGER,
                      source_sha256 TEXT,
                      old_bucket TEXT,
                      audit_locked INTEGER NOT NULL DEFAULT 0 CHECK(audit_locked IN (0,1)),
                      baseline_identity_unverifiable INTEGER NOT NULL DEFAULT 0
                        CHECK(baseline_identity_unverifiable IN (0,1)),
                      occurrence_state TEXT NOT NULL DEFAULT 'FROZEN',
                      discovered_at TEXT NOT NULL,
                      PRIMARY KEY(run_id, source_path),
                      FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE,
                      FOREIGN KEY(run_id, asset_id) REFERENCES run_items(run_id, asset_id) ON DELETE CASCADE,
                      CHECK(occurrence_state IN ('FROZEN','PROCESSED','MISSING','SOURCE_CHANGED','ADDED_DEFERRED'))
                    );
                    INSERT INTO run_occurrences_v4(
                      run_id,source_path,asset_id,source_size,source_mtime_ns,source_sha256,old_bucket,
                      audit_locked,baseline_identity_unverifiable,occurrence_state,discovered_at
                    ) SELECT
                      run_id,source_path,asset_id,source_size,source_mtime_ns,source_sha256,old_bucket,
                      audit_locked,baseline_identity_unverifiable,occurrence_state,discovered_at
                    FROM run_occurrences;
                    DROP TABLE run_occurrences;
                    ALTER TABLE run_occurrences_v4 RENAME TO run_occurrences;
                    CREATE INDEX run_occurrences_asset ON run_occurrences(run_id, asset_id);
                    CREATE INDEX run_occurrences_state ON run_occurrences(run_id, occurrence_state);
                    """
                )
                self.connection.execute("PRAGMA user_version=4")
        finally:
            self.connection.execute("PRAGMA foreign_keys=ON")

    def _require_run(self, run_id: str) -> None:
        if self.connection.execute(
            "SELECT 1 FROM analysis_runs WHERE run_id=?", (run_id,)
        ).fetchone() is None:
            raise KeyError(f"unknown run_id: {run_id}")

    def _execute_schema(self, script: str) -> None:
        """Execute DDL within the caller's transaction without executescript's implicit commit."""
        for statement in script.split(";"):
            if statement.strip():
                self.connection.execute(statement)

    def create_run(
        self,
        run_id: str,
        *,
        cutoff_at: str,
        config_fingerprint: str,
        code_version: str,
        model_manifest: dict[str, Any],
        resource_snapshot: dict[str, Any] | None = None,
        status: str = "CREATED",
    ) -> bool:
        """Create an immutable run identity; duplicate IDs are rejected without mutation."""
        with self._transaction():
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO analysis_runs("
                "run_id,created_at,cutoff_at,status,config_fingerprint,code_version,model_manifest_json,"
                "resource_snapshot_json) VALUES(?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    utc_now(),
                    cutoff_at,
                    status,
                    config_fingerprint,
                    code_version,
                    _canonical_json(model_manifest),
                    _canonical_json(resource_snapshot),
                ),
            )
        return cursor.rowcount == 1

    def upgrade_unstarted_run_identity(
        self, run_id: str, *, config_fingerprint: str, model_manifest: dict[str, Any]
    ) -> None:
        """Seal richer fingerprints before any item has ever been claimed."""
        with self._transaction():
            cursor = self.connection.execute(
                "UPDATE analysis_runs SET config_fingerprint=?,model_manifest_json=? "
                "WHERE run_id=? AND status='CREATED' AND NOT EXISTS("
                "SELECT 1 FROM run_items WHERE run_id=? AND attempts>0)",
                (config_fingerprint, _canonical_json(model_manifest), run_id, run_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("run identity can only be upgraded before processing starts")

    def set_run_status(
        self, run_id: str, status: str, *, statistics: dict[str, Any] | None = None
    ) -> None:
        with self._transaction():
            cursor = self.connection.execute(
                "UPDATE analysis_runs SET status=?, statistics_json=? WHERE run_id=?",
                (status, _canonical_json(statistics), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown run_id: {run_id}")

    def add_item(self, run_id: str, asset_id: str, source: SourceSnapshot) -> bool:
        """Add one unique content item without affecting another run's result."""
        with self._transaction():
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO run_items("
                "run_id,asset_id,source_path,source_size,source_mtime_ns,source_sha256,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    run_id,
                    asset_id,
                    str(source.path),
                    source.size,
                    source.mtime_ns,
                    source.sha256,
                    utc_now(),
                ),
            )
        return cursor.rowcount == 1

    def add_occurrence(
        self,
        run_id: str,
        asset_id: str,
        source: SourceSnapshot,
        *,
        old_bucket: str | None,
        audit_locked: bool,
        baseline_identity_unverifiable: bool = False,
    ) -> bool:
        """Freeze a path-level baseline; a duplicate path cannot silently be rewritten."""
        with self._transaction():
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO run_occurrences("
                "run_id,source_path,asset_id,source_size,source_mtime_ns,source_sha256,old_bucket,audit_locked,"
                "baseline_identity_unverifiable,discovered_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    str(source.path),
                    asset_id,
                    source.size,
                    source.mtime_ns,
                    source.sha256,
                    old_bucket,
                    int(audit_locked),
                    int(baseline_identity_unverifiable),
                    utc_now(),
                ),
            )
        return cursor.rowcount == 1

    def add_baseline_occurrence(
        self,
        run_id: str,
        source_path: Path,
        *,
        old_bucket: str | None,
        audit_locked: bool,
        occurrence_state: str,
        source: SourceSnapshot | None = None,
        asset_id: str | None = None,
        baseline_identity_unverifiable: bool = False,
    ) -> bool:
        """Persist any path-level baseline outcome, even when content is unavailable.

        A stable occurrence supplies both ``source`` and ``asset_id`` and is
        protected by the composite foreign key.  Missing, deferred, and
        pre-hash changed paths intentionally have no asset association.
        """
        states = {"FROZEN", "MISSING", "SOURCE_CHANGED", "ADDED_DEFERRED"}
        if occurrence_state not in states:
            raise ValueError("unsupported baseline occurrence state")
        if (source is None) != (asset_id is None):
            raise ValueError("source and asset_id must be supplied together")
        if source is not None and str(source.path) != str(source_path):
            raise ValueError("source snapshot path must match source_path")
        self._require_run(run_id)
        with self._transaction():
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO run_occurrences("
                "run_id,source_path,asset_id,source_size,source_mtime_ns,source_sha256,old_bucket,audit_locked,"
                "baseline_identity_unverifiable,occurrence_state,discovered_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    str(source_path),
                    asset_id,
                    source.size if source else None,
                    source.mtime_ns if source else None,
                    source.sha256 if source else None,
                    old_bucket,
                    int(audit_locked),
                    int(baseline_identity_unverifiable),
                    occurrence_state,
                    utc_now(),
                ),
            )
        return cursor.rowcount == 1

    def add_asset_occurrence(
        self,
        run_id: str,
        asset_id: str,
        source: SourceSnapshot,
        *,
        old_bucket: str | None,
        audit_locked: bool,
        baseline_identity_unverifiable: bool = False,
    ) -> tuple[bool, bool]:
        """Atomically add a unique item (if needed) and one frozen source occurrence."""
        with self._transaction():
            item = self.connection.execute(
                "INSERT OR IGNORE INTO run_items("
                "run_id,asset_id,source_path,source_size,source_mtime_ns,source_sha256,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (run_id, asset_id, str(source.path), source.size, source.mtime_ns, source.sha256, utc_now()),
            )
            occurrence = self.connection.execute(
                "INSERT OR IGNORE INTO run_occurrences("
                "run_id,source_path,asset_id,source_size,source_mtime_ns,source_sha256,old_bucket,audit_locked,"
                "baseline_identity_unverifiable,discovered_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    str(source.path),
                    asset_id,
                    source.size,
                    source.mtime_ns,
                    source.sha256,
                    old_bucket,
                    int(audit_locked),
                    int(baseline_identity_unverifiable),
                    utc_now(),
                ),
            )
        return item.rowcount == 1, occurrence.rowcount == 1

    def claim(
        self,
        run_id: str,
        worker_id: str,
        *,
        limit: int = 1,
        lease_seconds: int = 300,
        now: str | None = None,
    ) -> list[ReprocessItem]:
        """Lease pending/retryable items, safely recovering expired leases first."""
        if limit < 1 or lease_seconds < 1:
            raise ValueError("limit and lease_seconds must be positive")
        claimed_at = now or utc_now()
        lease_expires_at = (
            datetime.fromisoformat(claimed_at).astimezone(UTC) + timedelta(seconds=lease_seconds)
        ).isoformat(timespec="microseconds")
        lease_token = uuid4().hex
        with self._transaction():
            self._require_run(run_id)
            self.connection.execute(
                "UPDATE run_items SET state='PENDING',lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,updated_at=? "
                "WHERE run_id=? AND state='LEASED' AND lease_expires_at<=?",
                (claimed_at, run_id, claimed_at),
            )
            rows = self.connection.execute(
                "SELECT run_id,asset_id,source_path,source_size,source_mtime_ns,source_sha256,attempts "
                "FROM run_items WHERE run_id=? AND state IN ('PENDING','RETRYABLE_FAILED') "
                "ORDER BY asset_id LIMIT ?",
                (run_id, limit),
            ).fetchall()
            for row in rows:
                self.connection.execute(
                    "UPDATE run_items SET state='LEASED',attempts=attempts+1,lease_owner=?,lease_token=?,lease_expires_at=?,"
                    "updated_at=? WHERE run_id=? AND asset_id=?",
                    (worker_id, lease_token, lease_expires_at, claimed_at, run_id, row["asset_id"]),
                )
        return [
            ReprocessItem(
                run_id=row["run_id"],
                asset_id=row["asset_id"],
                source=SourceSnapshot(
                    Path(row["source_path"]),
                    int(row["source_size"]),
                    int(row["source_mtime_ns"]),
                    row["source_sha256"],
                ),
                attempts=int(row["attempts"]) + 1,
                lease_owner=worker_id,
                lease_token=lease_token,
                lease_expires_at=lease_expires_at,
            )
            for row in rows
        ]

    def complete(
        self,
        run_id: str,
        asset_id: str,
        *,
        metadata_evidence: dict[str, Any],
        features: dict[str, Any],
        lease_token: str,
        embedding_f16: bytes | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Atomically complete a leased item and mark its frozen paths processed."""
        if (embedding_f16 is None) != (embedding_dim is None):
            raise ValueError("embedding bytes and dimensions must be provided together")
        with self._transaction():
            self._transition_item(
                run_id,
                asset_id,
                "COMPLETE",
                metadata_evidence=metadata_evidence,
                features=features,
                embedding_f16=embedding_f16,
                embedding_dim=embedding_dim,
                lease_token=lease_token,
            )
            self.connection.execute(
                "UPDATE run_occurrences SET occurrence_state='PROCESSED' "
                "WHERE run_id=? AND asset_id=? AND occurrence_state='FROZEN'",
                (run_id, asset_id),
            )

    def fail_retryable(self, run_id: str, asset_id: str, error: Exception, *, lease_token: str) -> None:
        """Release a leased item for a retry without persisting sensitive error text."""
        with self._transaction():
            self._transition_item(
                run_id, asset_id, "RETRYABLE_FAILED", last_error=_safe_error(error), lease_token=lease_token
            )

    def fail_terminal(self, run_id: str, asset_id: str, error: Exception, *, lease_token: str) -> None:
        """Record an explicit non-retryable item failure."""
        with self._transaction():
            self._transition_item(
                run_id, asset_id, "FAILED", last_error=_safe_error(error), lease_token=lease_token
            )

    def source_changed(self, run_id: str, asset_id: str, *, lease_token: str) -> None:
        """Atomically stop a changed asset and preserve that outcome for all frozen paths."""
        with self._transaction():
            self._transition_item(run_id, asset_id, "SOURCE_CHANGED", lease_token=lease_token)
            self.connection.execute(
                "UPDATE run_occurrences SET occurrence_state='SOURCE_CHANGED' "
                "WHERE run_id=? AND asset_id=? AND occurrence_state='FROZEN'",
                (run_id, asset_id),
            )

    def source_changed_or_switch(self, run_id: str, asset_id: str, *, lease_token: str) -> bool:
        """Mark one changed path and retry another frozen duplicate when available."""
        with self._transaction():
            item = self.connection.execute(
                "SELECT source_path FROM run_items WHERE run_id=? AND asset_id=? "
                "AND state='LEASED' AND lease_token=?",
                (run_id, asset_id, lease_token),
            ).fetchone()
            if item is None:
                raise RuntimeError(f"item {asset_id} is not currently leased by a worker")
            self.connection.execute(
                "UPDATE run_occurrences SET occurrence_state='SOURCE_CHANGED' "
                "WHERE run_id=? AND asset_id=? AND source_path=? AND occurrence_state='FROZEN'",
                (run_id, asset_id, item["source_path"]),
            )
            replacement = self.connection.execute(
                "SELECT source_path,source_size,source_mtime_ns,source_sha256 FROM run_occurrences "
                "WHERE run_id=? AND asset_id=? AND occurrence_state='FROZEN' ORDER BY source_path LIMIT 1",
                (run_id, asset_id),
            ).fetchone()
            if replacement is None:
                self.connection.execute(
                    "UPDATE run_items SET state='SOURCE_CHANGED',lease_owner=NULL,lease_token=NULL,"
                    "lease_expires_at=NULL,updated_at=? WHERE run_id=? AND asset_id=?",
                    (utc_now(), run_id, asset_id),
                )
                return False
            self.connection.execute(
                "UPDATE run_items SET state='RETRYABLE_FAILED',source_path=?,source_size=?,source_mtime_ns=?,"
                "source_sha256=?,lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,last_error=NULL,updated_at=? "
                "WHERE run_id=? AND asset_id=?",
                (replacement["source_path"], replacement["source_size"], replacement["source_mtime_ns"],
                 replacement["source_sha256"], utc_now(), run_id, asset_id),
            )
            return True

    def mark_occurrence(self, run_id: str, source_path: Path, occurrence_state: str) -> None:
        """Record a path-level baseline outcome such as MISSING or SOURCE_CHANGED."""
        if occurrence_state not in {"MISSING", "SOURCE_CHANGED", "ADDED_DEFERRED"}:
            raise ValueError("unsupported path-level occurrence state")
        with self._transaction():
            cursor = self.connection.execute(
                "UPDATE run_occurrences SET occurrence_state=? WHERE run_id=? AND source_path=?",
                (occurrence_state, run_id, str(source_path)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown occurrence: {source_path}")

    def create_decision_run(
        self,
        decision_id: str,
        run_id: str,
        *,
        config_fingerprint: str,
        reference_fingerprint: str,
        human_labels_fingerprint: str,
        thresholds: dict[str, Any],
    ) -> bool:
        with self._transaction():
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO decision_runs("
                "decision_id,run_id,created_at,config_fingerprint,reference_fingerprint,human_labels_fingerprint,"
                "thresholds_json) VALUES(?,?,?,?,?,?,?)",
                (
                    decision_id,
                    run_id,
                    utc_now(),
                    config_fingerprint,
                    reference_fingerprint,
                    human_labels_fingerprint,
                    _canonical_json(thresholds),
                ),
            )
        return cursor.rowcount == 1

    def record_decision(
        self,
        decision_id: str,
        asset_id: str,
        *,
        proposed_bucket: str | None,
        candidate_identity: str | None,
        confidence: float | None,
        decision_state: str,
        reasons: dict[str, Any] | None = None,
    ) -> None:
        """Upsert one result inside a decision version, never across feature runs."""
        with self._transaction():
            asset = self.connection.execute(
                "SELECT 1 FROM decision_runs AS d JOIN run_items AS i ON i.run_id=d.run_id "
                "WHERE d.decision_id=? AND i.asset_id=?",
                (decision_id, asset_id),
            ).fetchone()
            if asset is None:
                raise KeyError(f"asset {asset_id} is not part of decision run {decision_id}")
            cursor = self.connection.execute(
                "INSERT INTO decision_results("
                "decision_id,asset_id,proposed_bucket,candidate_identity,confidence,decision_state,reasons_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(decision_id,asset_id) DO UPDATE SET "
                "proposed_bucket=excluded.proposed_bucket,candidate_identity=excluded.candidate_identity,"
                "confidence=excluded.confidence,decision_state=excluded.decision_state,"
                "reasons_json=excluded.reasons_json,created_at=excluded.created_at",
                (
                    decision_id,
                    asset_id,
                    proposed_bucket,
                    candidate_identity,
                    confidence,
                    decision_state,
                    _canonical_json(reasons),
                    utc_now(),
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown decision_id: {decision_id}")

    def decision_diffs(self, decision_id: str) -> list[dict[str, Any]]:
        """Return one deterministic path-level diff; audit locks preserve old state."""
        rows = self.connection.execute(
            "SELECT o.source_path,o.asset_id,o.old_bucket,o.audit_locked,o.occurrence_state,"
            "d.proposed_bucket,d.candidate_identity,d.confidence,d.decision_state,d.reasons_json,"
            "CASE WHEN o.audit_locked=1 THEN o.old_bucket ELSE d.proposed_bucket END AS effective_bucket "
            "FROM run_occurrences AS o JOIN decision_runs AS r ON r.run_id=o.run_id "
            "LEFT JOIN decision_results AS d ON d.decision_id=r.decision_id AND d.asset_id=o.asset_id "
            "WHERE r.decision_id=? ORDER BY o.source_path",
            (decision_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def status(self, run_id: str) -> dict[str, Any]:
        """Return compact run, item and path status suitable for stable JSON CLI output."""
        run = self.connection.execute(
            "SELECT run_id,status,cutoff_at,config_fingerprint,code_version,statistics_json "
            "FROM analysis_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(f"unknown run_id: {run_id}")
        item_counts = {
            str(row["state"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT state,count(*) AS count FROM run_items WHERE run_id=? GROUP BY state ORDER BY state",
                (run_id,),
            )
        }
        occurrence_counts = {
            str(row["occurrence_state"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT occurrence_state,count(*) AS count FROM run_occurrences WHERE run_id=? "
                "GROUP BY occurrence_state ORDER BY occurrence_state",
                (run_id,),
            )
        }
        return {
            "run_id": run["run_id"],
            "status": run["status"],
            "cutoff_at": run["cutoff_at"],
            "config_fingerprint": run["config_fingerprint"],
            "code_version": run["code_version"],
            "statistics": json.loads(run["statistics_json"]),
            "items": item_counts,
            "occurrences": occurrence_counts,
        }

    def integrity(self) -> dict[str, str]:
        """Expose SQLite integrity and FK checks without altering the database."""
        integrity = str(self.connection.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_key_rows = self.connection.execute("PRAGMA foreign_key_check").fetchall()
        return {"integrity_check": integrity, "foreign_key_check": "ok" if not foreign_key_rows else "failed"}

    def occurrences_for(self, run_id: str, asset_id: str) -> list[ReprocessOccurrence]:
        rows = self.connection.execute(
            "SELECT * FROM run_occurrences WHERE run_id=? AND asset_id=? ORDER BY source_path",
            (run_id, asset_id),
        ).fetchall()
        return [
            ReprocessOccurrence(
                row["run_id"],
                Path(row["source_path"]),
                SourceSnapshot(
                    Path(row["source_path"]), int(row["source_size"]), int(row["source_mtime_ns"]), row["source_sha256"]
                ) if row["asset_id"] is not None else None,
                row["asset_id"],
                row["old_bucket"],
                bool(row["audit_locked"]),
                bool(row["baseline_identity_unverifiable"]),
                row["occurrence_state"],
            )
            for row in rows
        ]

    def _require_run(self, run_id: str) -> None:
        if self.connection.execute("SELECT 1 FROM analysis_runs WHERE run_id=?", (run_id,)).fetchone() is None:
            raise KeyError(f"unknown run_id: {run_id}")

    def _transition_item(
        self,
        run_id: str,
        asset_id: str,
        state: str,
        *,
        metadata_evidence: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
        embedding_f16: bytes | None = None,
        embedding_dim: int | None = None,
        last_error: str | None = None,
        lease_token: str,
    ) -> None:
        cursor = self.connection.execute(
            "UPDATE run_items SET state=?,lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,metadata_evidence_json=?,"
            "features_json=?,embedding_f16=?,embedding_dim=?,last_error=?,updated_at=? "
            "WHERE run_id=? AND asset_id=? AND state='LEASED' AND lease_token=?",
            (
                state,
                _canonical_json(metadata_evidence),
                _canonical_json(features),
                embedding_f16,
                embedding_dim,
                last_error,
                utc_now(),
                run_id,
                asset_id,
                lease_token,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"item {asset_id} is not currently leased by a worker")
