import sqlite3
from pathlib import Path

import pytest

from image_curator.readonly import SourceSnapshot
from image_curator.reprocess_store import ReprocessStore


def _snapshot(path: Path, payload: bytes = b"pixels") -> SourceSnapshot:
    path.write_bytes(payload)
    return SourceSnapshot(path, len(payload), 42, "a" * 64)


def _run(store: ReprocessStore, run_id: str = "run-1") -> str:
    assert store.create_run(
        run_id,
        cutoff_at="2026-09-09T00:00:00+00:00",
        config_fingerprint="config-sha",
        code_version="0.3.0",
        model_manifest={"moat": "sha256:one", "nudenet": "sha256:two"},
        resource_snapshot={"gpu_workers": 2},
    )
    return run_id


def test_migration_is_non_destructive_and_enables_full_durability(tmp_path):
    database_path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE legacy_data(value TEXT NOT NULL)")
    connection.execute("INSERT INTO legacy_data VALUES('keep-me')")
    connection.execute("PRAGMA user_version=0")
    connection.commit()
    connection.close()

    with ReprocessStore(database_path) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 4
        assert store.connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert store.connection.execute("SELECT value FROM legacy_data").fetchone()[0] == "keep-me"
        assert store.integrity() == {"integrity_check": "ok", "foreign_key_check": "ok"}

    connection = sqlite3.connect(database_path)
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_v1_upgrade_appends_lease_token_without_rewriting_run_data(tmp_path):
    database_path = tmp_path / "v1.sqlite"
    source = _snapshot(tmp_path / "image.png")
    with ReprocessStore(database_path) as store:
        _run(store)
        store.add_asset_occurrence("run-1", "asset-1", source, old_bucket="old", audit_locked=False)

    connection = sqlite3.connect(database_path)
    connection.execute("ALTER TABLE run_items DROP COLUMN lease_token")
    connection.execute("PRAGMA user_version=1")
    connection.commit()
    connection.close()

    with ReprocessStore(database_path) as store:
        columns = {row["name"] for row in store.connection.execute("PRAGMA table_info(run_items)")}
        assert "lease_token" in columns
        assert store.connection.execute(
            "SELECT source_sha256 FROM run_items WHERE run_id='run-1' AND asset_id='asset-1'"
        ).fetchone()[0] == source.sha256


def test_v2_occurrence_upgrade_keeps_stable_rows_and_accepts_missing_baselines(tmp_path):
    database_path = tmp_path / "v2.sqlite"
    source = _snapshot(tmp_path / "image.png")
    with ReprocessStore(database_path) as store:
        _run(store)
        store.add_asset_occurrence("run-1", "asset-1", source, old_bucket="old", audit_locked=False)

    connection = sqlite3.connect(database_path)
    connection.execute("ALTER TABLE run_occurrences RENAME TO run_occurrences_v3")
    connection.executescript(
        """
        CREATE TABLE run_occurrences(
          run_id TEXT NOT NULL, source_path TEXT NOT NULL, asset_id TEXT NOT NULL,
          source_size INTEGER NOT NULL, source_mtime_ns INTEGER NOT NULL, source_sha256 TEXT NOT NULL,
          old_bucket TEXT, audit_locked INTEGER NOT NULL DEFAULT 0,
          baseline_identity_unverifiable INTEGER NOT NULL DEFAULT 0,
          occurrence_state TEXT NOT NULL DEFAULT 'FROZEN', discovered_at TEXT NOT NULL,
          PRIMARY KEY(run_id, source_path),
          FOREIGN KEY(run_id, asset_id) REFERENCES run_items(run_id, asset_id) ON DELETE CASCADE
        );
        INSERT INTO run_occurrences SELECT * FROM run_occurrences_v3;
        DROP TABLE run_occurrences_v3;
        PRAGMA user_version=2;
        """
    )
    connection.close()

    with ReprocessStore(database_path) as store:
        fields = {row["name"]: row["notnull"] for row in store.connection.execute("PRAGMA table_info(run_occurrences)")}
        assert fields["asset_id"] == 0
        assert store.add_baseline_occurrence(
            "run-1", tmp_path / "missing.png", old_bucket="old", audit_locked=False,
            occurrence_state="MISSING"
        )
        assert store.status("run-1")["occurrences"] == {"FROZEN": 1, "MISSING": 1}


def test_versioned_runs_keep_asset_results_independent(tmp_path):
    source = _snapshot(tmp_path / "image.png")
    with ReprocessStore(tmp_path / "runs.sqlite") as store:
        _run(store, "run-old")
        _run(store, "run-new")
        assert store.add_asset_occurrence("run-old", "asset-1", source, old_bucket="旧桶", audit_locked=False) == (True, True)
        assert store.add_asset_occurrence("run-new", "asset-1", source, old_bucket="旧桶", audit_locked=False) == (True, True)

        old_claim = store.claim("run-old", "worker-old")[0]
        assert old_claim.asset_id == "asset-1"
        store.complete(
            "run-old", "asset-1", metadata_evidence={"version": "old"}, features={"score": 1},
            lease_token=old_claim.lease_token,
        )
        new_claim = store.claim("run-new", "worker-new")[0]
        assert new_claim.asset_id == "asset-1"
        store.complete(
            "run-new", "asset-1", metadata_evidence={"version": "new"}, features={"score": 2},
            lease_token=new_claim.lease_token,
        )

        old = store.connection.execute(
            "SELECT features_json FROM run_items WHERE run_id='run-old' AND asset_id='asset-1'"
        ).fetchone()[0]
        new = store.connection.execute(
            "SELECT features_json FROM run_items WHERE run_id='run-new' AND asset_id='asset-1'"
        ).fetchone()[0]
        assert old == '{"score":1}'
        assert new == '{"score":2}'
        assert store.status("run-old")["items"] == {"COMPLETE": 1}
        assert store.status("run-new")["occurrences"] == {"PROCESSED": 1}


def test_duplicate_content_has_path_level_occurrences_and_audit_lock(tmp_path):
    one = _snapshot(tmp_path / "one.png")
    two = _snapshot(tmp_path / "two.png")
    with ReprocessStore(tmp_path / "runs.sqlite") as store:
        _run(store)
        assert store.add_asset_occurrence("run-1", "asset-1", one, old_bucket="草稿", audit_locked=False) == (True, True)
        assert store.add_asset_occurrence("run-1", "asset-1", two, old_bucket="已发帖", audit_locked=True) == (False, True)
        assert not store.add_occurrence("run-1", "asset-1", two, old_bucket="other", audit_locked=False)

        claim = store.claim("run-1", "worker")[0]
        store.complete("run-1", "asset-1", metadata_evidence={}, features={}, lease_token=claim.lease_token)
        assert store.create_decision_run(
            "decision-1",
            "run-1",
            config_fingerprint="decision-config",
            reference_fingerprint="references-sha",
            human_labels_fingerprint="labels-sha",
            thresholds={"accept": 0.95},
        )
        store.record_decision(
            "decision-1",
            "asset-1",
            proposed_bucket="可迁移",
            candidate_identity="character-a",
            confidence=0.99,
            decision_state="ACCEPTED",
            reasons={"model": "moat"},
        )
        with pytest.raises(KeyError, match="not part"):
            store.record_decision(
                "decision-1", "unrelated-asset", proposed_bucket=None, candidate_identity=None,
                confidence=None, decision_state="REVIEW"
            )
        diffs = store.decision_diffs("decision-1")

    assert [(row["source_path"], row["effective_bucket"]) for row in diffs] == [
        (str(one.path), "可迁移"),
        (str(two.path), "已发帖"),
    ]
    assert all(row["occurrence_state"] == "PROCESSED" for row in diffs)


def test_claim_recovers_expired_lease_and_retry_is_safe(tmp_path):
    source = _snapshot(tmp_path / "image.png")
    with ReprocessStore(tmp_path / "runs.sqlite") as store:
        _run(store)
        store.add_asset_occurrence("run-1", "asset-1", source, old_bucket=None, audit_locked=False)
        first = store.claim(
            "run-1", "worker-a", lease_seconds=10, now="2026-09-09T00:00:00+00:00"
        )
        assert first[0].attempts == 1
        assert store.claim("run-1", "worker-b", now="2026-09-09T00:00:05+00:00") == []
        reclaimed = store.claim("run-1", "worker-b", now="2026-09-09T00:00:11+00:00")
        assert reclaimed[0].attempts == 2
        with pytest.raises(RuntimeError, match="not currently leased"):
            store.complete(
                "run-1", "asset-1", metadata_evidence={}, features={}, lease_token=first[0].lease_token
            )
        store.fail_retryable(
            "run-1", "asset-1", RuntimeError("password=secret"), lease_token=reclaimed[0].lease_token
        )
        row = store.connection.execute(
            "SELECT state,last_error FROM run_items WHERE run_id='run-1' AND asset_id='asset-1'"
        ).fetchone()
        assert row["state"] == "RETRYABLE_FAILED"
        assert "secret" not in row["last_error"]
        assert store.claim("run-1", "worker-c")[0].attempts == 3


def test_source_change_and_path_level_missing_are_explicit_terminal_outcomes(tmp_path):
    source = _snapshot(tmp_path / "image.png")
    with ReprocessStore(tmp_path / "runs.sqlite") as store:
        _run(store)
        store.add_asset_occurrence("run-1", "asset-1", source, old_bucket="old", audit_locked=False)
        claim = store.claim("run-1", "worker")[0]
        store.source_changed("run-1", "asset-1", lease_token=claim.lease_token)
        assert store.status("run-1")["items"] == {"SOURCE_CHANGED": 1}
        assert store.status("run-1")["occurrences"] == {"SOURCE_CHANGED": 1}

        missing = _snapshot(tmp_path / "missing.png")
        store.add_asset_occurrence("run-1", "asset-2", missing, old_bucket=None, audit_locked=False)
        store.mark_occurrence("run-1", missing.path, "MISSING")
        assert store.status("run-1")["occurrences"] == {"MISSING": 1, "SOURCE_CHANGED": 1}


def test_constraints_reject_orphan_occurrence_and_complete_requires_lease(tmp_path):
    source = _snapshot(tmp_path / "image.png")
    with ReprocessStore(tmp_path / "runs.sqlite") as store:
        _run(store)
        with pytest.raises(sqlite3.IntegrityError):
            store.add_occurrence("run-1", "unknown", source, old_bucket=None, audit_locked=False)
        store.add_asset_occurrence("run-1", "asset-1", source, old_bucket=None, audit_locked=False)
        with pytest.raises(RuntimeError, match="not currently leased"):
            store.complete("run-1", "asset-1", metadata_evidence={}, features={}, lease_token="not-a-lease")


def test_hashless_occurrence_requires_real_run_and_cascades(tmp_path):
    with ReprocessStore(tmp_path / "paths.sqlite") as store:
        with pytest.raises(KeyError, match="unknown run_id"):
            store.add_baseline_occurrence(
                "missing-run", tmp_path / "missing.png", old_bucket=None,
                audit_locked=False, occurrence_state="MISSING",
            )
        _run(store)
        store.add_baseline_occurrence(
            "run-1", tmp_path / "missing.png", old_bucket=None,
            audit_locked=False, occurrence_state="MISSING",
        )
        store.connection.execute("DELETE FROM analysis_runs WHERE run_id='run-1'")
        store.connection.commit()
        assert store.connection.execute("SELECT count(*) FROM run_occurrences").fetchone()[0] == 0
