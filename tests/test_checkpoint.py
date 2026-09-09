import sqlite3

import pytest

from image_curator.checkpoint import CheckpointStore, WorkItem
from image_curator.evidence import metadata_evidence
from image_curator.readonly import SourceChangedError, read_verified, snapshot_source


def test_checkpoint_resumes_failed_item_and_stores_evidence(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"pixels")
    item = WorkItem("asset-001", snapshot_source(source))
    database_path = tmp_path / "checkpoint.sqlite"

    with CheckpointStore(database_path) as database:
        assert database.enqueue(item)
        assert not database.enqueue(item)
        database.fail_retryable(item.asset_id, ValueError("decode failed"))
        assert [pending.asset_id for pending in database.pending()] == [item.asset_id]
        evidence = metadata_evidence({"prompt": "private description", "workflow": {"version": 1}})
        database.complete(item.asset_id, metadata_evidence=evidence, features={"nudenet": {"explicit_score": 0.0}})
        assert database.status() == {"COMPLETE": 1}

    connection = sqlite3.connect(database_path)
    stored_evidence = connection.execute("SELECT metadata_evidence_json FROM work_items").fetchone()[0]
    assert "private description" not in stored_evidence
    assert "metadata_sha256" in stored_evidence


def test_verified_read_rejects_changed_source(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    snapshot = snapshot_source(source)
    source.write_bytes(b"second")

    with pytest.raises(SourceChangedError):
        read_verified(source, snapshot)


def test_verified_read_reads_source_once(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(b"one read")
    snapshot = snapshot_source(source)
    original = type(source).read_bytes
    reads = 0

    def counted(path):
        nonlocal reads
        reads += 1
        return original(path)

    monkeypatch.setattr(type(source), "read_bytes", counted)

    assert read_verified(source, snapshot) == b"one read"
    assert reads == 1


def test_opening_older_checkpoint_backfills_canonical_occurrence(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"pixels")
    database_path = tmp_path / "checkpoint.sqlite"
    with CheckpointStore(database_path) as database:
        database.enqueue(WorkItem("asset", snapshot_source(source)))
    connection = sqlite3.connect(database_path)
    connection.execute("DELETE FROM occurrences")
    connection.commit()
    connection.close()

    with CheckpointStore(database_path) as database:
        assert database.inventory_counts() == {"unique_assets": 1, "occurrences": 1}


def test_retry_error_message_is_not_persisted(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"pixels")
    item = WorkItem("asset-secret", snapshot_source(source))

    with CheckpointStore(tmp_path / "checkpoint.sqlite") as database:
        database.enqueue(item)
        database.fail_retryable(item.asset_id, RuntimeError("token=private-value"))
        stored = database.connection.execute(
            "SELECT last_error FROM work_items WHERE asset_id=?", (item.asset_id,)
        ).fetchone()[0]

    assert stored.startswith("RuntimeError:message_sha256=")
    assert "private-value" not in stored
