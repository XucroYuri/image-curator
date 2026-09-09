from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from image_curator.reprocess import (
    BaselineOccurrence,
    ProposedDecision,
    added_deferred_rows,
    build_diff_rows,
    build_machine_outputs,
    build_model_manifest,
    config_fingerprint,
    freeze_occurrence,
    load_migration_csv,
    select_review_queue,
    sha256_file,
    unique_assets,
)


def test_manifest_hashes_files_and_never_retains_absolute_paths(tmp_path):
    moat = tmp_path / "private" / "moat.onnx"
    tags = tmp_path / "private" / "tags.csv"
    nude = tmp_path / "private" / "nude.onnx"
    moat.parent.mkdir()
    moat.write_bytes(b"moat")
    tags.write_bytes(b"name")
    nude.write_bytes(b"nude")

    manifest = build_model_manifest(moat, tags, nude, ["CUDAExecutionProvider"])

    assert manifest["models"]["moat"] == {"name": "moat.onnx", "sha256": sha256_file(moat)}
    assert str(tmp_path) not in str(manifest)
    assert config_fingerprint({"model_path": moat, "manifest": manifest}) == config_fingerprint(
        {"model_path": Path("C:/another-host/moat.onnx"), "manifest": manifest}
    )


def test_load_csv_filters_non_images_and_internal_paths_and_locks_posted(tmp_path):
    migration = tmp_path / "migration.csv"
    migration.write_text(
        "source,destination,kind,bucket,size\n"
        "\\\\server\\share\\output\\a.png,target,image,待复核,1\n"
        "\\\\server\\share\\output\\_organized_v2\\b.png,target,image,待复核,1\n"
        "\\\\server\\share\\output\\_test_20260728\\c.png,target,image,待复核,1\n"
        "\\\\server\\share\\output\\posted.jpg,target,image,已发帖,2\n"
        "\\\\server\\share\\output\\sidecar.json,target,sidecar,待复核,1\n",
        encoding="utf-8",
    )

    rows = load_migration_csv(migration)

    assert [row.source_path for row in rows] == [
        r"\\server\share\output\a.png", r"\\server\share\output\posted.jpg"
    ]
    assert rows[1].audit_locked
    assert rows[0].baseline_identity_unverifiable


def test_freeze_uses_cutoff_and_stability_without_source_mutation(tmp_path):
    image = tmp_path / "item.png"
    image.write_bytes(b"pixels")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old = now - timedelta(minutes=10)
    timestamp = old.timestamp()
    import os
    os.utime(image, (timestamp, timestamp))
    occurrence = BaselineOccurrence(str(image), "待复核", baseline_sha256=sha256_file(image))

    frozen = freeze_occurrence(occurrence, cutoff=now - timedelta(minutes=1), now=now)

    assert frozen.status == "processed"
    assert frozen.sha256 == sha256_file(image)
    late = BaselineOccurrence(str(image), "待复核")
    os.utime(image, (now.timestamp(), now.timestamp()))
    assert freeze_occurrence(late, cutoff=now - timedelta(minutes=1), now=now).status == "added_deferred"
    assert image.read_bytes() == b"pixels"


def test_freeze_detects_baseline_mismatch_and_missing_and_unverifiable(tmp_path):
    image = tmp_path / "item.png"
    image.write_bytes(b"pixels")
    now = datetime.now(UTC)
    timestamp = (now - timedelta(minutes=10)).timestamp()
    import os
    os.utime(image, (timestamp, timestamp))

    changed = freeze_occurrence(BaselineOccurrence(str(image), "旧", baseline_sha256="0" * 64),
                                cutoff=now, now=now)
    no_hash = freeze_occurrence(BaselineOccurrence(str(image), "旧"), cutoff=now, now=now)
    missing = freeze_occurrence(BaselineOccurrence(str(tmp_path / "gone.png"), "旧"), cutoff=now, now=now)
    assert changed.status == "source_changed"
    assert no_hash.status == "baseline_identity_unverifiable"
    assert missing.status == "missing"


def test_duplicates_run_once_but_emit_path_level_locked_diffs(tmp_path):
    image = tmp_path / "item.png"
    image.write_bytes(b"pixels")
    now = datetime.now(UTC)
    import os
    os.utime(image, ((now - timedelta(minutes=10)).timestamp(),) * 2)
    digest = sha256_file(image)
    first = freeze_occurrence(BaselineOccurrence(str(image), "待复核", baseline_sha256=digest), cutoff=now, now=now)
    second_occurrence = BaselineOccurrence(str(image) + "-alias", "已发帖", baseline_sha256=digest)
    # Supply an already frozen duplicate to model two recorded paths sharing one content asset.
    second = type(first)(second_occurrence, "processed", first.size, first.mtime_ns, digest)
    assert unique_assets([first, second]) == {digest: tuple(sorted([str(image), str(image) + "-alias"], key=str.casefold))}
    rows = build_diff_rows([first, second], {digest: ProposedDecision(digest, "待发帖", "Tifa", .99)})
    assert rows[0]["effective_bucket"] == "待发帖"
    assert rows[1]["effective_bucket"] == "已发帖"
    assert rows[1]["action"] is None


def test_outputs_cover_every_status_and_deterministic_stratified_queue():
    rows = []
    for index, status in enumerate(("processed", "missing", "source_changed", "baseline_identity_unverifiable", "added_deferred")):
        rows.append({"occurrence_id": f"id-{index}", "asset_id": f"asset-{index}", "status": status,
                     "proposed_bucket": "待复核", "review_reasons": ["safety_borderline" if index % 2 else "identity_low"],
                     "review_strata": "safety" if index % 2 else "identity"})
    outputs = build_machine_outputs(rows, review_limit=3, review_seed="test")
    assert outputs["summary"]["by_status"] == {status: 1 for status in (
        "added_deferred", "baseline_identity_unverifiable", "missing", "processed", "source_changed"
    )}
    assert select_review_queue(rows, limit=3, seed="test") == outputs["review_rows"]
    assert len(outputs["review_rows"]) == 3


def test_added_rows_do_not_use_baseline_folder_hints():
    baseline = [BaselineOccurrence("C:/library/known.png", "待复核")]
    assert added_deferred_rows(["C:/library/known.png", "C:/library/new.png"], baseline) == [
        {"source_path": "C:/library/new.png", "status": "added_deferred", "reason": "not_in_baseline"}
    ]


def test_manifest_requires_valid_files_and_providers(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_model_manifest(tmp_path / "none.onnx", tmp_path / "tags.csv", None, ["CUDAExecutionProvider"])
