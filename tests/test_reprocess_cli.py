from __future__ import annotations

import csv
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from image_curator.cli import main
from image_curator.embeddings import moat_float16_blob
from image_curator.readonly import snapshot_source
from image_curator.reprocess_store import ReprocessStore


def _inputs(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    first = library / "first.png"
    posted = library / "posted.png"
    Image.new("RGB", (4, 4), "red").save(first)
    Image.new("RGB", (4, 4), "blue").save(posted)
    old = (datetime.now(UTC) - timedelta(minutes=10)).timestamp()
    os.utime(first, (old, old))
    os.utime(posted, (old, old))
    baseline = tmp_path / "migration.csv"
    with baseline.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "destination", "kind", "bucket", "size"])
        writer.writeheader()
        writer.writerow({"source": first, "destination": "x", "kind": "image", "bucket": "待复核",
                         "size": first.stat().st_size})
        writer.writerow({"source": posted, "destination": "y", "kind": "image", "bucket": "已发帖",
                         "size": posted.stat().st_size})
        writer.writerow({"source": library / "first.json", "destination": "z", "kind": "sidecar",
                         "bucket": "待复核", "size": 1})
    moat = tmp_path / "moat.onnx"
    tags = tmp_path / "tags.csv"
    nude = tmp_path / "nude.onnx"
    moat.write_bytes(b"model")
    tags.write_text("name\ngeneral\n", encoding="utf-8")
    nude.write_bytes(b"nude")
    return library, baseline, moat, tags, nude


def test_reprocess_create_status_and_diff_are_read_only(tmp_path, capsys):
    library, baseline, moat, tags, nude = _inputs(tmp_path)
    database = tmp_path / "output" / "run.sqlite"
    before = {path: path.read_bytes() for path in library.iterdir()}
    common = [
        str(database), "--run-id", "legacy-2026", "--moat-model", str(moat),
        "--wd14-tags", str(tags), "--nudenet-model", str(nude),
        "--provider", "CPUExecutionProvider",
    ]

    assert main(["reprocess", "create", *common, "--baseline", str(baseline),
                 "--root", str(library), "--expected-images", "2"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["read_only_sources"] is True
    assert created["status"]["items"] == {"PENDING": 2}

    assert main(["reprocess", "status", str(database), "--run-id", "legacy-2026"]) == 0
    assert json.loads(capsys.readouterr().out)["integrity_check"] == "ok"

    output = tmp_path / "reports"
    assert main(["reprocess", "diff", str(database), "--run-id", "legacy-2026",
                 "--output", str(output), "--review-limit", "2"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["occurrences"] == 2
    assert summary["audit_locked"] == 1
    with (output / "reprocess-diff.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    published = next(row for row in rows if row["old_bucket"] == "已发帖")
    assert published["effective_bucket"] == "已发帖"
    assert published["action"] == ""
    assert {path: path.read_bytes() for path in library.iterdir()} == before


def test_reprocess_create_rejects_wrong_fixed_scope(tmp_path, capsys):
    library, baseline, moat, tags, _ = _inputs(tmp_path)
    result = ["reprocess", "create", str(tmp_path / "run.sqlite"), "--run-id", "bad",
              "--baseline", str(baseline), "--root", str(library), "--expected-images", "3",
              "--moat-model", str(moat), "--wd14-tags", str(tags),
              "--provider", "CPUExecutionProvider"]
    try:
        main(result)
    except SystemExit as error:
        assert error.code == 2
    assert "baseline image count differs" in capsys.readouterr().err


def test_reprocess_create_rejects_duplicate_or_outside_paths(tmp_path, capsys):
    library, baseline, moat, tags, _ = _inputs(tmp_path)
    rows = list(csv.DictReader(baseline.open(encoding="utf-8")))
    with baseline.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([rows[0], rows[0]])
    arguments = [
        "reprocess", "create", str(tmp_path / "duplicate.sqlite"), "--run-id", "bad",
        "--baseline", str(baseline), "--root", str(library), "--expected-images", "2",
        "--moat-model", str(moat), "--wd14-tags", str(tags),
        "--provider", "CPUExecutionProvider",
    ]
    with pytest.raises(SystemExit):
        main(arguments)
    assert "duplicate image paths" in capsys.readouterr().err

    rows[1]["source"] = str(tmp_path / "outside.png")
    with baseline.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    arguments[2] = str(tmp_path / "outside.sqlite")
    with pytest.raises(SystemExit):
        main(arguments)
    assert "outside the authorized root" in capsys.readouterr().err


def test_authorized_root_rejects_ancestor_junction_escape(tmp_path, monkeypatch):
    from image_curator.reprocess import BaselineOccurrence
    from image_curator.reprocess_cli import _validate_authorized_paths

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "image.png").write_bytes(b"pixels")
    link = root / "link"
    link.mkdir()
    monkeypatch.setattr(os.path, "isjunction", lambda path: Path(path) == link, raising=False)
    with pytest.raises(ValueError, match="symlink or junction"):
        _validate_authorized_paths(root, [BaselineOccurrence(str(link / "image.png"), "待复核")])


def test_reprocess_decide_uses_explicit_labels_and_keeps_status_in_review(tmp_path, capsys):
    database = tmp_path / "decision.sqlite"
    sources = []
    with ReprocessStore(database) as store:
        store.create_run("features", cutoff_at=datetime.now(UTC).isoformat(), config_fingerprint="features",
                         code_version="test", model_manifest={"portable": True})
        for index, (vector, colour) in enumerate((((1.0, 0.0), "red"), ((0.0, 1.0), "blue"))):
            path = tmp_path / f"{index}.png"
            Image.new("RGB", (2, 2), colour).save(path)
            snapshot = snapshot_source(path)
            store.add_asset_occurrence("features", snapshot.sha256, snapshot,
                                       old_bucket="待分类", audit_locked=False)
            item = store.claim("features", "test", limit=1)[0]
            store.complete("features", item.asset_id, metadata_evidence={}, features={},
                           lease_token=item.lease_token,
                           embedding_f16=moat_float16_blob(vector, dimensions=2), embedding_dim=2)
            sources.append(snapshot)
        store.set_run_status("features", "COMPLETE")
    references = tmp_path / "references.json"
    labeled = tmp_path / "labeled.json"
    references.write_text(json.dumps({"A": [[1, 0]], "B": [[0, 1]]}), encoding="utf-8")
    labeled.write_text(json.dumps([
        {"label": "A", "vector": [1, 0]}, {"label": "B", "vector": [0, 1]},
        {"label": None, "vector": [-1, -1]},
    ]), encoding="utf-8")

    assert main(["reprocess", "decide", str(database), "--run-id", "features",
                 "--decision-id", "decision-1", "--references", str(references),
                 "--labeled", str(labeled), "--similarity", "0.8", "--margin", "0.1",
                 "--min-accepted", "2", "--min-unknown", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETE"
    assert result["results"] == 2
    with ReprocessStore(database) as store:
        rows = store.decision_diffs("decision-1")
    assert {row["candidate_identity"] for row in rows} == {"A", "B"}
    assert {row["proposed_bucket"] for row in rows} == {"待复核"}
