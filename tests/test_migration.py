import csv
import sqlite3
from pathlib import Path

from image_curator.migration_cli import handle_migrate
from image_curator.migration import destination_for, inherit_nested_context


def _args(**values):
    class Args: pass
    result = Args()
    for key, value in values.items(): setattr(result, key, value)
    return result


def _baseline(path: Path, source: Path, *, bucket="待分类"):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source", "kind", "entity", "subject", "source_engine", "bucket"])
        writer.writeheader(); writer.writerow({"source": str(source), "kind": "image", "entity": "IP", "subject": "角色", "source_engine": "Flux-v1", "bucket": bucket})


def test_journaled_move_is_hashed_and_resumable(tmp_path):
    root = tmp_path / "output"; root.mkdir(); source = root / "old" / "a.png"; source.parent.mkdir(); source.write_bytes(b"image")
    baseline = tmp_path / "plan.csv"; _baseline(baseline, source); database = tmp_path / "journal.sqlite"; target = root / "_organized_v3"
    created = handle_migrate(_args(migrate_command="create", database=database, run_id="one", baseline=baseline, root=root, target=target))
    assert created["planned"] == 1 and source.exists()
    applied = handle_migrate(_args(migrate_command="apply", database=database, run_id="one", execute=True, limit=None))
    assert applied["results"] == {"MOVED": 1} and not source.exists()
    destination = next(target.rglob("*.png")); assert destination.read_bytes() == b"image"
    # Re-running cannot duplicate or delete anything further.
    again = handle_migrate(_args(migrate_command="apply", database=database, run_id="one", execute=True, limit=None))
    assert again["attempted"] == 0


def test_changed_source_is_never_moved(tmp_path):
    root = tmp_path / "output"; root.mkdir(); source = root / "a.png"; source.write_bytes(b"one")
    baseline = tmp_path / "plan.csv"; _baseline(baseline, source); database = tmp_path / "journal.sqlite"
    handle_migrate(_args(migrate_command="create", database=database, run_id="one", baseline=baseline, root=root, target=root / "_organized_v3"))
    source.write_bytes(b"changed")
    result = handle_migrate(_args(migrate_command="apply", database=database, run_id="one", execute=True, limit=None))
    assert result["results"] == {"SOURCE_CHANGED": 1} and source.exists()


def test_published_path_remains_locked_status(tmp_path):
    root = tmp_path / "output"; root.mkdir(); source = root / "a.png"; source.write_bytes(b"x")
    baseline = tmp_path / "plan.csv"; _baseline(baseline, source, bucket="已发帖"); database = tmp_path / "journal.sqlite"
    handle_migrate(_args(migrate_command="create", database=database, run_id="one", baseline=baseline, root=root, target=root / "_organized_v3"))
    handle_migrate(_args(migrate_command="apply", database=database, run_id="one", execute=True, limit=None))
    assert "已发帖" in str(next((root / "_organized_v3").rglob("*.png")))


def test_nested_krea_context_skips_transport_and_date_folders(tmp_path):
    root = tmp_path / "output"
    source = root / "Krea2_turbo" / "剑星（StellarBlade）" / "EVE-伊芙" / "待复核" / "2026" / "07" / "x.png"
    entity, subject, engine = inherit_nested_context(source, root, entity="_未归档", subject="未识别", engine="other")
    assert (entity, subject, engine) == ("剑星（StellarBlade）", "EVE-伊芙", "Krea2_turbo")


def test_engine_is_in_filename_and_not_an_extra_directory_level(tmp_path):
    source = tmp_path / "Krea2_turbo" / "IP" / "角色" / "待复核" / "x.png"
    target, state = destination_for(source, root=tmp_path, target_root=tmp_path / "_organized_v3",
                                    entity="IP", subject="角色", engine="Krea2_turbo",
                                    old_bucket="待分类", audit_locked=False, sha256="a" * 64,
                                    mtime_ns=1767225600000000000)
    assert state == "待复核"
    assert target.parent.name == "01" and target.parent.parent.name == "2026"
    assert target.name.startswith("Krea2_turbo__")
    assert "Krea2_turbo" not in target.parts[-6:-1]


def test_unclassifiable_content_goes_to_single_sorting_tree(tmp_path):
    source = tmp_path / "Krea2_turbo" / "krea2_edit" / "x.png"
    target, state = destination_for(source, root=tmp_path, target_root=tmp_path,
                                    entity="_未归档", subject="krea2_edit", engine="Krea2_turbo",
                                    old_bucket="待分类", audit_locked=False, sha256="b" * 64,
                                    mtime_ns=1767225600000000000)
    assert target.parts[-5:] == ("待分拣", "待复核", "2026", "01", "Krea2_turbo__x.png")


def test_sync_plans_only_terminal_directories_and_processed_images(tmp_path):
    root = tmp_path / "output"; root.mkdir()
    first = root / "Flux" / "a.png"; first.parent.mkdir(); first.write_bytes(b"a")
    second = root / "Grok Imagine" / "b.png"; second.parent.mkdir(); second.write_bytes(b"b")
    baseline = tmp_path / "plan.csv"
    with baseline.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source", "kind", "entity", "subject", "source_engine"])
        writer.writeheader()
        writer.writerow({"source": str(first), "kind": "image", "entity": "IP", "subject": "角色", "source_engine": "Flux"})
        writer.writerow({"source": str(second), "kind": "image", "entity": "IP", "subject": "角色", "source_engine": "Grok Imagine"})
    feature = tmp_path / "feature.sqlite"
    with sqlite3.connect(feature) as db:
        db.execute("CREATE TABLE run_occurrences(run_id TEXT, source_path TEXT, occurrence_state TEXT, source_size INTEGER, source_mtime_ns INTEGER, source_sha256 TEXT)")
        db.executemany("INSERT INTO run_occurrences VALUES (?,?,?,?,?,?)", [("features", str(first), "PROCESSED", 1, first.stat().st_mtime_ns, "ca" * 32), ("features", str(second), "FROZEN", None, None, None)])
        db.commit()
    journal = tmp_path / "migration.sqlite"
    result = handle_migrate(_args(migrate_command="create", database=journal, run_id="one", baseline=baseline,
                                  root=root, target=root, feature_db=feature, feature_run_id="features"))
    assert result["planned"] == 1 and result["eligible_directories"] == ["Flux"]
