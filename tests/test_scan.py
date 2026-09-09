from image_curator.checkpoint import CheckpointStore
from image_curator.scan import scan_and_enqueue


def test_scan_enqueues_content_once_and_resumes_without_source_mutation(tmp_path):
    library = tmp_path / "library"
    nested = library / "nested"
    nested.mkdir(parents=True)
    first = library / "first.jpg"
    duplicate = nested / "duplicate.png"
    ignored = library / "notes.txt"
    first.write_bytes(b"same pixels")
    duplicate.write_bytes(b"same pixels")
    ignored.write_text("not an image", encoding="utf-8")
    original_mtime = first.stat().st_mtime_ns

    with CheckpointStore(tmp_path / "output" / "checkpoint.sqlite") as store:
        initial = scan_and_enqueue(store, [library])
        resumed = scan_and_enqueue(store, [library])

        assert initial.as_dict() == {"discovered": 2, "enqueued": 1, "duplicates": 1, "errors": 0}
        assert resumed.as_dict() == {"discovered": 2, "enqueued": 0, "duplicates": 2, "errors": 0}
        assert store.status() == {"PENDING": 1}
        assert store.inventory_counts() == {"unique_assets": 1, "occurrences": 2}
        pending = store.pending()[0]
        assert {location.path for location in store.occurrences_for(pending.asset_id)} == {
            first.resolve(), duplicate.resolve()
        }
    assert first.read_bytes() == b"same pixels"
    assert first.stat().st_mtime_ns == original_mtime
