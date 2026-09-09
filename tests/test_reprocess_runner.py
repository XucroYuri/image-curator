from __future__ import annotations

import os
from concurrent.futures import Future

import pytest
from PIL import Image

from image_curator.inference import AnalysisResult
from image_curator.readonly import snapshot_source
from image_curator.reprocess_runner import OnnxAdapterFactory, process_reprocess
from image_curator.reprocess_store import ReprocessStore
from image_curator.resources import ResourceProfile

_TEST_MANIFEST = {
    "name": "synthetic", "version": "1", "preprocessing": "none",
    "artifacts": [{"role": "fixture", "sha256": "0" * 64}],
}


class SuccessfulAdapter:
    name = "synthetic"

    def analyze(self, image, image_bytes, source):
        return AnalysisResult(embedding=(3.0, 4.0), features={"pixel": image.getpixel((0, 0))})


class IoThenSuccessAdapter(SuccessfulAdapter):
    attempts = 0

    def analyze(self, image, image_bytes, source):
        type(self).attempts += 1
        if type(self).attempts < 3:
            raise OSError("synthetic SMB interruption")
        return super().analyze(image, image_bytes, source)


class AlwaysIoAdapter(SuccessfulAdapter):
    def analyze(self, image, image_bytes, source):
        raise OSError("persistent synthetic SMB interruption")


class OomAdapter(SuccessfulAdapter):
    def analyze(self, image, image_bytes, source):
        raise RuntimeError("CUDA out of memory")


class ProviderLostAdapter(SuccessfulAdapter):
    def analyze(self, image, image_bytes, source):
        raise RuntimeError("CUDAExecutionProvider unavailable")


def _factory():
    return SuccessfulAdapter()


_factory_calls = 0


def _counting_factory():
    global _factory_calls
    _factory_calls += 1
    return SuccessfulAdapter()


def _profile(free=8000):
    return ResourceProfile(12, 32.0, 20.0, "test", 16000, free, 8000, 10)


def _store(tmp_path, image_path, run_id="run"):
    store = ReprocessStore(tmp_path / "run.sqlite")
    store.create_run(
        run_id, cutoff_at="2026-09-09T00:00:00+00:00", config_fingerprint="c",
        code_version="test", model_manifest=_TEST_MANIFEST,
    )
    source = snapshot_source(image_path)
    store.add_asset_occurrence(
        run_id, source.sha256, source, old_bucket="待复核", audit_locked=False
    )
    return store, source


def _png(path, colour=(10, 20, 30)):
    Image.new("RGB", (16, 16), colour).save(path)


def test_success_reads_decodes_once_and_never_changes_source(tmp_path, monkeypatch):
    source_path = tmp_path / "image.png"
    _png(source_path)
    store, source = _store(tmp_path, source_path)
    before = (source_path.stat().st_size, source_path.stat().st_mtime_ns, source_path.read_bytes())

    import image_curator.reprocess_runner as runner

    reads = 0
    opens = 0
    original_read = runner.read_verified
    original_open = runner.Image.open

    def counted_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_read(*args, **kwargs)

    def counted_open(*args, **kwargs):
        nonlocal opens
        opens += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(runner, "read_verified", counted_read)
    monkeypatch.setattr(runner.Image, "open", counted_open)
    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(), test_mode=True,
    )

    assert result["status"] == "COMPLETE"
    assert result["items"] == {"COMPLETE": 1}
    assert result["integrity"] == {"integrity_check": "ok", "foreign_key_check": "ok"}
    assert reads == opens == 1
    assert (source_path.stat().st_size, source_path.stat().st_mtime_ns, source_path.read_bytes()) == before
    row = store.connection.execute(
        "SELECT embedding_dim,length(embedding_f16),features_json FROM run_items"
    ).fetchone()
    assert (row["embedding_dim"], row["length(embedding_f16)"]) == (2, 4)
    assert source.sha256 not in row["features_json"]
    store.close()


def test_source_changed_and_decode_error_are_terminal(tmp_path):
    changed = tmp_path / "changed.png"
    broken = tmp_path / "broken.png"
    _png(changed)
    broken.write_bytes(b"not an image")
    store, changed_snapshot = _store(tmp_path, changed)
    broken_snapshot = snapshot_source(broken)
    store.add_asset_occurrence(
        "run", broken_snapshot.sha256, broken_snapshot, old_bucket=None, audit_locked=False
    )
    changed.write_bytes(changed.read_bytes() + b"changed")

    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(), test_mode=True,
    )
    assert result["items"] == {"FAILED": 1, "SOURCE_CHANGED": 1}
    assert result["occurrences"]["SOURCE_CHANGED"] == 1
    assert result["occurrences"]["FROZEN"] == 1
    assert changed_snapshot.sha256
    store.close()


def test_source_removed_after_freeze_is_source_changed(tmp_path):
    source_path = tmp_path / "removed.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    source_path.unlink()
    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(), test_mode=True,
    )
    assert result["items"] == {"SOURCE_CHANGED": 1}
    assert result["occurrences"] == {"SOURCE_CHANGED": 1}
    store.close()


def test_io_retries_at_most_three_and_then_completes(tmp_path):
    source_path = tmp_path / "retry.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    IoThenSuccessAdapter.attempts = 0

    result = process_reprocess(
        store, "run", adapter_factory=IoThenSuccessAdapter, workers=1,
        resource_profile=_profile(), test_mode=True,
    )
    row = store.connection.execute("SELECT state,attempts FROM run_items").fetchone()
    assert (row["state"], row["attempts"]) == ("COMPLETE", 3)
    assert result["session"]["retryable_failed"] == 2
    store.close()


def test_persistent_io_failure_stops_after_three_attempts(tmp_path):
    source_path = tmp_path / "failed.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    result = process_reprocess(
        store, "run", adapter_factory=AlwaysIoAdapter, workers=1,
        resource_profile=_profile(), test_mode=True,
    )
    assert result["items"] == {"FAILED": 1}
    assert store.connection.execute("SELECT attempts FROM run_items").fetchone()[0] == 3
    store.close()


def test_oom_retries_once_at_one_worker_then_pauses_resumably(tmp_path):
    source_path = tmp_path / "oom.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    result = process_reprocess(
        store, "run", adapter_factory=OomAdapter, workers=1,
        resource_profile=_profile(), test_mode=True,
    )
    assert result["status"] == "PAUSED"
    assert result["paused_reason"] == "gpu_oom_after_single_worker_retry"
    row = store.connection.execute("SELECT state,attempts,last_error FROM run_items").fetchone()
    assert (row["state"], row["attempts"]) == ("RETRYABLE_FAILED", 2)
    assert row["last_error"].startswith("GpuOutOfMemory:")
    store.close()


def test_cuda_provider_loss_pauses_without_cpu_fallback(tmp_path):
    source_path = tmp_path / "provider.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    result = process_reprocess(
        store, "run", adapter_factory=ProviderLostAdapter, workers=1,
        resource_profile=_profile(), test_mode=True,
    )
    assert result["status"] == "PAUSED"
    assert result["paused_reason"] == "cuda_provider_unavailable"
    assert result["items"] == {"RETRYABLE_FAILED": 1}
    store.close()


def test_expired_lease_is_recovered_on_resume(tmp_path):
    source_path = tmp_path / "leased.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    store.claim(
        "run", "dead-worker", lease_seconds=1, now="2020-01-01T00:00:00+00:00"
    )

    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(), test_mode=True,
    )
    assert result["items"] == {"COMPLETE": 1}
    assert store.connection.execute("SELECT attempts FROM run_items").fetchone()[0] == 2
    store.close()


def test_resource_and_foreground_gates_pause_before_claim(tmp_path):
    source_path = tmp_path / "gated.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(1024), test_mode=False,
    )
    assert result["status"] == "PAUSED"
    assert result["paused_reason"] == "free_vram_below_2_gib"
    assert result["items"] == {"PENDING": 1}
    store.close()


def test_maintenance_lock_pauses_before_claim(tmp_path):
    source_path = tmp_path / "maintenance.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    lock = tmp_path / "maintenance.lock"
    lock.touch()
    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(), maintenance_lock=lock, test_mode=False,
    )
    assert result["paused_reason"] == "nas_maintenance_lock"
    assert result["items"] == {"PENDING": 1}
    store.close()

    other = tmp_path / "other"
    other.mkdir()
    source_path = other / "locked.png"
    _png(source_path)
    store, _ = _store(other, source_path)
    lock = tmp_path / "foreground.lock"
    lock.touch()
    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(), foreground_lock=lock, test_mode=False,
    )
    assert result["paused_reason"] == "foreground_gpu_lock"
    assert store.connection.execute("SELECT attempts FROM run_items").fetchone()[0] == 0
    store.close()


def test_process_pool_keeps_sqlite_in_parent(tmp_path):
    source_path = tmp_path / "pool.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    owner_pid = os.getpid()
    result = process_reprocess(
        store, "run", adapter_factory=f"{__name__}:_factory", workers=2,
        resource_profile=_profile(), test_mode=True,
    )
    assert result["status"] == "COMPLETE"
    assert store.connection.execute("SELECT state FROM run_items").fetchone()[0] == "COMPLETE"
    assert os.getpid() == owner_pid
    store.close()


def test_changed_canonical_duplicate_switches_to_other_occurrence(tmp_path):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    _png(first)
    second.write_bytes(first.read_bytes())
    store, snapshot = _store(tmp_path, first)
    duplicate = snapshot_source(second)
    store.add_asset_occurrence(
        "run", duplicate.sha256, duplicate, old_bucket="已发帖", audit_locked=True
    )
    first.write_bytes(first.read_bytes() + b"changed")

    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(), test_mode=True,
    )

    assert result["items"] == {"COMPLETE": 1}
    assert result["occurrences"] == {"PROCESSED": 1, "SOURCE_CHANGED": 1}
    assert store.connection.execute("SELECT source_path FROM run_items").fetchone()[0] == str(second)
    store.close()


def test_explicit_cpu_run_ignores_gpu_memory_gate(tmp_path):
    source_path = tmp_path / "cpu.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1,
        resource_profile=_profile(0), test_mode=False, require_cuda=False,
        adapter_manifest=_TEST_MANIFEST,
    )
    assert result["items"] == {"COMPLETE": 1}
    store.close()


def test_dynamic_free_vram_gate_stops_before_next_claim(tmp_path, monkeypatch):
    store = ReprocessStore(tmp_path / "dynamic.sqlite")
    store.create_run("run", cutoff_at="2026-09-09T00:00:00+00:00", config_fingerprint="c",
                     code_version="test", model_manifest=_TEST_MANIFEST)
    for index, colour in enumerate(((10, 0, 0), (0, 10, 0), (0, 0, 10))):
        path = tmp_path / f"dynamic-{index}.png"
        _png(path, colour)
        snapshot = snapshot_source(path)
        store.add_asset_occurrence("run", snapshot.sha256, snapshot,
                                   old_bucket="待分类", audit_locked=False)
    profiles = iter((_profile(8000), _profile(1024)))
    import image_curator.reprocess_runner as runner
    monkeypatch.setattr(runner, "discover_resources", lambda: next(profiles))

    result = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1, max_in_flight=2,
        test_mode=False, require_cuda=True, adapter_manifest=_TEST_MANIFEST,
    )
    assert result["status"] == "PAUSED"
    assert result["paused_reason"] == "free_vram_below_2_gib"
    assert result["items"] == {"COMPLETE": 2, "PENDING": 1}
    store.close()


def test_process_pool_and_adapter_live_for_entire_run(tmp_path, monkeypatch):
    import image_curator.reprocess_runner as runner

    created = 0

    class ImmediateExecutor:
        def __init__(self, *, max_workers, initializer, initargs):
            nonlocal created
            created += 1
            initializer(*initargs)

        def submit(self, function, item):
            future = Future()
            future.set_result(function(item))
            return future

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(runner, "ProcessPoolExecutor", ImmediateExecutor)
    paths = []
    for number in range(3):
        path = tmp_path / f"pool-{number}.png"
        _png(path, (number + 1, 20, 30))
        paths.append(path)
    store, _ = _store(tmp_path, paths[0])
    for path in paths[1:]:
        snap = snapshot_source(path)
        store.add_asset_occurrence("run", snap.sha256, snap, old_bucket=None, audit_locked=False)
    global _factory_calls
    _factory_calls = 0

    result = process_reprocess(
        store, "run", adapter_factory=f"{__name__}:_counting_factory", workers=2,
        max_in_flight=2, resource_profile=_profile(), test_mode=True,
    )
    assert result["items"] == {"COMPLETE": 3}
    assert created == 1
    assert _factory_calls == 1
    store.close()


def test_invocation_limit_leaves_a_resumable_partial_run(tmp_path):
    paths = []
    for number in range(3):
        path = tmp_path / f"limit-{number}.png"
        _png(path, (number + 10, 20, 30))
        paths.append(path)
    store, _ = _store(tmp_path, paths[0])
    for path in paths[1:]:
        snap = snapshot_source(path)
        store.add_asset_occurrence("run", snap.sha256, snap, old_bucket=None, audit_locked=False)

    partial = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1, limit=2,
        resource_profile=_profile(), test_mode=True,
    )
    assert partial["status"] == "PAUSED"
    assert partial["paused_reason"] == "invocation_limit_reached"
    assert partial["session"]["attempted"] == 2
    assert partial["items"] == {"COMPLETE": 2, "PENDING": 1}
    resumed = process_reprocess(
        store, "run", adapter_factory=_factory, workers=1, limit=2,
        resource_profile=_profile(), test_mode=True,
    )
    assert resumed["status"] == "COMPLETE"
    assert resumed["items"] == {"COMPLETE": 3}
    store.close()


def test_onnx_factory_is_pickle_safe_and_forwards_explicit_inputs(tmp_path, monkeypatch):
    import image_curator.onnx_adapters as adapters

    captured = {}
    sentinel = object()

    def fake_from_paths(moat, tags, *, nudenet_model, providers):
        captured.update(moat=moat, tags=tags, nudenet=nudenet_model, providers=providers)
        return sentinel

    monkeypatch.setattr(adapters.WD14MoatNudeNetAdapter, "from_paths", fake_from_paths)
    for path in (tmp_path / "moat.onnx", tmp_path / "tags.csv", tmp_path / "nude.onnx"):
        path.write_bytes(path.name.encode())
    from image_curator.reprocess import build_model_manifest
    expected = build_model_manifest(
        tmp_path / "moat.onnx", tmp_path / "tags.csv", tmp_path / "nude.onnx",
        ("CUDAExecutionProvider",),
    )
    factory = OnnxAdapterFactory(
        tmp_path / "moat.onnx", tmp_path / "tags.csv", tmp_path / "nude.onnx",
        ("CUDAExecutionProvider",), expected_manifest=expected,
    )
    assert factory() is sentinel
    assert captured == {
        "moat": tmp_path / "moat.onnx",
        "tags": tmp_path / "tags.csv",
        "nudenet": tmp_path / "nude.onnx",
        "providers": ("CUDAExecutionProvider",),
    }


def test_onnx_factory_adds_explicit_cuda_dll_directories(tmp_path, monkeypatch):
    import image_curator.onnx_adapters as adapters
    import image_curator.reprocess_runner as runner

    dll_dir = tmp_path / "cuda"
    dll_dir.mkdir()
    added = []
    sentinel = object()
    monkeypatch.setattr(os, "add_dll_directory", lambda path: added.append(path) or sentinel,
                        raising=False)
    monkeypatch.setattr(adapters.WD14MoatNudeNetAdapter, "from_paths", lambda *args, **kwargs: object())
    runner._DLL_DIRECTORY_HANDLES.clear()
    (tmp_path / "model").write_bytes(b"model")
    (tmp_path / "tags").write_bytes(b"tags")
    from image_curator.reprocess import build_model_manifest
    expected = build_model_manifest(tmp_path / "model", tmp_path / "tags", None,
                                    ("CUDAExecutionProvider",))

    factory = OnnxAdapterFactory(tmp_path / "model", tmp_path / "tags", cuda_dll_dirs=(dll_dir,),
                                 expected_manifest=expected)
    factory()

    assert added == [str(dll_dir.resolve())]
    assert runner._DLL_DIRECTORY_HANDLES == [sentinel]
    assert os.environ["PATH"].split(os.pathsep)[0] == str(dll_dir.resolve())


def test_invalid_cuda_dll_directory_is_rejected_before_claim(tmp_path):
    from image_curator.reprocess import build_model_manifest

    model = tmp_path / "model.onnx"
    tags = tmp_path / "tags.csv"
    model.write_bytes(b"model")
    tags.write_bytes(b"tags")
    expected = build_model_manifest(model, tags, None, ("CUDAExecutionProvider",))
    source_path = tmp_path / "image.png"
    _png(source_path)
    store = ReprocessStore(tmp_path / "invalid-dll.sqlite")
    store.create_run("run", cutoff_at="2026-09-09T00:00:00+00:00", config_fingerprint="c",
                     code_version="test", model_manifest=expected)
    snapshot = snapshot_source(source_path)
    store.add_asset_occurrence("run", snapshot.sha256, snapshot, old_bucket=None, audit_locked=False)
    factory = OnnxAdapterFactory(model, tags, providers=("CUDAExecutionProvider",),
                                 cuda_dll_dirs=(tmp_path / "absent",), expected_manifest=expected)
    with pytest.raises(ValueError, match="CUDA DLL path"):
        process_reprocess(store, "run", adapter_factory=factory, workers=1,
                          resource_profile=_profile(), test_mode=False)
    assert store.connection.execute("SELECT attempts FROM run_items").fetchone()[0] == 0
    store.close()


def test_broken_process_pool_pauses_and_releases_entire_claim(tmp_path, monkeypatch):
    import image_curator.reprocess_runner as runner

    class BrokenExecutor:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args, **kwargs):
            raise runner.BrokenProcessPool("synthetic native crash")

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(runner, "ProcessPoolExecutor", BrokenExecutor)
    source_path = tmp_path / "pool-crash.png"
    _png(source_path)
    store, _ = _store(tmp_path, source_path)
    result = process_reprocess(
        store, "run", adapter_factory=f"{__name__}:_factory", workers=2,
        resource_profile=_profile(), test_mode=True,
    )
    assert result["status"] == "PAUSED"
    assert result["paused_reason"] == "worker_pool_broken"
    assert result["items"] == {"RETRYABLE_FAILED": 1}
    store.close()
