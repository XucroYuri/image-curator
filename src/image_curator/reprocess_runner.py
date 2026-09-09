"""Resource-bounded execution for versioned, read-only analysis runs.

Only the parent process receives a :class:`ReprocessStore`.  Worker processes
receive immutable source snapshots and return compact evidence, keeping SQLite
ownership and all state transitions in one process.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from .embeddings import moat_float16_blob
from .evidence import metadata_evidence
from .inference import AnalysisAdapter, AnalysisResult, InferenceAdapter
from .readonly import SourceChangedError, read_verified
from .reprocess_store import ReprocessItem, ReprocessStore
from .resources import ResourceProfile, choose_resource_plan, discover_resources
from .technical import technical_evidence

AdapterFactory = str | Callable[[], object]


def _validate_custom_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(value)
    if not manifest.get("name") or not manifest.get("version") or not manifest.get("preprocessing"):
        raise ValueError("custom adapter manifest requires name, version, and preprocessing")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("custom adapter manifest requires fingerprinted artifacts")
    for artifact in artifacts:
        digest = artifact.get("sha256") if isinstance(artifact, dict) else None
        if (not artifact.get("role") if isinstance(artifact, dict) else True) or not isinstance(digest, str):
            raise ValueError("custom adapter artifacts require role and SHA-256")
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError("custom adapter artifact has an invalid SHA-256")
        if "path" in artifact:
            raise ValueError("custom adapter manifests must not persist local paths")
    return manifest


class CudaProviderUnavailable(RuntimeError):
    """The requested CUDA execution provider disappeared or fell back."""


class GpuOutOfMemory(RuntimeError):
    """Inference exhausted GPU memory."""


class AdapterInitializationError(RuntimeError):
    """A runtime or immutable input failed before per-image inference."""


@dataclass(frozen=True)
class OnnxAdapterFactory:
    """Pickle-safe factory for explicit, caller-verified local ONNX inputs."""

    moat_model: Path
    wd14_tags: Path
    nudenet_model: Path | None = None
    providers: tuple[str, ...] = ("CUDAExecutionProvider",)
    cuda_dll_dirs: tuple[Path, ...] = ()
    expected_manifest: Mapping[str, Any] | None = None

    def __call__(self) -> object:
        # Python-installed CUDA runtimes are intentionally outside Windows'
        # default DLL search path.  Keep the directory cookies alive for the
        # lifetime of this worker before importing/initialising ONNX Runtime.
        for directory in self.cuda_dll_dirs:
            resolved = directory.resolve(strict=True)
            if not resolved.is_dir():
                raise ValueError(f"CUDA DLL path is not a directory: {resolved}")
            os.environ["PATH"] = f"{resolved}{os.pathsep}{os.environ.get('PATH', '')}"
            if hasattr(os, "add_dll_directory"):
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(resolved)))

        from .onnx_adapters import WD14MoatNudeNetAdapter
        from .reprocess import build_model_manifest

        before = build_model_manifest(
            self.moat_model, self.wd14_tags, self.nudenet_model, self.providers
        )
        if self.expected_manifest is not None and before != dict(self.expected_manifest):
            raise ValueError("local model inputs changed before worker initialization")

        adapter = WD14MoatNudeNetAdapter.from_paths(
            self.moat_model,
            self.wd14_tags,
            nudenet_model=self.nudenet_model,
            providers=self.providers,
        )
        after = build_model_manifest(
            self.moat_model, self.wd14_tags, self.nudenet_model, self.providers
        )
        if before != after:
            raise ValueError("local model inputs changed during worker initialization")
        return adapter


@dataclass(frozen=True)
class WorkerResult:
    asset_id: str
    outcome: str
    metadata: dict[str, Any] | None = None
    features: dict[str, Any] | None = None
    embedding_f16: bytes | None = None
    embedding_dim: int | None = None
    error_type: str | None = None
    error_digest: str | None = None


_WORKER_ADAPTER: object | None = None
_WORKER_FACTORY: AdapterFactory | None = None
_WORKER_REQUIRE_CUDA = True
_DLL_DIRECTORY_HANDLES: list[object] = []


def _safe_digest(error: BaseException) -> str:
    return hashlib.sha256(str(error).encode("utf-8", errors="replace")).hexdigest()[:16]


def _make_adapter(factory: AdapterFactory) -> object:
    if callable(factory):
        return factory()
    module_name, separator, factory_name = factory.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("adapter factory must use module:factory syntax")
    creator = getattr(import_module(module_name), factory_name, None)
    if not callable(creator):
        raise ValueError(f"adapter factory not found: {factory}")
    return creator()


def _worker_init(factory: AdapterFactory, require_cuda: bool) -> None:
    global _WORKER_ADAPTER, _WORKER_FACTORY, _WORKER_REQUIRE_CUDA
    _WORKER_FACTORY = factory
    _WORKER_REQUIRE_CUDA = require_cuda
    _WORKER_ADAPTER = None  # Initialise lazily so startup failures become typed results.


def _adapter() -> object:
    global _WORKER_ADAPTER
    if _WORKER_ADAPTER is None:
        if _WORKER_FACTORY is None:  # pragma: no cover - defensive worker invariant
            raise RuntimeError("worker adapter factory was not configured")
        try:
            _WORKER_ADAPTER = _make_adapter(_WORKER_FACTORY)
        except BaseException as error:
            raise AdapterInitializationError(str(error)) from error
    return _WORKER_ADAPTER


def _is_provider_error(error: BaseException) -> bool:
    message = str(error).lower()
    return ("cuda" in message and "provider" in message) or "cudaexecutionprovider" in message


def _is_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda_error_out_of_memory" in message


def _check_cuda_evidence(value: Any) -> None:
    """Reject an explicitly reported provider fallback while allowing generic test adapters."""
    if isinstance(value, dict):
        providers = value.get("providers")
        if isinstance(providers, list) and providers and providers[0] != "CUDAExecutionProvider":
            raise CudaProviderUnavailable("adapter reported a non-CUDA primary provider")
        for child in value.values():
            _check_cuda_evidence(child)
    elif isinstance(value, list):
        for child in value:
            _check_cuda_evidence(child)


def _analyze(item: ReprocessItem) -> WorkerResult:
    """Read and decode exactly once, returning no source pixels to the parent."""
    try:
        image_bytes = read_verified(item.source.path, item.source)
    except (SourceChangedError, FileNotFoundError) as error:
        return WorkerResult(item.asset_id, "SOURCE_CHANGED", error_type=type(error).__name__)
    except OSError as error:
        return WorkerResult(
            item.asset_id, "IO_RETRYABLE", error_type=type(error).__name__, error_digest=_safe_digest(error)
        )
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            opened.load()
            raw_metadata = dict(opened.info)
            image = opened.convert("RGB")
            features: dict[str, Any] = {
                "width": opened.width,
                "height": opened.height,
                "format": opened.format,
                "mode": opened.mode,
                "technical": technical_evidence(image),
            }
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return WorkerResult(
            item.asset_id, "DECODE_OR_INVALID", error_type=type(error).__name__, error_digest=_safe_digest(error)
        )
    try:
        adapter = _adapter()
        if isinstance(adapter, AnalysisAdapter):
            analysis = adapter.analyze(image, image_bytes, item.source.path)
        elif isinstance(adapter, InferenceAdapter):
            analysis = AnalysisResult(embedding=adapter.embed(image_bytes, item.source.path))
        else:
            raise TypeError("adapter factory must return an analysis or inference adapter")
        if _WORKER_REQUIRE_CUDA:
            _check_cuda_evidence(analysis.evidence)
        evidence = metadata_evidence(raw_metadata)
        if analysis.evidence:
            evidence["analysis"] = analysis.evidence
        features["inference"] = {"adapter": adapter.name, "state": "complete"}
        if analysis.features:
            features["analysis"] = analysis.features
        blob = None
        dimensions = None
        if analysis.embedding is not None:
            vector = tuple(analysis.embedding)
            dimensions = len(vector)
            blob = moat_float16_blob(vector, dimensions=dimensions)
        return WorkerResult(item.asset_id, "COMPLETE", evidence, features, blob, dimensions)
    except AdapterInitializationError as error:
        return WorkerResult(
            item.asset_id, "ADAPTER_INIT_FAILED", error_type=type(error).__name__,
            error_digest=_safe_digest(error)
        )
    except ValueError as error:
        if _is_provider_error(error):
            outcome = "CUDA_PROVIDER_LOST"
        elif _is_oom(error):
            outcome = "OOM"
        else:
            outcome = "DECODE_OR_INVALID"
        return WorkerResult(
            item.asset_id, outcome, error_type=type(error).__name__, error_digest=_safe_digest(error)
        )
    except OSError as error:
        return WorkerResult(
            item.asset_id, "IO_RETRYABLE", error_type=type(error).__name__, error_digest=_safe_digest(error)
        )
    except BaseException as error:  # A worker must always return a parent-actionable result.
        if _is_provider_error(error):
            outcome = "CUDA_PROVIDER_LOST"
        elif _is_oom(error):
            outcome = "OOM"
        else:
            outcome = "FAILED"
        return WorkerResult(
            item.asset_id, outcome, error_type=type(error).__name__, error_digest=_safe_digest(error)
        )


def _stored_error(result: WorkerResult) -> RuntimeError:
    # The original message never crosses the persistence boundary.
    message = f"{result.error_type or 'WorkerError'}:digest={result.error_digest or 'none'}"
    if result.outcome == "OOM":
        return GpuOutOfMemory(message)
    if result.outcome == "CUDA_PROVIDER_LOST":
        return CudaProviderUnavailable(message)
    return RuntimeError(message)


def _gate_reason(profile: ResourceProfile, foreground_lock: Path | None,
                 maintenance_lock: Path | None = None, *, test_mode: bool,
                 require_cuda: bool = True, startup: bool = True) -> str | None:
    if test_mode:
        return None
    if foreground_lock is not None and foreground_lock.exists():
        return "foreground_gpu_lock"
    if maintenance_lock is not None and maintenance_lock.exists():
        return "nas_maintenance_lock"
    if require_cuda and profile.gpu_free_mib < 2048:
        return "free_vram_below_2_gib"
    if profile.ram_available_gib and profile.ram_available_gib < 2:
        return "free_ram_below_2_gib"
    if require_cuda and startup and profile.gpu_util_percent > 95:
        return "gpu_utilization_above_95_percent"
    return None


def _result_statistics(
    store: ReprocessStore,
    run_id: str,
    *,
    profile: ResourceProfile,
    workers: int,
    maximum_in_flight: int,
    session: dict[str, int],
    paused_reason: str | None,
) -> dict[str, Any]:
    status = store.status(run_id)
    return {
        "run_id": run_id,
        "status": status["status"],
        "workers": workers,
        "max_in_flight": maximum_in_flight,
        "paused_reason": paused_reason,
        "session": {key: session[key] for key in sorted(session)},
        "items": status["items"],
        "occurrences": status["occurrences"],
        "resource_profile": asdict(profile),
        "integrity": store.integrity(),
    }


def process_reprocess(
    store: ReprocessStore,
    run_id: str,
    *,
    adapter_factory: AdapterFactory,
    workers: int | None = None,
    resource_profile: ResourceProfile | None = None,
    foreground_lock: Path | None = None,
    maintenance_lock: Path | None = None,
    lease_seconds: int = 300,
    max_in_flight: int | None = None,
    limit: int | None = None,
    test_mode: bool = False,
    require_cuda: bool = True,
    adapter_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run or resume one analysis run and return deterministic JSON-ready statistics.

    ``adapter_factory`` is a previously manifest-validated ``module:factory`` hook.
    A callable is accepted for the explicit single-worker test mode only.
    """
    if lease_seconds < 1:
        raise ValueError("lease_seconds must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    profile = resource_profile or discover_resources()
    plan = choose_resource_plan(profile)
    worker_count = workers if workers is not None else plan.workers
    if worker_count not in (1, 2):
        raise ValueError("workers must be 1 or 2")
    if callable(adapter_factory) and not isinstance(adapter_factory, OnnxAdapterFactory) and worker_count != 1:
        raise ValueError("callable adapter factories require workers=1; use module:factory for processes")
    capacity = max_in_flight if max_in_flight is not None else worker_count * plan.prefetch_batches
    if capacity < worker_count:
        raise ValueError("max_in_flight must be at least workers")
    session = {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "retryable_failed": 0,
        "source_changed": 0,
    }
    dynamic_profile = resource_profile is None
    paused_reason = _gate_reason(profile, foreground_lock, maintenance_lock, test_mode=test_mode,
                                 require_cuda=require_cuda)
    if paused_reason is not None:
        store.set_run_status(run_id, "PAUSED", statistics={"paused_reason": paused_reason})
        return _result_statistics(
            store, run_id, profile=profile, workers=worker_count,
            maximum_in_flight=capacity, session=session, paused_reason=paused_reason
        )

    stored_manifest = json.loads(store.connection.execute(
        "SELECT model_manifest_json FROM analysis_runs WHERE run_id=?", (run_id,)
    ).fetchone()[0])
    if isinstance(adapter_factory, OnnxAdapterFactory):
        for directory in adapter_factory.cuda_dll_dirs:
            if not directory.is_dir():
                raise ValueError(f"CUDA DLL path is not a directory: {directory}")
        from .reprocess import build_model_manifest

        actual_manifest = build_model_manifest(
            adapter_factory.moat_model, adapter_factory.wd14_tags,
            adapter_factory.nudenet_model, adapter_factory.providers,
        )
        if (adapter_factory.expected_manifest is None
                or actual_manifest != dict(adapter_factory.expected_manifest)
                or actual_manifest != stored_manifest):
            raise ValueError("adapter inputs do not match the immutable run manifest")
    elif not test_mode:
        if adapter_manifest is None or _validate_custom_manifest(adapter_manifest) != stored_manifest:
            raise ValueError("formal custom adapter manifest does not match the immutable run")

    store.set_run_status(run_id, "RUNNING")
    worker_serial = 0
    executor: ProcessPoolExecutor | None = None
    direct_initialized = False

    try:
        while True:
            if limit is not None and session["attempted"] >= limit:
                limited_status = store.status(run_id)["items"]
                if limited_status.get("PENDING", 0) or limited_status.get("RETRYABLE_FAILED", 0):
                    paused_reason = "invocation_limit_reached"
                break
            claim_limit = capacity
            if limit is not None:
                claim_limit = min(claim_limit, limit - session["attempted"])
            claimed = store.claim(
                run_id, f"runner-{worker_serial}", limit=claim_limit, lease_seconds=lease_seconds
            )
            if not claimed:
                break
            session["attempted"] += len(claimed)
            if worker_count == 1:
                if not direct_initialized:
                    _worker_init(adapter_factory, require_cuda and not test_mode)
                    direct_initialized = True
                completed_results = [(item, _analyze(item)) for item in claimed]
            else:
                completed_results: list[tuple[ReprocessItem, WorkerResult]] = []
                if executor is None:
                    executor = ProcessPoolExecutor(
                        max_workers=worker_count,
                        initializer=_worker_init,
                        initargs=(adapter_factory, require_cuda and not test_mode),
                    )
                try:
                    futures: dict[Future[WorkerResult], ReprocessItem] = {
                        executor.submit(_analyze, item): item for item in claimed
                    }
                    for future in as_completed(futures):
                        item = futures[future]
                        try:
                            completed_results.append((item, future.result()))
                        except BrokenProcessPool as error:
                            raise error
                        except BaseException as error:
                            completed_results.append(
                                (item, WorkerResult(item.asset_id, "IO_RETRYABLE",
                                                    error_type=type(error).__name__,
                                                    error_digest=_safe_digest(error)))
                            )
                except BrokenProcessPool as error:
                    completed_results = [
                        (item, WorkerResult(item.asset_id, "WORKER_POOL_BROKEN",
                                           error_type=type(error).__name__,
                                           error_digest=_safe_digest(error)))
                        for item in claimed
                    ]

            downshift = False
            for item, result in completed_results:
                if result.outcome == "COMPLETE":
                    store.complete(
                        run_id, item.asset_id, metadata_evidence=result.metadata or {},
                        features=result.features or {}, lease_token=item.lease_token,
                        embedding_f16=result.embedding_f16, embedding_dim=result.embedding_dim,
                    )
                    session["completed"] += 1
                elif result.outcome == "SOURCE_CHANGED":
                    switched = store.source_changed_or_switch(
                        run_id, item.asset_id, lease_token=item.lease_token
                    )
                    if switched:
                        session["retryable_failed"] += 1
                    else:
                        session["source_changed"] += 1
                elif result.outcome == "DECODE_OR_INVALID":
                    store.fail_terminal(run_id, item.asset_id, _stored_error(result), lease_token=item.lease_token)
                    session["failed"] += 1
                elif result.outcome == "IO_RETRYABLE":
                    if item.attempts < 3:
                        store.fail_retryable(run_id, item.asset_id, _stored_error(result), lease_token=item.lease_token)
                        session["retryable_failed"] += 1
                    else:
                        store.fail_terminal(run_id, item.asset_id, _stored_error(result), lease_token=item.lease_token)
                        session["failed"] += 1
                elif result.outcome == "OOM":
                    previous_error = store.connection.execute(
                        "SELECT last_error FROM run_items WHERE run_id=? AND asset_id=?",
                        (run_id, item.asset_id),
                    ).fetchone()[0]
                    store.fail_retryable(run_id, item.asset_id, _stored_error(result), lease_token=item.lease_token)
                    session["retryable_failed"] += 1
                    if previous_error and str(previous_error).startswith("GpuOutOfMemory:"):
                        paused_reason = "gpu_oom_after_single_worker_retry"
                    else:
                        worker_count = 1
                        capacity = max(1, min(capacity, plan.prefetch_batches))
                        downshift = True
                elif result.outcome == "CUDA_PROVIDER_LOST":
                    store.fail_retryable(run_id, item.asset_id, _stored_error(result), lease_token=item.lease_token)
                    session["retryable_failed"] += 1
                    paused_reason = "cuda_provider_unavailable"
                elif result.outcome == "ADAPTER_INIT_FAILED":
                    store.fail_retryable(run_id, item.asset_id, _stored_error(result), lease_token=item.lease_token)
                    session["retryable_failed"] += 1
                    paused_reason = "adapter_initialization_failed"
                elif result.outcome == "WORKER_POOL_BROKEN":
                    store.fail_retryable(run_id, item.asset_id, _stored_error(result), lease_token=item.lease_token)
                    session["retryable_failed"] += 1
                    paused_reason = "worker_pool_broken"
                else:
                    store.fail_terminal(run_id, item.asset_id, _stored_error(result), lease_token=item.lease_token)
                    session["failed"] += 1
            worker_serial += 1
            if downshift and executor is not None:
                executor.shutdown(wait=True)
                executor = None
                direct_initialized = False
            if paused_reason is not None:
                break
            if dynamic_profile and not test_mode:
                profile = discover_resources()
                paused_reason = _gate_reason(
                    profile, foreground_lock, maintenance_lock, test_mode=False,
                    require_cuda=require_cuda, startup=False
                )
                if paused_reason is not None:
                    break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    current = store.status(run_id)
    if paused_reason is None and current["items"].get("LEASED", 0):
        paused_reason = "active_leases"
    final_status = "PAUSED" if paused_reason is not None else "COMPLETE"
    preliminary = {
        "paused_reason": paused_reason,
        "session": {key: session[key] for key in sorted(session)},
    }
    store.set_run_status(run_id, final_status, statistics=preliminary)
    return _result_statistics(
        store, run_id, profile=profile, workers=worker_count,
        maximum_in_flight=capacity, session=session, paused_reason=paused_reason
    )


def process_reprocess_json(*args: Any, **kwargs: Any) -> str:
    """Return the stable runner result as canonical JSON for CLI integration."""
    return json.dumps(process_reprocess(*args, **kwargs), sort_keys=True, separators=(",", ":"))
