"""CLI orchestration for read-only, versioned historical reprocessing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .calibration import calibrate_thresholds
from .classification import ClassificationConfig, classify_open_set
from .embeddings import moat_vector_from_blob
from .readonly import SourceSnapshot
from .reprocess import (
    BaselineOccurrence,
    FrozenOccurrence,
    ProposedDecision,
    build_diff_rows,
    build_machine_outputs,
    build_model_manifest,
    config_fingerprint,
    freeze_occurrence,
    is_excluded_path,
    load_migration_csv,
    sha256_file,
)
from .reprocess_store import ReprocessStore
from .resources import discover_resources
from .scan import IMAGE_EXTENSIONS


def add_reprocess_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("reprocess", help="versioned, read-only historical reprocessing")
    actions = parser.add_subparsers(dest="reprocess_command", required=True)

    create = actions.add_parser("create", help="freeze a legacy CSV baseline into a new analysis run")
    create.add_argument("database", type=Path)
    create.add_argument("--run-id", required=True)
    create.add_argument("--baseline", type=Path, required=True)
    create.add_argument("--root", type=Path, required=True,
                        help="authorized image root and added-path inventory scope")
    create.add_argument("--expected-images", type=int)
    create.add_argument("--cutoff", help="ISO-8601 cutoff; defaults to current UTC time")
    create.add_argument("--stable-seconds", type=int, default=120)
    create.add_argument("--freeze-workers", type=int, choices=(1, 2, 3, 4), default=4)
    _add_model_arguments(create)

    process = actions.add_parser("process", help="process or resume one immutable analysis run")
    process.add_argument("database", type=Path)
    process.add_argument("--run-id", required=True)
    process.add_argument("--workers", type=int, choices=(1, 2))
    process.add_argument("--foreground-lock", type=Path)
    process.add_argument("--maintenance-lock", type=Path)
    process.add_argument("--cuda-dll-dir", type=Path, action="append", default=[])
    process.add_argument("--lease-seconds", type=int, default=300)
    process.add_argument("--limit", type=int)
    _add_model_arguments(process)

    status = actions.add_parser("status", help="print one versioned run's state")
    status.add_argument("database", type=Path)
    status.add_argument("--run-id", required=True)

    decide = actions.add_parser("decide", help="create a calibrated identity decision version")
    decide.add_argument("database", type=Path)
    decide.add_argument("--run-id", required=True)
    decide.add_argument("--decision-id", required=True)
    decide.add_argument("--references", type=Path, required=True)
    decide.add_argument("--labeled", type=Path, required=True)
    decide.add_argument("--similarity", type=float, action="append")
    decide.add_argument("--margin", type=float, action="append")
    decide.add_argument("--top-k", type=int, default=3)
    decide.add_argument("--target-accepted-accuracy", type=float, default=0.95)
    decide.add_argument("--max-unknown-false-accept", type=float, default=0.02)
    decide.add_argument("--min-accepted", type=int, default=20)
    decide.add_argument("--min-unknown", type=int, default=20)

    diff = actions.add_parser("diff", help="write path-level JSON/CSV audit and review queue")
    diff.add_argument("database", type=Path)
    diff.add_argument("--run-id", required=True)
    diff.add_argument("--decision-id")
    diff.add_argument("--output", type=Path, required=True)
    diff.add_argument("--review-limit", type=int, default=500)

    prune = actions.add_parser("prune", help="remove heavy derived payloads while retaining manifests")
    prune.add_argument("database", type=Path)
    prune.add_argument("--run-id", required=True)
    prune.add_argument("--dry-run", action="store_true")


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--moat-model", type=Path, required=True)
    parser.add_argument("--wd14-tags", type=Path, required=True)
    parser.add_argument("--nudenet-model", type=Path)
    parser.add_argument("--provider", action="append", required=True)


def _iso(value: str | None) -> str:
    moment = datetime.now(UTC) if value is None else datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _manifest_for(args: argparse.Namespace) -> dict[str, Any]:
    return build_model_manifest(args.moat_model, args.wd14_tags, args.nudenet_model, args.provider)


def _existing_run(store: ReprocessStore, run_id: str) -> dict[str, Any] | None:
    row = store.connection.execute("SELECT * FROM analysis_runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def _validate_authorized_paths(root: Path, baseline: list[BaselineOccurrence]) -> Path:
    lexical_root = Path(os.path.abspath(root))
    resolved_root = lexical_root.resolve(strict=True)
    root_key = os.path.normcase(str(lexical_root))
    resolved_root_key = os.path.normcase(str(resolved_root))
    for occurrence in baseline:
        lexical = Path(os.path.abspath(occurrence.source_path))
        try:
            if os.path.commonpath((root_key, os.path.normcase(str(lexical)))) != root_key:
                raise ValueError("baseline contains a path outside the authorized root")
        except ValueError as error:
            raise ValueError("baseline contains a path outside the authorized root") from error
        relative = lexical.relative_to(lexical_root)
        cursor = lexical_root
        for component in relative.parts[:-1]:
            cursor /= component
            is_junction = getattr(os.path, "isjunction", lambda _: False)(cursor)
            if cursor.is_symlink() or is_junction:
                raise ValueError("baseline path crosses a symlink or junction")
        resolved = lexical.resolve(strict=False)
        try:
            if os.path.commonpath((resolved_root_key, os.path.normcase(str(resolved)))) != resolved_root_key:
                raise ValueError("baseline resolves outside the authorized root")
        except ValueError as error:
            raise ValueError("baseline resolves outside the authorized root") from error
    return lexical_root


def _create(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_migration_csv(args.baseline)
    root = _validate_authorized_paths(args.root, baseline)
    normalized_paths = [os.path.normcase(os.path.abspath(item.source_path)) for item in baseline]
    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError("baseline contains duplicate image paths")
    if args.expected_images is not None and len(baseline) != args.expected_images:
        raise ValueError(f"baseline image count differs: expected {args.expected_images}, got {len(baseline)}")
    manifest = _manifest_for(args)
    configuration = {
        "baseline_sha256": sha256_file(args.baseline),
        "baseline_images": len(baseline),
        "stable_seconds": args.stable_seconds,
        "model_manifest": manifest,
        "read_only": True,
    }
    fingerprint = config_fingerprint(configuration)
    with ReprocessStore(args.database) as store:
        existing = _existing_run(store, args.run_id)
        cutoff = existing["cutoff_at"] if existing else _iso(args.cutoff)
        if existing:
            existing_manifest = json.loads(existing["model_manifest_json"])
            legacy_manifest = {key: manifest[key] for key in ("adapter", "models", "providers")}
            if existing_manifest == legacy_manifest and existing["code_version"] == __version__:
                store.upgrade_unstarted_run_identity(
                    args.run_id, config_fingerprint=fingerprint, model_manifest=manifest
                )
                existing = _existing_run(store, args.run_id)
            if existing["config_fingerprint"] != fingerprint or existing_manifest not in (manifest, legacy_manifest):
                raise ValueError("run ID already exists with a different immutable configuration")
        else:
            store.create_run(
                args.run_id, cutoff_at=cutoff, config_fingerprint=fingerprint,
                code_version=__version__, model_manifest=manifest,
                resource_snapshot=asdict(discover_resources()),
            )
        known = {str(row[0]).casefold() for row in store.connection.execute(
            "SELECT source_path FROM run_occurrences WHERE run_id=?", (args.run_id,)
        )}
        prior_states = {str(row[0]).casefold(): int(row[1]) for row in store.connection.execute(
            "SELECT occurrence_state,count(*) FROM run_occurrences WHERE run_id=? GROUP BY occurrence_state",
            (args.run_id,),
        )}
        counts = {
            "baseline": len(baseline),
            "frozen": prior_states.get("frozen", 0) + prior_states.get("processed", 0),
            "missing": prior_states.get("missing", 0),
            "source_changed": prior_states.get("source_changed", 0),
            "added_deferred": prior_states.get("added_deferred", 0),
            "resumed": len(known),
        }
        cutoff_value = datetime.fromisoformat(cutoff)
        pending = [item for item in baseline if item.source_path.casefold() not in known]

        def freeze(item: BaselineOccurrence) -> FrozenOccurrence:
            return freeze_occurrence(item, cutoff=cutoff_value, stable_seconds=args.stable_seconds)

        with ThreadPoolExecutor(max_workers=args.freeze_workers,
                                thread_name_prefix="baseline-freeze") as executor:
            frozen_rows = executor.map(freeze, pending)
            for completed, frozen in enumerate(frozen_rows, 1):
                occurrence = frozen.occurrence
                if frozen.sha256 and frozen.status in {"processed", "baseline_identity_unverifiable"}:
                    source = SourceSnapshot(Path(occurrence.source_path), int(frozen.size),
                                            int(frozen.mtime_ns), frozen.sha256)
                    store.add_asset_occurrence(
                        args.run_id, frozen.sha256, source, old_bucket=occurrence.old_bucket,
                        audit_locked=occurrence.audit_locked,
                        baseline_identity_unverifiable=occurrence.baseline_identity_unverifiable,
                    )
                    counts["frozen"] += 1
                else:
                    state = {
                        "missing": "MISSING", "source_changed": "SOURCE_CHANGED",
                        "added_deferred": "ADDED_DEFERRED",
                    }.get(frozen.status, "SOURCE_CHANGED")
                    store.add_baseline_occurrence(
                        args.run_id, Path(occurrence.source_path), old_bucket=occurrence.old_bucket,
                        audit_locked=occurrence.audit_locked, occurrence_state=state,
                        baseline_identity_unverifiable=occurrence.baseline_identity_unverifiable,
                    )
                    counts[state.casefold()] += 1
                visited = len(known) + completed
                if visited % 250 == 0:
                    print(json.dumps({"event": "freeze_progress", "run_id": args.run_id,
                                      "visited": visited, **counts}, ensure_ascii=False),
                          file=sys.stderr, flush=True)
        _add_deferred_inventory(store, args.run_id, root, baseline, known, counts)
        status = store.status(args.run_id)
        store.set_run_status(args.run_id, "CREATED", statistics=counts)
        return {"read_only_sources": True, "model_paths_persisted": False,
                "status": status | {"statistics": counts}, **store.integrity()}


def _add_deferred_inventory(store: ReprocessStore, run_id: str, root: Path,
                            baseline: list[BaselineOccurrence], known: set[str],
                            counts: dict[str, int]) -> None:
    baseline_paths = {item.source_path.casefold() for item in baseline}
    for base, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories[:] = [name for name in directories if not is_excluded_path(Path(base) / name)]
        for name in files:
            path = Path(base) / name
            key = str(path).casefold()
            if path.suffix.casefold() not in IMAGE_EXTENSIONS or key in baseline_paths or key in known:
                continue
            if is_excluded_path(path):
                continue
            store.add_baseline_occurrence(
                run_id, path, old_bucket=None, audit_locked=False,
                occurrence_state="ADDED_DEFERRED",
            )
            counts["added_deferred"] += 1


def _load_references(path: Path) -> dict[str, list[list[float]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("references must map labels to vectors")
    result = {}
    for label, examples in value.items():
        if examples and isinstance(examples[0], (int, float)):
            examples = [examples]
        result[str(label)] = [[float(number) for number in example] for example in examples]
    return result


def _load_labeled(path: Path):
    from .calibration import LabeledVector
    value = json.loads(path.read_text(encoding="utf-8"))
    return [LabeledVector(record.get("label"), tuple(float(number) for number in record["vector"]),
                          str(record.get("id", index))) for index, record in enumerate(value)]


def _decide(args: argparse.Namespace) -> dict[str, Any]:
    references = _load_references(args.references)
    labeled = _load_labeled(args.labeled)
    report = calibrate_thresholds(
        labeled, references, similarity_thresholds=args.similarity or [0.75, 0.8, 0.85, 0.9],
        margin_thresholds=args.margin or [0.0, 0.05, 0.1], top_k=args.top_k,
        min_accepted_accuracy=args.target_accepted_accuracy,
        max_unknown_false_accept_rate=args.max_unknown_false_accept,
        min_accepted=args.min_accepted, min_unknown=args.min_unknown,
    )
    thresholds = report["recommended"]
    decision_config = {
        "calibration": report["target_gates"], "top_k": args.top_k,
        "similarity_grid": args.similarity or [0.75, 0.8, 0.85, 0.9],
        "margin_grid": args.margin or [0.0, 0.05, 0.1],
        "thresholds": thresholds,
    }
    decision_fingerprint = config_fingerprint(decision_config)
    reference_fingerprint = sha256_file(args.references)
    labels_fingerprint = sha256_file(args.labeled)
    with ReprocessStore(args.database) as store:
        feature_status = store.status(args.run_id)
        if feature_status["status"] != "COMPLETE":
            raise ValueError("feature run must be COMPLETE before creating decisions")
        created = store.create_decision_run(
            args.decision_id, args.run_id, config_fingerprint=decision_fingerprint,
            reference_fingerprint=reference_fingerprint,
            human_labels_fingerprint=labels_fingerprint, thresholds=thresholds or {},
        )
        if not created:
            existing = store.connection.execute(
                "SELECT * FROM decision_runs WHERE decision_id=?", (args.decision_id,)
            ).fetchone()
            matches = (
                existing is not None and existing["run_id"] == args.run_id
                and existing["config_fingerprint"] == decision_fingerprint
                and existing["reference_fingerprint"] == reference_fingerprint
                and existing["human_labels_fingerprint"] == labels_fingerprint
            )
            if not matches or existing["status"] not in {"CREATED", "COMPLETE"}:
                raise ValueError("decision ID already exists with incompatible inputs or state")
            if existing["status"] == "COMPLETE":
                result_count = store.connection.execute(
                    "SELECT count(*) FROM decision_results WHERE decision_id=?", (args.decision_id,)
                ).fetchone()[0]
                return {"decision_id": args.decision_id, "status": "COMPLETE",
                        "results": int(result_count), "calibration": report, "resumed": True}
        if thresholds is None:
            store.connection.execute("UPDATE decision_runs SET status='FAILED' WHERE decision_id=?",
                                     (args.decision_id,))
            store.connection.commit()
            return {"decision_id": args.decision_id, "status": "FAILED", "calibration": report}
        config = ClassificationConfig(float(thresholds["min_similarity"]),
                                      float(thresholds["min_margin"]), args.top_k)
        rows = store.connection.execute(
            "SELECT asset_id,embedding_f16,embedding_dim FROM run_items "
            "WHERE run_id=? AND state='COMPLETE' AND embedding_f16 IS NOT NULL ORDER BY asset_id",
            (args.run_id,),
        ).fetchall()
        complete_count = int(store.connection.execute(
            "SELECT count(*) FROM run_items WHERE run_id=? AND state='COMPLETE'", (args.run_id,)
        ).fetchone()[0])
        if len(rows) != complete_count:
            raise ValueError("feature payloads were pruned or are incomplete; decision run cannot continue")
        for row in rows:
            vector = moat_vector_from_blob(row["embedding_f16"], dimensions=int(row["embedding_dim"]))
            decision = classify_open_set(vector, references, config=config)
            store.record_decision(
                args.decision_id, row["asset_id"], proposed_bucket="待复核",
                candidate_identity=decision.label, confidence=decision.similarity,
                decision_state="ACCEPTED" if decision.state == "accepted" else "UNKNOWN",
                reasons={"reasons": ["status_policy_not_calibrated"] if decision.label else ["identity_unknown"],
                         "margin": decision.margin},
            )
        store.connection.execute("UPDATE decision_runs SET status='COMPLETE' WHERE decision_id=?",
                                 (args.decision_id,))
        store.connection.commit()
        return {"decision_id": args.decision_id, "status": "COMPLETE",
                "results": len(rows), "calibration": report, "resumed": not created}


def _diff(args: argparse.Namespace) -> dict[str, Any]:
    with ReprocessStore(args.database) as store:
        run = _existing_run(store, args.run_id)
        if run is None:
            raise KeyError(f"unknown run_id: {args.run_id}")
        result_rows = {}
        decision_fingerprint = None
        if args.decision_id:
            decision_run = store.connection.execute(
                "SELECT * FROM decision_runs WHERE decision_id=? AND run_id=?",
                (args.decision_id, args.run_id),
            ).fetchone()
            if decision_run is None:
                raise KeyError(f"unknown decision_id for run: {args.decision_id}")
            decision_fingerprint = decision_run["config_fingerprint"]
            result_rows = {row["asset_id"]: row for row in store.connection.execute(
                "SELECT * FROM decision_results WHERE decision_id=?", (args.decision_id,)
            )}
        frozen = []
        decisions = {}
        evidence_by_asset: dict[str, dict[str, Any]] = {}
        rows = store.connection.execute(
            "SELECT o.*,i.state AS item_state,i.features_json FROM run_occurrences AS o LEFT JOIN run_items AS i "
            "ON i.run_id=o.run_id AND i.asset_id=o.asset_id WHERE o.run_id=? ORDER BY o.source_path",
            (args.run_id,),
        ).fetchall()
        for row in rows:
            occurrence = BaselineOccurrence(
                row["source_path"], row["old_bucket"] or "待复核",
                baseline_sha256=row["source_sha256"], audit_locked=bool(row["audit_locked"]),
                baseline_identity_unverifiable=bool(row["baseline_identity_unverifiable"]),
            )
            occurrence_state = row["occurrence_state"]
            item_state = row["item_state"]
            if occurrence_state == "PROCESSED" or item_state == "COMPLETE":
                status = "baseline_identity_unverifiable" if row["baseline_identity_unverifiable"] else "processed"
            elif item_state == "FAILED":
                status = "error"
            elif item_state in {"PENDING", "LEASED", "RETRYABLE_FAILED"}:
                status = item_state.casefold()
            else:
                status = occurrence_state.casefold()
            frozen.append(FrozenOccurrence(
                occurrence, status, row["source_size"], row["source_mtime_ns"], row["source_sha256"]
            ))
            decision_row = result_rows.get(row["asset_id"])
            if decision_row:
                reasons = json.loads(decision_row["reasons_json"])
                decisions[row["asset_id"]] = ProposedDecision(
                    row["asset_id"], decision_row["proposed_bucket"], decision_row["candidate_identity"],
                    decision_row["confidence"], tuple(reasons.get("reasons", [])),
                    run["config_fingerprint"], decision_fingerprint,
                )
            if row["asset_id"] and row["features_json"]:
                evidence_by_asset[row["asset_id"]] = _compact_feature_evidence(
                    json.loads(row["features_json"])
                )
        diff_rows = build_diff_rows(frozen, decisions)
        for row in diff_rows:
            evidence = evidence_by_asset.get(row["asset_id"], {})
            row["evidence"] = evidence
            strata = ["published_lock"] if row["audit_locked"] else []
            if (evidence.get("nudenet_explicit_score") or 0) > 0 or (evidence.get("rating_explicit") or 0) >= 0.1:
                strata.append("safety_signal")
            if (evidence.get("rating_sensitive") or 0) >= 0.2 or (evidence.get("rating_questionable") or 0) >= 0.1:
                strata.append("safety_boundary")
            if evidence.get("quality_boundary"):
                strata.append("quality_boundary")
            if row["processing_state"] == "processed" and not row["identity"]:
                strata.append("unknown_identity")
                if not row["audit_locked"]:
                    row["proposed_bucket"] = "待复核"
                    row["effective_bucket"] = "待复核"
                    row["review_reasons"] = sorted(set(row["review_reasons"] + [
                        "identity_reference_decision_pending", "status_policy_not_calibrated"
                    ]))
            row["review_strata"] = strata or ["source_outcome"]
        outputs = build_machine_outputs(diff_rows, review_limit=args.review_limit,
                                        review_seed=args.run_id)
        path_lookup = {row["occurrence_id"]: row["source_path"] for row in diff_rows}
        for row in outputs["review_rows"]:
            row["source_path"] = path_lookup.get(row["occurrence_id"])
        outputs["summary"].update({"run_id": args.run_id, "decision_id": args.decision_id,
                                   "read_only_sources": True, **store.integrity()})
    _write_outputs(args.output, outputs)
    return outputs["summary"] | {"output": str(args.output)}


def _compact_feature_evidence(features: dict[str, Any]) -> dict[str, Any]:
    analysis = features.get("analysis", {})
    wd14 = analysis.get("wd14_moat", {})
    nudenet = analysis.get("nudenet", {})
    technical = features.get("technical", {})
    quality_boundary = (
        technical.get("entropy_bits", 9) < 2
        or technical.get("dark_clip_ratio", 0) > 0.1
        or technical.get("bright_clip_ratio", 0) > 0.1
    )
    return {
        "rating_general": wd14.get("rating_general"),
        "rating_sensitive": wd14.get("rating_sensitive"),
        "rating_questionable": wd14.get("rating_questionable"),
        "rating_explicit": wd14.get("rating_explicit"),
        "top_tags": wd14.get("top_tags", [])[:8],
        "nudenet_explicit_score": nudenet.get("explicit_score", 0.0),
        "nudenet_intimate_covered_score": nudenet.get("intimate_covered_score", 0.0),
        "technical": technical,
        "quality_boundary": quality_boundary,
    }


def _write_outputs(directory: Path, outputs: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(directory / "reprocess-summary.json", outputs["summary"])
    _write_json_atomic(directory / "review-queue.json", outputs["review_rows"])
    fields = sorted({key for row in outputs["diff_rows"] for key in row})
    temporary = directory / "reprocess-diff.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in outputs["diff_rows"]:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(directory / "reprocess-diff.csv")


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _prune(args: argparse.Namespace) -> dict[str, Any]:
    with ReprocessStore(args.database) as store:
        unfinished = int(store.connection.execute(
            "SELECT count(*) FROM decision_runs WHERE run_id=? AND status='CREATED'", (args.run_id,)
        ).fetchone()[0])
        if unfinished and not args.dry_run:
            raise ValueError("cannot prune while a decision run is unfinished")
        row = store.connection.execute(
            "SELECT count(*) AS items,coalesce(sum(length(embedding_f16)),0)"
            "+coalesce(sum(length(features_json)),0)+coalesce(sum(length(metadata_evidence_json)),0) AS bytes "
            "FROM run_items WHERE run_id=? AND (embedding_f16 IS NOT NULL OR features_json!='{}')",
            (args.run_id,),
        ).fetchone()
        result = {"run_id": args.run_id, "items": int(row["items"]), "derived_bytes": int(row["bytes"]),
                  "dry_run": bool(args.dry_run)}
        if not args.dry_run:
            store.connection.execute(
                "UPDATE run_items SET metadata_evidence_json='{}',features_json='{}',embedding_f16=NULL,embedding_dim=NULL "
                "WHERE run_id=?", (args.run_id,),
            )
            store.connection.commit()
        return result


def handle_reprocess(args: argparse.Namespace) -> dict[str, Any]:
    if args.reprocess_command == "create":
        return _create(args)
    if args.reprocess_command == "status":
        with ReprocessStore(args.database) as store:
            return store.status(args.run_id) | store.integrity()
    if args.reprocess_command == "decide":
        return _decide(args)
    if args.reprocess_command == "diff":
        return _diff(args)
    if args.reprocess_command == "prune":
        return _prune(args)
    if args.reprocess_command == "process":
        from .reprocess_runner import OnnxAdapterFactory, process_reprocess

        manifest = _manifest_for(args)
        with ReprocessStore(args.database) as store:
            run = _existing_run(store, args.run_id)
            if (run is None or json.loads(run["model_manifest_json"]) != manifest
                    or run["code_version"] != __version__):
                raise ValueError("local model artifacts do not match the immutable run manifest")
            factory = OnnxAdapterFactory(args.moat_model, args.wd14_tags, args.nudenet_model,
                                         tuple(args.provider), tuple(args.cuda_dll_dir), manifest)
            return process_reprocess(
                store, args.run_id, adapter_factory=factory, workers=args.workers,
                foreground_lock=args.foreground_lock, maintenance_lock=args.maintenance_lock,
                lease_seconds=args.lease_seconds,
                limit=args.limit, require_cuda=args.provider[0] == "CUDAExecutionProvider",
            )
    raise ValueError(f"unsupported reprocess command: {args.reprocess_command}")
