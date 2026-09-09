"""Command-line interface for inspecting curation pipeline state."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Sequence

from .calibration import LabeledVector, calibrate_thresholds
from .checkpoint import CheckpointStore
from .classification import ClassificationConfig, classify_open_set
from .extract import extract_pending
from .inference import CombinedAnalysisAdapter, load_adapter
from .onnx_adapters import OnnxDependencyError, WD14MoatNudeNetAdapter
from .readonly import snapshot_source
from .reprocess_cli import add_reprocess_parser, handle_reprocess
from .resources import discover_resources, plan_as_dict
from .routing import route
from .scan import IMAGE_EXTENSIONS, scan_and_enqueue


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_vector(path: Path) -> list[float]:
    value = _load_json(path)
    if isinstance(value, dict):
        value = value.get("vector")
    if not isinstance(value, list):
        raise ValueError("vector JSON must be a list or an object with a 'vector' list")
    return [float(number) for number in value]


def _load_references(path: Path) -> dict[str, list[list[float]]]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError("references JSON must map labels to vectors")
    references: dict[str, list[list[float]]] = {}
    for label, examples in value.items():
        if not isinstance(label, str) or not isinstance(examples, list):
            raise ValueError("each reference label must map to a vector or a list of vectors")
        if examples and isinstance(examples[0], (int, float)):
            examples = [examples]
        if not all(isinstance(example, list) for example in examples):
            raise ValueError("each reference example must be a numeric vector")
        references[label] = [[float(number) for number in example] for example in examples]
    return references


def _load_labeled_vectors(path: Path) -> list[LabeledVector]:
    value = _load_json(path)
    if not isinstance(value, list):
        raise ValueError("labeled JSON must be a list of {'label': ..., 'vector': [...]} records")
    records: list[LabeledVector] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict) or "label" not in record or "vector" not in record:
            raise ValueError("each labeled record requires explicit 'label' and 'vector' fields")
        label = record["label"]
        vector = record["vector"]
        if label is not None and not isinstance(label, str) or not isinstance(vector, list):
            raise ValueError("label must be a string or null, and vector must be a list")
        records.append(LabeledVector(label, tuple(float(number) for number in vector),
                                     str(record.get("id", index))))
    return records


def _doctor(database: Path | None) -> dict[str, object]:
    optional = {name: importlib.util.find_spec(name) is not None
                for name in ("numpy", "onnxruntime", "torch", "transformers")}
    return {
        "python_optional_dependencies": optional,
        "database": str(database) if database else None,
        "database_exists": database.exists() if database else None,
        "weights_bundled": False,
        "inference": "Provide model weights and an InferenceAdapter explicitly; no model is downloaded by this package.",
        "read_only_sources": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image-curator", description="Read-only image curation pipeline utilities.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="print a conservative plan for detected host resources")
    init_db = commands.add_parser("init-db", help="create a local SQLite checkpoint database")
    init_db.add_argument("database", type=Path)
    status = commands.add_parser("status", help="print checkpoint state counts")
    status.add_argument("database", type=Path)
    verify = commands.add_parser("verify-source", help="fingerprint a source without modifying it")
    verify.add_argument("source", type=Path)
    doctor = commands.add_parser("doctor", help="report optional runtime support without downloading models")
    doctor.add_argument("--database", type=Path)
    scan = commands.add_parser("scan", help="recursively enqueue image files into a local SQLite checkpoint")
    scan.add_argument("database", type=Path)
    scan.add_argument("roots", type=Path, nargs="+")
    scan.add_argument("--extension", action="append", help="image extension to include; repeat to override defaults")
    scan.add_argument("--follow-symlinks", action="store_true", help="follow directory and file symlinks")
    extract = commands.add_parser("extract", help="extract Pillow metadata and optional embeddings for pending assets")
    extract.add_argument("database", type=Path)
    extract.add_argument("--adapter", help="user-provided module:factory InferenceAdapter; no weights are bundled")
    extract.add_argument("--moat-model", type=Path, help="explicit local WD14 MoAT ONNX model path")
    extract.add_argument("--wd14-tags", type=Path, help="explicit local WD14 tag CSV path")
    extract.add_argument("--nudenet-model", type=Path, help="optional explicit local NudeNet ONNX model path")
    extract.add_argument("--provider", action="append",
                         help="ONNX Runtime provider; repeat for fallback order (default: CPUExecutionProvider)")
    extract.add_argument("--limit", type=int, help="maximum pending unique assets to process")
    classify = commands.add_parser("classify", help="open-set classify a vector against user-provided references")
    classify.add_argument("--references", type=Path, required=True, help="JSON mapping labels to vectors")
    classify.add_argument("--vector", type=Path, required=True, help="JSON vector or {'vector': [...]} object")
    classify.add_argument("--min-similarity", type=float, default=0.80)
    classify.add_argument("--min-margin", type=float, default=0.05)
    classify.add_argument("--top-k", type=int, default=3)
    classify.add_argument("--rule-accept", action="store_true", help="route directly through a caller-approved rule")
    classify.add_argument("--rule-reason", default="")
    calibrate = commands.add_parser("calibrate", help="audit open-set thresholds using explicitly labeled JSON")
    calibrate.add_argument("--references", type=Path, required=True, help="JSON mapping labels to reference vectors")
    calibrate.add_argument("--labeled", type=Path, required=True,
                           help="JSON list of explicit label/vector validation records")
    calibrate.add_argument("--similarity", type=float, action="append",
                           help="candidate similarity threshold; repeat as needed")
    calibrate.add_argument("--margin", type=float, action="append", help="candidate margin threshold; repeat as needed")
    calibrate.add_argument("--top-k", type=int, default=3)
    calibrate.add_argument("--target-accepted-accuracy", type=float, default=0.90)
    calibrate.add_argument("--max-unknown-false-accept", type=float, default=0.05)
    calibrate.add_argument("--min-accepted", type=int, default=5)
    calibrate.add_argument("--min-unknown", type=int, default=1)
    add_reprocess_parser(commands)
    return parser


def build_extract_adapter(args: argparse.Namespace):
    """Combine explicit local ONNX and caller-plugin adapters without model discovery."""
    adapters = []
    if args.adapter:
        adapters.append(load_adapter(args.adapter))
    if bool(args.moat_model) != bool(args.wd14_tags):
        raise ValueError("--moat-model and --wd14-tags must be supplied together")
    if args.nudenet_model and not args.moat_model:
        raise ValueError("--nudenet-model requires --moat-model and --wd14-tags")
    if args.moat_model:
        adapters.append(WD14MoatNudeNetAdapter.from_paths(
            args.moat_model, args.wd14_tags, nudenet_model=args.nudenet_model,
            providers=args.provider or ["CPUExecutionProvider"],
        ))
    if not adapters:
        return None
    return adapters[0] if len(adapters) == 1 else CombinedAnalysisAdapter(*adapters)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        _print_json(plan_as_dict(discover_resources()))
    elif args.command == "init-db":
        with CheckpointStore(args.database):
            pass
        _print_json({"database": str(args.database), "created": True})
    elif args.command == "status":
        with CheckpointStore(args.database) as database:
            _print_json({"database": str(args.database), "states": database.status(),
                         **database.inventory_counts()})
    elif args.command == "verify-source":
        snapshot = snapshot_source(args.source)
        _print_json({"path": str(snapshot.path), "size": snapshot.size, "mtime_ns": snapshot.mtime_ns,
                     "sha256": snapshot.sha256, "mutation_allowed": False})
    elif args.command == "doctor":
        _print_json(_doctor(args.database))
    elif args.command == "scan":
        extensions = args.extension or IMAGE_EXTENSIONS
        with CheckpointStore(args.database) as database:
            result = scan_and_enqueue(database, args.roots, extensions=extensions,
                                      follow_symlinks=args.follow_symlinks)
            _print_json({"database": str(args.database), "read_only_sources": True,
                         "follow_symlinks": args.follow_symlinks, **result.as_dict(),
                         "states": database.status(), **database.inventory_counts()})
    elif args.command == "extract":
        try:
            adapter = build_extract_adapter(args)
        except (FileNotFoundError, OnnxDependencyError, TypeError, ValueError) as error:
            parser.error(str(error))
        with CheckpointStore(args.database) as database:
            result = extract_pending(database, adapter=adapter, limit=args.limit)
            _print_json({"database": str(args.database), "adapter": adapter.name if adapter else None,
                         **result.as_dict(), "states": database.status(), **database.inventory_counts()})
    elif args.command == "classify":
        config = ClassificationConfig(args.min_similarity, args.min_margin, args.top_k)
        decision = classify_open_set(_load_vector(args.vector), _load_references(args.references), config=config)
        _print_json({"classification": decision.as_dict(),
                     "route": route(rule_accepted=args.rule_accept, rule_reason=args.rule_reason,
                                    classification=decision).as_dict(),
                     "weights_bundled": False})
    elif args.command == "calibrate":
        report = calibrate_thresholds(_load_labeled_vectors(args.labeled), _load_references(args.references),
                                      similarity_thresholds=args.similarity or [0.75, 0.8, 0.85, 0.9],
                                      margin_thresholds=args.margin or [0.0, 0.05, 0.1], top_k=args.top_k,
                                      min_accepted_accuracy=args.target_accepted_accuracy,
                                      max_unknown_false_accept_rate=args.max_unknown_false_accept,
                                      min_accepted=args.min_accepted, min_unknown=args.min_unknown)
        _print_json(report)
    elif args.command == "reprocess":
        try:
            _print_json(handle_reprocess(args))
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
            parser.error(str(error))
    return 0
