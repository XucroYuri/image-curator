"""CLI for the explicitly opt-in, journaled NAS layout migration."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from .migration import (MigrationRow, MigrationStore, destination_for, execute_row,
                        inherit_nested_context, ready_directories, source_directory, validate_roots)
from .reprocess import is_excluded_path, sha256_file
from .scan import IMAGE_EXTENSIONS


def add_migrate_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("migrate", help="journaled, same-share migration into the v3 layout")
    actions = parser.add_subparsers(dest="migrate_command", required=True)
    create = actions.add_parser("create", help="freeze a complete CSV into a local migration journal")
    create.add_argument("database", type=Path)
    create.add_argument("--run-id", required=True)
    create.add_argument("--baseline", type=Path, required=True)
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--target", type=Path, required=True)
    create.add_argument("--feature-db", type=Path,
                        help="completed feature-run SQLite; only settled directories are planned")
    create.add_argument("--feature-run-id", help="feature run whose terminal directories may migrate")
    create.add_argument("--directory", action="append",
                        help="restrict to a source top-level directory; repeatable")
    sync = actions.add_parser("sync", help="append directories that have reached a terminal feature state")
    sync.add_argument("database", type=Path); sync.add_argument("--run-id", required=True)
    sync.add_argument("--baseline", type=Path, required=True)
    sync.add_argument("--root", type=Path, required=True); sync.add_argument("--target", type=Path, required=True)
    sync.add_argument("--feature-db", type=Path, required=True); sync.add_argument("--feature-run-id", required=True)
    sync.add_argument("--directory", action="append")
    status = actions.add_parser("status", help="show journal state counts")
    status.add_argument("database", type=Path); status.add_argument("--run-id", required=True)
    apply = actions.add_parser("apply", help="rename planned files after identity verification")
    apply.add_argument("database", type=Path); apply.add_argument("--run-id", required=True)
    apply.add_argument("--execute", action="store_true", help="required: permits NAS rename operations")
    apply.add_argument("--limit", type=int)


def _field(row: dict[str, str], *keys: str) -> str:
    lookup = {str(key).casefold(): (value or "").strip() for key, value in row.items()}
    return next((lookup.get(key.casefold(), "") for key in keys if lookup.get(key.casefold(), "")), "")


def _planned_rows(baseline: Path, root: Path, target: Path, *, directories: set[str] | None = None,
                  processed_sources: set[str] | None = None) -> list[MigrationRow]:
    rows: list[MigrationRow] = []
    seen_sources: set[str] = set()
    seen_targets: set[str] = set()
    with baseline.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames: raise ValueError("baseline CSV requires a header")
        for raw in reader:
            text = _field(raw, "source", "source_path", "path")
            if not text or is_excluded_path(text): continue
            source = Path(text)
            try: source.relative_to(root)
            except ValueError: raise ValueError("baseline path is outside authorized root")
            if source.is_symlink(): raise ValueError(f"symbolic link is forbidden: {source}")
            if directories is not None and source_directory(source, root) not in directories: continue
            if processed_sources is not None and source.suffix.casefold() in IMAGE_EXTENSIONS and str(source).casefold() not in processed_sources:
                continue
            if not source.is_file(): continue  # missing historical paths remain in reprocess audit
            source_key = str(source).casefold()
            if source_key in seen_sources: continue
            info = source.stat(); digest = sha256_file(source)
            entity, subject, engine = inherit_nested_context(
                source, root, entity=_field(raw, "entity", "ip", "project"),
                subject=_field(raw, "subject", "character", "name"),
                engine=_field(raw, "source_engine", "engine", "model"))
            destination, state = destination_for(
                source, root=root, target_root=target, entity=entity,
                subject=subject, engine=engine,
                old_bucket=_field(raw, "bucket", "old_bucket", "status"),
                audit_locked=_field(raw, "bucket", "old_bucket", "status").casefold() in {"已发帖", "posted", "published"},
                sha256=digest, mtime_ns=info.st_mtime_ns)
            # Identical stem/engine sources may collide.  Retain both safely.
            destination_key = str(destination).casefold()
            if destination_key in seen_targets:
                destination = destination.with_name(f"{destination.stem}__{digest[:12]}{destination.suffix}")
                destination_key = str(destination).casefold()
            if destination_key in seen_targets: raise ValueError(f"unresolvable target collision: {destination}")
            seen_sources.add(source_key); seen_targets.add(destination_key)
            rows.append(MigrationRow(source, destination, digest, info.st_size, info.st_mtime_ns, state))
    return rows


def handle_migrate(args: argparse.Namespace) -> dict[str, Any]:
    if args.migrate_command == "create":
        root, target = validate_roots(args.root, args.target)
        directories = None
        processed_sources = None
        feature_db = getattr(args, "feature_db", None)
        feature_run_id = getattr(args, "feature_run_id", None)
        if feature_db:
            if not feature_run_id:
                raise ValueError("--feature-run-id is required with --feature-db")
            directories = ready_directories(feature_db, feature_run_id, root)
            with sqlite3.connect(feature_db) as connection:
                processed_sources = {str(row[0]).casefold() for row in connection.execute(
                    "SELECT source_path FROM run_occurrences WHERE run_id=? AND occurrence_state='PROCESSED'",
                    (feature_run_id,))}
        requested_directories = getattr(args, "directory", None)
        if requested_directories:
            directories = (set(requested_directories) if directories is None
                           else directories & set(requested_directories))
        planned = _planned_rows(args.baseline, root, target, directories=directories,
                                processed_sources=processed_sources)
        with MigrationStore(args.database) as store:
            count = store.create(args.run_id, root, target, planned)
            return {"run_id": args.run_id, "planned": count, "eligible_directories": sorted(directories) if directories is not None else None,
                    "database": str(args.database),
                    "target": str(target), "source_mutated": False}
    if args.migrate_command == "sync":
        root, target = validate_roots(args.root, args.target)
        directories = ready_directories(args.feature_db, args.feature_run_id, root)
        with sqlite3.connect(args.feature_db) as connection:
            processed_sources = {str(row[0]).casefold() for row in connection.execute(
                "SELECT source_path FROM run_occurrences WHERE run_id=? AND occurrence_state='PROCESSED'",
                (args.feature_run_id,))}
        if args.directory:
            directories &= set(args.directory)
        planned = _planned_rows(args.baseline, root, target, directories=directories,
                                processed_sources=processed_sources)
        with MigrationStore(args.database) as store:
            count = store.add_rows(args.run_id, planned)
            return {"run_id": args.run_id, "appended": count,
                    "eligible_directories": sorted(directories), "states": store.summary(args.run_id),
                    "source_mutated": False}
    with MigrationStore(args.database) as store:
        if args.migrate_command == "status":
            return {"run_id": args.run_id, "states": store.summary(args.run_id), "database": str(args.database)}
        if args.migrate_command == "apply":
            if not args.execute: raise ValueError("--execute is required before any NAS rename")
            rows = [row for row in store.rows(args.run_id) if row["state"] in {"PLANNED", "FAILED"}]
            if args.limit is not None: rows = rows[:args.limit]
            results: dict[str, int] = {}
            for row in rows:
                state = execute_row(store, args.run_id, row)
                results[state] = results.get(state, 0) + 1
            return {"run_id": args.run_id, "attempted": len(rows), "results": results,
                    "states": store.summary(args.run_id), "source_mutated": bool(rows)}
    raise ValueError(f"unsupported migrate command: {args.migrate_command}")
