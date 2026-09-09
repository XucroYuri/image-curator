# image-curator

[English](README.md) · [简体中文](README.zh-CN.md)

Resource-aware, resumable, metadata-first curation for large and messy image libraries. `image-curator` keeps safety, quality, identity, and publishing value separate while making dirty data, missing metadata, and model uncertainty explicit.

The project is currently Alpha. Interfaces, checkpoint schemas, and example configuration may change. Start with a copy or a small test library, and retain the configuration and audit output for each run.

## Quick start

Python 3.11 or newer is required. The commands below create empty example directories and do not depend on a real image library.

```bash
git clone https://github.com/XucroYuri/image-curator.git
cd image-curator
python -m venv .venv
python -m pip install -e ".[dev]"
python -c "from pathlib import Path; Path('sample-library').mkdir(exist_ok=True); Path('curation-output').mkdir(exist_ok=True)"
python -m image_curator plan
python -m image_curator doctor --database ./curation-output/checkpoint.sqlite
python -m image_curator init-db ./curation-output/checkpoint.sqlite
python -m image_curator scan ./curation-output/checkpoint.sqlite ./sample-library
python -m image_curator extract ./curation-output/checkpoint.sqlite --limit 100
python -m image_curator status ./curation-output/checkpoint.sqlite
```

Pass your own image-library path as the final argument to `scan`. Scanning, hashing, metadata reads, and checkpoint writes do not modify source files; keep the output database in a separate directory.

`extract --limit` supports batches and checkpoint resume. Without an adapter, it still validates images and extracts dimensions, format, mode, and redacted metadata evidence. `verify-source` fingerprints one file, `classify` performs open-set decisions with caller-supplied vectors and references, and `calibrate` evaluates thresholds against explicit labels.

```bash
python -m image_curator verify-source ./sample-library/photo.jpg
python -m image_curator --help
```

## Current scope

- Resource discovery and conservative resource planning.
- Read-only scanning, duplicate occurrence tracking, content-hash checkpoints, and resumable extraction.
- Bounded technical-quality signals and metadata evidence summaries.
- Caller-supplied WD14 MoAT/NudeNet ONNX analysis.
- Versioned historical reprocessing with immutable model/configuration fingerprints and path-level audit locks.
- Open-set reference classification, threshold calibration, and routing primitives.
- A runner-neutral policy template in `configs/readonly.example.yaml`.

The current CLI uses explicit arguments and does not yet orchestrate a complete end-to-end YAML run. The `reprocess` commands orchestrate a frozen historical baseline, local feature extraction, calibrated identity decisions, and read-only diff reports. They never move, rename, delete, or publish files, and do not treat model output as ground truth. Human review and audit records are required before publication.

This repository does not distribute or download MoAT, NudeNet, SigLIP, or VLM weights. Users must obtain models and matching WD14 tag CSV files independently and verify model, tag, and service licenses.

## Versioned reprocessing

`reprocess create` freezes image rows from a legacy migration CSV into a new run. Sidecar rows are ignored for inference, identical image content is processed once, and every observed path remains a separate audit occurrence. Paths modified after the cutoff, missing paths, and newly discovered paths receive explicit states instead of being guessed or silently dropped.

```bash
python -m image_curator reprocess create ./curation-output/reprocess.sqlite \
  --run-id legacy-2026-09 \
  --baseline ./legacy-migration.csv \
  --root ./sample-library \
  --expected-images 100 \
  --freeze-workers 4 \
  --moat-model /path/to/model.onnx \
  --wd14-tags /path/to/selected_tags.csv \
  --nudenet-model /path/to/nudenet.onnx \
  --provider CUDAExecutionProvider \
  --cuda-dll-dir /path/to/cuda/bin \
  --cuda-dll-dir /path/to/cudnn/bin

python -m image_curator reprocess process ./curation-output/reprocess.sqlite \
  --run-id legacy-2026-09 \
  --moat-model /path/to/model.onnx \
  --wd14-tags /path/to/selected_tags.csv \
  --nudenet-model /path/to/nudenet.onnx \
  --provider CUDAExecutionProvider

python -m image_curator reprocess status ./curation-output/reprocess.sqlite --run-id legacy-2026-09
python -m image_curator reprocess diff ./curation-output/reprocess.sqlite \
  --run-id legacy-2026-09 --output ./curation-output/audit
```

`--root` is a mandatory authorization boundary; out-of-root and duplicate CSV paths are rejected before any image is read. The process command verifies model, tag, preprocessing, and postprocessing fingerprints against the immutable run manifest, again before and after each worker loads them. On Windows, repeat `--cuda-dll-dir` when pip-installed CUDA/cuDNN DLLs are outside the default search path; these local paths are never persisted. It uses up to two isolated inference workers (CUDA workers when that provider is selected) with one parent SQLite writer and resumes expired leases safely. A path whose old bucket is `published` remains locked at that effective bucket; new scores are audit evidence only. `reprocess decide` requires explicit reference vectors and human-labelled validation vectors. If the configured accuracy and unknown false-accept gates cannot be met, it records a failed decision version rather than recommending thresholds.

## Principles

- **Read-only by default:** scanning, hashing, metadata extraction, and inference do not move, rename, or delete files. Derived indexes go to a separate output directory.
- **Resumable:** each asset has a stable identity and checkpoint state; completed work is not repeated after interruption.
- **Metadata first:** filesystem and embedded metadata are evidence; raw values, normalized values, and parse errors are kept separate.
- **Staged inference:** MoAT, SigLIP, and VLM stages are ordered by cost and confidence. A cheap screening stage is not a semantic verdict.
- **Open-set `unknown`:** uncertain, uncovered, or conflicting samples remain `unknown` instead of being forced into a known class.
- **Separate value axes:** `safety`, `quality`, `identity`, and `publish_value` are independent fields and decision axes.

## Configuration

`configs/readonly.example.yaml` is a copyable baseline. Every path is a replaceable example and represents no real deployment:

```yaml
scan:
  roots: ["./sample-library"]
  follow_symlinks: false
  include_extensions: [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"]
  exclude: ["**/.git/**", "**/.cache/**"]
  read_only: true
checkpoint:
  path: "./curation-output/checkpoint.sqlite"
  resume: true
metadata:
  extract_embedded: true
  preserve_raw: false
stages:
  moat: {enabled: true, min_confidence: 0.80}
  siglip: {enabled: false, min_confidence: 0.75}
  vlm: {enabled: false, min_confidence: 0.85}
policy:
  unknown_action: "review"
  allow_file_mutation: false
outputs:
  directory: "./curation-output"
```

`unknown_action: review` only creates review records. Any file mutation must be explicitly recorded in policy, command-line arguments, and the run audit, then tested on a copy first. Raw metadata is not retained by default; model weights, access tokens, and local paths must never be committed.

## Local ONNX analysis

Install `.[onnx]` for CPU execution or `.[onnx-gpu]` for NVIDIA GPU execution, but do not install both. Users must match CUDA/cuDNN and ONNX Runtime versions to their local environment.

The GPU provider must be explicit. If CUDA is unavailable, the command fails instead of silently falling back to CPU. Models and tag files are supplied by the caller; the project does not download or cache them.

```bash
python -m pip install -e ".[onnx-gpu]"
python -m image_curator extract ./curation-output/checkpoint.sqlite \
  --moat-model /path/to/model.onnx \
  --wd14-tags /path/to/selected_tags.csv \
  --nudenet-model /path/to/nudenet.onnx \
  --provider CUDAExecutionProvider \
  --provider CPUExecutionProvider \
  --limit 100
```

MoAT produces WD14 ratings, top tags, and a normalized vector in one ONNX call. NudeNet reuses the same Pillow decode, applies classwise NMS, and stores `explicit_score` separately from `intimate_covered_score`. The technical probe measures luminance, entropy, clipping, and edge variance on a thumbnail bounded to a 512-pixel long edge. These values require human-truth calibration and must not directly drive deletion.

`--adapter module:factory` loads caller-provided Python code. That code has the current process's file and network permissions and is therefore a trusted extension boundary. Use audited adapters and run untrusted extensions under a restricted account or container.

## Output model

Each resource record contains at least a stable ID, path snapshot, content hash, file and embedded metadata, stage state, confidence, and error information.

| Field | Meaning |
| --- | --- |
| `safety` | Safety assessment; `unknown` means the evidence is insufficient |
| `quality` | Sharpness, integrity, and technical-quality signals |
| `identity` | Reference-set and open-set identity assessment |
| `publish_value` | Independent score and rationale for publishing or reuse |
| `unknown` | Reserved state for open-set, conflict, or insufficient evidence |

The result is an auditable candidate set, not an automatic publishing or deletion list. In production, retain checkpoints, configuration fingerprints, and run logs outside the repository.

## Privacy boundary

The public repository contains only general-purpose code, synthetic test fixtures, and replaceable examples. Real images, crops, embeddings, checkpoints, reports, logs, model weights, credentials, private network addresses, NAS/SMB details, and machine-specific absolute paths must remain outside the repository.

Before publishing, inspect the staged index and diff. `.gitignore` is only a convenience and is not a security boundary. See [docs/privacy-boundary.md](docs/privacy-boundary.md) for the complete policy.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

Install `.[numerics]` for local vector operations, and use `.[onnx]` or `.[onnx-gpu]` for ONNX CPU/GPU runtimes. These extras do not contain model weights and do not send images to external services.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. Report security issues according to [SECURITY.md](SECURITY.md). The project is licensed under Apache-2.0; see [LICENSE](LICENSE).

## License

Copyright 2026 image-curator contributors. Licensed under the [Apache License, Version 2.0](LICENSE).
