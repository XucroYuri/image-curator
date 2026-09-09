# image-curator

资源感知、可断点、元数据优先的大型图片库整理框架。它把安全、质量、身份和发布价值分开评估，让脏数据、缺失元数据和模型不确定性都能被明确记录。

`image-curator` is a resource-aware, resumable, metadata-first framework for organizing large and messy image libraries. It keeps safety, quality, identity, and publishing value separate, while making dirty data, missing metadata, and model uncertainty explicit.

本项目目前处于 Alpha 阶段。接口、checkpoint schema 和示例配置可能变化；请先在副本或小型测试库上运行，并保留配置与审计输出。

This project is currently Alpha. Interfaces, checkpoint schemas, and example configuration may change. Start with a copy or a small test library, and retain the configuration and audit output for each run.

## 快速开始 | Quick start

需要 Python 3.11 或更新版本。以下命令只创建空的示例目录，不依赖任何真实图片库。

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

把自己的图片库作为 `scan` 的最后一个参数传入。扫描、哈希、`extract` 元数据读取和 checkpoint 写入不会修改输入文件；输出数据库应放在独立目录。

Pass your own image-library path as the final argument to `scan`. Scanning, hashing, metadata reads, and checkpoint writes do not modify source files; keep the output database in a separate directory.

`extract --limit` 可分批运行并从 checkpoint 续跑；不指定适配器时，它仍会校验图片并提取尺寸、格式、模式和脱敏元数据证据。`verify-source` 可对单个文件生成指纹，`classify` 可用用户提供的向量和参考集执行开放集判断，`calibrate` 可用显式标注数据审核阈值。

`extract --limit` supports batches and checkpoint resume. Without an adapter, it still validates images and extracts dimensions, format, mode, and redacted metadata evidence. `verify-source` fingerprints one file, `classify` performs open-set decisions with caller-supplied vectors and references, and `calibrate` evaluates thresholds against explicit labels.

```bash
python -m image_curator verify-source ./sample-library/photo.jpg
python -m image_curator --help
```

## 当前支持边界 | Current scope

- 已提供资源发现与保守资源规划、只读扫描、重复文件位置追踪、内容哈希 checkpoint、可断点元数据/适配器提取、有界技术质量信号、元数据证据摘要、用户自备权重的 WD14 MoAT/NudeNet ONNX 分析、开放集参考集分类、阈值校准和路由原语。
- `configs/readonly.example.yaml` 是跨运行器的策略模板；当前 CLI 使用显式参数，尚未把 YAML 配置自动编排成完整端到端运行。
- Alpha 版本不会自动移动、重命名、删除或发布文件，也不会把模型输出当作最终事实。人工复核和审计记录是发布前置条件。
- 仓库不分发 MoAT、NudeNet、SigLIP 或 VLM 权重，也不替用户下载权重。使用者须自行取得模型、匹配的 WD14 标签 CSV，并核对模型、标签和服务许可证。

- The framework provides resource discovery and conservative planning, read-only scanning, duplicate occurrence tracking, content-hash checkpoints, resumable metadata/adapter extraction, bounded technical-quality signals, metadata evidence summaries, caller-supplied WD14 MoAT/NudeNet ONNX analysis, open-set reference classification, threshold calibration, and routing primitives.
- `configs/readonly.example.yaml` is a runner-neutral policy template. The current CLI uses explicit arguments and does not yet orchestrate a complete end-to-end YAML run.
- The Alpha CLI does not automatically move, rename, delete, or publish files, and does not treat model output as ground truth. Human review and audit records are required before publication.
- This repository does not distribute or download MoAT, NudeNet, SigLIP, or VLM weights. Users must obtain models and matching WD14 tag CSV files independently and verify model, tag, and service licenses.

## 设计原则 | Principles

- **默认只读 | Read-only by default**：扫描、哈希、元数据抽取和模型推理不会移动、重命名或删除文件。派生索引写入独立输出目录；任何文件操作都必须由显式策略开启。
- **可断点 | Resumable**：每个资源以稳定标识记录状态；中断后可从 checkpoint 继续，已完成工作不会重复执行。
- **元数据优先 | Metadata first**：文件系统和嵌入元数据先作为证据读取；原始值、规范化值和解析错误分开保存。
- **分级推理 | Staged inference**：MoAT、SigLIP 和 VLM 按成本与置信度分级运行。便宜的筛选阶段不能伪装成最终语义结论。
- **开放集 unknown | Open-set unknown**：模型不确定、未覆盖或互相冲突的样本保留为 `unknown`，不会被强行归入已知类别。
- **价值分离 | Separate value axes**：`safety`、`quality`、`identity` 和 `publish_value` 是独立字段和决策轴。

## 配置要点 | Configuration

`configs/readonly.example.yaml` 是可复制的基线。所有路径都是可替换的示例，不代表任何真实部署：

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

配置中的 `unknown_action: review` 只生成待复核记录。要启用任何文件变更，必须在策略、命令行和运行审计中明确记录，并先在副本上验证。配置默认不保留原始元数据值；模型权重、访问令牌和本地路径不应提交到仓库。

`unknown_action: review` only creates review records. Any file mutation must be explicitly recorded in policy, command-line arguments, and the run audit, then tested on a copy first. Raw metadata is not retained by default; model weights, access tokens, and local paths must never be committed.

## 本地 ONNX 分析 | Local ONNX analysis

CPU 运行可安装 `.[onnx]`，NVIDIA GPU 运行可安装 `.[onnx-gpu]`。两者不要同时安装；CUDA/cuDNN 与 ONNX Runtime 的版本必须由使用者按本机环境匹配。

Install `.[onnx]` for CPU execution or `.[onnx-gpu]` for NVIDIA GPU execution, but do not install both. Users must match CUDA/cuDNN and ONNX Runtime versions to their local environment.

GPU provider 必须显式指定；如果 CUDA 不可用，命令会失败而不会自动降级到 CPU。权重和标签文件由调用方提供，项目不会自动下载或缓存它们。

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

MoAT 在同一次 ONNX 调用中产生 WD14 评级、top tags 和归一化向量。NudeNet 使用同一次 Pillow 解码结果，逐类别做 NMS，并把 `explicit_score` 与 `intimate_covered_score` 分开保存。技术质量探针只在最长边 512 的缩略图上计算亮度、熵、明暗截断和边缘方差；这些值需要用人工真值校准，不能直接作为删除规则。

MoAT produces WD14 ratings, top tags, and a normalized vector in one ONNX call. NudeNet reuses the same Pillow decode, applies classwise NMS, and stores `explicit_score` separately from `intimate_covered_score`. The technical probe measures luminance, entropy, clipping, and edge variance on a thumbnail bounded to a 512-pixel long edge. These values require human-truth calibration and must not directly drive deletion.

`--adapter module:factory` 会加载调用方提供的 Python 代码。该代码与当前用户拥有相同的文件和网络权限，属于受信任扩展边界；只使用经过审计的 adapter，并在受限账户或容器中运行不受信任扩展。

`--adapter module:factory` loads caller-provided Python code. That code has the current process's file and network permissions and is therefore a trusted extension boundary. Use audited adapters and run untrusted extensions under a restricted account or container.

## 输出模型 | Output model

每个资源的记录至少包含稳定 ID、路径快照、内容哈希、文件与嵌入元数据、阶段状态、置信度和错误信息。结果按以下字段分别写入：

Each resource record contains at least a stable ID, path snapshot, content hash, file and embedded metadata, stage state, confidence, and error information. Results are kept in separate fields:

| 字段 | 含义 | Field | Meaning |
| --- | --- | --- | --- |
| `safety` | 风险与安全检查结果；`unknown` 表示无法可靠判断 | `safety` | Safety assessment; `unknown` means the evidence is insufficient |
| `quality` | 清晰度、完整性、技术质量等信号 | `quality` | Sharpness, integrity, and technical-quality signals |
| `identity` | 参考集和开放集身份判断 | `identity` | Reference-set and open-set identity assessment |
| `publish_value` | 面向发布或再利用的独立评分与理由 | `publish_value` | Independent score and rationale for publishing or reuse |
| `unknown` | 开放集、冲突或证据不足时的保留状态 | `unknown` | Reserved state for open-set, conflict, or insufficient evidence |

结果是可审计的候选集，不是自动发布或删除清单。生产环境应保留 checkpoint、配置指纹和运行日志，并将它们存放在仓库之外。

The result is an auditable candidate set, not an automatic publishing or deletion list. In production, retain checkpoints, configuration fingerprints, and run logs outside the repository.

## 隐私边界 | Privacy boundary

公开仓库只包含通用代码、合成测试夹具和可替换示例。真实图片、裁剪图、嵌入向量、checkpoint、报告、日志、模型权重、凭据、私网地址、NAS/SMB 信息和机器特定绝对路径必须留在仓库之外。

The public repository contains only general-purpose code, synthetic test fixtures, and replaceable examples. Real images, crops, embeddings, checkpoints, reports, logs, model weights, credentials, private network addresses, NAS/SMB details, and machine-specific absolute paths must remain outside the repository.

发布前请检查 staged index 和 diff；`.gitignore` 只是便利规则，不是安全边界。完整约定见 [docs/privacy-boundary.md](docs/privacy-boundary.md)。

Before publishing, inspect the staged index and diff. `.gitignore` is only a convenience and is not a security boundary. See [docs/privacy-boundary.md](docs/privacy-boundary.md) for the complete policy.

## 开发 | Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

需要本地向量运算可安装 `.[numerics]`；ONNX CPU/GPU 分别使用 `.[onnx]`、`.[onnx-gpu]`。这些 extras 不包含模型权重，也不会向外部服务发送图片。

Install `.[numerics]` for local vector operations, and use `.[onnx]` or `.[onnx-gpu]` for ONNX CPU/GPU runtimes. These extras do not contain model weights and do not send images to external services.

提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 报告。项目采用 Apache-2.0，详见 [LICENSE](LICENSE)。

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. Report security issues according to [SECURITY.md](SECURITY.md). The project is licensed under Apache-2.0; see [LICENSE](LICENSE).

## License

Copyright 2026 image-curator contributors. Licensed under the [Apache License, Version 2.0](LICENSE).
