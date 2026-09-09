# image-curator

资源感知、可断点、元数据优先的大型图片库整理工具。它把安全、质量与发布价值分开评估，让脏数据、缺失元数据和模型不确定性都能被明确记录。

Resource-aware, resumable curation for large and messy image libraries. `image-curator` keeps safety, quality, and publishing value as separate signals, records uncertainty explicitly, and is safe to trial on an existing library.

本项目目前处于 **Alpha**。接口、checkpoint schema 和示例配置可能变化；请先在副本或小型测试库上运行，并保留配置与审计输出。

## 快速开始 | Quick start

需要 Python 3.11 或更新版本。下面的流程会创建空的示例目录，扫描命令因此可以安全地复制执行：

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

把自己的图片库作为 `scan` 的最后一个参数传入。扫描、哈希、`extract` 元数据读取和 checkpoint 写入不会修改输入文件；输出数据库应放在独立目录。`extract --limit` 可分批运行并从 checkpoint 续跑；不指定适配器时，它仍会校验图片并提取尺寸、格式、模式和脱敏元数据证据。`verify-source` 可对单个文件生成指纹，`classify` 可用用户提供的向量和参考集执行开放集判断，`calibrate` 可用显式标注数据审核阈值。

```bash
python -m image_curator verify-source ./sample-library/photo.jpg
python -m image_curator --help
```

## 当前支持边界 | Current scope

- 已提供资源发现与保守资源规划、只读扫描、重复文件位置追踪、内容哈希 checkpoint、可断点元数据/适配器提取、有界技术质量信号、元数据证据摘要、用户自备权重的 WD14 MoAT/NudeNet ONNX 分析、开放集参考集分类、阈值校准和路由原语。
- `configs/readonly.example.yaml` 是跨运行器的策略模板；当前 CLI 使用显式参数，尚未把 YAML 配置自动编排成完整端到端运行。
- Alpha 版本不会自动移动、重命名、删除或发布文件，也不会把模型输出当作最终事实。人工复核和审计记录是发布前置条件。
- 仓库不分发 MoAT、NudeNet、SigLIP 或 VLM 权重，也不替用户下载权重。使用者须自行取得模型、匹配的 WD14 标签 CSV，并核对模型、标签和服务许可证。
- `siglip`、`vlm` 等 optional extras 只提供常见运行时依赖；安装依赖不会自动下载模型权重或向外部服务发送图片。

## 设计原则 | Principles

- **默认只读**：扫描、哈希、元数据抽取和模型推理不会移动、重命名或删除文件。需要写入派生索引时，输出到单独目录；任何文件操作都必须由显式策略开启。
- **可断点**：每个资源以稳定标识记录状态；中断后可从 checkpoint 继续，已完成的工作不会重复执行。Alpha CLI 尚不比较配置指纹，改变模型或预处理配置时请使用新的 checkpoint。
- **元数据优先**：先读取文件系统和嵌入元数据，再按需运行模型；原始值、规范化值和解析错误分开保存。
- **分级推理**：MoAT、SigLIP 和 VLM 按成本与置信度分级运行。便宜的筛选阶段不能伪装成最终语义结论。
- **开放集 unknown**：模型不确定、未覆盖或互相冲突的样本保留为 `unknown`，不会被强行归入已知类别。
- **价值分离**：`safety`、`quality`、`publish_value` 是独立字段和决策轴；低质量不等于不安全，也不等于不值得发布。

## 配置要点 | Configuration

`configs/readonly.example.yaml` 是可复制的基线：

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

配置中的 `unknown_action: review` 只生成待复核记录。要启用任何文件变更，必须在策略、命令行和运行审计中明确记录，并先在副本上验证。

配置默认不保留原始元数据值，只保留字段摘要和校验信息；如果业务确实需要原始值，必须由使用者在受控环境中显式开启，并单独评估隐私与留存期限。模型权重、访问令牌和本地路径不应提交到仓库。

## 本地 ONNX 分析 | Local ONNX analysis

CPU 运行可安装 `.[onnx]`，NVIDIA GPU 运行可安装 `.[onnx-gpu]`。两者不要同时安装；CUDA/cuDNN 与 ONNX Runtime 的版本必须由使用者按本机环境匹配。GPU provider 必须显式指定；如果 CUDA 不可用，命令会失败而不会自动降级到 CPU。`onnx-gpu` extra 只是 Windows/Linux 的安装便利项，本项目的实机 GPU 验证范围是 Windows x86-64，其他平台和架构请自行安装兼容的 ONNX Runtime build。

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

MoAT 在同一次 ONNX 调用中产生 WD14 评级、top tags 和归一化向量。NudeNet 使用同一次 Pillow 解码结果，逐类别做 NMS，并把 `explicit_score` 与 `intimate_covered_score` 分开保存。默认技术质量探针只在最长边 512 的缩略图上计算亮度、熵、明暗截断和边缘方差；这些值都需要用自己的人工真值校准，不能直接作为删除规则。

`--adapter module:factory` 会加载调用方提供的 Python 代码。该代码与当前用户拥有相同的文件和网络权限，因此属于受信任扩展边界；核心程序会把同一次读取和解码的 bytes/Pillow 图像传给 adapter，但无法阻止恶意或错误的第三方 adapter 再次读取、写入或发送源文件。只使用经过审计的本地 adapter，并在受限账户或容器中运行不受信任扩展。

## 输出模型 | Output model

每个资源的记录至少包含稳定 ID、路径快照、内容哈希、文件与嵌入元数据、阶段状态、置信度和错误信息。决策结果按以下字段分别写入：

| 字段 | 含义 |
| --- | --- |
| `safety` | 风险与安全检查结果；`unknown` 表示无法可靠判断 |
| `quality` | 清晰度、完整性、技术质量等信号 |
| `publish_value` | 面向发布或再利用的独立评分与理由 |
| `unknown` | 开放集、冲突或证据不足时的保留状态 |

结果是可审计的候选集，不是自动发布或删除清单。请在生产环境中保留 checkpoint、配置指纹和运行日志。

处理阶段、证据边界与模型策略见 [docs/architecture.md](docs/architecture.md) 和 [docs/model-policy.md](docs/model-policy.md)。

公开仓库的隐私边界、合成示例约定和发布前检查见 [docs/privacy-boundary.md](docs/privacy-boundary.md)。仓库中的路径、文件名和配置值均为可替换的示例，不代表任何真实部署。

## 开发 | Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

需要本地向量运算可安装 `.[numerics]`，ONNX CPU/GPU 分别使用 `.[onnx]`、`.[onnx-gpu]`；需要 SigLIP 或 VLM 适配器时分别考虑 `.[siglip]`、`.[vlm]`。这些 extras 不包含模型权重。

提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 报告。项目采用 Apache-2.0，详见 [LICENSE](LICENSE)。

## License

Copyright 2026 image-curator contributors. Licensed under the [Apache License, Version 2.0](LICENSE).
