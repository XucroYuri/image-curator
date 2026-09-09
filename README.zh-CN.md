# image-curator

[English](README.md) · [简体中文](README.zh-CN.md)

资源感知、可断点、元数据优先的大型图片库整理框架。`image-curator` 将安全、质量、身份和发布价值分开评估，让脏数据、缺失元数据和模型不确定性都能被明确记录。

本项目目前处于 Alpha 阶段。接口、checkpoint schema 和示例配置可能变化；请先在副本或小型测试库上运行，并保留每次运行的配置与审计输出。

## 快速开始

需要 Python 3.11 或更新版本。以下命令只创建空的示例目录，不依赖任何真实图片库。

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

将自己的图片库路径作为 `scan` 的最后一个参数传入。扫描、哈希、元数据读取和 checkpoint 写入不会修改输入文件；输出数据库应放在独立目录。

`extract --limit` 支持分批运行并从 checkpoint 续跑。不指定适配器时，它仍会校验图片并提取尺寸、格式、模式和脱敏元数据证据。`verify-source` 可对单个文件生成指纹，`classify` 可使用调用方提供的向量和参考集执行开放集判断，`calibrate` 可使用显式标注数据审核阈值。

```bash
python -m image_curator verify-source ./sample-library/photo.jpg
python -m image_curator --help
```

## 当前支持边界

- 资源发现与保守资源规划。
- 只读扫描、重复文件位置追踪、内容哈希 checkpoint 和可断点提取。
- 有界技术质量信号和元数据证据摘要。
- 使用调用方自备权重的 WD14 MoAT/NudeNet ONNX 分析。
- 使用不可变模型/配置指纹和逐路径审计锁的版本化历史重处理。
- 开放集参考集分类、阈值校准和路由原语。
- `configs/readonly.example.yaml` 中提供跨运行器的策略模板。

当前 CLI 使用显式参数，尚未把 YAML 配置自动编排成完整端到端运行。`reprocess` 命令可以编排冻结历史基线、本地特征提取、校准后的身份决策和只读差异报告，但不会移动、重命名、删除或发布文件，也不会把模型输出当作最终事实。发布前必须经过人工复核并保留审计记录。

仓库不分发或下载 MoAT、NudeNet、SigLIP 或 VLM 权重。使用者须自行取得模型和匹配的 WD14 标签 CSV，并核对模型、标签和服务许可证。

## 版本化重处理

`reprocess create` 将旧迁移 CSV 中的图片行冻结成新的运行。sidecar 不参与推理；相同图片内容只处理一次，但每个实际路径都保留独立审计记录。截止时间后变化、缺失或新发现的路径会获得明确状态，不会按文件名猜测或静默丢弃。

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

`--root` 是强制授权边界；CSV 中的越界路径或重复路径会在任何图片读取前被拒绝。`process` 会验证再次提供的模型、标签、预处理和后处理指纹与不可变运行清单一致，并在每个 worker 加载前后再次核对。Windows 中 pip 安装的 CUDA/cuDNN DLL 不在默认搜索路径时，可重复提供 `--cuda-dll-dir`；这些本地路径不会写入数据库。执行器最多使用两个隔离推理 worker（选择 CUDA provider 时为 CUDA worker），由父进程独占 SQLite 写入，并可安全回收过期租约。旧状态为“已发帖”的路径始终保持有效状态锁定，新评分只作为审计证据。`reprocess decide` 必须提供显式参考向量和人工标注验证向量；若准确率与未知误接纳率门禁不满足，只记录失败的决策版本，不推荐阈值。

## 设计原则

- **默认只读：** 扫描、哈希、元数据抽取和推理不会移动、重命名或删除文件；派生索引写入独立输出目录。
- **可断点：** 每个资源有稳定标识和 checkpoint 状态；中断后不会重复已完成工作。
- **元数据优先：** 文件系统和嵌入元数据作为证据读取；原始值、规范化值和解析错误分开保存。
- **分级推理：** MoAT、SigLIP 和 VLM 按成本与置信度分级运行，便宜筛选阶段不伪装成语义结论。
- **开放集 `unknown`：** 对不确定、未覆盖或互相冲突的样本保留 `unknown`，不强行归入已知类别。
- **价值分离：** `safety`、`quality`、`identity` 和 `publish_value` 是独立字段和决策轴。

## 配置要点

`configs/readonly.example.yaml` 是可复制的基线。所有路径都是可替换的示例，不代表任何真实部署：

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

`unknown_action: review` 只生成待复核记录。启用任何文件变更前，必须在策略、命令行参数和运行审计中明确记录，并先在副本上验证。配置默认不保留原始元数据；模型权重、访问令牌和本地路径不得提交到仓库。

## 本地 ONNX 分析

CPU 运行安装 `.[onnx]`，NVIDIA GPU 运行安装 `.[onnx-gpu]`，两者不要同时安装。CUDA/cuDNN 与 ONNX Runtime 的版本必须由使用者按本机环境匹配。

GPU provider 必须显式指定。如果 CUDA 不可用，命令会失败而不会静默降级到 CPU。模型和标签文件由调用方提供，项目不会自动下载或缓存它们。

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

MoAT 在一次 ONNX 调用中产生 WD14 评级、top tags 和归一化向量。NudeNet 复用同一次 Pillow 解码，逐类别执行 NMS，并分别保存 `explicit_score` 与 `intimate_covered_score`。技术质量探针只在最长边 512 像素的缩略图上计算亮度、熵、明暗截断和边缘方差；这些值需要用人工真值校准，不能直接作为删除规则。

`--adapter module:factory` 会加载调用方提供的 Python 代码。该代码拥有当前进程的文件和网络权限，属于受信任扩展边界；只使用经过审计的 adapter，并在受限账户或容器中运行不受信任扩展。

## 输出模型

每个资源记录至少包含稳定 ID、路径快照、内容哈希、文件与嵌入元数据、阶段状态、置信度和错误信息。

| 字段 | 含义 |
| --- | --- |
| `safety` | 安全检查结果；`unknown` 表示证据不足 |
| `quality` | 清晰度、完整性和技术质量信号 |
| `identity` | 参考集和开放集身份判断 |
| `publish_value` | 面向发布或再利用的独立评分与理由 |
| `unknown` | 开放集、冲突或证据不足时的保留状态 |

结果是可审计的候选集，不是自动发布或删除清单。生产环境应保留 checkpoint、配置指纹和运行日志，并将它们放在仓库之外。

## 隐私边界

公开仓库只包含通用代码、合成测试夹具和可替换示例。真实图片、裁剪图、嵌入向量、checkpoint、报告、日志、模型权重、凭据、私网地址、NAS/SMB 信息和机器特定绝对路径必须留在仓库之外。

发布前请检查 staged index 和 diff；`.gitignore` 只是便利规则，不是安全边界。完整约定见 [docs/privacy-boundary.md](docs/privacy-boundary.md)。

## 开发

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

需要本地向量运算可安装 `.[numerics]`；ONNX CPU/GPU 分别使用 `.[onnx]`、`.[onnx-gpu]`。这些 extras 不包含模型权重，也不会向外部服务发送图片。

提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。安全问题请按 [SECURITY.md](SECURITY.md) 报告。项目采用 Apache-2.0，详见 [LICENSE](LICENSE)。

## 许可证

Copyright 2026 image-curator contributors. Licensed under the [Apache License, Version 2.0](LICENSE).
