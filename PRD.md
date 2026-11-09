# qwen-tts-cli 产品需求文档（PRD）

| 项目 | 内容 |
|---|---|
| 文档版本 | v0.5（字幕默认逐句；cue 分段策略与质量门禁拆为 P1 显式需求） |
| 日期 | 2026-11-03 |
| 状态 | Draft，待评审 |
| 参考实现 | [`../Qwen3-TTS`](../Qwen3-TTS)（`qwen-tts==0.1.1`） |
| 目标产物 | 可安装的本地 CLI + Python 包 |

---

## 1. 背景与问题

Qwen3-TTS 官方仓库提供了 Python API 和示例脚本，但直接使用存在四个断层：

1. **入口分散**：`generate_voice_clone`、`generate_voice_design`、`generate_custom_voice` 三套签名与模型组合，用户必须理解模型差异才能选对。
2. **资产不沉淀**：克隆 prompt、参考音频、设计描述散落在脚本和输出目录里，无法命名、复用、迁移、审计。
3. **生产链路缺失**：长文本切分、批量任务、断点续跑、失败重试、字幕时间轴、质量校验都要自己写。
4. **结果不可复现**：seed、模型 revision、采样参数、voice prompt 没有统一落盘，同一稿件换台机器结果漂移。

因此需要 `qwen-tts-cli`：把 Qwen3-TTS 的三种生成能力、强制对齐器、ASR 校验和音频后处理，收敛成一套可脚本化、可复现、可验收的命令行工具。

---

## 2. 产品定位与目标

### 2.1 一句话定位

> **以原文为内容真值的本地语音制作工具**：统一音色资产，一键生成音频与时间轴，用 ASR 和音频指标验收，产物可复现。

项目招牌不是「能调三个模型」，而是**可验收的制作闭环**：

```
原文/脚本 → GenerationPlan → 音频 + 时间轴 → ASR/音频质检 → 可交付产物 + 可复现配方
```

音色库是资产层，不是护城河；真正的差异化在长音频切窗、原文到时间轴的单调映射、多语言规范化、CER/WER 门禁、坏 take 处理和可复现产物。

### 2.2 目标用户

| 用户 | 典型诉求 |
|---|---|
| 内容创作者 | 把文章/脚本变成配音，带字幕，能批量出多期 |
| 播客/有声书制作者 | 多角色、长文本、章节化输出、统一响度 |
| 应用开发者 / Agent | 单条命令或 JSONL 流式调用，返回结构化结果和稳定 job_id |
| 研究者 / 评测者 | 固定配方复现实验，批量跑 benchmark，比较模型与参数 |

### 2.3 成功指标

| 指标 | MVP 目标 |
|---|---|
| 首次可用时间（从 clone 到第一段音频） | ≤ 10 分钟（不含模型下载） |
| 单条命令完成「文本 → 音频 → 字幕」 | 是 |
| 同一 recipe 重跑结果一致（同 seed/环境） | 音频哈希一致或指标差异 < 阈值 |
| 批量任务中断后可续跑 | 是，且已完成项不重算 |
| 字幕时间轴可用率（人工抽检） | ≥ 95% 条目边界正确 |
| 参数错误在加载模型前报出 | 100% 的已知不支持组合 |
| 本机性能基线 | M0 产出 RTF、峰值内存、加载耗时报告 |
| 双语验收 | 中文、英文各一组文本可完整生成，字幕文本与原文一致 |

### 2.4 非目标

- 不做 GUI / Web 编辑器（`serve` 只提供 API）。
- 不做本地 token 级流式生成（当前 API 不支持，见 §3）。
- 不做音素级 SSML 控制（模型无此接口，只做词典替换和停顿插入）。
- 不做模型训练平台（只包装官方 `finetuning/` 工作流）。
- 不承诺跨后端 bit-perfect 一致（vLLM / transformers / DashScope 数值路径不同）。

---

## 3. 技术事实与约束

以下结论已对照 `../Qwen3-TTS` 源码与 README 核对。

| 事实 | 影响 | 依据 |
|---|---|---|
| `generate_*` 均返回完整 `(wavs, sample_rate)`，无音频块迭代器 | 不提供 `--stream`；需要边生成边播时只能分块生成 | `qwen_tts/inference/qwen3_tts_model.py` |
| `non_streaming_mode=False` 只模拟流式**文本输入**，不是流式生成 | 不承诺本地低延迟；真流式走 DashScope Realtime | `qwen3_tts_model.py:513` |
| 三个 `generate_*` 原生接受 list | batch 可一次推理多项；但需自行分桶限批 | README batch 示例 |
| 无 `max_inference_batch_size` | CLI 必须按模型/prompt/语言/长度分桶，避免 OOM 导致整批失败 | 源码 |
| `max_new_tokens` 默认 2048，单位是 codec token | 12Hz ≈ 12.5 token/秒；长文本必须按预算切段 | `qwen3_tts_model.py:329` |
| VoiceDesign 仅 1.7B，支持 instruct；`instruct` 形参必填但允许空串 | 需要区分「未给」与「空」 | README 模型表 |
| Base 不支持 instruct；`ref_text` 仅 ICL 模式必需 | `x_vector_only_mode=True` 可免 ref_text，质量略降 | README:243 |
| 0.6B CustomVoice 不支持 instruct，代码**静默丢弃** | CLI 必须在参数层报错，不能静默 | `qwen3_tts_model.py:799` |
| design→clone 是两个独立推理阶段，不是 API 硬约束 | 可并发驻留、懒加载或拆 worker；不写进数据模型 | 源码 |
| 对齐器为 `Qwen/Qwen3-ForcedAligner-0.6B`，依赖 `qwen-asr` | 独立可选依赖；11 种语言；单样本约 5 分钟上限；支持 batch/路径/URL/base64/ndarray | 对齐器 README |
| `scripts/align_srt.py` 使用 `aligner_processor` 内部接口 | 不复用；改为逐段对齐 + 偏移累加 | `scripts/align_srt.py:149` |
| 对齐器不能替代 ASR | CER/WER 与自动 `ref_text` 需要 Qwen3-ASR | 模型职责 |
| PyPI 已有 `qwen-tts` | distribution 用 `qwen-tts-cli`，import package 用 `qtts_cli`，命令用 `qtts` | — |

### 3.1 模型能力矩阵（CLI 必须在加载前校验）

| 模型 | 模式 | instruct | ref_text | 备注 |
|---|---|---|---|---|
| 1.7B-VoiceDesign | design | 支持（可空） | — | 设计后可 clone 固化 |
| 1.7B-CustomVoice | builtin | 支持 | — | 9 个内置音色 |
| 1.7B-Base | clone | 不支持 | ICL 需要；xvec 不需要 | 可微调 |
| 0.6B-CustomVoice | builtin | **不支持** | — | 传 instruct 直接报错 |
| 0.6B-Base | clone | 不支持 | 同上 | 显存友好 |
| ForcedAligner-0.6B | align | — | — | 独立模型，单样本 ≤ 5 分钟 |

### 3.2 首个目标环境

- **首个目标设备：本机 macOS**。用户已验证 Qwen3-TTS 可在本机运行，不再把「能否在 Mac 上跑」当作风险。
- M0 必须产出本机性能基线：模型加载耗时、RTF、峰值内存、长文本稳定性，作为后续模型选择与后端排期的依据。
- `doctor` 负责报告设备与基线；性能不可接受时再评估 0.6B 模型或远程/DashScope 后端。
- 本机 `models/` 现有：`1.7B-Base`、`1.7B-CustomVoice`、`1.7B-VoiceDesign`、`ForcedAligner-0.6B`。**没有 0.6B CustomVoice**，MVP 的 9 个内置音色由 1.7B-CustomVoice 提供。

### 3.3 依赖兼容性（已实测：混装可用）

- 元数据钉版不一致：`qwen-tts==0.1.1` 要求 `transformers==4.57.3`；PyPI `qwen-asr==0.0.6` 要求 `transformers==4.57.6`（其仓库 `pyproject.toml` 写的是 `>=4.57,<4.58`，与发布元数据不一致）。
- **本机实测（2026-11-02，`qwen3-asr` 环境：transformers 4.57.6 / torch 2.15.0 / MPS）：**
  - TTS：本地 1.7B CustomVoice 加载 5.89s，生成 4.00s 中文音频 7.60s，正常。
  - Align：本地 0.6B ForcedAligner 加载 1.86s，对齐 0.51s，11 个 token 时间戳合理。
  - **同进程先 TTS 后 align 成功**，无 Auto 注册冲突、无 traceback、无 CPU 回退。
  - 4.57.3 对照组 TTS 同样正常；未观察到 4.57.6 相对 4.57.3 的行为差异（生成有随机性，时长不同不构成证据）。
  - `pip check` 仅报钉版不一致（transformers 4.57.3 → 实际 4.57.6）及缺 flask/gradio/pytz/qwen-omni-utils；align 路径不引用 `qwen_omni_utils`，不影响主流程。
- **结论：这是元数据钉版不一致，不是运行时不兼容。单环境混装可用于 MVP。**
- **M0 方案**：单环境安装 `qwen-tts` + `qwen-asr`，锁定实测通过的版本组合（lock 文件），把「同进程 TTS + align」smoke test 作为回归项。
- **保留 fallback**：当升级 transformers、需要 ASR/Web UI 的 flask/gradio 依赖、或出现新冲突时，`align`/`check` 可切到独立 qwen-asr 环境 + 子进程桥接（`--asr-python` / `QTTS_ASR_PYTHON`）。
- **未验证项**：ASR 转写侧在 transformers 4.57.3 下的兼容性；本机 `models/` 无 0.6B CustomVoice，实测使用 1.7B。

---

## 4. 设计原则

1. **原文是真值**：字幕内容来自原文，对齐器只产时间轴，ASR 只做校验，不用 ASR 结果覆盖原文。
2. **先校验，后加载**：模型能力、参数组合、参考音频质量在加载权重前检查。
3. **一切可命名**：音色、take、项目、生成配方都有稳定 ID 和落盘格式。
4. **一切可复现**：seed + recipe + sidecar 足以重建结果；缓存键与配方分离。
5. **统一 IR**：所有输入（TEXT/文件/脚本/manifest/SRT）先编译成 `GenerationPlan`，再交给执行器。
6. **CLI 优先**：`--json`、stdin 输入、stdout 音频、stderr 进度、稳定退出码；HTTP 是后置能力。
7. **不静默降级**：不支持的能力报错，不偷偷忽略参数。

---

## 5. 功能需求

优先级：**P0 = MVP 必须**，**P1 = 生产可用**，**P2 = 成品化与生态**。
复杂度：S ≈ 数天，M ≈ 1–2 周，L > 2 周。

### 5.1 P0 — 决定工具是否可用

#### FR-P0-01 统一生成入口 `say`

- 一条命令覆盖三种模式，通过 voice 引用路由：
  - `builtin:Vivian` → CustomVoice
  - `voice:alice` → Base + 已保存 clone prompt
  - `design:描述` → VoiceDesign 即时生成
- 参数校验：Base / 0.6B CustomVoice 收到 `--instruct` 直接报错；未知音色、缺失 ref_text（ICL）报错。
- 输入：位置参数 TEXT、`-`（stdin）、`--file`、`--script`。
- 输出：`--out file.wav` 或 `-`（stdout），支持 wav/flac（P0），其他格式转 P2。
- M0 验收语言：**中文 + 英文**；其他语言沿用模型支持列表，不作为 M0 门槛。
- 复杂度：M

#### FR-P0-02 音色资产库

- `voice add` 支持 `--kind clone|design`；`voice list/inspect/preview/rm`。
- clone：保存 ref 音频哈希、ref_text、模式（icl/xvec）、模型 revision、tokenizer 类型、dtype。
- design：保存自然语言描述、示例文本、语言；`--materialize-clone` 执行 design→clone 固化。
- preview：固定试音句生成 `preview.wav`，便于 A/B。
- M0 直接支持内置**全部 9 个音色**（`builtin:*`）。音色由模型侧定义，CLI 只是枚举和校验，不做数量裁剪。
- P0 仅本地目录；`export/import` 归 P1。
- 复杂度：M

#### FR-P0-03 参考音频准入检查

- 时长、采样率、声道、削波比例、静音比例、有效语音占比、底噪估计、语言/文本覆盖。
- 不合格给出可操作建议（重录/裁剪/降噪/改用 xvec）。
- 保存 `consent` / `license` 元数据；未确认授权的克隆可要求 `--i-have-rights`。
- 复杂度：S

#### FR-P0-04 文本流水线与预算控制

- 规范化：数字、日期、金额、单位、URL、emoji、全半角、中英混排。
- 语义分段：按句/标点/段落切分，不在引号、数字、缩写中间切。
- token/时长预算：按 `max_new_tokens`（默认 2048，12Hz）估算上限，超限自动分段并提示。
- 逐段语言标记：`auto` 只在段级生效，不整篇一个语言。
- 复杂度：M

#### FR-P0-05 已知文本对齐与直出字幕

- `align AUDIO --text FILE|--srt|--script`，输出 SRT/VTT/ASS/JSON。
- `say --subtitle srt,vtt,json`：生成音频后直接对齐，逐段累加时间偏移。
- 粒度 `sentence|char|word`，默认 `sentence`：每个句子一个 cue；`char` 用于逐字校对/卡拉 OK，`word` 用于英文逐词校对。`--srt` 输入始终保留原 cue 边界。
- P0 输出原始时间轴；超长句二次切分、短句合并、CPS 与行宽门禁在 P1 实现（FR-P1-05a / FR-P1-05b）。
- 依赖 `qwen-asr`，M0 默认与主环境同装（已实测可用，见 §3.3）；未安装时给出明确指引，不静默失败。子进程桥接保留为冲突时的 fallback。
- 复杂度：M

#### FR-P0-06 生成配方与 sidecar

- 每次生成写 sidecar JSON：seed、模型 ID/revision、dtype、采样参数、语言、instruct、文本哈希、参考音频哈希、voice prompt 哈希、qtts 版本、耗时。
- 缓存键与配方分离：缓存键用于命中，配方用于复现/审计。
- 复杂度：S

#### FR-P0-07 `--dry-run` 计划

- 不加载模型，输出：分段列表、每段模型/音色/语言、预计 token 与音频时长、显存风险、缓存命中率、将写入的文件。
- 复杂度：S

#### FR-P0-08 模型管理与 `doctor`

- `models pull/list/prune`，支持 HF / ModelScope 源与 revision pin。
- `doctor`：Python/torch/设备、ffmpeg/SoX、磁盘、模型完整性、qwen-tts/qwen-asr/transformers 版本组合与钉版告警、能力矩阵自检；若启用桥接，校验桥接解释器。
- `doctor --bench`：记录本机模型加载耗时、RTF、峰值内存，输出可对比的基线报告。
- `--offline` 禁止联网下载。
- 复杂度：M

### 5.2 P1 — 决定是否适合生产

| ID | 功能 | 要求 | 复杂度 |
|---|---|---|---|
| FR-P1-01 | JSONL 批处理 | `batch plan/run`；分桶、`--jobs`、`--resume`、`--retry`、`--fail-fast`、结构化错误、逐项状态 | M |
| FR-P1-02 | 多角色脚本 | YAML/JSON 声明角色、音色、语言、instruct、停顿；`script lint/compile` 产出 plan | M |
| FR-P1-03 | ASR 质检 | 自动 `ref_text`（仅 ICL）；`check` 计算规范化 CER/WER，定位错误片段；与 `align` 共用 qwen-asr 集成形态 | L |
| FR-P1-04 | 多 take 资产库 | `--takes N` 显式开启；`--select cer,speaker-sim,lufs`；`takes list/compare/promote` | L |
| FR-P1-05a | 字幕 cue 分段策略 | 默认逐句；超长句按次级标点切分并受最大字数/CPS 约束；过短句可合并且不跨段；始终保持 `join(cue.text) == normalized_text` | S |
| FR-P1-05b | 字幕质量门禁 | CPS、最短/最长展示时长、重叠、行宽、标点悬挂；不达标非零退出 | S |
| FR-P1-06 | 发音词典与导演标记 | `--lexicon dict.json`；停顿/重音/速度用可重复的标记，不依赖 instruct | M |
| FR-P1-07 | 音频 QC | 削波、LUFS、真峰值、DC offset、静音、语速；speaker similarity 复用 Base `extract_speaker_embedding()` | L |
| FR-P1-08 | Agent 友好 stdin 模式 | `say --stdin --jsonl`：一行请求一行结果，含 `id/status/output/metrics` | S |
| FR-P1-09 | 项目锁文件 | `project init/lock` 固定模型、音色、文本规则、采样参数、qtts 版本 | M |
| FR-P1-10 | 音色导入导出 | `voice export/import`，单文件归档 `*.qttsvoice` | S |
| FR-P1-11 | 长音频对齐 | > 5 分钟的分窗策略（见 §6.3） | L |

### 5.3 P2 — 成品化与生态

| ID | 功能 | 要求 | 复杂度 |
|---|---|---|---|
| FR-P2-01 | `serve` | 本地队列/并发/取消/异步 job；可选 OpenAI 兼容 `/v1/audio/speech`；默认只监听 127.0.0.1 | L |
| FR-P2-02 | 可插拔后端 | transformers / DashScope Realtime / vLLM-Omni（仅离线 batch） | L |
| FR-P2-03 | 有声书 builder | 章节、停顿策略、章节标记、BGM ducking、M4B | L |
| FR-P2-04 | 视频配音 `dub` | SRT/EDL 导入、时长适配、多角色、stem 导出 | L |
| FR-P2-05 | 音频后处理 | 响度标准化、真峰值限制、重采样、格式转换、静音修剪 | M |
| FR-P2-06 | benchmark | RTF、首段延迟、显存、批量吞吐、CER、speaker similarity | M |
| FR-P2-07 | 微调工作流 | 数据准备、12Hz Base 单说话人训练、评估、注册为 voice profile | L |
| FR-P2-08 | 插件 SDK | 后端/后处理插件协议与版本兼容层 | L |
| FR-P2-09 | btts 音色导入 | `voice import-btts` 消费 `btts voice export --mode full` 目录/zip；校验 manifest 哈希与路径安全；clone 复用参考音频+文本，design 必须显式 `--auto-ref-text`/`--ref-text`/`--mode xvec`；要求 `--i-have-rights`，保留 btts provenance | M |

#### FR-P2-09 btts 音色导入

- 命令形态：
  `qtts voice import-btts BUNDLE (--voice NAME ... | --all) [--name NEW] [--i-have-rights] [--auto-ref-text | --ref-text TEXT] [--mode icl|xvec] [--asr-model PATH|ID] [--model NAME] [--force]`。
  `BUNDLE` 是 `btts voice export --mode full` 产出的目录或 `.zip`（zip 可含单一顶层目录）；`--mode wav` 导出明确拒绝。
- bundle 校验：`manifest.json` 存在时必须逐文件校验 SHA-256 与大小，并校验 manifest 与 profile 的 voice 列表/kind/language/profile 哈希一致；拒绝绝对路径、盘符、反斜杠、`..`、符号链接、特殊文件、大小写冲突、未知文件、超限成员与异常压缩比；`profile.json` 必须 `schema_version=1`，`name`/`kind`/`language` 合法且与目录、manifest 一致；clone 的 `reference.audio`/`reference.transcript` 及其 profile 哈希必须匹配实际文件。目录 bundle 缺少 manifest 时允许导入，但结果必须显式报告 `manifest.present=false`，不能静默当作已验证。
- clone 音色：默认 ICL，使用 `reference.audio` + `reference.transcript`，复用 `voice add --kind clone` 的 prompt 生成与 staging 逻辑（不 shell out）；`--mode xvec` 允许无文本。参考音频仍通过准入检查。
- design 音色：btts 只有 `design.yaml` + `preview.wav`，没有参考文本。默认报错，要求显式选择：`--auto-ref-text`（Qwen3-ASR 转写 `preview.wav`）、`--ref-text TEXT`（仅单音色）、或 `--mode xvec`。禁止把 `preview.wav` 当 ICL 参考却编造文本。design 导入后是 qtts clone profile；btts 的 design 指令只进入 provenance，不写 `default_instruct`。
- `--name` 只用于单音色重命名；`--all` 或多于一个 `--voice` 时退出 2。
- 同意与授权：必须显式 `--i-have-rights`；btts profile 的 `consent` 只作为 provenance 记录，不能替代 qtts 的同意记录。qtts `consent.license` 记录 `btts:<model license>`。
- provenance：`voice.json.source.origin="btts"`，`source.btts` 保留 btts name、voice_id、kind、language、license、model id/digest、profile sha256、manifest 摘要、reference/design 元数据与各文件 sha256。qtts `model` 始终写实际生成本地 prompt 的 Qwen Base 模型，不伪造 btts 模型。
- 原子性：所有选中音色先完成校验、staging 与 prompt 生成，再统一发布；`--force` 采用备份-替换-回滚，任一失败回滚已发布音色，不留下半成品 profile。
- 退出码：用法/组合错误 2；bundle/路径/哈希/授权/重名/准入失败 4；模型或 ASR 不可用 3；中断 130。
- 非目标：不解析 `design.yaml` 重建设计流程，不导入 btts 模型权重/检查点，不绕过授权确认，不把 `--mode wav` 导出当作 profile 导入。

### 5.4 明确不做

- 本地 token 级流式输出。
- 音素级 SSML / 韵律标注。
- 自动绕过克隆授权确认。
- 在 P0/P1 引入独立说话人模型（复用 Base embedding）。

---

## 6. 关键流程

### 6.1 `qtts say`

```
解析参数
  → 能力矩阵校验（模型 × instruct × ref_text × language）
  → 建立 GenerationPlan（文本规范化、分段、预算检查）
  → --dry-run 则输出计划并退出
  → 查询缓存（命中则复用音频/take）
  → 懒加载模型（必要时卸载冲突模型）
  → 逐段生成（batch 内可合并）
  → 拼接/插入停顿
  → 可选：对齐生成字幕
  → 可选：ASR/音频 QC
  → 写音频 + sidecar + 更新缓存
  → 输出结果（文本或 --json）
```

### 6.2 design → clone 固化

1. VoiceDesign 生成参考音频与 `preview.wav`。
2. 用户确认后（或 `--yes`），以生成音频作为参考，走 Base `create_voice_clone_prompt`。
3. 保存为 `voice:<name>`，记录来源为 `design`，保留原描述。
4. 内存策略：可同时驻留两个 1.7B；显存不足时先卸载 VoiceDesign 再加载 Base，**不把顺序加载写进数据模型**。

### 6.3 长音频对齐（> 5 分钟）

单样本 5 分钟是硬上限，batch 不能绕过。策略：

1. 以 20ms 帧、-45 dBFS、至少 0.6s 检测合格静音；静音切点取区间中点，并优先满足 240–285s 的 core、约 270s 的目标时长与至少 2s 的重叠。全局 DP 选路，避免贪心在尾部留下无解短窗。
2. 音频窗可重叠，但 core 覆盖 `[0, duration)` 且不重叠；每个音频窗（含单侧左上下文）不超过 300s。
3. 默认路径逐窗用 Qwen3-ASR 做**无时间戳**转写，对已知原文与各窗转录做规范化后的顺序单调匹配，得到每窗 spoken 原文区间；匹配必须全量覆盖、唯一锚定，插入/删除/替换只在破坏唯一性时失败。
4. 每窗用 `slice_prepared(global_prepared, display_start, display_end)` 切出 display/spoken 文本，禁止逐窗重新投影词典；切点落在词典 term 内必须吸附到 term 边界，否则退出 4。
5. 每窗调用 `Qwen3ForcedAligner.align`；结果映射回全局 display 字符位置。合并采用字符归属制：每字符恰好选一个候选，优先 core 覆盖，再按窗口序号/时间确定性选择；缺失覆盖、越界或非容差时间倒退退出 4。最后只调用一次全局 `build_cues`/`build_cues_for_ranges`。
6. SRT 输入以 cue 边界为硬锚点，不跑 ASR；cue 超 285s、边界不在合格静音内或跨窗都退出 4，窗口绝不拆分 cue。

禁止按固定时长或字符比例硬切；跨窗漂移会导致整体错位。若既无 SRT 也不引入 ASR，`--approximate` 按每个窗口实际送入 aligner 的 audio 区间（含左重叠）分配文本，允许文本区间重叠；静音计划无解时退化为确定性固定间隔并明确标注「边界不保证」；时间倒退做单调裁剪并报告次数，无法避开 term 时退出 4。`--srt --approximate` 在模型加载前退出 2，因为 SRT cue 边界已是硬锚点。

### 6.4 批处理

1. 读取 JSONL manifest，逐项校验并编译为 plan。
2. 按模型、voice prompt、语言、预计长度分桶。
3. 桶内合并推理，桶间按 `--jobs` 并发。
4. 每项独立状态：`pending/running/done/failed/skipped`；失败隔离。
5. `--resume` 依据 `(缓存键, 输出存在性, 配方哈希)` 跳过已完成项。
6. 结束时输出汇总；部分失败返回退出码 6。

---

## 7. 命令规范

### 7.1 全局参数

```text
qtts [--config FILE] [--data-dir DIR] [--model-dir DIR] [--cache-dir DIR]
     [--device auto|cpu|mps|cuda:N] [--dtype auto|bf16|fp16|fp32]
     [--offline] [--seed N] [--json] [--quiet] [--log-level LEVEL]
     <command>
```

- 默认数据目录用 `platformdirs` 的 user data dir（macOS: `~/Library/Application Support/qtts`；Linux: `~/.local/share/qtts`）。
- 所有命令支持 `--json`；进度写 stderr，音频/结果写 stdout。
- 交互确认只在 TTY 下出现；非 TTY 必须显式 `--yes`。

### 7.2 P0 命令

```bash
qtts say TEXT | - | --file FILE | --script FILE
  --voice builtin:Vivian|voice:alice|design:描述
  --language auto|Chinese|English|...
  --instruct TEXT
  --out out.wav|-
  --format wav|flac
  --subtitle srt,vtt,ass,json
  --max-new-tokens N
  --dry-run
  --yes

qtts voice add NAME --kind clone  --ref-audio FILE --ref-text TEXT [--mode icl|xvec]
qtts voice add NAME --kind design --description TEXT --sample-text TEXT
  --language Chinese [--materialize-clone] [--yes]
qtts voice list [--json]
qtts voice inspect NAME [--json]
qtts voice preview NAME --text TEXT --out preview.wav
qtts voice rm NAME [--yes]

qtts align AUDIO
  --text TEXT | --text-file FILE | --srt FILE | --script FILE
  --language Chinese
  --granularity sentence|char|word
  --format srt,vtt,ass,json
  --out-prefix OUT

qtts models pull MODEL_ID --source hf|modelscope --revision REV
qtts models list
qtts models prune
qtts doctor [--json]
```

### 7.3 P1 命令

```bash
qtts voice clone NAME --ref-audio FILE --auto-ref-text
qtts voice export NAME -o alice.qttsvoice
qtts voice import alice.qttsvoice

qtts script lint FILE
qtts script compile FILE -o build.plan.json

qtts batch plan MANIFEST.jsonl --out-dir OUT
qtts batch run  MANIFEST.jsonl --out-dir OUT
  --jobs 2 --resume --retry 2 --fail-fast --watch
  --backend transformers|dashscope

qtts check AUDIO --text TEXT|--srt FILE
  --asr-model Qwen/Qwen3-ASR-0.6B
  --metric auto|cer|wer --threshold 0.03
  --report report.json

qtts say ... --takes 4 --select cer,speaker-sim,lufs
qtts takes list JOB_ID
qtts takes compare JOB_ID --ids 1,2,3
qtts takes promote JOB_ID --id 2 --out final.wav

qtts project init
qtts project lock
```

### 7.4 P2 命令

```bash
qtts serve --backend transformers|dashscope --api native|openai \
  --host 127.0.0.1 --port 8000 --max-concurrency 1 --queue-size 32

qtts audiobook build project.yaml --m4b --bgm bgm.wav
qtts dub build timeline.srt --video input.mp4 --stems-out stems/
qtts post INPUT.wav --lufs -16 --true-peak -1 --format m4a|mp3|flac
qtts bench --suite smoke|batch|longform --json
qtts finetune prepare|train|eval --project project.yaml
qtts voice import-btts BUNDLE --all --i-have-rights
qtts voice import-btts BUNDLE --voice alice --auto-ref-text --i-have-rights
```

### 7.5 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 参数/用法错误 |
| 3 | 模型或后端错误（加载、显存、依赖） |
| 4 | 输入校验失败（音频不合格、能力不支持） |
| 5 | 质量门禁未通过（CER/WER/CPS/音频 QC） |
| 6 | 批处理部分失败 |
| 130 | 用户中断 |

---

## 8. 数据模型

### 8.1 音色 profile

```
<data-dir>/voices/<name>/
  voice.json          # 元数据（不含二进制）
  prompt.safetensors  # clone prompt tensors（禁止 pickle）
  ref.wav             # 原始参考音频（clone/design 固化）
  preview.wav         # 固定试音句
```

`voice.json`：

```json
{
  "schema_version": 1,
  "name": "alice",
  "kind": "clone",
  "created_at": "2026-11-02T21:00:00Z",
  "source": {"type": "clone", "ref_audio_sha256": "...", "ref_text": "...", "mode": "icl"},
  "model": {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "revision": "..."},
  "tokenizer": {"type": "12hz", "codec_rate_hz": 12},
  "dtype": "bf16",
  "language": "Chinese",
  "default_instruct": null,
  "tags": ["narration"],
  "consent": {"rights_confirmed": true, "license": "self-owned", "note": ""},
  "qtts_version": "0.1.0"
}
```

`source` 可包含 `origin` 与 `btts` 子树（FR-P2-09）。从 btts 导入的 profile 记录 `origin="btts"`，并在 `source.btts` 保留 btts name、voice_id、kind、language、license、model id/digest、profile sha256、manifest 摘要、reference/design 元数据与各文件 sha256；`model` 字段始终是 qtts 实际用于生成 prompt 的本地 Qwen Base 模型。`consent` 是 qtts 自己的同意记录（`--i-have-rights`），btts 的 consent 只作为 provenance。

### 8.2 GenerationPlan IR

所有输入先编译为该结构：

```json
{
  "schema_version": 1,
  "project": "demo",
  "defaults": {
    "voice": "voice:alice",
    "language": "auto",
    "format": "wav",
    "seed": 1234,
    "max_new_tokens": 2048
  },
  "segments": [
    {
      "id": "s001",
      "text": "第一段原文。",
      "role": "narrator",
      "voice": "voice:alice",
      "language": "Chinese",
      "instruct": null,
      "pause_after_ms": 300,
      "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
      "estimated_tokens": 42,
      "cache_key": "sha256:..."
    }
  ],
  "output": {"path": "out.wav", "format": "wav"},
  "subtitle": {"enabled": true, "formats": ["srt", "json"], "granularity": "sentence"},
  "qc": {"asr": false, "audio": true}
}
```

### 8.3 缓存键

```
cache_key = sha256(canonical_json({
  text_normalized,
  voice_prompt_sha256 or builtin_speaker,
  model_id, model_revision, dtype,
  language, instruct,
  sampling: {do_sample, top_k, top_p, temperature, repetition_penalty, max_new_tokens},
  seed
}))
```

- 缓存键**不包含** qtts 版本；版本变化通过配方哈希与 sidecar 记录，升级后可用 `cache prune --stale` 清理。
- 缓存目录分层存储，支持 `--no-cache`。

### 8.4 sidecar 配方

与每个音频同目录：`<audio_stem>.recipe.json`（例如 `a.wav` → `a.recipe.json`；`--out -` 时落到 data-dir 的 recipes 目录）

```json
{
  "schema_version": 1,
  "qtts_version": "0.1.0",
  "created_at": "...",
  "input": {"text_sha256": "...", "source": "script.yaml"},
  "voice": {"ref": "voice:alice", "prompt_sha256": "..."},
  "model": {"id": "...", "revision": "...", "dtype": "bf16"},
  "sampling": {"seed": 1234, "temperature": 0.9, "top_p": 1.0},
  "segments": [{"id": "s001", "cache_key": "...", "duration_ms": 2310}],
  "postprocess": {"lufs": -16, "true_peak": -1.0},
  "metrics": {"cer": 0.012, "speaker_similarity": 0.93}
}
```

### 8.5 项目锁文件

`qtts project lock` 生成 `qtts.lock.json`：固定模型 revision、voice profile 哈希、文本规则版本、采样参数、qtts 版本。用于团队与 CI 的结果一致性。

---

## 9. 非功能需求

| 类别 | 要求 |
|---|---|
| 兼容性 | Python 3.10–3.12；macOS / Linux 优先；Windows 尽力支持；实测版本组合 `transformers 4.57.6 + qwen-tts 0.1.1 + qwen-asr 0.0.6` 写入 lock 文件 |
| 设备 | 首个目标为 macOS 本机（已验证可运行 Qwen3-TTS）；`auto` 按平台探测 mps/cuda/cpu，不支持时明确报错，不静默降级 |
| 性能 | 模型在单进程内复用；batch 分桶避免 OOM；`--dry-run` 不加载权重 |
| 可靠性 | 批处理失败隔离；断点续跑不重算；写文件原子替换（tmp + rename） |
| 可观测性 | `--json` 结构化输出；`--verbose` 显示阶段耗时；sidecar 记录 metrics |
| 可测试性 | 核心逻辑（分段、预算、缓存键、能力矩阵、对齐合并）不依赖 GPU 即可单测；提供 fake backend |
| 安全 | 默认离线可用；`serve` 默认 127.0.0.1；不执行用户文本中的任何内容；归档导入校验路径穿越 |
| 合规 | 克隆需确认授权；sidecar 可写入 AI 生成标记；不内置规避水印的能力 |
| 国际化 | 文本规范化与语言检测按段处理；错误信息中英双语可配置 |

---

## 10. 里程碑

### M0 — MVP（约 2 周）

范围：`say`、`voice add/list`、`align`、`doctor`、`--dry-run`、sidecar。

固定项：

- 验收语言：中文 + 英文
- 内置音色：全部 9 个
- `align` 集成：M0 采用单环境混装（qwen-tts + qwen-asr + transformers 4.57.6，已实测可用）；锁定版本组合，同进程 smoke test 作回归；桥接作为 fallback
- `doctor --bench` 产出本机性能基线

命令集：

```bash
qtts say
qtts voice add NAME --kind clone|design
qtts voice list
qtts align
```

验收：见 §11。

### M1 — 生产可用（约 4–6 周）

- batch manifest + 缓存 + 续跑
- 多角色脚本与 `script compile`
- `check`（ASR + CER/WER）
- takes 资产库
- 字幕 cue 分段策略（超长句切分 / 短句合并）与 CPS / 行宽门禁
- `voice export/import`
- `project init/lock`

### M2 — 成品化（约 6–10 周）

- `serve` + OpenAI 兼容接口
- 可插拔后端（DashScope / vLLM-Omni batch）
- 有声书 / dub / 后处理 / bench
- 微调工作流包装
- 插件 SDK

---

## 11. MVP 验收标准

1. 空环境执行 `qtts doctor`，能给出缺失依赖与修复建议。
2. `qtts voice add alice --kind clone --ref-audio ref.wav --ref-text "..."` 成功，`voice list` 可见，`voice preview` 可播放。
3. `qtts say "你好，世界。" --voice voice:alice --out a.wav` 产出可播放音频与 `a.recipe.json`。
4. `qtts say "..." --voice builtin:Vivian --instruct "温柔"` 在 0.6B CustomVoice 下**报错退出码 4**，不静默忽略。
5. `qtts say "..." --voice voice:alice --subtitle srt` 产出音频与 SRT；SRT 文本与原文完全一致。
6. `qtts align a.wav --text-file script.txt --format srt,vtt,json` 产出三种格式，跨格式时间一致。
7. `--dry-run` 不触发模型加载（可通过日志/耗时验证），输出分段与预算。
8. 同一 recipe 重跑两次，sidecar 中 seed/参数一致，音频时长差异 ≤ 50ms。
9. 单元测试覆盖：能力矩阵、分段器、token 预算、缓存键、字幕合并、参数校验；无需 GPU。
10. 已知不支持组合（Base + instruct、0.6B CustomVoice + instruct、ICL 缺 ref_text）全部在加载前失败。
11. 英文文本走通完整链路：`say --language English` 生成音频，`--subtitle srt` 的字幕文本与原文逐字一致。
12. `voice list` 能枚举并试听全部 9 个内置音色；任一 `builtin:<speaker>` 可用于 `say`。
13. 同进程 smoke test：先 TTS 生成 wav，再对同一 wav + 原文调用 ForcedAligner，全部成功；该测试纳入回归。
14. 默认 `--granularity sentence`：未触发 split/merge 时 cue 数等于句数；任何模式下 `join(cue.text)` 与规范化原文逐字一致；显式 `char` / `word` 行为与 0.1.0 一致；`align --srt` 仍保留原 cue 边界且不重新分组。

---

## 12. 风险与开放问题

| 风险 | 影响 | 应对 |
|---|---|---|
| **本机性能基线未知** | 1.7B 在本机的 RTF、峰值内存、长文本稳定性尚未量化 | M0 用 `doctor --bench` 建立基线；若 RTF 不可接受，再评估 0.6B 或远程 backend |
| 对齐器单样本 5 分钟上限 | 长音频需要 ASR 锚点，引入额外依赖与误差 | 分窗策略 + 显式「边界不保证」降级；M1 完成 |
| ASR 与对齐器版本耦合 | `qwen-asr` 升级可能破坏接口 | 固定 revision；把对齐器做成可选 extra；fake backend 测试 |
| batch 分桶不当导致 OOM | 整批失败 | 长度/语言/prompt 分桶 + 保守默认 `--jobs 1` + 失败隔离 |
| 多后端数值差异 | 同一配方跨后端结果不同 | 文档声明不承诺 bit-perfect；配方记录 backend |
| 克隆授权与滥用 | 合规风险 | 准入检查 + consent 字段 + 导出时保留标记 |
| 命名/包冲突 | 安装冲突 | dist `qwen-tts-cli`、import `qtts_cli`、命令 `qtts` |
| 依赖钉版不一致 | `qwen-tts` 要求 4.57.3、`qwen-asr` 要求 4.57.6；实测 4.57.6 下混装可用，但 `pip check` 会报警 | 锁定实测版本组合；把同进程 TTS+align smoke test 作为回归；升级或出现冲突时切换子进程桥接 |

**待确认 / 待验证**

1. ASR 转写侧（`Qwen3ASRModel`）在 `transformers==4.57.3` 下的兼容性未验证；M0 只需 TTS + align，若后续 `check` 走 ASR，需要补测或直接用 4.57.6 组合。
2. 子进程桥接仅作为 fallback 保留；若启用，需确定协议（stdin/stdout JSON 或临时文件）、对齐模型是否常驻、长音频分窗归属、错误码与重试。

---

## 13. 附录

### 13.1 命名

| 项 | 名称 |
|---|---|
| PyPI distribution | `qwen-tts-cli` |
| Python import | `qtts_cli` |
| 命令 | `qtts` |
| 音色归档扩展名 | `.qttsvoice` |
| 项目锁文件 | `qtts.lock.json` |

### 13.2 参考实现映射

| qtts 能力 | 参考实现 |
|---|---|
| CustomVoice 生成 | `Qwen3TTSModel.generate_custom_voice` |
| 语音复刻 | `generate_voice_clone` + `create_voice_clone_prompt` |
| 音色设计 | `generate_voice_design` |
| 字幕对齐 | `Qwen3ForcedAligner.align`（`qwen-asr`） |
| SRT 切分参考 | `scripts/align_srt.py`（仅参考逻辑，不复用内部接口） |
| 微调 | `finetuning/` |

### 13.3 术语

- **ICL**：in-context learning 克隆模式，需要 `ref_text`。
- **xvec**：仅使用说话人向量的克隆模式，不需要 `ref_text`。
- **take**：同一段文本的一次生成结果。
- **recipe**：可复现一次生成所需的全部参数与哈希。
- **plan**：编译后的生成计划（`GenerationPlan`），是所有执行器的唯一输入。
- **btts**：本机 `breeze-tts` CLI（`btts`），用 `breeze-tts-2` 模型设计/克隆音色。
- **btts bundle**：`btts voice export --mode full` 产出的目录或 zip，含 `manifest.json`、根许可证文件和每音色的 `profile.json`/`preview.wav`（clone 另有 `ref.wav`/`ref.txt`，design 另有 `design.yaml`）。
- **btts provenance**：qtts `voice.json.source.btts` 中保留的 btts 原始身份、许可证、模型与文件哈希，不代表 qtts 模型或同意记录。

### 13.4 P1 Wave 1 实现注记

本节记录实现选择，不改变上面的需求定义。

**字幕边界（FR-P1-05a）**

- 三级优先级：`--srt` 输入 cue > TTS 计划段 > 原句。每个 span 带 `parent_ids`（如 `("seg:<id>", "sent:<n>")` 或 SRT cue id）；只有 `parent_ids` 完全一致时才合并。`--srt` 永不重新分组。
- 流程是完整链路：文本初切 → 对齐 → 按约束分组/合并 → QC。切分后由字符时间表回填每个 span 的起止时间，再按 `--min-cue-seconds` / `--max-cue-seconds` / `--max-cue-chars` / `--max-cps` 重新平衡。
- 保留字符偏移，不重新 `find()` cue 文本；不 strip；`join(cue.text) == normalized_text` 由测试覆盖重复文本、标点和空白。
- 硬切只在没有自然断点时使用，并写 warning、记录切分策略；引号内、数字和拉丁单词内不硬切。

**字幕策略与 QC（FR-P1-05b）**

- 默认 `SubtitlePolicy(max_cue_chars=42, max_line_width=42, max_cps=20.0, min_cue_seconds=0.8, max_cue_seconds=7.0, max_merge_gap_ms=300, qc_enabled=True)`；`say` / `align` / `project init` 共用同一组参数。切分宽度取 `min(max_cue_chars, max_line_width)`，因为渲染器不自动换行。
- 短 cue 合并和 CPS 平衡只在间隔不超过 `max_merge_gap_ms` 时进行，避免把相隔很远的片段合并成异常长 cue；任一侧缺少对齐时间时视为未知，不合并。
- CPS = 非空白可见显示宽度 / `max(0, 渲染毫秒 / 1000)`。显示宽度按 Unicode：`east_asian_width` 的 W/F 计 2，组合字符计 0；时间先量化到毫秒，等于上限不算违规。
- CPS 始终计算。若 `duration_s < min_cue_seconds` 且 `display_width <= max_cps * min_cue_seconds`，说明该 cue 在最短展示时长下本可满足 CPS，问题在时长，CPS 违规降为 warning（“好。”这类不可合并短句不退出 5）；长文本塞进短时长仍为 error。
- 结构问题（重叠、越界、非正时长、非单调）始终修复并记录计数；无法修复退出 4。`--no-subtitle-qc` 只关闭阈值判定，不关闭结构修复，配方记录 `enabled:false` 和 `repairs`。
- 过短/不可合并的 cue（如 `好。`）只告警，不退出 5；只有“可修复但修复后仍超限”的格式/质量失败退出 5。
- 退出 5 的产物契约：不写字幕文件；`--json` 在 stderr 输出错误，`details.subtitle_qc` 含完整报告；音频与 recipe 可保留，便于定位。
- `char` / `word` 只做结构检查；`--srt` 输入跳过 CPS/行宽等格式强校验。

**JSONL stdin（FR-P1-08）**

- `say --stdin --jsonl` 一行请求一行结果；`id` 必填且流内唯一（重复时首个保留、后续报错）；坏 JSON 行以 `id:null` 报错并继续。
- 任意单条错误使最终退出码为 6；零错误为 0；空 stdin 或启动参数错误为 2。
- 输出文件名不直接使用 raw `id`：安全 ASCII 直接用，否则 `qtts-<sha16>`；同时做 casefold 冲突检查。`--jsonl` 禁止 `--out -`、`--recipe-out -`、`--subtitle-out-prefix -`、`--json`；逐条 `out:"-"` 在处理前拒绝。
- 模型按 `(path, device, dtype)` 在流内复用；测试断言同一模型只加载一次。

**音色归档（FR-P1-10）**

- `.qttsvoice` 是 zip；成员白名单固定为 `manifest.json`、`voice.json`、`prompt.safetensors`、`ref.wav`、`preview.wav`，实际成员必须与 manifest 列表完全一致。
- 拒绝绝对路径、盘符、反斜杠、`..`、重复/大小写冲突、加密、目录/特殊成员；限制成员数、单成员大小、总大小和压缩比；从不 `extractall`，流式复制并计算 sha256。
- `prompt.safetensors` 用 safetensors 安全 API 校验；`--force` 采用备份-重命名-回滚；`--name` 在哈希校验后改写 `voice.json.name` 再重新校验；导出先写临时文件再 `os.replace`。

**项目锁（FR-P1-09）**

- 锁文件必须记录所有引用模型权重/配置的**完整 sha256**，catalog fingerprint 只作为辅助；记录所有音色（含用户音色的 `voice.json`、prompt、参考音频哈希）。只要 `ref.wav` 存在就 hash 实际文件，即使 `voice.json` 未声明；声明了 `ref_audio` 但文件缺失则失败。
- `qwen-asr` 缺失记录为 `available:false`/null；若项目声明字幕 formats，即使模型目录里有对齐器，也会先检查包可用性并直接报错（退出 3，等价 `--require-align`）。
- 记录有效采样参数、seed、规范化版本、字幕策略、dtype/device、完整 Python 与包版本。`--project` 中的相对 `model_dir` 相对项目文件所在目录解析。
- 锁文件为排序 JSON、无易变时间戳；所有校验和哈希在原子写入前完成；生成锁时不加载权重。

### 13.5 P1 Wave 2 实现注记

本节记录实现选择，不改变上面的需求定义。

**通用计划与执行（FR-P1-02 基础）**

- `SegmentPlan` 保存完整的 `VoiceResolution` 快照（ref/mode/model id/revision/fingerprint/speaker/description/ref_text/icl/prompt sha256/ref audio sha256/identity）；`GenerationPlan.voice` 只作为默认与旧 JSON 兼容字段。预算切分后的 id 形如 `s001.p2`，同时保存 `source_segment_id`；字幕 parent id 使用 source id，不依赖执行器临时编号。
- 执行器始终只保留一个 TTS 模型槽；切换模型前释放旧 engine 并清空 clone prompt，prompt 按 `(model path, voice identity)` 缓存。所有 segment 的 voice/model/language/instruct/ref_text 在第一次 `TtsEngine.load()` 前完成校验。`--model` 与多模型脚本冲突时退出 2。
- 多段文本聚合为 `"".join(display_text)`，不插入额外换行或空格，确保 `ensure_full_text()` 可验证。

**多角色脚本（FR-P1-02）**

- 脚本支持 YAML/JSON，`pyproject.toml` 声明 `PyYAML>=6.0`；缺依赖退出 3。YAML 使用 `safe_load` 等价路径，并拒绝重复 mapping key、多文档、任何 anchor/alias、超大文件与节点数；JSON 拒绝重复 key。
- `defaults` 可被 segment 覆盖 `voice/language/instruct/seed/max_new_tokens/pause_after_ms`；`role` 仅透传，不自动映射 voice；`format` 为作业级参数，segment 中出现即 schema 错。缺省 id 自动生成 `s001...`；重复 id、未知字段、字段类型/整数错误、顶层未支持字段（含 `project`）均退出 2。一个 source segment 产出多个 piece 时，全部 piece 使用稳定后缀 id（`s001.p1`、`s001.p2`），`source_segment_id` 保持源 id。
- `lint`/`compile` 都不加载模型，且先校验脚本语法/schema 再扫描模型。`compile` 输出排序 key、无时间戳、无 cache 状态/耗时/绝对 `script_source` 的 JSON；保留 revision/fingerprint。`lint` 语法/schema 错退出 2，未知 voice/能力冲突退出 4，错误 details 带 segment id 与字段路径。
- `say --script` 复用同一编译器与单条执行器；`--dry-run` 的 plan 与 compile 的 segments 一致。TEXT/`--stdin`/`--file` 与 `--script` 互斥（退出 2）；显式 `--language/--instruct/--max-new-tokens/--pause-ms` 覆盖脚本 `defaults` 并进入 plan/recipe。
- segment 的首尾 `[[pause:MS]]` 不丢弃：首段 leading、末段 trailing 落到计划全局字段，中间段合并进相邻 `pause_after_ms`。

**文本投影、词典与导演标记（FR-P1-06）**

- 固定顺序：raw → 抽取 director marker → 词典保护的 display 投影 → spoken 投影 → 分句/预算。marker 必须在数字规范化之前解析，否则 `[[pause:500]]` 的 `500` 会被改写。
- 词典匹配使用原始文本的 NFKC+casefold canonical key（大小写/全角归一），最长匹配优先；重复 term（归一后相同）退出 4。未匹配片段各自 `normalize_text`，匹配项 display 保留原始 `term`、spoken 使用 `spoken`。
- 映射是显式 `MapBlock` 字符区间，支持 1:1、1:N、N:1；构造后断言完整、连续、单调、无重叠、覆盖 display 与 spoken 两端。预算切分只在词条边界外切，切完不重新匹配词典。禁止用 `find()` 二次定位。
- 对齐器读 spoken 文本，`spoken_items_to_display()` 顺序消费 aligner item（不使用 `find`）把时间投影到 display 字符，再调用现有 `build_cues()`；字幕渲染层不做替换。顺序定位只允许跳过标点/空白，跳过内容字符时该 item 视为未映射，交给 coverage/QC 暴露。`align --text/--text-file/--srt --lexicon` 同样映射回 display，且 coverage 一律按 aligner 实际读取的 spoken 文本计算。
- `[[pause:MS]]` 只支持非负整数，可位于句首、句间、句尾或连续出现；连续标记求和，句首记 `leading_pause_ms`、句尾记 `trailing_pause_ms`，执行器用静音实现，字幕不含 marker。`speed/rate/pitch/emphasis/stress`、未知标记、未闭合/嵌套残留 `[[`/`]]` 均退出 2，不静默忽略。
- 词典内容哈希进入缓存键；recipe 记录 `lexicon_sha256`。ICL clone 的 voice identity 另含 `ref_text_sha256`，修改 `voice.json` 的 `ref_text` 会使缓存失效；xvec 不受影响。

**批量清单（FR-P1-01）**

- manifest JSONL 字段与 `say --stdin --jsonl` 对齐：`id`（必填唯一）、`text`、`voice`、`language`、`instruct`、`out`、`format`、`subtitle`、`seed`、`max_new_tokens`、`pause_ms`。未知字段、坏 JSON/重复 key（退出 2）、重复 id、输出路径冲突、未知 voice、能力冲突都在任何生成前失败（退出 4）。`out` 只要有后缀就必须是 `.wav`/`.flac`，且与显式 `format` 一致。
- `batch plan` 全量编译并写确定性 `plan.json` 与初始 `pending` 的 `status.jsonl`，不加载模型。`job_key` 覆盖 effective request（text、voice/model identity、language、instruct、seed、max_new_tokens、format、subtitle、pause_ms）、`lexicon_sha256` 与规范化输出路径，不使用 cache key。
- `batch run` 复用单条 `say` 执行器与错误结构。任务按 model/voice/language/estimated length 排序分桶；`--jobs` 只并发调度，模型推理由全局锁串行（MPS 同样）。状态 `pending/running/done/failed/skipped` 由单线程调度器维护有序内存状态，串行原子重写 `status.jsonl`；每行记录 `last_error` 与 `transient`。
- `--resume` 校验 `status=done`、job_key、输出存在且 sha256 匹配、recipe 存在且其中稳定 job_key 匹配；recipe 的 `created_at`/耗时/绝对路径不进入 job_key。`running` 恢复为 `pending`，`skipped` 下次重新调度；失败的 job 只有 `transient=true` 时才会在 `--resume --retry N` 下重新调度。
- `--retry N` 只重试执行期 transient 失败（ModelError/OSError/RuntimeError），普通异常立即失败；不重试 Usage/Input/Capability/QC。记录 attempts、last_error 与 transient。`--fail-fast` 只停止调度新任务，在跑任务完成，未启动项标 `skipped`。
- 进度与 `--watch` 只走 stderr；stdout 只有最终 summary。任一任务失败退出 6。

### 13.6 P1 Wave 3a 实现注记

本节记录实现选择，不改变上面的需求定义。

**ASR 模型发现与选择（FR-P1-03）**

- `Qwen3-ASR-*` 与 `Qwen3-ForcedAligner-*` 的 `config.json` 都可能是 `model_type=qwen3_asr`，因此 `catalog.scan()` 先按目录名判定含 `aligner` 的为 `align`，再判定目录名 `Qwen3-ASR-*` 或 `model_type == "qwen3_asr"` 的为 `asr`；不接受任意含 `asr` 子串的第三方 `model_type`。ASR 的 `size` 从目录名解析（如 `0b6`、`1b7`），不依赖 `Qwen3-TTS` 的 `tts_model_size`。
- ASR 模型的完整性检查单独处理：只要求存在 `*.safetensors`，不要求 TTS 的 `speech_tokenizer` 目录。
- `select_asr(models, spec=None)` 独立于 TTS 的 `find_model`：`spec` 支持本地目录、`org/id`、basename 与精确名称；无 `spec` 时选最高版本，同版本多个候选退出 2；缺失或命中的不是 ASR 模型退出 3（不是 `find_model` 的 4）。

**ASR 运行时与报告（FR-P1-03）**

- `AsrEngine` 惰性导入 `qwen_asr`；`transcribe()` 把模型返回规范化成 `{"text", "language", "items"}`，接受对象、单元素列表或字符串。Wave 3a 固定 `timestamps=False`，不加载 forced aligner，`errors[].start/end` 恒为 `null`。
- ASR 模型按 `(model path, device, dtype, fingerprint, aligner path/fingerprint)` 进程内缓存，并加 `RLock`；aligner 参与缓存键，避免未来启用时间戳时复用错误实例。`clear_asr_cache()` 清引用并尽力释放 MPS/CUDA 缓存。
- `--language auto` 在调用 qwen-asr 前转为 `None`；显式语言先按冻结的 Qwen3-ASR 语言表大小写不敏感 canonicalize，再按 `ModelInfo.languages` 校验，非法值退出 2。`check` 在读取音频或选择模型之前完成全局表校验，因此缺音频、缺模型时非法语言仍先退出 2；语言在全局表但所选模型不支持时也在加载权重前退出 2。
- `check` 在加载任何模型前完成参考输入、音频（存在、非空、有限、≤300s）校验；ASR 缺失在加载前退出 3。参考文本与 ASR 文本使用同一 `qc_language` 与归一化管道：`normalize_text()` → 去掉 Unicode `P*`/`S*` → 折叠空白 → `casefold()`；`qc_language()` 与 `normalize_for_qc()` 入口先 canonicalize，小写 `chinese`/`english` 与规范名产生相同数字/日期/货币读法。CER 去空白按字符，WER 按空白切词。`ratio = distance / max(1, 参考单元数)`，`ratio == threshold` 通过。
- 错误片段用 `SequenceMatcher(autojunk=False)`，块区间半开；错误块不合并跨多个操作，便于定位。
- `--json` 与 `--report` 都写同一 `schema_version: 1` 文档；排序 key、`ensure_ascii=False`、`allow_nan=False`、末尾换行，`audio.path` 只记 basename，模型只记 id/revision。`asr.language` 保留模型返回值（未检测到就是 `null`），另有 `comparison_language` 记录 QC 实际使用的语言。

**音频 QC（FR-P1-07）**

- `audio_qc.analyze()` 是纯函数，任何指标无法可靠计算时返回 `null`，并在 `unavailable` 写稳定原因（如 `scipy_unavailable`、`digital_silence`、`too_short_for_lufs`、`below_absolute_gate`、`cjk_no_word_boundaries`、`no_text`），不编造数值。
- 音频按原始声道读取；mono/stereo 支持，超过 2 声道退出 4。数字静音时 `peak_dbfs`/`lufs_integrated`/`true_peak_dbtp` 为 `null`。
- 响度按 ITU-R BS.1770-4：每声道重采样到 48kHz 后做两级 K-weighting，400ms 块、100ms hop，施加 -70 LUFS 绝对门限与 -10 LU 相对门限。true peak 至少 4x 过采样。scipy 为基础依赖；不可用时这两项降级为 `null` + `scipy_unavailable`。
- 静音检测复用 20ms 帧 RMS，默认 -50 dBFS、最短 0.2s；帧按完整时长补零，避免尾部短静音被漏算。DC offset 取各声道 |mean| 的最大值。
- 所有 QC 阈值 argparse 默认 `None`；不显式给出就只测量并退出 0。语速门禁要求非空文本：空字符串或全空白 `--text`/`--text-file` 视为未提供并退出 2。越界项以 `{metric, op, value|None, threshold, reason?}` 写入 `violations`，`ok=false` 并退出 5；阈值不可用视为越界，`op=unavailable` 并带枚举原因。
- `--reference-voice` 要求 clone profile 且有 prompt 或 `ref.wav`；`TtsEngine.require_speaker_embedding()` 在 `load()` 之前做 runtime/类级能力探测，`extract_speaker_embedding()` 也先探测再加载权重。Base 模型选择失败（路径不存在、非 clone、无可用 Base）统一映射为退出 3。非 24kHz 音频重采样后返回扁平 CPU float32。

**`say --qc` 与自动 `ref_text`**

- `say --qc` 对最终拼装音频（含首尾停顿）跑同一 `analyze()`；指标写入 `--json` 的 `qc` 与 recipe 的 `qc`。阈值越界时音频、recipe 与报告仍完整写出，payload `ok=false`、退出 5。
- 不带 `--reference-voice` 时 `--qc` 不加载额外模型；`--reference-voice` 未与 `--qc` 同用退出 2。
- `voice add --kind clone --auto-ref-text` 先把参考音频复制到 staging，再从 staging 转写，确保 ASR 与 clone prompt 读取同一份字节；`--asr-model` 只能与 `--auto-ref-text` 同用（否则退出 2），并与 `check` 共用 `select_asr` 解析/消歧。ASR 语言非法时在 staging 内、加载权重前退出 2，失败路径删除 staging。转写成功后清空 ASR 缓存再加载 TTS Base。任何失败删除 staging，不提交半成品 profile。`voice.json.source` 记录 `ref_text`、`ref_text_source=asr` 与 `asr_model{id,revision,fingerprint}`。
- `say --qc --reference-voice` 先完成纯参数/输出组合校验（例如 `--out -` 与 `--json` 冲突），再在任何生成或写音频前完成 profile 文件级校验、Base 模型解析与类级 embedding 能力探测（不加载权重）。`--dry-run` 只做 profile 文件级校验，不导入运行时。
- 集成测试复用一条 TTS 生成音频，覆盖真实 ASR 的 CER、故意改字后的退出 5、QC 指标量级与自动 `ref_text`，控制总时长并避免重复加载模型。模型目录只从 `QTTS_SMOKE_MODEL_DIR` 读取；未设置时整组跳过，不把个人绝对路径写入仓库。

### 13.7 P1 Wave 3b 实现注记

本节记录长音频对齐的实现选择，不改变 §6.3 的需求定义。

**窗口与切点**

- `AudioWindow` 以样本号为准，暴露 `audio_*`（可重叠）与 `core_*`（不重叠、覆盖全时长）两套半开区间；秒数只由样本号派生。`plan_windows()` 只规划音频/核心区间和 SRT 字符锚点，不负责文本匹配。
- 静音按 20ms 帧、`<= -45 dBFS`、至少 0.6s 检测；不足一帧的尾部仍作为部分帧计入。非末窗 core 限制在 240–285s；末窗非空且不超过 285s；音频窗只带左侧上下文，任一窗不超过 300s。候选切点用全局 DP 选择，代价为偏离目标窗长的绝对值，并按路径字典序确定性打破平局。静音计划无解且 `--approximate` 时，从样本 0 起按 `round(window_s * sr)` 固定间隔切分，报告 `plan="fixed"` 并同时向 stderr 与 JSON warnings 写入固定间隔 warning；非近似模式仍退出 4，且只有该模式错误文本包含 `use --approximate to override`。
- SRT 模式只接受 cue 边界作为切点；cue 必须整体落在同一 core 内。cue >285s、时间重叠/越界、边界不在合格静音内或无法映射到 cue 边界时退出 4，不做 cue 拆分或时间修正。

**文本锚点与字符归属**

- `mapped_text.mapped_normalize()` 复用 QC 规范化规则，但为每个规范化字符保留原文源区间；`monotonic_window_ranges(reference, window_transcripts)` 在 spoken 坐标系做顺序后缀匹配，边界通过源区间回到原文，不把规范化偏移当原文偏移。
- 每个匹配窗的区间必须非空、首窗从 0 开始、末窗到参考末尾、相邻区间有序且无空洞；重复文本用 occurrence 序列确认唯一单调锚点，歧义直接退出 4。
- 每窗 aligner 输出先乘全局样本偏移并映射为 `WindowItem(global_display_start/end, start_s, end_s)`。`merge_window_items()` 为每个 display 字符建立候选，用字符级 DP 按 `(非 core 候选数, 窗口序号序列, 时间序列)` 字典序求全局最优，而不是逐字符贪心；标点/空白可继承邻字时间，内容字符缺失、原始 `end < start`、候选越界或时间倒退（超出数值容差）退出 4。近似模式对倒退做 `start=max(start,last_end)`、`end=max(end,start)` 的确定性裁剪并记录 `monotonic_adjustments`；最后一窗允许最多 `max(overlap,1s)` 的尾部溢出并裁剪到音频时长，非末窗同幅溢出仍退出 4。
- 合并完成后只调用一次全局 `build_cues()` 或 `build_cues_for_ranges()`；SRT 模式最后恢复输入 cue 的 index。`ensure_full_text()` 在 QC 之前校验全局文本不变量。

**词典与 SRT 边界**

- `project_text()` 在入口只调用一次。`PreparedText` 现在同时携带 `source_spans`：每个 display 字符对应原始文本的 `[start, end)`；SRT cue 的原始边界通过该映射转成 display 边界，若落在展开后的数字/日期或词典 term 内部则退出 4。
- 逐窗只调用 `slice_prepared()`，切片拒绝落在 lexicon term 内部；`spoken_items_to_display_positions()` 保留全局字符位置，避免重复文本被 `find()` 错位。

**模型驻留与输出**

- 执行顺序固定为：参数/文本/词典/SRT/时长/format/stdout 冲突校验 → 选模型 → 逐窗 ASR（默认路径）→ `engine.model=None`、删除引用、`gc.collect()`、`clear_asr_cache()`、清 MPS/CUDA → 确认 aligner 缓存为空 → 加载 aligner → 对齐 → 合并 → QC → 写出。
- 新增 `align_engine.clear_aligner_cache()`；长路径在 ASR 前和 aligner 构造前都清理。`AsrEngine.transcribe_samples()` 直接接收 `(samples, sr)` 且固定 `return_time_stamps=False`，不构造常驻 forced aligner。
- 字幕与 recipe 先全部序列化到同目录临时文件，QC、全文校验和 `window` 报告全部成功后再 `os.replace` 提交；任何失败删除临时文件，不留下半套字幕/recipe。
- `align --json` 与 recipe 增加 `windowing`：`enabled`、`approximate`、`window_seconds`、`overlap_seconds`、`min/max_window_seconds`、ASR model/revision/fingerprint、warnings，以及每窗的样本号、时间、core/display/spoken 区间、转录 sha256、item 数。短音频保持 `{"enabled": false}` 且不选择 ASR；长音频 length、窗口计划和 cue 在固定 fake 输入下确定，recipe 的 `created_at`/timing 仍是诊断字段。
- `--approximate` 不构造 ASR，按每窗实际 `audio_start_s/audio_end_s` 分配字符（允许相邻区间重叠），边界落在词典 term 内部时 start 吸附 term start、end 吸附 term end；静音计划无解时使用固定间隔并在 payload/recipe/stderr 报告固定间隔 warning。`--srt --approximate` 在模型加载前退出 2。缺 ASR 模型时默认长路径退出 3；`select_asr()` 与 `select_aligner()` 都在任何 ASR 权重加载前完成，aligner 缺失时 ASR 调用次数为 0。
- 长路径覆盖率来自逐窗 aligner 的原始逐字符 item（不是合并后补齐全部字符的结果），写入 `windowing.matched_ratio` 与 recipe `coverage`，并复用短路径的 `MIN_COVERAGE=0.3` 退出 4、`WARN_COVERAGE=0.8` warning 门禁。

### 13.8 P1 Wave 3c 实现注记

本节记录多 take 资产库的实现选择，不改变 §13 的需求定义。

**请求身份与 seed**

- `job_id` 由规范化请求的 `sha256_json` 前 16 位加 `tk_` 前缀得到。canonical request 覆盖 schema/特征/归一化版本、文本与 display/词典哈希、voice identity、语言、instruct、采样参数、codec 采样率、max tokens、停顿、format、运行时身份、逐段文本哈希/voice/seed、模型身份、seed 向量、选择准则/`lufs_target`、speaker-sim 参考身份与打分模型；不覆盖输出路径、recipe 路径、QC 阈值、字幕开关、耗时与脚本路径。
- `runtime_identity()` 不导入 torch：显式 device 原样使用；`auto` 在 Darwin/arm64 预测 `mps`，否则有 `nvidia-smi` 预测 `cuda:0`，再否则 `cpu`；dtype 按显式配置或预测设备给出。dry-run 与实际运行调用同一函数，因此 job_id 一致；加载后若真实 dtype 不同，仍按既有 `rekey_plan` 处理缓存，recipe 记录实际 runtime。
- speaker-sim 的参考身份包含 `name`、`kind`、`prompt_sha256`、`ref_audio_sha256`、`embedding_source`（prompt/ref_wav）与 `scoring_version="speaker-sim.v1"`；显式 `--reference-voice` 与单一 clone 自动选择都会写入 canonical request 与每个 take recipe。仅 QC 使用 `--reference-voice` 时不影响 job 身份。
- take 1 沿用用户给出的 plan seed 与逐段 seed；take `t` 在 plan 与每个 segment 上同时加 `t−1`。派生 seed 必须仍在 `[0, 2**31−1]`，否则在生成前退出 2。`reseed_plan()` 深拷贝计划并重算每段 `cache_key`；缓存键只依赖有效 seed，不依赖 job/take id。
- 同一 job 已完整发布时直接复用，不加载模型；`scoring_failed` 状态只重试评分，不重新生成音频。`job.json` 与请求哈希不一致视为 canonical mismatch 退出 4。

**存储、锁与发布**

- 资产根为 `<data-dir>/takes/`；每个 job 使用 `<job_id>/job.json`、`take-NN.wav|flac`、`take-NN.recipe.json`。生成先写 `.staging-<job_id>-<random>/`，完成后 fsync 并 rename 发布；文档不宣称目录替换是原子事务。
- 每个 job 用 `.locks/<job_id>.lock` 上的 `flock` 串行化生成、评分与 promote。已存在但损坏/缺文件的 job 在锁内隔离为 `.damaged-<random>` 后重建；发布失败恢复旧目录，重建期间生成失败且目标不存在时同样恢复隔离目录。读取拒绝符号链接；take 音频按 `audio_sha256` 校验，recipe 校验 `job_id`、`take_id`、`audio.file` 与去前缀后的哈希。
- 路径碰撞键为 resolve 后的路径，Darwin 上额外做 casefold，Unicode 统一 NFC，避免大小写不敏感文件系统绕过覆盖保护。

**评分与模型驻留**

- 准则顺序即优先级：`cer` 升序、`speaker-sim` 降序、`lufs` 取 `abs(lufs − lufs_target)` 升序；全部使用未舍入值，最后按 take id 打破平局，排序完成后才写入舍入展示值。`rank_takes()` 显式接收 `lufs_target`，不在函数体内引用模块常量。
- 预检在任何 TTS 之前完成：`cer` 选择并校验 ASR 语言，并通过 `preflight_asr_runtime()` 在 dry-run 之外确认 `qwen_asr`/torch 可导入（缺失退出 3）；`speaker-sim` 要求恰好一个 voice identity（或显式 `--reference-voice` 的 clone profile），design/builtin 且无参考退出 2，Base 能力缺失退出 3；`lufs` 无外部依赖。`--reference-voice` 在 takes 模式下必须与 `--qc` 或 `speaker-sim` 同用。
- 驻留顺序固定为：全部 TTS → 释放 → ASR → 释放 → Base 说话人/QC → 释放 → 选中 take 的 aligner。CER 对每个 segment 使用 `seg.text` 转写，微平均 `sum(distance)/sum(reference_units)`；单段超过 300 秒退出 4。speaker-sim 按记录的 segment 时长切片，参考 embedding 只提取一次、Base 引擎只加载一次，统计加权均值与最小值，绝不对整条长 take 计算。
- 每个 take 始终记录 LUFS 与 true peak；未选准则的指标为 `null` 并写 `unavailable` 稳定原因。字幕与 QC 只针对选中 take；下游 QC 失败退出 5 时不删除已完成的 job。

**CLI 与限制**

- `--takes N --select ...` 必须成对出现，`N` 为 2–16，select 非空、去重、是 `cer,speaker-sim,lufs` 的子集。与 `--stdin --jsonl`、`--jsonl`、batch 清单互斥并在模型加载/生成前退出 2；单文件 `--script` 允许。
- `--dry-run` 输出 N 个 take 的 seed 向量、选择依赖/预检、预计输出路径，不创建 takes 目录，也不导入 torch/qwen_tts/qwen_asr。
- `takes list`、`takes compare [--ids]` 在输出前校验全部 take 的音频哈希与 recipe，损坏退出 4 且不输出成功 payload。`takes promote --id N --out FILE|- [--format]` 不加载模型，只复制原始字节、不转码；目标后缀必须与 job 格式一致；不得覆盖 job.json、任一 take 音频或 recipe（已选中且目标即源文件是唯一的幂等 no-op）。promote 是 `scoring_failed` job 的人工转正路径：先校验音频/recipe 并验证更新后的 job，再把状态改为 `complete`、写入 selected，最后才落盘；job 写入失败不产生目标文件，替换失败会尽力回滚 job。
- `_downstream()` 在任何 QC/字幕/输出副作用之前构建 job 资产保护集合；`--out`、`--recipe-out`、字幕前缀命中 `job.json`、任一 take 音频或 recipe 时退出 2。选中 take 的 recipe 在复用与写出前统一校验，损坏时退出 4 且 stdout 不产生音频字节。
- `JOB_ID` 限制为 `[A-Za-z0-9._-]{1,64}` 并只在 `<data-dir>/takes/` 下解析；绝对路径、穿越、符号链接退出 2，job/take 缺失、损坏、哈希或音频结构异常退出 4，评分/模型失败退出 3。

### 13.9 P2 btts 导入实现注记

本节记录 FR-P2-09 的实现选择，不改变上面的需求定义。

**bundle 解析与安全**

- 新增 `qtts_cli/voice_btts.py`，只做读取/校验，不加载模型、不写音色库。目录 bundle 用 `iterdir`+`lstat` 逐项检查；zip bundle 支持 btts 的单一顶层目录（如 `export/...`）和扁平布局，拒绝混合布局、嵌套音色目录、目录成员以外的特殊成员。
- 所有文件先按上限扫描，再流式计算 sha256：最多 512 个成员（约 100 个 btts 音色）、单文件 64 MiB、总大小 512 MiB、压缩比 500:1；`manifest.json` 与 `profile.json` 同样限制 1 MiB。`manifest.json` 存在时，`root_files` 与 `voices[].files` 的并集必须与实际文件完全一致（manifest 自身除外），每个 `sha256`/`bytes` 必须匹配；manifest 的 `schema_version=1`、`kind="btts-voice-export"`、`mode="full"` 必须成立。真实 `btts voice export --mode wav` 产物没有 profile 目录，根目录包含预览 `*.wav` 以及 `manifest.json`/`LICENSE.txt`/`README.txt`；导入时会依据 manifest 的 `mode` 得到“请用 `--mode full` 重新导出”的定向错误，而不是泛化的未知根文件错误。
- 路径检查拒绝绝对路径、盘符、反斜杠、NUL、`.`/`..`、首尾空白、大小写冲突和符号链接；目录读取用 `O_NOFOLLOW`，zip 成员用 `ZipFile.open` 流式读取，从不 `extractall`。
- 畸形 zip 的解析异常（损坏 central directory/local header、`extract_version` 超限、UTF-8 标志配非法文件名字节、`LargeZipFile`、`struct.error` 等）统一包装成携带 `bundle`/`file` details 的 `InputError`，退出 4；`zipfile` 的 `UserWarning`（如重叠成员 `Overlapped entries`）在读取时被提升为异常并同样包装，保证默认环境和 `PYTHONWARNINGS=error` 下都不会 traceback，`--json` 的 stderr 始终只有一个错误对象。
- `profile.json` 限制 1 MiB，必须是 UTF-8 JSON 对象且 `schema_version=1`；`name` 必须等于目录名，`kind` 为 `clone|design`，`language` 通过 btts 代码（`zh`/`en` 等）或 qtts 显示名映射到 qtts 语言。clone 的 `reference.audio`/`reference.transcript` 必须是安全 basename，其 sha256 与 profile 一致；transcript 按 btts 规则做 CRLF 归一化并去掉尾部换行，再与 `transcript_sha256` 比对。design 不要求 `design.yaml`（btts 本身只警告），缺失时结果带 warning。
- 目录 bundle 缺少 `manifest.json` 时允许继续，但 `manifest.present=false`/`verified=false` 会写入 JSON 和 provenance，并在 `warnings` 中说明；zip 缺少 manifest 直接拒绝。这是唯一允许的“降级”，且完全显式。

**clone 与 design 的处理**

- 把 `_add_clone` 的 prompt 生成/元数据/原子 staging 抽成 `_freeze_clone_prompt`，`voice add` 与 `voice import-btts` 共用；导入不 shell out，也不复制 prompt 生成逻辑。`_freeze_clone_prompt` 新增 `allow_existing`，仅在 `--force` 预检通过后跳过 `VoiceStore.create` 的“已存在”检查，实际替换仍由统一的提交事务完成。
- clone 默认 ICL：staging 内先写入 bundle 的 `ref.wav`，再跑准入检查与 `create_clone_prompt_items`；`--mode xvec` 不传 `ref_text`，但仍要求 bundle 有完整 transcript（bundle 完整性校验，不静默跳过）。
- design 统一转换为 qtts clone profile：`preview.wav` 作为参考音频，`--auto-ref-text` 用 ASR 转写同一份 staging 字节，`--ref-text` 仅允许单个 design 音色，`--mode xvec` 是显式免文本路径。默认（ICL 且无文本）退出 4 并在错误中列出三个选项。选择 clone 转换而不是 qtts `kind=design`，是因为 btts 与 qwen3-tts 是不同的模型：用 btts 的设计描述重新跑 qwen3 的 VoiceDesign 不会得到等价音色；而 preview 的音频 + 真实转写可以在 qtts 中冻结出等价说话人。btts 的 design 指令只进 provenance，不写 `default_instruct`，避免悄悄改变生成行为。
- 混合选择（`--all`）时，`--auto-ref-text` 只作用于 design 音色；clone 音色始终使用 bundle 的确认转写。只选中 clone 时给出 `--auto-ref-text` 退出 2，避免无意义的静默 no-op。

**同意、provenance 与原子性**

- `--i-have-rights` 是硬性要求，不弹 TTY 确认；btts profile 的 `consent` 只复制到 `source.btts.consent` 供审计，不能替代 qtts `consent.rights_confirmed`。qtts `consent.license` 记为 `btts:<model license>`，note 说明来源。
- `source.origin="btts"`；`source.btts` 保存 bundle 路径/格式、manifest 摘要、btts name/voice_id/kind/language、license、model id/digest、profile sha256、reference/design 元数据、tags 与每个导入文件的 sha256。值经过类型与长度裁剪（字符串 256/4096 字符，consent 最多 4 KiB），避免 `voice.json` 被异常 profile 撑大。
- 所有选中音色先完成 bundle 校验、staging、ASR（如有）和 prompt 生成，再进入统一提交：逐个 `os.replace`，已存在目标先移到 `.name.backup.<random>`；任一提交失败按逆序删除新目标并恢复备份，全部成功后删除备份。ASR 缓存会在加载 Base 前清空；staging 目录在 `finally` 中删除，失败不会留下半成品 profile。该回滚覆盖 Python 异常路径；进程在备份与替换之间被 SIGKILL/掉电时可能只留下隐藏备份目录，需要人工恢复，不承诺崩溃安全。
- `--name` 只在恰好一个选中音色时可用；`--all` 或多 `--voice` 直接退出 2。重名检查枚举现有 store 目录并按 `casefold` 比较，不依赖文件系统大小写语义，先于模型加载报出；`--force` 遇到仅大小写不同的已有目录时替换该目录，而不是在大小写敏感文件系统上创建第二个 profile。

**测试**

- `tests/test_voice_import_btts.py` 与 `tests/test_review_voice_import_btts.py` 使用合成目录/zip bundle、sine 参考音频、fake TTS/ASR 与 fake `save_prompt_tensors`，不联网、不加载权重、不依赖 torch。覆盖 clone 目录/zip、design 三路径、manifest 哈希/模式与元数据不一致、真实 `--mode wav` 布局、路径穿越/盘符/反斜杠、符号链接、压缩比 bomb、大小写冲突、超大 manifest、重名/`--force` 与提交回滚、`--name`、`--all`、ASR 成功/空转写/抛错与 staging 清理，以及畸形 zip（unsupported `extract_version`、UTF-8 标志配非法文件名、损坏 local header、重叠成员 `UserWarning`）统一退出 4 且不 traceback；另覆盖模拟大小写敏感文件系统下 existing store 的 `casefold` 重名检测与 `--force` 替换。
