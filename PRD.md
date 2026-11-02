# qwen-tts-cli 产品需求文档（PRD）

| 项目 | 内容 |
|---|---|
| 文档版本 | v0.4（P0 实现完成；sidecar 路径澄清为 `<stem>.recipe.json`） |
| 日期 | 2026-11-02 |
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
- 粒度 `char|word`；CPS 与行宽约束在 P1 加门禁，P0 先输出原始时间轴。
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
| FR-P1-05 | 字幕质量门禁 | CPS、最短/最长展示时长、重叠、行宽、标点悬挂；不达标非零退出 | S |
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

1. 用 Qwen3-ASR 转写长音频，得到粗粒度文本锚点。
2. 对已知原文与 ASR 文本做规范化后的**单调序列匹配**，确定每段音频对应的原文范围。
3. 在长静音处切窗，每窗约 4–4.5 分钟，保留 1 条字幕或数秒重叠。
4. 每窗只送入对应范围内的原文，调用 `Qwen3ForcedAligner.align`。
5. 合并结果，加全局时间偏移，裁掉重叠区。

禁止按固定时长或字符比例硬切；跨窗漂移会导致整体错位。若既无 SRT/脚本也不引入 ASR，只提供近似切分并明确标注「边界不保证」。

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
  --granularity char|word
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
  "subtitle": {"enabled": true, "formats": ["srt", "json"], "granularity": "char"},
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
- 字幕质量门禁
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
