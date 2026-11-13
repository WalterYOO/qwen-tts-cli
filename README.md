# qwen-tts-cli

基于 Qwen3-TTS 的本地语音制作命令行工具：一条命令完成**文本 → 音频 + 字幕时间轴 + 可复现配方**，并能对参考音频、模型能力和运行环境做前置校验。

> **原文是真值。** 字幕内容来自输入原文，ForcedAligner 只负责时间轴，ASR 只用于质检而不是改写内容。

当前状态：**P2 已全部实现并发布 0.3.0**，包括音频后处理、性能基准、可插拔后端与插件 SDK、HTTP 服务、有声书、视频配音和微调工作流。见[路线图](#路线图)。

---

## 目录

- [特性](#特性)
- [环境要求](#环境要求)
- [安装](#安装)
- [模型准备](#模型准备)
- [快速开始](#快速开始)
- [HTTP 服务](#http-服务)
- [命令参考](#命令参考)
- [音色引用](#音色引用)
- [后端与插件](#后端与插件)
- [输出与数据目录](#输出与数据目录)
- [配置文件](#配置文件)
- [退出码](#退出码)
- [开发与测试](#开发与测试)
- [设计约束](#设计约束)
- [路线图](#路线图)
- [相关文档](#相关文档)
- [License](#license)

---

## 特性

| 能力 | 说明 |
|---|---|
| **统一生成入口** | `qtts say` 通过 `builtin:` / `voice:` / `design:` 三种引用自动路由到 CustomVoice、Base 克隆或 VoiceDesign |
| **音色资产库** | 克隆音色、设计音色、设计后固化为克隆 profile；支持列出、检查、试听、删除 |
| **结构化音色设计** | `design init/lint/explore/report/freeze/prune` 用 YAML 描述音色身份与交付基线，离线编译/评分，候选矩阵生成、指标排序、盲听听审、recipe 固化与候选清理；模板覆盖 zh/en/ja/ko/de/fr/ru/pt/es/it |
| **参考音频准入** | 时长、采样率、削波、静音比例、有效语音占比等检查；保存 consent/license 元数据 |
| **文本流水线** | 数字/日期/金额/URL/emoji 规范化，语义分段，token/时长预算，逐段语言标记 |
| **字幕与对齐** | `align` 和 `say --subtitle` 输出 SRT / VTT / ASS / JSON；内容逐字来自规范化原文 |
| **字幕分段与门禁** | 超长句按次级标点切分、过短句同段合并；CPS / 行宽 / 时长 / 重叠 / 标点悬挂检查，可复现配方记录策略与 QC 报告 |
| **可复现配方** | 每次生成写 `*.recipe.json`：seed、模型、dtype、采样参数、文本/音频/prompt 哈希 |
| **缓存** | 内容哈希缓存；命中时不加载模型；缓存键包含模型 fingerprint 与实际 dtype |
| **JSONL stdin** | `say --stdin --jsonl` 一行请求一行结果，失败隔离，模型在流内复用 |
| **多角色脚本** | `script lint/compile` 与 `say --script` 接受 YAML/JSON；逐段音色、模型、语言、instruct、seed 与预算，生成稳定、无时间戳的计划 |
| **词典与导演标记** | `--lexicon` 让 TTS 读 `spoken`、字幕显示原文 `term`；`[[pause:MS]]` 只影响音频与时间轴，不进入字幕 |
| **批量清单** | `batch plan/run` 全量预校验 JSONL、确定性计划、单线程原子状态、并发调度、`--resume/--retry/--fail-fast` |
| **ASR 质检** | `check AUDIO` 用 Qwen3-ASR 转写，按语言自动选 CER/WER，输出编辑距离、错误片段与可选阈值门禁 |
| **音频 QC** | `qc AUDIO` 测量响度（BS.1770 LUFS）、true peak、削波、DC、静音与语速；阈值全部可选，`say --qc` 复用同一实现 |
| **音频后处理** | `post` 完成静音修剪、声道转换、重采样、响度归一化、`lookahead-min-4x` true-peak 限幅与 wav/flac/mp3/m4a 编码，最终 true peak 以发布文件解码测量为准 |
| **视频配音** | `dub build` 解析 SRT/CMX3600 子集 EDL，逐句合成、按帧放置、静音补齐与 `atempo` 时长适配；支持逐角色音色/stems、视频背景 bed、mux、响度与 true-peak |
| **模型微调** | `finetune prepare/train/eval` 调用官方脚本准备 JSONL、训练与评测；校验/选择 checkpoint，注册为 `voice:` profile，并以 CER/说话人相似度门禁评测 |
| **自动参考文本** | `voice add --kind clone --auto-ref-text` 用 ASR 转写参考音频，作为 ICL 的 `ref_text` 并记录来源与模型 |
| **音色归档** | `voice export/import` 单文件 `.qttsvoice`，白名单成员、流式校验、原子替换；`voice export --all` / `import-bundle` 多音色目录（bundle schema 2，崩溃可恢复） |
| **btts 导入** | `voice import-btts` 直接消费 `btts voice export --mode full` 目录/zip，校验 manifest 与路径安全；design 默认离线导入 recipe，clone 复用参考音频+文本，保留模板别名与完整 btts provenance |
| **项目锁文件** | `project init/lock` 固定模型全量 sha256、音色、采样参数、字幕策略与运行时版本 |
| **计划与诊断** | `--dry-run` 不加载模型即可输出分段与预算；`doctor --bench` 产出本机性能基线 |
| **机器可读** | 所有命令支持 `--json`；稳定退出码；错误走 stderr |

---

## 环境要求

- Python 3.10–3.12
- macOS 或 Linux（Windows 未验证）
- 本地推理需要 `qwen-tts` 与 `qwen-asr`（ForcedAligner / ASR）
- 基础依赖包含 `scipy`，用于 LUFS K-weighting 与 true peak 重采样；scipy 不可用时这两项为 `null` 并在 `unavailable` 给出原因
- 可选：`ffmpeg` / `SoX`（格式转换与音频检查）

**实测通过的版本组合（macOS / Apple Silicon / MPS）：**

| 包 | 版本 |
|---|---|
| `qwen-tts` | 0.1.1 |
| `qwen-asr` | 0.0.6 |
| `transformers` | 4.57.6 |
| `torch` | 2.15.0 |

> **已知依赖冲突**：`qwen-tts 0.1.1` 在元数据中要求 `transformers==4.57.3`，而 `qwen-asr 0.0.6` 要求 `transformers==4.57.6`。两者**无法同时满足**，但实测在 4.57.6 下 TTS 与 ForcedAligner 同进程均可用。`pip check` 会报告 pin 不一致，这是预期现象。详见 [PRD §3.3](PRD.md#33-依赖兼容性已实测混装可用)。

---

## 安装

```bash
git clone <repo-url> qwen-tts-cli
cd qwen-tts-cli

python -m venv .venv
source .venv/bin/activate

# 1) 安装 CLI 本体（numpy + scipy + soundfile + PyYAML）
pip install -e .

# 2) 依次安装运行时；第二条会把 transformers 升到 4.57.6
pip install "qwen-tts==0.1.1"
pip install "qwen-asr==0.0.6"

# 3) 验证
qtts --version
qtts doctor
```

可选 extras：

```bash
pip install -e ".[hf]"          # 使用 Hugging Face 下载模型
pip install -e ".[modelscope]"  # 使用 ModelScope 下载模型
pip install -e ".[dev]"         # pytest

# 可选后端（不装也能使用默认 transformers 后端）
pip install "dashscope>=2.0.1"  # DashScope 实时 TTS
pip install "vllm-omni"         # vLLM-Omni 离线 batch，仅 Linux + GPU
```

如果 `pip check` 报 `qwen-tts requires transformers==4.57.3`，属于上述已知冲突；只要 `doctor` 与集成测试通过即可继续。

不安装也行，可以直接用模块方式运行：

```bash
PYTHONPATH=. python -m qtts_cli --help
```

---

## 模型准备

CLI 不内置模型。通过 `--model-dir` 指向一个包含以下子目录的本地目录：

```text
models/
├── Qwen3-TTS-12Hz-1.7B-Base/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
└── Qwen3-ForcedAligner-0.6B/
```

示例：

```bash
export QTTS_MODEL_DIR=/path/to/Qwen3-TTS/models
qtts models list --model-dir "$QTTS_MODEL_DIR"
```

`QTTS_MODEL_DIR`（或 `QWEN_TTS_MODEL_DIR`）会被 CLI 自动读取，也可以改用 `--model-dir` 显式覆盖。

从远端下载：

```bash
qtts models pull Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --source hf --revision main --model-dir "$QTTS_MODEL_DIR"
```

`--offline` 会禁止下载，并尽量让加载器只读本地文件。

> 0.6B CustomVoice 可选，但不是必需。本机 M0 使用 1.7B CustomVoice 提供全部 9 个内置音色。

---

## 快速开始

### 1. 内置音色 + 字幕

```bash
qtts say "你好，世界。这是一次测试。" \
  --voice builtin:Vivian \
  --language Chinese \
  --model-dir "$QTTS_MODEL_DIR" \
  --out hello.wav \
  --subtitle srt

# 产出：hello.wav、hello.srt、hello.recipe.json
```

### 2. 先看计划，不加载模型

```bash
qtts say "第一段。第二段。" \
  --voice builtin:Vivian \
  --model-dir "$QTTS_MODEL_DIR" \
  --out out.wav --subtitle srt --dry-run
```

### 3. 克隆音色

```bash
qtts voice add alice \
  --kind clone \
  --ref-audio ref.wav \
  --ref-text "参考音频对应的文字" \
  --mode icl \
  --i-have-rights \
  --model-dir "$QTTS_MODEL_DIR"

qtts voice preview alice --model-dir "$QTTS_MODEL_DIR" --out alice-preview.wav
qtts say "你好，我是 Alice。" --voice voice:alice \
  --model-dir "$QTTS_MODEL_DIR" --out alice.wav
```

### 4. 设计音色并固化

```bash
qtts voice add narrator \
  --kind design \
  --description "低沉、沙哑的中年男声，语速偏慢" \
  --sample-text "这是试音句。" \
  --language Chinese \
  --materialize-clone \
  --model-dir "$QTTS_MODEL_DIR"
```

### 5. 对已有音频做对齐

```bash
qtts align hello.wav \
  --text-file script.txt \
  --language Chinese \
  --format srt,vtt,ass,json \
  --model-dir "$QTTS_MODEL_DIR"
```

### 6. 诊断与性能基线

```bash
qtts doctor --bench --model-dir "$QTTS_MODEL_DIR"
```

### 7. 多角色脚本

```yaml
schema_version: 1
defaults:
  language: Chinese
  format: wav
  seed: 1234
segments:
  - id: narrator
    text: "[[pause:300]]夜色很深。"
    voice: "builtin:Vivian"
  - id: guest
    text: "谁在外面？"
    voice: "design:低沉沙哑的中年男声"
    pause_after_ms: 500
```

```bash
qtts script lint roles.yaml
qtts script compile roles.yaml -o build.plan.json
qtts say --script roles.yaml --out roles.wav --subtitle srt
```

`script lint` 只校验，不写计划、不加载模型；`compile` 输出排序、无时间戳、不含运行期缓存状态的 `GenerationPlan` JSON；`say --script` 复用同一编译器。每段可独立覆盖 `voice/language/instruct/seed/max_new_tokens/pause_after_ms`；`format` 是作业级参数，不能按段覆盖。一个脚本 segment 含多句、多个 marker 块或预算切分时，所有 piece 都获得稳定后缀 id（`s001.p1`、`s001.p2`），`source_segment_id` 保持 `s001`。段首/段尾 `[[pause:MS]]` 进入计划的 `leading_pause_ms`/`trailing_pause_ms` 音频静音；中间段的段首/段尾停顿合并到相邻 segment 的 `pause_after_ms`。`say --script` 可与 `--language/--instruct/--max-new-tokens/--pause-ms` 组合，这些显式 CLI 值覆盖脚本 `defaults`；与 TEXT/`--stdin`/`--file` 组合则退出 2。任何脚本语法/schema 错误（含重复 id、未知字段、未闭合 marker、顶层 `project`）都在扫描模型前退出 2；未知 voice、缺失 ICL ref_text、能力冲突退出 4。

### 8. 词典与暂停标记

```json
{"schema_version": 1, "entries": [{"term": "AI", "spoken": "人工智能"}]}
```

```bash
qtts say "AI发布。" --voice "builtin:Vivian" --lexicon lexicon.json \
  --subtitle srt --out ai.wav
```

最长匹配优先，匹配前做大小写与全角归一；TTS 使用 `spoken`，字幕仍显示原文 `term`。导演标记只支持 `[[pause:MS]]`（可出现在句首、句间、句尾或连续出现）；`speed/rate/pitch/emphasis/stress` 等标记以及未闭合的 `[[`/`]]` 都会显式退出 2，不会静默忽略。`align --text/--text-file/--srt` 同样接受 `--lexicon`，把外部音频的时间轴映射回显示文本；coverage 始终按 aligner 实际读取的 spoken 文本计算。

### 9. 批量清单

```jsonl
{"id":"a","text":"第一段。","voice":"builtin:Vivian","language":"Chinese"}
{"id":"b","text":"第二段。","voice":"design:温柔女声","subtitle":"srt"}
```

```bash
qtts batch plan manifest.jsonl --out-dir build
qtts batch run manifest.jsonl --out-dir build --jobs 2 --resume --retry 2 --watch
```

`batch plan` 启动时完整解析、校验并编译全部条目，写 `build/plan.json` 与初始 `build/status.jsonl`，不加载模型。`batch run` 复用单条 `say` 执行器；`job_key` 覆盖文本、音色/模型身份、语言、instruct、采样、格式、字幕、词典内容哈希与输出路径。`--resume` 只在 `status=done`、`job_key`、输出 sha256 和 recipe 的 `job_key` 全部匹配时跳过；只有被标记为 transient 的失败才可在 `--resume --retry` 下重新调度。`status.jsonl` 由单线程调度器原子重写，记录 `last_error`/`transient`，崩溃遗留的 `running` 会恢复为 `pending`。部分失败退出 6。

### 10. ASR 质检与音频 QC

```bash
# 转写音频并与参考文本比较；中文默认 CER，英文默认 WER
qtts check out.wav --text "你好，世界。" --threshold 0.05 --report check.json

# 只测量音频，不设阈值时永远退出 0
qtts qc out.wav --json --report qc.json

# 生成后立即质检；阈值与 qc 子命令同名
qtts say "你好。" --voice "builtin:Vivian" --qc \
  --max-clipping-ratio 0.001 --target-lufs -16 --lufs-tolerance 1.0 --out hello.wav
```

`check` 的参考文本与 ASR 文本走同一归一化管道（标点/符号移除、空白折叠、大小写折叠）；`ratio = distance / max(1, 参考单元数)`，`ratio <= threshold` 通过（默认 0.03），否则退出 5。归一化、编辑距离与错误片段均可复现；Wave 3a 冻结时间戳，`errors[].start/end` 固定为 `null`。`check` 单次音频上限 300 秒，超过在加载模型前退出 4。`check --qc-metrics basic|full` 可附带声学指标（`--qc-recipe` 必须与 `--qc-metrics` 同用），结果只写入嵌套 `qc`，不参与 CER/WER 门禁。

`qc` 的阈值全部可选，未显式给出的指标只测量不判定。`lufs_integrated` 用 BS.1770-4（K-weighting、400ms 块、绝对 -70 LUFS 与相对 -10 LU 门限），`true_peak_dbtp` 至少 4x 过采样；数字静音、音频过短或 scipy 不可用时返回 `null`，并在 `unavailable` 中以稳定枚举说明原因，绝不编造数值。`--reference-voice` 额外计算说话人余弦相似度，需要 Base 模型。`--json` 与 `--report` 生成逐字节相同的排序 JSON，不含绝对路径、时间戳或耗时。

### 11. 音频后处理

```bash
# 归一化到 -16 LUFS，并按 -1 dBTP 限幅；输出 FLAC
qtts post raw.wav --out final.flac --lufs -16 --true-peak -1

# 批量重采样、转单声道并裁掉首尾静音
qtts post a.wav b.flac --out-dir build --format wav \
  --sample-rate 48000 --channels mono --trim-silence

# 只计算与测量，不写文件
qtts post raw.wav --out-dir build --format mp3 --bitrate 192k --dry-run --json
```

`post` 的输入只接受 `.wav`/`.flac`。处理顺序固定为：读取 → 静音修剪 → 声道转换 → 重采样 → 响度增益 → true-peak 限幅 → 最终测量 → 编码。`--lufs` 未配 `--true-peak` 时默认使用 -1.0 dBTP；限幅采用 `lookahead-min-4x`（4x 过采样 gain 包络、5ms attack、50ms release，不做 sample clipping），必要时再施加静态安全衰减。`true_peak_ceiling_scope` 固定为 `encoded`：WAV/FLAC 写临时文件后读回测量，mp3/m4a 编码后解码回 24-bit WAV 测量；超出上限时最多重编码 2 次，仍不达标则该文件失败且不会替换最终输出。`wav/flac` 默认 PCM_24（可 `--bit-depth 16`），`mp3/m4a` 使用 ffmpeg 的 `libmp3lame`/`aac`，默认 192k。输出后缀与 `--format` 必须一致；`--out-dir` 要求显式 `--format`。`--dry-run` 完成全部预检与内存 DSP，但不创建目录、不编码、不写报告；`--json` 与 `--report` 逐字节一致。路径碰撞（输入/输出/报告、symlink、大小写与 NFC）在写任何文件前退出 2。

### 12. 性能基准

```bash
# 冒烟基准：内置中英各一句，重复 3 次
qtts bench --suite smoke --runs 3 --json

# 批量短句 + CER 门禁（需要本地 ASR）
qtts bench --suite batch --count 16 --cer --max-cer 0.05

# 长文本分段基准，并计算与参考音色的相似度
qtts bench --suite longform --text-file script.txt \
  --speaker-sim --reference-voice alice --min-speaker-similarity 0.75

# 只解析模型/音色与计划，不加载模型、不写文件
qtts bench --suite smoke --dry-run

# 每个 case 附带声学指标（只报告，不影响 CER/相似度门禁）
qtts bench --suite smoke --qc-metrics basic --json
```

`bench` 的计时使用外部 `time.perf_counter()` 未舍入值聚合：`runs[].wall_s` 不含加载，`metrics.wall_s` 含加载但不含评分/报告，`rtf` 取每 run 中位数，`wall_rtf = 全场 wall_s / 全场 audio_s`。默认 `--suite smoke` 跑内置中英各一句；`batch` 从 16 条短句按语言过滤后循环取 `--count`；`longform` 用 `--text` > `--text-file` > 内置长文，经 `plan.build_plan()` 分段后逐段生成。`--dry-run` 只做目录/元数据/文件存在性与计划，不 import torch/qwen、不加载权重、不创建文件。真实运行时严格按 TTS → 释放 → ASR（可选）→ 释放 → Base（可选）单驻留顺序，`finally` 释放所有已加载模型。CER 按字符编辑距离、逐 case 语言转写；speaker-sim 优先使用 clone profile 的 `ref_spk_embedding`，否则用同一 Base 从 `ref.wav` 提取，并在报告中记录 `reference_embedding_source`。`--json` 与 `--report` 逐字节一致；门禁失败输出完整 payload、`ok=false`、`gate_failures[]` 并退出 5；报告写失败退出 4。`--qc-metrics basic|full` 为每个 case 生成 `qc` 指标，仅报告、不改变退出码；与 `--dry-run` 互斥。

### 13. 多 take 资产库

```bash
# 连续生成 2 个 take，按响度接近 -16 LUFS 选择，选中结果写到 hello.wav
qtts say "你好。" --voice "builtin:Vivian" --takes 2 --select lufs \
  --out hello.wav --json

# 不加载模型地查看、比较与提升 take
qtts takes list tk_0123456789abcdef --json
qtts takes compare tk_0123456789abcdef --ids 1,2
qtts takes promote tk_0123456789abcdef --id 2 --out chosen.wav
```

`--takes N --select cer,speaker-sim,lufs` 成对出现，`N` 为 2–16；`--select` 是逗号分隔、去重、非空的 `cer,speaker-sim,lufs` 子集，顺序即优先级。每个 take 在基准 seed 上依次加 `t−1`（同时作用于 plan 与所有 segment），因此同一请求可复现。take 1 保留用户给出的全部 seed。

生成结束后按准则排序：`cer` 升序、`speaker-sim` 降序（按段时长加权均值，另记最小值）、`lufs` 取 `abs(lufs − (-16.0))` 升序；并列按 take id。无论是否被选中，LUFS 与 true peak 都会记录；未选准则的指标为 `null` 并附 `unavailable` 原因。`cer` 需要本地 ASR：缺失 `qwen_asr` 会在任何 TTS 生成之前退出 3。`speaker-sim` 要求单个克隆音色（或用 `--reference-voice` 指定参考，且必须同时带 `--qc` 或 `speaker-sim`）。显式或隐含的参考音色身份（名称、kind、prompt/ref 音频哈希、embedding 来源、评分版本）进入 canonical request 和 job_id；切换参考音色不会复用旧 job。

资产库位于 `<data-dir>/takes/`：

```text
takes/.locks/<job_id>.lock
takes/<job_id>/job.json
takes/<job_id>/take-01.wav
takes/<job_id>/take-01.recipe.json
...
```

`job_id` 由规范化请求哈希得到（`tk_` + 16 位十六进制）。请求完全一致时直接复用已完成的 job；仅评分失败时只重试评分，不重新生成音频。`job.json` 损坏或缺文件时旧目录被隔离为 `.damaged-*` 后重建；重建期间生成失败会把隔离目录恢复回原位。`takes list/compare` 在输出前校验每个 take 的音频 sha256 与 recipe 身份/哈希，任何损坏退出 4 且不输出成功 payload。`takes list/compare/promote` 都不加载模型；`promote` 只复制所选 take 的原始字节，不转码，因此目标后缀必须与 job 格式一致。`promote` 也是 `scoring_failed` job 的人工转正路径：先校验音频与 recipe，再把状态改为 `complete` 并写入 selected，最后才落盘；任一步失败不会留下半写的目标文件。`--out`、`--recipe-out` 与字幕前缀不得覆盖 `job.json`、任一 take 音频或 recipe。`--takes` 与 `--stdin --jsonl`、`--jsonl` 和 batch 清单互斥（退出 2）。

### 14. 有声书

```bash
# 从项目文件构建整本有声书（默认输出 build/<project>.wav）
qtts audiobook build book.yaml

# M4B（AAC + 章节元数据）、逐章响度归一化、背景音乐
qtts audiobook build book.yaml --out-dir build --m4b --lufs -18 \
  --bgm music.ogg --ducking-db -12

# 只校验与打印计划：不加载模型、不创建目录、不写文件
qtts audiobook build book.yaml --dry-run --json
```

项目文件接受 YAML/JSON，严格拒绝未知字段、重复键、alias/anchor 与多文档；`chapters[]` 每章必须且只能有一个 `source`（项目相对或绝对 UTF-8 文件）或 `text`。段落按空行切分，Markdown 不解释。`--title/--author/--album/--voice/--bgm`、暂停、采样率与响度均可覆盖项目值。

每个段落独立合成并写入 `<out-dir>/segments/<chapter>-<paragraph>.wav` 与原子 sidecar；指纹只覆盖合成输入（文本、音色身份与 clone 参考音频哈希、模型文件、后端身份、有效 device/dtype、语言、instruct、采样参数、seed、归一化版本），因此改采样率、暂停、BGM 或输出格式不会重新合成。`--force`/`--no-cache` 绕过复用；损坏的缓存 WAV/sidecar 视为 miss 并自动重生成（不是错误）。

BGM 以书本第 0 个采样为相位锚点循环，20 ms 帧、-35 dB 阈值、0.05/0.30 s attack/release 的单极点包络进行 ducking，仅在书本首尾淡入淡出。启用响度时逐章归一化并在编码前校验 true peak；m4a/m4b/mp3 编码后再解码测一次并写入 manifest。发布采用 journal（`prepared` → `published`），旧 final/manifest 独立备份与回滚；下一次构建会自动恢复被中断的事务。SIGINT/SIGTERM 在提交点之前回滚并以 130 退出，提交点之后保留完整新文件并打印 `interrupted after publish`。

### 15. 视频配音（dub）

```bash
# 用 SRT/EDL 时间轴把每句台词合成到对应位置（输出 dub.wav + dub.dub.json）
qtts dub build subtitles.srt --voice builtin:Vivian --out dub.wav

# 项目文件定义默认音色、语言、响度与逐角色音色；EDL 指定 25 fps
qtts dub build timeline.edl --project dub.yaml --fps 25 --out out.flac

# 以视频音轨作为背景 bed（可选），并按视频时长裁剪；同时导出每个角色的 stem
qtts dub build subs.srt --project dub.yaml --video movie.mp4 --bed-db -12 \
  --stems-out stems --out dub.wav

# 只校验输入、音色与时间轴：不加载模型、不建目录、不写文件
qtts dub build subs.srt --project dub.yaml --dry-run --json
```

时间轴支持严格的 SRT 与 CMX3600 子集 EDL（`NNN REEL TRACK TYPE ...`，仅 `C` 事件、轨道含 `V`，`* SPEAKER`/`* TEXT` 注释；非 drop-frame；`--fps` 默认 25）。SRT 以空行或只含空白的行分隔 cue，每个 block 必须恰好是索引行、时间行和非空文本；缺时间行、空文本、文本中出现另一条时间行、文本行只有一个正整数（通常是漏了分隔符的下一索引）或末尾悬空索引都退出 4。SRT/EDL 的 cue 按半开区间 `[start, end)` 放置；cue 文本可写 `speaker: text`，音色优先级为 `--speaker-voice` → 项目 `speakers` → `--voice` → 项目 `voice`，未映射的说话人必须有默认音色，否则退出 4。`--language` 作用于每个 cue，优先级为 CLI → 项目 `language` → `auto`；`--instruct` 是全局默认值（CLI → 项目顶层 `instruct`），但项目 `speakers.<name>.instruct` 仍优先于该说话人的 CLI 全局 instruct。三者的生效值都进入 fingerprint、dry-run payload、`voices` 与 manifest/`timeline.json`。`__anonymous__` 是无 speaker cue 的保留键，出现在时间轴 speaker 名中一律退出 4，无论是否请求 stems。

每句独立合成并缓存到 `<out-stem>.dub-work/cues/`。指纹只覆盖合成输入（文本、音色身份与 clone 参考音频哈希、模型文件、后端身份、有效 device/dtype、语言、instruct、采样参数、seed、归一化版本），改采样率、拉伸上下限或输出格式不会重新合成；损坏的缓存 WAV/sidecar 视为 miss。`--force` 同时覆盖输出并刷新缓存；`--no-cache` 跳过缓存读取、强制重新合成但仍写入新的 cue WAV/sidecar；`--keep-work` 保留中间文件。

时长适配先按 `目标帧 / 渲染帧` 求速度比，再夹到 `--min-speed`（默认 0.85）与 `--max-speed`（默认 1.25）；仅在速度与 1 的差超过 0.005 时调用一次 `ffmpeg atempo`，随后精确重采样到目标帧并在尾部补静音。渲染仍超出目标帧时按 `--overflow warn|fail` 处理：`warn`（默认）保留音频并给出 warning，`fail` 在发布任何输出前以退出码 5 失败。提供 `--video` 时混音长度等于视频时长，超出视频尾部的 cue 走同一 overflow 策略（warn 截断）。

`--bed-db` 是显式 opt-in，必须同时给 `--video`；视频音轨以指定增益混入，缺失音轨退出 4。默认启用响度归一化（`--lufs -18 --true-peak -1.5`），`--no-loudness` 关闭限幅但仍测量实际 true peak 并记录 warning。输出为 wav/flac（soundfile）或 m4a/mp3（ffmpeg），`--mux` 额外写出 `<out-stem>.mp4`。manifest 写到 `<out-stem>.dub.json`；`--stems-out` 还写 `timeline.json` 与每个说话人的 WAV。stems 按 speaker 排序逐个物化并立即写盘释放，峰值内存保持 master + 一个 stem（+ 当前读回的 cue），不随说话人数量线性增长。已存在的空 stems 目录无需 `--force` 即可使用，但目录中已存在的 stem 文件或 `timeline.json` 仍会退出 4（除非 `--force`）。输出写入、read-back 哈希或 manifest/timeline.json 写入失败均按模型/IO 失败退出 3。

发布采用固定顺序：最终音频 → stems 与 `timeline.json` → mux 的 mp4 → manifest 最后。每个文件都先写 UUID 临时文件再原子替换，但**没有跨文件 journal**：这是逐文件原子性，不是整组事务。SIGINT/SIGTERM 在第一次替换前会清理临时文件并让命令以 130 退出；第一次替换成功后信号被推迟到整组发布完成，然后打印 `interrupted after publish` 并以 130 退出。若进程被 SIGKILL 或机器掉电中断在多次替换之间，磁盘上可能留下新旧混合的一组文件；重跑同一命令（必要时 `--force`）即可重建。用法错误退出 2，输入/路径/时间轴/锁错误退出 4，模型/后端/ffmpeg/DSP 失败退出 3，overflow fail 退出 5。

### 16. 微调音色（finetune）

```bash
# 1) 校验数据集音频并生成官方 prepare_data.py 所需的 raw JSONL，
#    再调用官方脚本产出带 audio_codes 的训练 JSONL
qtts finetune prepare --project finetune.yaml

# 2) 调用官方 sft_12hz.py；成功后校验 checkpoint、原子发布输出目录
#    并把选中的 checkpoint 注册为 voice profile
qtts finetune train --project finetune.yaml --register alice

# 3) 用保留文本评测：CER（本地 ASR）与说话人相似度（Base 模型）
qtts finetune eval --project finetune.yaml --voice alice \
  --asr-model Qwen3-ASR-0.6B --max-cer 0.1

# 只校验/规划，不启动子进程、不加载模型、不取锁、不写文件
qtts finetune train --project finetune.yaml --dry-run --json
```

项目文件为严格 YAML/JSON（`schema_version: 1`，未知/重复键、alias、多文档、NUL 与超限输入均退出 4），包含 `dataset`（参考音频与逐条文本）、可选 `prepare`、`train`（官方仓库目录、初始化模型、输出目录、speaker 名）、`register`（目标 profile 名与 `consent.rights_confirmed`）与 `eval`（文本条目、可选 ASR/Base 模型与阈值）。所有路径展开 `~` 并相对项目文件解析；输出目录不得包含或位于数据集、官方仓库、模型、JSONL、`data-dir` 或目标 profile 内，参与发布的路径必须两两不同。

`prepare` 先校验每条音频为 24 kHz 单声道 WAV、时长 0.5–30 s（参考音频 2–30 s）、文本非空且不超过 2000 字符，写出确定性的 `<project>.raw.jsonl`，再以 `--device`（默认 `auto`，仅作为元数据传递，不在进程内解析）调用 `prepare_data.py`；输出写入 `<output>.tmp`，校验每行 16 列 `audio_codes` 后再原子替换。`train` 以固定 argv 调用 `sft_12hz.py`，子进程输出实时转发到 stderr（`--quiet` 抑制），非零退出保留 `<output_dir>.partial` 并退出 3。校验通过的 checkpoint 目录为 `checkpoint-epoch-<N>`（规范十进制），必须包含 `config.json`、非空 `model.safetensors`、`speech_tokenizer/`，且 `catalog.load_model` 识别为 `custom_voice` builtin、包含目标 speaker。发布使用 `.old` 备份与延迟信号处理器：提交点前失败/中断回滚旧目录并保留 `.partial`，提交点后保留新目录并以 130 退出。注册只写 profile 元数据（权重/config 哈希、数据集路径哈希、训练参数与 consent），不复制、移动或链接 checkpoint；`--register-only` 只为已有输出注册或重新注册。

`eval` 从项目/epoch、`--checkpoint` 或 `--voice` 三选一解析 checkpoint；`--voice` 走与 `say` 相同的 profile 完整性检查（权重 size 与 config sha256）并使用 profile 的 speaker，不能与 `--checkpoint` 同用。逐条按 item → eval → dataset 语言优先级解析语言，复用单个 TTS engine，生成后立即释放并清缓存，再按需加载 ASR/Base；音频写到 `<audio-dir>/<index:04d>.wav`（原子替换）。`metrics` 的 CER 与相似度键集严格按是否请求出现，聚合不可用时为 `null`；任一阈值失败输出完整 payload 并退出 5。`--dry-run` 只解析并报告 checkpoint/speaker/模型/阈值。

并发保护：每次非 dry-run 调用按固定顺序取项目锁 `.<project>.finetune.lock` 与 artifact 锁（`<output>.lock`、`<jsonl>.lock` 或 checkpoint 父目录），均为非阻塞 `flock`；冲突退出 4，锁文件保留。SIGINT/SIGTERM 首次转发给子进程组，第二次恢复默认行为；提交阶段信号延迟到 rename/回滚完成。`prepare`/`train` 成功时 stdout 只写一个 JSON（`--json`），所有进度与子进程输出走 stderr；中断返回 130 且 stdout 为空。

---

## HTTP 服务

`qtts serve` 启动一个只用 Python 标准库实现的本地 TTS 服务，同时提供原生 JSON API 和 OpenAI 兼容端点。导入 `qtts_cli.serve` 不会加载 torch/qwen_tts/dashscope/vllm，启动阶段也只解析后端元数据，不加载权重。

### 启动

```bash
# 只监听回环地址，端口由系统分配；stdout 恰好输出一行 compact JSON
qtts serve --port 0 --json

# 固定端口，同时启用两套 API（默认 both）
qtts serve --host 127.0.0.1 --port 8000

# 只解析后端与模型，不绑定端口、不加载模型
qtts serve --dry-run --json
```

非 `--json` 启动行形如：

```text
qtts serve listening on http://127.0.0.1:8000 (api=both, backend=transformers)
```

`--port 0` 时启动行给出实际端口。`--json` 启动对象包含 `host/port/api/backend/allowed_hosts/max_concurrency/effective_max_concurrency/queue_size/max_connections`。本地 backend 的 `effective_max_concurrency` 恒为 1，即使传入更大的 `--max-concurrency`；远程 backend 使用该值作为 worker 数。

### 原生 API

```bash
# 提交作业（默认格式 wav，可请求 srt/vtt/ass/json 字幕）
curl -sS http://127.0.0.1:8000/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，世界。","voice":"builtin:Vivian","subtitle":["srt"]}'

# 轮询状态，再下载音频与字幕
curl -sS http://127.0.0.1:8000/v1/jobs/job_…
curl -sS -o hello.wav http://127.0.0.1:8000/v1/jobs/job_…/audio
curl -sS http://127.0.0.1:8000/v1/jobs/job_…/subtitles/srt

# 列表、过滤与取消
curl -sS 'http://127.0.0.1:8000/v1/jobs?status=queued,running&limit=20'
curl -sS -X DELETE http://127.0.0.1:8000/v1/jobs/job_…
```

状态为 `queued/running/cancel_requested/done/failed/cancelled`；内部等锁状态对 API 显示为 `queued`，`/health` 的 `queued` 计数包含它，`running` 不包含。`DELETE` 对排队作业立即返回 200 并置 `cancelled`，对运行中作业返回 202 并置 `cancel_requested`，终态返回 409。音频/字幕下载对未完成作业返回 409，对失败/取消/已清理作业返回 404。

请求校验在创建作业前完成：未知字段、错误类型/范围、非法 JSON、重复 key、`NaN`/`Infinity`、错误的 `Content-Length`/`Transfer-Encoding`/`Content-Type` 都会返回 4xx。`text` 上限 200000 字符，`instruct` 上限 20000，`metadata` 规范化后上限 4096 字节且顶层最多 64 键，请求体上限由 `--max-body-bytes` 控制（默认 1 MiB）。`queue_size > 0` 时队列满返回 429 + `Retry-After: 1`；`queue_size == 0` 时只有空闲 worker 才直接接收，否则同样 429。

### OpenAI 兼容 API

```bash
curl -sS http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Hello world","voice":"alloy","response_format":"wav"}' \
  -o speech.wav

curl -sS http://127.0.0.1:8000/v1/models
```

`/v1/audio/speech` 同步等待作业结束并一次性返回完整音频，支持 `wav`/`flac`。OpenAI voice 别名（`alloy/echo/fable/onyx/nova/shimmer/ash/sage/coral/ballad/verse/marin`）映射到内置音色；带 `builtin:`/`voice:`/`design:` 前缀的值原样使用，其他值按 `builtin:NAME` 解释。`speed` 只接受缺省或 `1/1.0`，`stream_format` 只接受缺省或 `"audio"`；`mp3/opus/aac/pcm`、`speed != 1`、`sse` 和未知字段返回 400。`/v1/models` 对本地 backend 返回全部模型，对远程 backend 返回当前 `--backend-model`（或默认模型）。

### Token 与网络安全

- 默认只绑定 `127.0.0.1`。绑定非 loopback 地址（含 `0.0.0.0`/`::`）时必须提供 token，否则退出 2；token 可用 `--token` 或 `QTTS_SERVE_TOKEN`，空字符串视为非法。
- 配置 token 后，除“loopback 绑定下的 `/health`”外的所有请求都需要 `Authorization: Bearer <token>`；非 loopback 绑定下 `/health` 同样鉴权。校验使用常量时间比较，token 不会出现在日志、响应或作业记录中。
- 所有请求都校验 `Host`。允许集合为 loopback、显式绑定地址（通配地址除外）与全部 `--allowed-host`；通配绑定必须至少给一个 `--allowed-host`。`--allowed-host` 不能带端口或 scheme。
- `--max-connections` 限制并发 HTTP 连接，超限的新连接立即收到 503；`--keep-jobs` 控制每个终态 `done/failed/cancelled` 作业目录的保留数量，`0` 表示不清理。
- 服务端生成的错误、指标、警告、日志与 traceback 都经过统一脱敏（绝对路径、token），不会回传 traceback；客户端提交的 `text/voice/metadata` 原样回显。

### 取消与关闭

`say.run(cancel_check=...)` 在预检后、模型加载前、每个分段开始时、写字幕前、写音频前与写 recipe 前检查取消。请求取消后，运行中作业在下一个检查点抛出内部 `CancelledError` 并置 `cancelled`；如果作业恰好先完成，则按 `done` 结束。等待本地模型锁的作业可以在拿到锁前后被原子取消，绝不会短暂进入 `running`。

收到 SIGINT/SIGTERM 后，服务停止接受新连接，取消全部排队与等锁作业，等待运行中作业最多 `--shutdown-timeout` 秒。超时仍未返回的作业会以 `cancelled` 强制发布（只含 `request.json`、`result.json` 与脱敏 `run.log`，不含音频/字幕），worker 线程是 daemon，因此进程可以立即退出；第二次 SIGINT 立即以 130 退出。作业产物位于：

```text
<data-dir>/serve/staging/<job_id>/      # 生成期间
<data-dir>/serve/jobs/<job_id>/         # 终态：request.json、result.json、
                                        # audio.<fmt>、字幕、run.log（仅失败/取消）
```

每次启动都会清空残留的 `staging/` 并按 mtime 清理超出保留数量的孤儿终态目录；清理在后台线程执行，不阻塞监听。

---

## 命令参考

全局参数适用于所有子命令：

```text
qtts [--config FILE] [--data-dir DIR] [--model-dir DIR] [--cache-dir DIR]
     [--device auto|cpu|mps|cuda|cuda:0|cuda:1]
     [--dtype auto|bf16|fp16|fp32]
     [--seed N] [--offline] [--json] [--quiet] [--no-cache]
     [--log-level debug|info|warning|error]
     <command>
```

### `qtts say`

```text
qtts say [TEXT | - | --file FILE | --script FILE | --stdin --jsonl]
  [--voice builtin:NAME | voice:NAME | design:DESCRIPTION | --design FILE.yaml]
  [--lexicon FILE.json]
  [--language auto|Chinese|English|...]
  [--instruct TEXT]
  [--out FILE | -] [--out-dir DIR]
  [--format wav|flac]
  [--subtitle srt,vtt,ass,json]
  [--subtitle-out-prefix PREFIX | -]
  [--recipe-out FILE | -]
  [--granularity sentence|char|word]
  [--max-cue-chars N] [--max-line-width N] [--max-cps N]
  [--min-cue-seconds N] [--max-cue-seconds N] [--max-merge-gap-ms N]
  [--no-subtitle-qc]
  [--max-new-tokens N]
  [--pause-ms N]
  [--model MODEL]
  [--backend transformers|dashscope|vllm-omni|PLUGIN] [--backend-model ID]
  [--takes N --select cer,speaker-sim,lufs] [--asr-model PATH|ID]
  [--qc] [--reference-voice NAME]
  [--max-clipping-ratio N] [--target-lufs N --lufs-tolerance N]
  [--max-true-peak-dbtp N] [--max-dc-offset N] [--max-silence-ratio N]
  [--min-speech-rate N] [--max-speech-rate N]
  [--dry-run] [--yes]
```

要点：

- `--script` 接受 `.yaml/.yml/.json` 多角色脚本（见[多角色脚本](#7-多角色脚本)）；纯文本脚本仍按普通输入读取。脚本模式下 `--voice` 只作为 defaults 缺省值，`--model` 若与多模型脚本冲突会退出 2；与 TEXT/`--stdin`/`--file` 同时出现会退出 2；显式传入的 `--language/--instruct/--max-new-tokens/--pause-ms` 覆盖脚本 `defaults` 并体现在 plan/recipe 中。
- `--design FILE.yaml` 先严格解析并 lint 设计文件，再把编译后的 instruction 作为 `design:` 音色描述；与 `--voice`/`--instruct`/`--script`/`--jsonl` 互斥。recipe 额外记录 `design.design_yaml_sha256`、`rendered_instruction_sha256` 与 `compiler_version`。
- `--lexicon` 使用 `{"schema_version":1,"entries":[{"term","spoken"}]}`；最长匹配优先，匹配前做大小写/全角归一，重复 term 直接退出 4。TTS 读 `spoken`，字幕与 `join(cue.text)` 保持原文 `term`；缓存键包含词典内容哈希。
- `--instruct` 只支持 1.7B CustomVoice / VoiceDesign；Base、0.6B CustomVoice 收到会直接退出 4。
- `--language auto` 按段判定并逐段校验。
- `--granularity` 默认 `sentence`：初始一句一 span；超长句按次级标点切分，过短句仅在 `parent_ids` 完全相同时合并。`char` 逐字（卡拉 OK/精修），`word` 按英文词。未触发切分/合并时 cue 数等于句数。
- 字幕策略默认：单 cue 最大 42 显示宽度、单行最大 42、CPS 20、最短 0.8s、最长 7s、合并最大静音间隔 300ms。CPS = 非空白可见显示宽度 / 渲染秒数（毫秒先量化；CJK/全角/emoji 宽 2，组合字符宽 0）；恰等于上限不算违规。短 cue 合并与 CPS 平衡都不会跨过超过 `--max-merge-gap-ms` 的静音。
- 短句（如 `好。`）只告警；若其显示宽度不超过 `max_cps * min_cue_seconds`，即使当前时长导致 CPS 偏高，也只作为 warning。只有“可修复但仍超限”的格式/质量失败才退出 5。退出 5 时不写字幕文件，仍可写音频与配方，并在 `--json` 的 `details.subtitle_qc` 给出完整报告。
- `--no-subtitle-qc` 只关闭阈值判定；重叠、越界、非正时长、非单调仍会修复，无法修复则退出 4，配方记录 `enabled:false` 与修复计数。
- `--qc` 对最终音频（含首尾停顿）运行同一套 `audio_qc.analyze()`，指标写入 `--json` 的 `qc` 字段与 recipe；不指定阈值时只测量，越界时音频与报告仍完整写出并退出 5。`--reference-voice` 必须与 `--qc` 同时使用，且会额外加载 Base 模型计算说话人相似度；不带它时 `--qc` 不加载任何额外模型。
- `--qc --reference-voice` 在纯参数/输出组合校验之后、任何生成或写音频之前，依次做 profile 文件级校验、Base 模型解析与类级 embedding 能力探测（不加载权重）；profile 不存在、Base 缺失或运行时无该能力时不会产生音频或 recipe。`--dry-run` 只做 profile 文件级校验，不导入 torch/qwen_tts。
- `--takes N --select ...` 生成 2–16 个 take 并写入 `<data-dir>/takes/` 资产库（见[多 take 资产库](#12-多-take-资产库)）。两者必须同时出现；不能与 `--stdin --jsonl`、`--jsonl` 或 batch 清单同用，可与单文件 `--script` 和 `--dry-run` 同用。`--asr-model` 只用于 `--select cer` 选择 ASR 模型；`--reference-voice` 在 takes 模式下需要 `--qc` 或 `speaker-sim`，否则退出 2。
- `--stdin --jsonl` 每行一个请求：`{"id","text","voice"?,"out"?}`；每行结果 `{"id","status":"ok","output","recipe","metrics"}` 或 `{"id","status":"error","error":{"code","message","details"}}`。单条失败继续，全部成功退出 0、任一失败退出 6、空 stdin 或启动参数错误退出 2。`--jsonl` 不能与 `--json`、`--out`、`--recipe-out`、`--subtitle-out-prefix`、`--dry-run`、位置文本或 `--out-dir -` 组合；`out` 必须是文件路径，不能是 `-`。
- stdout 一次只能承载一个产物：`--out -`、`--json`、`--subtitle-out-prefix -`、`--recipe-out -` 互斥。
- `--out -` 时 stdout 是纯音频，摘要写 stderr。未指定 `--recipe-out` 时，recipe 写到 data-dir 的 `recipes/<request hash>.recipe.json`；hash 覆盖 voice identity、seed、sampling、language、instruct 与逐段 cache key，避免相同文本的不同请求互相覆盖。

### `qtts design`

```text
qtts design init FILE [--template T] [--name NAME]
  [--language zh|en|ja|ko|de|fr|ru|pt|es|it] [--force]
qtts design lint FILE [--show-rendered] [--strict] [--json]
qtts design explore FILE --out RUN_DIR [--resume] [--wide]
  [--seeds 19,97] [--sampling default|stable|expressive]...
  [--max-new-tokens N] [--retry-failed] [--yes] [--dry-run]
qtts design report RUN_DIR [--metrics basic|full]
  [--csv/--no-csv] [--html/--no-html] [--reel/--no-reel] [--gap 1.0]
  [--require-full-metrics] [--fail-on-degraded] [--json]
qtts design freeze RUN_DIR --take TAKE --name NAME --as clone|design
  [--confirm-transcript] [--ref-text TEXT] [--allow-long-ref]
  [--allow-incomplete] [--force] [--tag TAG]... [--i-have-rights]
qtts design prune RUN_DIR --keep N [--dry-run | --yes] [--json]
```

- 设计 YAML 是 btts 风格的结构化字段（`identity`/`delivery_baseline`/`scene`/`avoid`/`covers`）加 qtts 模板；`description` 与 `instruction_override` 是逃生口，优先级 `instruction_override > description > compiler`。YAML 拒绝 anchor/alias、重复 key、超过 256 KiB 与非 UTF-8。
- `lint` 离线确定性运行；解析错误退出 4，block 退出 1，`--strict` 下 warning 也退出 1。输出五维评分与 `green/yellow/red`；有 block 时强制 red。
- `explore` 默认 2 个 seed × 2 个采样 profile × 1 条 cover；`--wide` 叠加第二条 cover，候选上限 16（显式更大矩阵需 `--yes`）。每次候选写 WAV 与 recipe，`run.json` 是唯一索引；`--dry-run` 不加载模型、不写盘。run 锁在 POSIX 用 `fcntl.flock`、Windows 用 `msvcrt.locking`，冲突退出 2。
- `report` 只读 run 目录，按 `failure_count → completion 严重度 → clipping → silence → hnr → voiced_ratio → take_id` 排序，生成 `report.json`、`metrics.csv`、`report.md`、`audition.html`（盲听模式）与 `all_candidates.wav`。缺失指标为 `null` 且不计 failure；`--require-full-metrics` 或 `--fail-on-degraded` 失败退出 5。
- `freeze --as design` 保存可重生成的配方（YAML + rendered instruction + 哈希），`source.description` 直接可用；`--as clone` 以选中 take 为参考构建 ICL prompt，要求 `--confirm-transcript`，拒绝失败/疑似截断的候选，克隆还需 `--i-have-rights`。
- `prune` 按 report 排序保留前 `--keep` 个候选，其余标记为 `pruned`（`wav/recipe=null`，记录 sha256）；默认 dry-run，`--yes` 执行。两阶段移动 + 原子 manifest，崩溃后按 journal 恢复；pruned 候选在 report/CSV 中排最后且不参与 best 统计。`--yes` 与 explore 共用 run 锁；RUN_DIR 视为单写者目录，不应由其他用户/进程并发写入。symlink 防护是 best-effort（会校验 resolve 后的目标仍在 RUN_DIR 内），不宣称已消除跨进程 TOCTOU。
- 退出码：lint block 1、usage 2、模型 3、输入/解析 4、质量门禁 5、中断 130。

### `qtts script`

```text
qtts script lint FILE [--model MODEL] [--lexicon FILE.json]
qtts script compile FILE -o PLAN.json [--model MODEL] [--lexicon FILE.json]
  [--backend transformers|dashscope|vllm-omni|PLUGIN] [--backend-model ID]
```

- 脚本 schema：`schema_version: 1`；`defaults` 支持 `voice/language/format/seed/max_new_tokens/pause_ms/instruct`；`segments[]` 支持 `id/text/role/voice/language/instruct/pause_after_ms/seed/max_new_tokens`。
- 缺省 `id` 自动生成 `s001...`，重复 id、未知字段、字段类型/整数错误与顶层未知字段（包括未支持的 `project`）退出 2；YAML 额外拒绝重复 key、多文档、anchor/alias、超大文件，JSON 拒绝重复 key；缺 PyYAML 退出 3。
- 每段独立解析 voice/model 并在任何 `TtsEngine.load()` 前完成语言、instruct、ICL ref_text 能力校验。一个 source segment 产出多个 piece（多句、marker 分块或预算切分）时，全部 piece 使用稳定后缀 id（`s001.p1`、`s001.p2`），并保存 `source_segment_id`；单 piece 保留原 id。
- `compile` 输出按 key 排序、无时间戳、无 `cache`/绝对 `script_source` 的计划；同一环境重复编译字节稳定。`lint` 语法/schema 错退出 2，未知 voice/能力冲突退出 4。

### `qtts batch`

```text
qtts batch plan MANIFEST.jsonl --out-dir DIR [--model MODEL] [--lexicon FILE.json]
  [--backend ...] [--backend-model ID]
qtts batch run MANIFEST.jsonl --out-dir DIR
  [--jobs N] [--resume] [--retry N] [--fail-fast] [--watch]
  [--model MODEL] [--lexicon FILE.json]
  [--backend ...] [--backend-model ID]
```

- manifest 每行字段与 `say --stdin --jsonl` 对齐：`id`（唯一必填）、`text`、`voice`、`language`、`instruct`、`out`、`format`、`subtitle`、`seed`、`max_new_tokens`、`pause_ms`。默认输出为 `<out-dir>/<safe id>.<format>`。
- `plan` 对所有行先解析、校验、解析 voice/能力并编译，再写 `plan.json` 与初始全 `pending` 的 `status.jsonl`；不加载模型。坏 JSON/重复 key 退出 2，重复 id/未知 voice/能力冲突退出 4。`out` 只要有后缀就必须是 `.wav`/`.flac`，且与显式 `format` 一致。
- `run` 复用单条 `say` 执行器与错误结构；默认 `--jobs 1`。任务按模型/音色/语言/估算时长分桶排序，调度可并发，但模型推理由全局锁串行（MPS 同样只保留一个模型槽）。状态为 `pending/running/done/failed/skipped`，由单线程调度器原子重写；`job_key` 包含 `lexicon_sha256`。
- `--resume` 要求 `status=done`、`job_key` 相同、输出存在且 sha256 匹配、recipe 存在且其 `job_key` 匹配；`running` 恢复为 `pending`，`skipped` 下次重新调度。`--retry N` 只重试执行期 transient 失败（ModelError/OSError/RuntimeError），普通异常立即失败；status 记录 `attempts`、`last_error` 与 `transient`，`--resume --retry` 只重新调度 `transient=true` 的失败；`--fail-fast` 只停止调度新任务，未启动项标 `skipped`。
- 进度与 `--watch` 只写 stderr；stdout 只有最终 summary（`--json` 为单个对象）。任一任务失败退出 6。

### `qtts audiobook`

```text
qtts audiobook build PROJECT.yaml|PROJECT.json
  [--out FILE | --out-dir DIR]
  [--format wav|flac|m4a|m4b|mp3] [--m4b] [--bitrate KBPS]
  [--title TEXT] [--author TEXT] [--album TEXT]
  [--voice REF] [--model ID]
  [--bgm FILE] [--bgm-volume-db DB] [--ducking-db DB] [--bgm-fade-ms MS]
  [--lufs DB | --no-loudness] [--true-peak DB]
  [--paragraph-pause-ms MS] [--chapter-pause-ms MS]
  [--sample-rate HZ] [--keep-stems] [--force]
  [--dry-run] [--json] [--quiet]
```

- `--out` 的扩展名决定格式；无扩展名时按 `--format`（默认 wav）补齐。`--out` 与 `--out-dir` 互斥；`--out -` 退出 2。
- `--bitrate` 仅用于 m4a/m4b/mp3（32–320；AAC 默认 128，MP3 默认 192）。m4a/m4b/mp3 需要 ffmpeg 与对应编码器，缺失退出 3。
- 响度默认对 m4a/m4b/mp3 与含 BGM 的项目启用，wav/flac 默认关闭；`--lufs` 开启、`--no-loudness` 关闭，二者互斥。`--true-peak` 需要响度归一化。
- manifest 写到 `<out-dir>/<project-stem>.json`；输出已存在时需 `--force`。`--keep-stems` 保留 `segments/`、`chapters/` 与 `recipes/` 以便下次复用。
- 用法与参数冲突退出 2；项目/输入/BGM/路径/锁错误退出 4；模型、后端、ffmpeg 或编码失败退出 3；取消退出 130。

### `qtts dub`

```text
qtts dub build TIMELINE.srt|TIMELINE.edl
  --out FILE [--format wav|flac|m4a|mp3]
  [--voice REF] [--speaker-voice SPEAKER=REF ...] [--project FILE]
  [--language LANG] [--instruct TEXT]
  [--video FILE] [--bed-db DB] [--mux]
  [--stems-out DIR]
  [--fps N] [--sample-rate HZ]
  [--min-speed N] [--max-speed N] [--overflow warn|fail]
  [--lufs DB | --no-loudness] [--true-peak DB]
  [--model ID] [--keep-work] [--force] [--dry-run] [--json] [--quiet]
```

- `--out` 省略时为 `dub.wav`；无后缀按 `--format` 补齐；后缀与 `--format` 一致，`--out -` 退出 2。manifest 为 `<out-stem>.dub.json`。
- 时间轴必须是 `.srt` 或 `.edl`；EDL 用 `--fps`（1–120，默认 25）。对象、路径、时间轴与锁错误退出 4；参数冲突/超范围退出 2。
- 音色优先级：`--speaker-voice`（malformed 退出 2）→ 项目 `speakers` → `--voice` → 项目 `voice`；引用本身无效退出 4。未映射且无默认音色的说话人退出 4。
- `--language` 覆盖项目 `language`（默认 `auto`），作用于所有 cue；`--instruct` 覆盖项目顶层 `instruct`，但项目逐 speaker 的 `instruct` 仍优先。两者都参与合成与缓存指纹。
- `__anonymous__` 是保留 speaker 名，出现在时间轴中退出 4。已存在的 stems 目录可直接复用；已存在的 stem 文件或 `timeline.json` 需要 `--force` 才能覆盖。
- `--bed-db`（-40..0）必须与 `--video` 同时使用；`--mux` 也要求 `--video` 且视频含视频流。
- 采样率 12000–48000；`--min-speed` 与 `--max-speed` 均在 0.5–2.0 且必须 min < max；`--lufs` -40..-5、`--true-peak` -9..0。
- 默认响度开启；`--lufs`/`--no-loudness` 互斥。wav/flac 用 soundfile，m4a/mp3 需要 ffmpeg；任意 atempo 拉伸、mux 或 bed 也需要 ffmpeg，缺失退出 3。
- `--dry-run` 不加载模型、不建目录、不取锁，返回计划、cue 与解析后的音色/fingerprint（device 为 auto 时 fingerprint 为 null）。`--force` 覆盖已有输出并刷新缓存；`--keep-work` 保留 `<out-stem>.dub-work/`。
- 输出存在且未给 `--force` 退出 4；发布顺序、逐文件原子性与 SIGKILL 说明见[视频配音](#15-视频配音dub)。输出写入/回读哈希/manifest 写入失败退出 3，时间轴 block 结构非法退出 4。

### `qtts takes`

```text
qtts takes list JOB_ID [--json]
qtts takes compare JOB_ID [--ids 1,2,3] [--json]
qtts takes promote JOB_ID --id N (--out FILE | --out -) [--format wav|flac] [--json]
```

- `JOB_ID` 必须匹配 `[A-Za-z0-9._-]{1,64}`，只解析 `<data-dir>/takes/` 下的目录；绝对路径、`..` 穿越或符号链接退出 2，job 不存在退出 4。三个子命令都不加载任何模型。
- `list` 输出 job 状态、选择准则、每个 take 的 seed/rank/指标与 selected 标记；`--json` 直接输出原始 `job.json`。
- `compare` 默认比较全部 take，`--ids` 可限定并保持 job 内顺序；输出含指标、`unavailable` 原因与 `rank`。未知/重复 id 分别退出 4/2。
- `promote` 把指定 take 的原始字节复制到 `--out`，或写到 stdout（`--out -`，此时不能与 `--json` 同用）。promote 不转码：目标后缀必须与 job 格式一致，未知后缀或格式不符退出 2；不允许覆盖 `job.json`、任一 take 音频或 recipe（提升到自身是幂等 no-op）。成功后更新 `selected`/`selected_at` 并校验源文件 sha256。
- 资产库布局与 job 复用语义见[多 take 资产库](#12-多-take-资产库)；job/take 缺失、损坏、哈希不匹配或音频结构异常退出 4，评分/模型失败退出 3。

### `qtts check`

```text
qtts check AUDIO (--text TEXT | --text-file FILE | --srt FILE)
  [--language auto|Chinese|English|...]
  [--asr-model PATH|ID]
  [--metric auto|cer|wer] [--threshold 0.03]
  [--report FILE] [--json]
```

- 参考输入三者互斥且必须给一个，否则退出 2；AUDIO 不存在/不可读退出 4；音频超过 300 秒在模型加载前退出 4。
- `auto` 时中文/日文选择 CER，空格分词语言选择 WER；也可显式 `--metric`。CJK 的 CER 去掉所有空白后按字符比较。
- `--language` 大小写不敏感并 canonicalize 到 Qwen3-ASR 冻结语言表；非法语言在读取音频或选择模型之前退出 2。语言在全局表但所选 ASR 模型不支持时也在加载权重前退出 2；小写 `chinese`/`english` 与规范名产生相同的数字/日期/货币读法。
- ASR 模型按 `--asr-model`、本地路径或 `Qwen/Qwen3-ASR-0.6B` 这类 id 解析；缺失退出 3，多个同版本候选退出 2。
- 报告包含 `schema_version/ok/audio/metric/threshold/ratio/distance/reference_units/units/reference/hypothesis/errors/asr/comparison_language/normalization`。`asr.language` 保留 ASR 原始返回值（未检测到就是 `null`），`comparison_language` 是实际用于归一化与选 metric 的语言。`--json` 与 `--report` 逐字节一致，`audio.path` 仅 basename。
- `ratio > threshold` 时 `ok=false` 并退出 5；`ratio == threshold` 通过。Wave 3a 不启用时间戳，`errors[].start/end` 固定 `null`。

### `qtts qc`

```text
qtts qc AUDIO [--text TEXT|--text-file FILE] [--language auto]
  [--reference-voice NAME] [--model MODEL]
  [--max-clipping-ratio N] [--target-lufs N --lufs-tolerance N]
  [--max-true-peak-dbtp N] [--max-dc-offset N] [--max-silence-ratio N]
  [--min-speech-rate N] [--max-speech-rate N]
  [--report FILE] [--json]
```

- 只测量时退出 0；任何显式阈值越界时全部指标照常输出、`ok=false`、退出 5。
- 指标：`peak_dbfs`、`clipping_ratio/count`、`dc_offset`（逐声道最大 |mean|）、`silence_ratio`、`longest_silence_s`、`lufs_integrated`、`true_peak_dbtp`、`speech_rate`、`speaker_similarity`。
- 音频按原始声道读取；mono/stereo 支持，超过 2 声道退出 4。数字静音或能力缺失的指标为 `null`，原因写在 `unavailable`。
- 语速阈值需要非空 `--text`/`--text-file`（全空白等同于未提供，退出 2）；`--target-lufs` 与 `--lufs-tolerance` 必须成对出现。
- `--reference-voice` 必须是 clone profile，需要 Base 模型提供 `extract_speaker_embedding()`；该能力在加载权重前做类级探测。`--model` 路径不存在、非 clone 或无可用 Base 统一退出 3。

### `qtts post`

```text
qtts post INPUT [INPUT ...] (--out FILE | --out-dir DIR)
  [--lufs N] [--true-peak DBTP]
  [--sample-rate HZ] [--channels keep|mono|stereo]
  [--format wav|flac|mp3|m4a] [--bitrate 192k] [--bit-depth 16|24]
  [--trim-silence] [--trim-threshold-dbfs -50] [--trim-min-silence-s 0.2]
  [--trim-padding-ms 50] [--trim-mode leading|trailing|both]
  [--plugin NAME] [--plugin-param KEY=VALUE ...]
  [--dry-run] [--report FILE] [--json]
```

- 输入只支持 `.wav`/`.flac`（后缀大小写不敏感）；mp3/m4a 输入、无后缀、未知后缀或目录退出 2。`--out` 只接受一个输入；`--out` 与 `--out-dir` 互斥且必须给一个。
- 输出格式解析：`--out` 有已知后缀时 `--format` 若给出必须一致，否则用后缀；`--out` 无后缀时必须给 `--format` 并追加后缀；`--out` 未知后缀即使给了 `--format` 也退出 2；`--out-dir` 必须给 `--format`，文件名为 `<input stem>.<format>`。
- `--out -`、`--out-dir -`、`--report -` 和 `--dry-run --report` 均退出 2。输入/输出/report 在写文件前做规范化碰撞检查（`resolve(strict=False)`、NFC、大小写折叠，含 symlink），任一碰撞退出 2；多输入还校验规范化 basename 唯一。
- 全局错误（参数范围、路径/格式、ffmpeg/encoder 缺失）退出 2/3 且 0 写入；逐文件错误（解码、指标不可用、trim 后为空、DSP/编码/IO）写入 `files[].error`。全部成功退出 0；部分失败退出 6；全部失败退出输入顺序第一个错误码；单输入失败返回该错误码。
- `--lufs` 范围 [-70, 0]，`--true-peak` ≤ 0，`--sample-rate` 范围 [8000, 192000]；`--bit-depth` 只用于 wav/flac，`--bitrate` 只用于 mp3/m4a，trim 细项必须与 `--trim-silence` 同用，违反退出 2。静音且未请求响度/true-peak 时可透传；显式请求的指标不可用、或 trim 后为空时该文件退出 4。
- 报告 `schema_version` 为 1，顶层含 `request`（含 `true_peak_ceiling_scope: "encoded"`）与 `files[]`；每项含 `input_format/output_format/codec/encoder/dsp_version`、`before/after_dsp/after_encoded`、`applied`（`gain_db/limiter_reduction_db/safety_attenuation_db/limiter_algorithm/encode_retries/encoder_margin_db`）与稳定 `warnings`。`--json` 与 `--report` 逐字节一致；进程 `exit_code` 不写入 canonical 文档。
- `--dry-run` 仍做路径/碰撞/ffmpeg 预检和内存 DSP，但不创建目录、不编码、不写报告；`output` 为计划路径，`output_sha256` 与 `after_encoded` 为 `null`。`--report` 写失败退出 4，但已成功写出的音频保留，stdout/JSON 仍包含同一 `files[]` 并在顶层给出 `error`。
- `--plugin NAME` 在重采样之后、响度增益之前运行已安装的 `qtts.postprocess` 插件；`--plugin-param KEY=VALUE` 可重复，值先按 JSON 解析，`NaN/Infinity` 与重复 key 退出 2。插件必须返回 `(samples, sr)` 且保持声道数、采样率与有限样本，时长变化限制在 0.5x–2.0x，否则该文件退出 4 且不写输出；插件抛异常退出 3。应用信息写入 `applied.postprocess_plugin`。

### `qtts plugins`

```text
qtts plugins list [--strict] [--json]
```

- 只列出通过 entry point 发现的后端与后处理插件，不调用 `models()/create()/process()`，也不加载任何运行时。
- 每个条目含 `kind/name/distribution/version/status/reason`；`status` 为 `ok`、`unsupported_api`、`broken`、`invalid` 或 `duplicate_name`。发现阶段的异常只产生 warning，列表仍退出 0；`--strict` 在任何插件不可用时退出 3。

### `qtts bench`

```text
qtts bench --suite smoke|batch|longform
  [--model ID] [--voice builtin:NAME|voice:NAME]
  [--language auto|Chinese|English]
  [--text TEXT | --text-file FILE]
  [--count N] [--runs N]
  [--cer] [--asr-model ID]
  [--speaker-sim --reference-voice NAME]
  [--max-cer R] [--min-speaker-similarity S]
  [--dry-run] [--report FILE] [--json]
```

- `--suite` 必填。`--text`/`--text-file` 只用于 `longform`；`--count` 只用于 `batch`（默认 8，范围 1–64）；`--runs` 范围 1–10。`--voice` 只接受 `builtin:`/`voice:`，`design:` 退出 2；缺省为选中 builtin 模型的第一个 speaker。
- `--cer` 选本地 ASR（`--asr-model` 只能与它同用）；`--speaker-sim` 必须配 `--reference-voice`；`--max-cer`/`--min-speaker-similarity` 必须分别配 `--cer`/`--speaker-sim`。参数组合错误退出 2。
- 模型选择：显式 `--model` 未找到/不支持退出 3，同名歧义退出 2，`find_model` 不静默取第一个；未知 builtin speaker、voice profile/prompt/ref 缺失、voice 与 model kind 不兼容退出 4；显式 `--model` 为 Base/Design 且未给 `--voice` 退出 2；ASR 未找到 3、歧义 2。
- 指标：`load_ms`、`cold_first_segment_ms`、`warm_first_segment_ms`、`wall_s`、`generate_s`、`audio_s`、`rtf`、`wall_rtf`、`segments_per_run/total`、`segments_per_s`、`process_peak_rss_bytes`、`peak_rss_delta_bytes`、可选 `device_memory_mb/device_memory_kind`、`cer/cer_distance/cer_units`、`speaker_similarity/speaker_segments_scored/speaker_segments_total`。`cases[]` 含所有 run 的 id、文本 sha256、语言、instruct、seed、max_new_tokens 与计时。
- `--dry-run` 不加载模型、不 import torch/qwen_tts/qwen_asr、不写文件；`metrics` 全 `null`、`runs=[]`、`cases[].run=null`，并给出 `requirements`。`--report -` 与 `--dry-run --report` 退出 2。
- 门禁失败不抛异常：完整 payload 照常输出/写报告，`ok=false`、`gate_failures[]`、退出 5；指标为 `null` 且显式给门禁同样退出 5。报告写失败退出 4，已算出的 payload 仍输出。`--json` 与 `--report` 逐字节一致，canonical 不含 `exit_code`。

### `qtts finetune`

```text
qtts finetune prepare --project FILE
  [--tokenizer-model-path DIR] [--raw-jsonl FILE] [--output-jsonl FILE]
  [--device auto|CPU] [--force] [--dry-run] [--offline] [--quiet] [--json]

qtts finetune train --project FILE
  [--train-jsonl FILE] [--init-model-path DIR] [--output-dir DIR]
  [--finetuning-dir DIR] [--python CMD_OR_PATH] [--speaker-name NAME]
  [--batch-size N] [--lr FLOAT] [--epochs N] [--checkpoint last|N|DIR]
  [--register NAME] [--register-only] [--force] [--dry-run] [--offline] [--json]

qtts finetune eval --project FILE
  [--voice NAME | --checkpoint last|N|DIR]
  [--asr-model SPEC] [--base-model SPEC]
  [--max-cer R] [--min-speaker-similarity S] [--audio-dir DIR]
  [--dry-run] [--json]
```

- `--project` 必填且后缀为 `.yaml`/`.yml`/`.json`。所有数值选项按官方默认回退：batch size 2、lr `2e-05`、epochs 3；`--python` 缺省为当前解释器，含路径分隔符时解析为绝对路径，否则用 `shutil.which`。
- `--force` 仅 `prepare` 与 `train` 接受：覆盖输出 JSONL、删除 `.partial` 并替换已有输出目录；`eval` 没有 `--force`，生成的音频总是原子替换。
- `train --register-only` 不启动训练、不重命名输出，只为已存在的 checkpoint 写入 profile；没有 `register.name`/`--register` 时退出 2。
- 退出码：参数/组合错误 2；子进程/checkpoint/哈希/IO 失败 3；项目、路径、锁、选择器与 profile 冲突 4；评测门禁 5；中断 130。
- `--offline` 只向子进程环境加入 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`；`--dry-run` 不 import torch/qwen_tts/qwen_asr/dashscope/vllm_omni，不取锁、不建目录、不写文件。

### `qtts serve`

```text
qtts serve [--backend transformers|dashscope|vllm-omni|PLUGIN]
  [--backend-model ID] [--model NAME]
  [--api native|openai|both]
  [--host 127.0.0.1] [--port 8000] [--allowed-host HOST ...]
  [--max-concurrency 1] [--queue-size 32] [--max-connections 32]
  [--keep-jobs 100] [--max-body-bytes 1048576]
  [--token TOKEN] [--shutdown-timeout 10.0] [--dry-run]
```

- 数值越界、空 token、非 loopback/通配绑定缺少 token 或缺少 `--allowed-host`、`--allowed-host` 带端口/scheme、`--offline` 配远程 backend 均在绑定前退出 2；后端未配置退出 3；绑定失败退出 4。
- 启动只解析后端元数据；`--dry-run` 不 bind、不加载模型、不写数据目录，只输出解析结果并退出 0。成功启动向 stdout 写恰好一行；服务结束后 `cli.main` 不再追加 summary。
- 运行期单作业失败不会终止服务。请求日志逐行写入 stderr 的 JSON，包含时间、方法、无 query 的路径、状态、`job_id` 与耗时。

### `qtts voice`

```text
qtts voice add NAME --kind clone|design ...
qtts voice list
qtts voice inspect NAME
qtts voice preview NAME [--text TEXT] [--out FILE] [--format wav|flac]
qtts voice rm NAME [--yes]
qtts voice export NAME -o FILE.qttsvoice [--force] [--dry-run]
qtts voice export --all --out-dir DIR [--force] [--dry-run]
qtts voice import FILE.qttsvoice [--name NAME] [--force]
qtts voice import-bundle DIR [--force] [--dry-run]
qtts voice import-btts BUNDLE (--voice NAME ... | --all) [--name NAME]
  [--i-have-rights] [--design-mode auto|recipe|clone] [--dry-run]
  [--auto-ref-text | --ref-text TEXT] [--asr-model PATH|ID]
  [--mode icl|xvec] [--model NAME] [--force]
```

`.qttsvoice` 是 zip 容器，只允许 `manifest.json`、`voice.json`、`prompt.safetensors`、`ref.wav`、`preview.wav`；导入逐项校验路径、大小、压缩比、sha256 与 safetensors 结构，不使用 `extractall`，替换已有音色失败时回滚。

`voice export --all --out-dir DIR` 输出目录包：`bundle.json`（`format="qtts-voice-bundle"`、`bundle_schema_version=2`）列出每个单音色 `.qttsvoice`（archive schema 1）的 `name/kind/archive/sha256/size`；`voice import-bundle DIR` 校验全部归档后一次性发布，`--force` 替换、`--dry-run` 只校验。导出拒绝大小写冲突、非托管文件与 symlink 输出；导入拒绝 symlink、大小/哈希不符与大小写冲突。两侧都写 journal（`.bundle-export.json` / `.bundle-import.json`），崩溃后下一次运行自动回滚未提交事务或保留已提交结果。

#### 与 btts 互操作：`voice import-btts`

`qtts` 可以直接消费 `btts voice export --mode full` 的目录或 `.zip`：

```bash
# 先在本机用 btts 导出（目录或 --zip 均可）
btts voice export alice narrator -o ./btts-bundle
# 或：btts voice export --all --zip -o ./btts-bundle.zip

# 默认 auto：clone 音色用 bundle 里的 ref.wav + ref.txt（ICL）；
# design 音色离线导入 recipe（不加载模型、不跑 ASR）
qtts voice import-btts ./btts-bundle --all --i-have-rights

# 需要旧版 design -> clone 行为时显式选择
qtts voice import-btts ./btts-bundle --voice narrator \
  --i-have-rights --design-mode clone --auto-ref-text \
  --model-dir "$QTTS_MODEL_DIR"
# 或：--ref-text "preview.wav 对应的文字"（仅单音色）
# 或：--mode xvec（不使用参考文本，质量可能略降）

# 只看计划，不写 profile、不加载模型
qtts voice import-btts ./btts-bundle.zip --all --i-have-rights --dry-run
```

btts bundle 的布局是根目录 `manifest.json` + `LICENSE.txt` + `README.txt` + 每音色的 `profile.json`/`preview.wav`；clone 音色另有 `ref.wav`/`ref.txt`，design 音色另有 `design.yaml`。`--zip` 包可含单一顶层目录。

- **默认行为**：`--design-mode auto`（默认）。design 音色导入为 qtts `kind=design` recipe；clone 音色仍走 clone。显式传 `--auto-ref-text` / `--ref-text` / `--asr-model` / `--mode` / `--model` 时自动按 `clone` 处理，以兼容旧脚本；`--design-mode recipe|auto` 与这些 clone-only 选项互斥（退出 2）。需要旧行为就显式 `--design-mode clone`。
- **recipe**：直接使用 btts 实际回放的 `design.instruction`（为空时回退 `identity.rendered_instruction`），保留 `identity.structured`、模板、seed/cfg_scale/max_new_tokens 与内嵌 `design.yaml` 原文及哈希。qtts 不重放 btts 的 seed/cfg，也不承诺跨模型声学等价；输出带 `acoustic-equivalence-not-possible` warning。design 没有可用的 instruction 时退出 4。
- **模板**：6 个 btts 模板 ID 与 `btts/...` 都是输入别名，写入时 canonicalize 为 `qtts/...`；原始 btts ID 保留在 `source.design.template_id`。未知模板不猜测：`template=None`、保留 raw ID 并 warning。新增 `qtts/zh-documentary-narrator/v1`、`qtts/zh-anime-character/v1`、`qtts/en-audiobook-narrator/v1`、`qtts/en-game-character/v1`。
- **preview**：btts `preview.wav` 复制为 profile preview，`source.preview.origin="btts"`；`voice preview` 覆盖后更新为 `origin="qtts"` 与新 hash，原始 hash 仍保留在 `source.btts.preview_sha256`。
- **安全校验**：有 `manifest.json` 时逐文件校验 SHA-256 和大小；拒绝绝对路径、盘符、反斜杠、`..`、符号链接、特殊文件、大小写冲突、未知文件、重叠 zip 成员与超限成员。zip 读取期间的 `UserWarning` 会被提升为输入错误（退出 4），不会破坏 `--json` 的单错误对象契约。zip 必须带 manifest；目录 bundle 缺 manifest 时允许导入，但 JSON 与 `voice.json` 会显式记录 `manifest.present=false`/`verified=false` 并给出 warning。
- **clone**：ICL 使用 `reference.audio` + `reference.transcript`，走与 `voice add --kind clone` 相同的 prompt 生成与 staging；`--mode xvec` 可跳过参考文本。参考音频仍要通过准入检查。
- **同意**：必须显式 `--i-have-rights`。btts profile 的 `consent` 只作为 provenance 保留，不能替代 qtts 的同意记录；`voice.json` 的 `consent.license` 记为 `btts-import`，模型许可证另存于 `source.btts.license`。
- **provenance**：`source.origin="btts"`，`source.btts` 保留 name、voice_id、kind、language、tags、license、consent、model id/digest、profile sha256、manifest 摘要、reference/identity/design/defaults 元数据与文件哈希；超长字段截断并写入 `source.btts.truncations`（含长度与 sha256）。bundle 只记录 basename，不写绝对路径。
- **原子性**：所有选中音色先校验、staging，再通过 `.btts-import.json` journal + `.btts-import.lock` 统一发布；安装中途失败会逆序回滚已替换/新装 profile 并清理 staging，进程崩溃后下一次运行先恢复 journal。`--dry-run` 不写 store、不 scan 模型、不加载 TTS/ASR；存在 pending journal 时拒绝 dry-run。`--name` 只能用于单音色；`--all` 或多于一个 `--voice` 时使用 `--name` 退出 2。
- **限制**：只支持 `btts voice export --mode full`；`--mode wav` 导出没有 profile，会明确提示用 `--mode full` 重新导出。单个 bundle 最多 512 个成员（约 100 个音色），单文件上限 64 MiB、总大小 512 MiB、压缩比上限 500:1；`manifest.json` 与其他 `profile.json` 一样限制 1 MiB。不导入 btts 模型权重/检查点；btts 语言代码（`zh`/`en` 等）会映射为 qtts 显示名。全程不需要联网。

`add` 的 clone 选项：

```text
--ref-audio FILE (--ref-text TEXT | --auto-ref-text) [--mode icl|xvec]
[--language LANG] [--asr-model PATH|ID]
--i-have-rights --license TEXT --consent-note TEXT --tags a,b
```

`--auto-ref-text` 先用 Qwen3-ASR 转写参考音频，再把文本作为 ICL `ref_text`；与 `--ref-text` 同时给出或与 `--mode xvec` 组合退出 2。`--asr-model` 只能与 `--auto-ref-text` 一起使用，否则退出 2；它与 `check` 使用同一套解析/消歧规则（路径、id、目录名；同版本多个候选必须显式指定）。转写发生在 staging 目录内，ASR 模型在加载 TTS Base 前释放；ASR/模型缺失退出 3，转写为空退出 4，失败不会留下半成品 profile。`voice.json` 记录 `source.ref_text_source=asr` 与 `source.asr_model`。

`add` 的 design 选项：

```text
--description TEXT --sample-text TEXT [--language Chinese]
[--materialize-clone]
```

### `qtts align`

```text
qtts align AUDIO
  (--text TEXT | --text-file FILE | --srt FILE | --script FILE)
  [--lexicon FILE.json]
  [--language Chinese]
  [--granularity sentence|char|word]
  [--format srt,vtt,ass,json]
  [--out-prefix PREFIX]
  [--max-cue-chars N] [--max-line-width N] [--max-cps N]
  [--min-cue-seconds N] [--max-cue-seconds N] [--max-merge-gap-ms N]
  [--no-subtitle-qc]
  [--window-seconds 270] [--window-overlap 3]
  [--approximate] [--asr-model MODEL]
  [--recipe-out FILE | -]
```

- 默认 `--granularity sentence`，按句子出 cue；`char` / `word` 可显式选择。
- 音频 ≤300s 走原有单次对齐路径，不选择也不加载 ASR；报告里只有 `windowing: {"enabled": false}`。
- 音频 >300s 默认按静音切窗：每个 core 窗 240–285s（`--window-seconds`，默认 270），相邻音频窗保留至少 2s 重叠（`--window-overlap`，默认 3），单窗音频不超过 300s。每窗先用 Qwen3-ASR 无时间戳转写，把 ASR 文本单调锚定到原文，再逐窗调用 forced aligner；因此长路径需要 `Qwen3-ASR-*` 与 `Qwen3-ForcedAligner-*` 两个模型。最后按全局字符位置合并时间，只构建一次 cue。
- `--approximate` 跳过 ASR，按每个窗口**实际送入 aligner 的 audio 区间**（含左重叠）比例分配原文，相邻文本区间因此允许重叠；碰到词典保护项时吸附到 term 边界。静音切窗无解时退化为确定性固定间隔（`windowing.plan="fixed"`，并给出 warning），不再退出 4；逐字符归属仍优先 core，时间倒退会被确定性单调裁剪并报告调整次数。边界不保证，报告与 stderr 会给出 `long-audio boundaries are approximate; subtitle boundaries are not guaranteed`。无法避开词典项时退出 4。
- `--approximate` 不能与 `--srt` 同用：SRT cue 边界已是硬锚点，冲突在加载任何模型前退出 2。
- `--srt` 输入不会重新分组：输入 cue 边界优先于 TTS 计划段和原句边界，只重算时间；cue 文本保持逐字原文。长 SRT 的 cue 边界同时作为切窗硬锚点，不使用 ASR；cue 超过 285s 或边界不在合格静音内退出 4，窗口绝不切开 cue。
- `--lexicon` 时对齐器仍读 `spoken` 文本，时间轴再按字符映射回 display；`--text/--text-file/--srt` 都支持，SRT 模式下词典项不能跨越 cue 边界。词典只做一次全局 `project_text`，逐窗只切片，不逐窗重新投影。
- 长路径使用 aligner 原始逐字符 item 计算真实覆盖率，写入 `windowing.matched_ratio` 与 recipe `coverage`，并复用短路径的 30% 拒绝/80% warning 门禁；末窗允许最多 `max(--window-overlap, 1s)` 的有界尾部溢出并裁剪到音频时长。
- 字幕策略与质量门禁和 `say` 相同；`--srt` 输入不做 CPS/行宽等格式强校验，只做结构检查。
- 无法修复的结构问题（重叠、越界、非正时长、非单调）退出 4；`--no-subtitle-qc` 下可修复的仍然修复并记录。

### `qtts project`

```text
qtts project init [DIR] [--force]
  --voice REF [--language LANG] [--format wav|flac] [--max-new-tokens N]
  [--subtitle srt,vtt,ass,json] [--granularity sentence|char|word]
  [--max-cue-chars N] [--max-line-width N] [--max-cps N]
  [--min-cue-seconds N] [--max-cue-seconds N] [--max-merge-gap-ms N]
  [--no-subtitle-qc]

qtts project lock [--project FILE] [--out FILE]
```

- `init` 写 `qtts.project.json`（默认当前目录），已存在时需 `--force`。
- `lock` 解析项目中的相对 `model_dir`（相对项目文件所在目录），为所有引用模型和音色记录**全量文件 sha256 与大小**；同时记录有效采样参数、seed、字幕策略、dtype/device、Python 与 `qwen-tts`/`qwen-asr`/`transformers`/`torch` 版本。`qwen-asr` 缺失时记为 `available:false`；需要字幕对齐但本机没有对齐器会退出 3。
- 锁文件为排序后的 JSON，不含时间戳；所有校验和哈希在原子写入前完成。

### `qtts models`

```text
qtts models list [--model-dir DIR]
qtts models pull MODEL_ID [--source hf|modelscope] [--revision REV]
qtts models prune [--yes]
```

### `qtts doctor`

```text
qtts doctor [--bench] [--json]
```

检查 Python/torch/设备、ffmpeg/SoX、磁盘、模型完整性、依赖版本组合与钉版告警、能力矩阵。`--bench` 会真实加载模型并输出 RTF、加载耗时和峰值 RSS。

### `qtts completion`

```text
qtts completion [auto|bash|zsh|fish] [--install|--uninstall|--status]
                [--dir DIRECTORY] [--rc-file PATH] [--follow-symlinks]
                [--dry-run] [--json] [--quiet]
```

为 bash、zsh、fish 生成自适应补全脚本；不带动作时把脚本写到 stdout，可直接 `eval` 或重定向。`--install` 幂等写入独立补全文件，只有 shell 不能自动加载该目录时才在 rc 文件里加一个很小的 marker block；`--uninstall` 删除补全文件和该 block，`--status` 报告 `current`/`stale`/`not_installed`/`malformed`/`unsupported_version`/`symlink_refused`（非 `current` 退出 4）。

```bash
# 当前 shell 临时启用
eval "$(qtts completion bash)"

# 永久安装（默认检测 $SHELL / 父进程 / 平台）
qtts completion --install

# 指定 shell / 自定义 rc / 只写补全文件 / 先预览
qtts completion zsh --install
qtts completion bash --install --rc-file ~/.bashrc
qtts completion zsh --install --dir ~/.zfunc
qtts completion bash --install --dry-run --rc-file /tmp/qtts.bashrc

# 查询与卸载
qtts completion --status --json
qtts completion --uninstall
```

- 自动检测顺序：显式 SHELL 参数 > `QTTS_COMPLETION_SHELL` > `$SHELL` > 父进程链 > 平台回退（macOS zsh、Linux bash）。无法确定时退出 2，不会猜成 bash。
- **zsh**：优先写入 Oh My Zsh 的 `custom/completions/_qtts`，或 `.zshrc`/`.zshenv`/`.zprofile`/`.zlogin` 中已有且可写的 `fpath` 目录；都没有时才写 `~/.zfunc/_qtts`，并在 rc 里追加三行 `fpath` + `compinit` block，而不是把整个补全脚本塞进 rc。
- **bash**：优先写入 `BASH_COMPLETION_USER_DIR/completions/qtts` 或 `~/.local/share/bash-completion/completions/qtts`；未安装 bash-completion 时写 `~/.local/share/qtts/completion.bash` 并在 rc 里追加一行 `source` block。macOS 的 Terminal 可能以 login shell 读取 `~/.bash_profile`；若它不 source `~/.bashrc`，请用 `--rc-file ~/.bash_profile`。
- **fish**：写入 `${XDG_CONFIG_HOME:-~/.config}/fish/completions/qtts.fish`，fish 会自动加载，不修改 rc。
- `--dir` 只写补全文件、不碰 shell 配置，适合 dotfiles 管理。
- 默认拒绝符号链接；dotfiles 管理场景显式加 `--follow-symlinks` 写入其 target。marker 不完整/重复时退出 4 且零写入。重复 `--install` 不改动任何字节；升级时会替换旧 marker block 并清理旧布局文件。
- 补全由隐藏命令 `qtts __complete` 驱动，始终读取实时 argparse 树；TAB 时只读扫描本地音色/模型元数据，不加载模型、不访问网络、不写缓存。已知限制：PowerShell 尚未实现；`--opt=value` 形式的文件路径不作原生补全（改用 `--opt value`）。

---

## 后端与插件

`say`、`script compile`、`batch plan/run` 都接受 `--backend` 与 `--backend-model`。默认后端是本地 `transformers`；后端身份（`backend.identity`）会写入 cache key、`batch` 的 `job_key`、recipe 与 JSON payload，因此同一模型/音色在不同后端之间不会错误复用缓存。

| 后端 | `--backend-model` | 能力 | 运行时 |
|---|---|---|---|
| `transformers`（默认） | 不接受 | builtin / clone / design、instruct、全部 sampling | 本地 `qwen-tts` |
| `dashscope` | 远程 model id，默认 `qwen3-tts-flash-realtime` | 仅 `builtin:`；不转发 instruct；无 sampling | `dashscope>=2.0.1` |
| `vllm-omni` | 不接受（用 `--model` 选本地模型） | builtin / clone / design；clone 只需 `ref.wav` | Linux + GPU 的 `vllm-omni`，离线 batch |
| 插件 | 插件自定义 | 由插件 `capabilities()` 声明 | 插件自行管理 |

- `--model` 与 `--backend-model` 互斥；`transformers`/`vllm-omni` 拒绝 `--backend-model`，`dashscope` 与插件拒绝 `--model`，违反退出 2。`Config` 也支持文件级 `backend`/`backend_model`，CLI 显式值优先。
- `dashscope` 读取 `DASHSCOPE_API_KEY`、可选的 `DASHSCOPE_WORKSPACE` 与 `DASHSCOPE_REGION`（默认 `cn-beijing`）。key 永不出现在 identity/报告；`--dry-run` 不要求 key 并输出 `backend.configured=false`，真实生成缺 key 退出 3，与 `--offline` 同用退出 2。`voice:`/`design:`/`--instruct` 在创建 client 前退出 4。`--takes`、`--qc --reference-voice` 与 `--select speaker-sim` 需要本地 `transformers` 后端，否则退出 2。
- `vllm-omni` clone 从 `voice.json` 的 `source.ref_audio` 读取参考音频；路径 resolve 后必须仍在音色目录内，缺失/越界退出 4。ICL 模式还要求非空 `ref_text`。模块导入、`--dry-run` 与 `plugins list` 都不会 import `dashscope`/`vllm_omni`/`torch`/`qwen_tts`。
- 本地 aligner、speaker-similarity QC 与 takes 始终使用 `--model-dir` 扫描到的本地模型，与 TTS 后端无关。远程后端的 payload/recipe 中 `device="remote"`、`dtype=null`。
- `batch --resume` 会先读取已有 `plan.json`；`backend_identity` 与本次不同（旧 plan 无该字段视为 `transformers`）时在 `mkdir`/写文件前退出 2。

### 插件

插件只通过 Python entry point 发现，安装插件等于允许其代码在 qtts 进程中执行；qtts 自身不会主动安装或导入第三方运行时。API 版本必须为整数 `1`，名称必须是小写 `^[a-z][a-z0-9_-]{0,31}$`。

```toml
# pyproject.toml
[project.entry-points."qtts.backends"]
demo = "demo_qtts.backend:DemoBackend"

[project.entry-points."qtts.postprocess"]
denoise = "demo_qtts.postprocess:Denoise"
```

```python
# demo_qtts/backend.py
from pathlib import Path
from qtts_cli.backends.base import BackendCapabilities
from qtts_cli.catalog import ModelInfo

class DemoBackend:
    api_version = 1
    name = "demo"
    def capabilities(self):
        return BackendCapabilities(
            modes=frozenset({"builtin"}), instruct_modes=frozenset(),
            sampling=frozenset(), remote=True,
        )
    def models(self, cfg, backend_model=None):
        return [ModelInfo(path=Path("<demo:model>"), name="demo-model",
                          model_type="qwen3_tts", kind="builtin",
                          speakers=["Vivian"])]
    def create(self, cfg, model):
        return DemoBackendInstance(cfg, model)  # load/generate/release
    # 可选: preflight(plan, cfg, args)
```

```python
# demo_qtts/postprocess.py
import numpy as np

class Denoise:
    api_version = 1
    name = "denoise"
    def process(self, samples, sr, params):
        return np.asarray(samples, dtype=np.float32), sr
```

通过 `--backend demo --backend-model custom` 使用后端插件；后处理插件用 `post ... --plugin denoise --plugin-param strength=0.5`。后端插件 `models(cfg, backend_model=None)` 收到 `--backend-model` 原值；后处理插件在固定的 DSP 顺序中运行，返回结果会被严格校验。

## 音色引用

| 形式 | 路由 | 说明 |
|---|---|---|
| `builtin:Vivian` | CustomVoice | 9 个内置音色，可用 `qtts voice list` 查看 |
| `voice:alice` | Base + clone prompt | 本地音色库中的克隆/固化音色 |
| `design:低沉沙哑的中年男声` | VoiceDesign | 即时生成，不落盘；可用 `voice add --kind design --materialize-clone` 固化 |

音色库位置：`<data-dir>/voices/<name>/`

```text
voice.json          # 元数据
prompt.safetensors  # clone prompt（不使用 pickle）
ref.wav             # 参考音频
preview.wav         # 试音
```

---

## 输出与数据目录

默认数据目录由 `platformdirs` 决定：

- macOS: `~/Library/Application Support/qtts`
- Linux: `~/.local/share/qtts`

可用 `--data-dir` 覆盖。目录结构：

```text
<data-dir>/
├── voices/<name>/
├── cache/
└── recipes/          # --out - 时的配方落点
```

项目文件由 `project init` 写到指定目录（默认当前目录）：`qtts.project.json`；`project lock` 默认在同目录写 `qtts.lock.json`。

生成文件：

| 文件 | 说明 |
|---|---|
| `<out>.wav` / `.flac` | 音频 |
| `<out_stem>.recipe.json` | 可复现配方（`a.wav` → `a.recipe.json`） |
| `<prefix>.srt/.vtt/.ass/.json` | 字幕（按 `--subtitle` 选择） |

---

## 配置文件

`--config FILE` 接受 JSON，命令行参数优先：

```json
{
  "data_dir": "/path/to/data",
  "model_dir": "/path/to/models",
  "cache_dir": "/path/to/cache",
  "device": "auto",
  "dtype": "auto",
  "offline": false,
  "seed": 1234,
  "no_cache": false
}
```

---

## 退出码

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 2 | 参数/用法错误 |
| 3 | 模型或后端错误（加载、显存、依赖） |
| 4 | 输入校验失败（音频不合格、能力不支持、对齐覆盖率过低、takes job/take 缺失或损坏） |
| 5 | 质量门禁未通过（`check` 错误率超阈值、`qc`/`say --qc` 显式阈值越界、字幕 QC） |
| 6 | 批处理部分失败（`say --stdin --jsonl`、`batch run`） |
| 130 | 用户中断 |

---

## 开发与测试

```bash
# 单元测试：不需要 GPU / 模型 / torch
PYTHONPATH=. pytest tests -q

# 集成测试：需要本地模型与 MPS/CUDA（含 Qwen3-ASR-0.6B）
QTTS_SMOKE_MODEL_DIR=/path/to/models RUN_INTEGRATION=1 PYTHONPATH=. \
  pytest tests/test_integration_smoke.py tests/test_integration_wave3a.py \
    tests/test_integration_wave3c.py -q

# lint
ruff check qtts_cli tests
```

项目结构：

```text
qtts_cli/
├── cli.py                 # argparse、全局参数、退出码
├── config.py              # 目录/设备/配置解析
├── catalog.py             # 模型与 9 个内置音色枚举、fingerprint
├── capabilities.py        # 能力矩阵与路由校验
├── textnorm.py            # 文本规范化
├── segmenter.py           # 语义分段与预算
├── budget.py              # token/时长估算
├── plan.py                # GenerationPlan IR、逐段 voice/预算、语言与 prompt 预检
├── lexicon.py             # 词典投影、导演标记、spoken→display 映射
├── script.py              # 多角色 YAML/JSON 校验、编译与稳定计划
├── batch.py               # JSONL 清单、plan/status、resume/retry/调度
├── cache.py               # 缓存键与命中
├── voice_store.py         # 音色资产与原子提交
├── admit.py               # 参考音频准入检查
├── tts_engine.py          # qwen-tts 懒加载
├── align_engine.py        # qwen-asr ForcedAligner 懒加载
├── subtitles.py           # cue 分段/合并、时间轴清洗与 SRT/VTT/ASS/JSON
├── subtitle_qc.py         # CPS/行宽/时长/结构门禁与 QC 报告
├── sidecar.py             # recipe 读写
├── voice_archive.py       # .qttsvoice 导出/导入与安全校验
├── voice_btts.py          # btts voice export bundle 解析与安全校验（FR-P2-09）
├── voice_import_journal.py # btts 多音色安装 journal、锁与崩溃恢复
├── project.py             # project init/lock
├── postprocess.py         # 响度/true-peak/重采样/静音修剪纯 DSP
├── benchmark.py           # bench 语料、计时聚合、CER/speaker 评分
├── doctor.py              # 环境诊断与 doctor --bench
├── models_ops.py          # models list/pull/prune
└── commands/              # say / script / batch / voice / align / project / models / doctor / post / bench

tests/
├── test_capabilities.py
├── test_text_pipeline.py
├── test_subtitles.py
├── test_store_cache.py
├── test_cli.py
├── test_p1_subtitles.py   # P1 字幕分段与门禁
├── test_p1_jsonl.py       # P1 JSONL stdin
├── test_p1_script.py      # P1 Wave 2 多角色脚本
├── test_p1_lexicon.py     # P1 Wave 2 词典/导演标记
├── test_p1_batch.py       # P1 Wave 2 批量清单
├── test_p1_voice_archive.py
├── test_voice_import_btts.py  # btts 导入、安全校验与原子性
├── test_voice_import_journal.py  # journal 崩溃窗口与恢复
├── test_p1_project.py
├── test_review_p0.py      # 审查回归
├── test_review_p1.py      # 审查回归
└── test_integration_smoke.py
```

---

## 设计约束

- **没有本地 token 级流式**：`non_streaming_mode` 只模拟流式文本输入，API 返回完整音频。需要低延迟请走 DashScope Realtime。
- **模型能力差异**：VoiceDesign 仅 1.7B；Base 不支持 instruct；0.6B CustomVoice 的 instruct 会被静默丢弃，因此 CLI 在参数层直接报错。
- **对齐器单样本约 5 分钟**：`say` 按段对齐，限制按段计算；`align` 对 >300s 的音频使用静音切窗、ASR 文本锚点与全局字符归属合并，单窗仍不超过 300s。
- **长音频默认需要 ASR**：`align` 的默认长路径先无时间戳转写再逐窗对齐；`--srt` 用 cue 边界硬锚定、`--approximate` 按比例降级，这两条路径不需要 ASR，但 approximate 的边界不保证。
- **字幕三级边界**：`--srt` 输入 cue > TTS 计划段 > 原句。只有 `parent_ids` 完全一致时才合并；`--srt` 永不重新分组。切分/合并后仍逐字保留原文，`join(cue.text) == normalized_text`。
- **硬切是最后手段**：只有找不到自然断点时才硬切，并记录策略/告警；引号内、数字和拉丁单词内不硬切。
- **投影顺序固定**：原文 → 抽取导演标记 → 词典保护的 display 投影 → spoken 投影 → 分句/预算。词典匹配发生在数字/日期规范化之前，映射是显式字符区间，禁止事后 `find()` 二次定位；`join(cue.text) == display_text`。
- **单模型槽**：脚本/批量即便跨模型也只有一个 TTS engine 驻留；切换前释放旧 model 与 clone prompt。`--jobs>1` 只并发调度，推理仍全局串行。
- **MPS 不保证 bit-perfect**：同 seed 可复现趋势，但跨设备/后端结果会漂移。
- **依赖钉版冲突**：见[环境要求](#环境要求)；M0 采用单环境混装，桥接方案保留为 fallback。

---

## 路线图

### M0（已完成）

`say`、`voice add/list/inspect/preview/rm`、`align`、`models list/pull/prune`、`doctor [--bench]`、`--dry-run`、sidecar、缓存、能力矩阵、字幕四格式。

### P1 Wave 1（已提交）

- 字幕 cue 分段策略（超长句切分 / 短句合并且不跨段）与 CPS / 行宽 / 时长门禁
- `say --stdin --jsonl`（流内模型复用、逐条失败隔离、退出码 0/2/6）
- `voice export/import`（`.qttsvoice` 安全归档）
- `project init/lock`（全量模型 sha256、音色与运行时锁定）

### P1 Wave 2（已提交）

- 多角色脚本 `script lint/compile` 与 `say --script`（逐段音色/模型/语言/instruct/seed）
- 发音词典 `--lexicon` 与 `[[pause:MS]]` 导演标记（spoken/display 映射）
- JSONL 批处理 `batch plan/run`（确定性计划、原子状态、`--jobs/--resume/--retry/--fail-fast`）

### P1 Wave 3（已提交）

- 音频 QC 与 ASR `check`（CER/WER、自动 `ref_text`、speaker similarity）
- 长音频对齐分窗、静音切窗与近似文本映射
- 多 take 资产库 `--takes/--select` 与 `takes list/compare/promote`

FR-P1-01 至 FR-P1-11 均已实现并在本地提交；P1 期间版本号保持 0.1.x。

### M2

- 音频后处理 `post`（FR-P2-05，已提交：响度归一化、true-peak 限幅、重采样、声道转换、静音修剪与编码）
- 可复现性能基准 `bench`（FR-P2-06，已提交：smoke/batch/longform、CER、speaker-sim、dry-run 与 canonical 报告）
- 可插拔后端（FR-P2-02，已提交：transformers / DashScope / vLLM-Omni batch）
- 插件 SDK（FR-P2-08，已提交：`qtts.backends` / `qtts.postprocess` entry points）
- `serve` 与 OpenAI 兼容 `/v1/audio/speech`（已提交）
- 有声书（已提交）与视频配音 `dub`（已实现）
- 微调工作流 `finetune prepare|train|eval`（已提交）
- btts 音色导入 `voice import-btts`（FR-P2-09，已提交：0.3.2）

### 0.4.x 与 Phase 5（已完成）

- 结构化音色设计 `design init/lint/explore/report/freeze`，full 指标与 completion 判定，voice 管理面与单音色归档
- Phase 5：`design prune`、多语言模板、Windows run 锁、`check/bench --qc-metrics`、`voice export --all` / `voice import-bundle`（bundle schema 2）

---

## 相关文档

- [PRD.md](PRD.md) — 产品需求、功能清单、命令规范、数据模型、验收标准
- [WORKLOG.md](WORKLOG.md) — 从需求讨论到 P0 交付的工作进程记录
- [CHANGELOG.md](CHANGELOG.md) — 版本变更记录与已知限制

---

## License

Apache-2.0
