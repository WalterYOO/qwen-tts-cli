# qwen-tts-cli

基于 Qwen3-TTS 的本地语音制作命令行工具：一条命令完成**文本 → 音频 + 字幕时间轴 + 可复现配方**，并能对参考音频、模型能力和运行环境做前置校验。

> **原文是真值。** 字幕内容来自输入原文，ForcedAligner 只负责时间轴，ASR 只用于质检而不是改写内容。

当前状态：**P1 已全部实现并发布 0.2.0；P2 Wave 1a 的音频后处理 `post` 与 Wave 1b 的 `bench` 已提交**。其余 P2 能力（serve、audiobook/dub、插件与后端抽象、微调）尚未实现，见[路线图](#路线图)。

---

## 目录

- [特性](#特性)
- [环境要求](#环境要求)
- [安装](#安装)
- [模型准备](#模型准备)
- [快速开始](#快速开始)
- [命令参考](#命令参考)
- [音色引用](#音色引用)
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
| **自动参考文本** | `voice add --kind clone --auto-ref-text` 用 ASR 转写参考音频，作为 ICL 的 `ref_text` 并记录来源与模型 |
| **音色归档** | `voice export/import` 单文件 `.qttsvoice`，白名单成员、流式校验、原子替换 |
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

`check` 的参考文本与 ASR 文本走同一归一化管道（标点/符号移除、空白折叠、大小写折叠）；`ratio = distance / max(1, 参考单元数)`，`ratio <= threshold` 通过（默认 0.03），否则退出 5。归一化、编辑距离与错误片段均可复现；Wave 3a 冻结时间戳，`errors[].start/end` 固定为 `null`。`check` 单次音频上限 300 秒，超过在加载模型前退出 4。

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
```

`bench` 的计时使用外部 `time.perf_counter()` 未舍入值聚合：`runs[].wall_s` 不含加载，`metrics.wall_s` 含加载但不含评分/报告，`rtf` 取每 run 中位数，`wall_rtf = 全场 wall_s / 全场 audio_s`。默认 `--suite smoke` 跑内置中英各一句；`batch` 从 16 条短句按语言过滤后循环取 `--count`；`longform` 用 `--text` > `--text-file` > 内置长文，经 `plan.build_plan()` 分段后逐段生成。`--dry-run` 只做目录/元数据/文件存在性与计划，不 import torch/qwen、不加载权重、不创建文件。真实运行时严格按 TTS → 释放 → ASR（可选）→ 释放 → Base（可选）单驻留顺序，`finally` 释放所有已加载模型。CER 按字符编辑距离、逐 case 语言转写；speaker-sim 优先使用 clone profile 的 `ref_spk_embedding`，否则用同一 Base 从 `ref.wav` 提取，并在报告中记录 `reference_embedding_source`。`--json` 与 `--report` 逐字节一致；门禁失败输出完整 payload、`ok=false`、`gate_failures[]` 并退出 5；报告写失败退出 4。

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
  [--voice builtin:NAME | voice:NAME | design:DESCRIPTION]
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
  [--takes N --select cer,speaker-sim,lufs] [--asr-model PATH|ID]
  [--qc] [--reference-voice NAME]
  [--max-clipping-ratio N] [--target-lufs N --lufs-tolerance N]
  [--max-true-peak-dbtp N] [--max-dc-offset N] [--max-silence-ratio N]
  [--min-speech-rate N] [--max-speech-rate N]
  [--dry-run] [--yes]
```

要点：

- `--script` 接受 `.yaml/.yml/.json` 多角色脚本（见[多角色脚本](#7-多角色脚本)）；纯文本脚本仍按普通输入读取。脚本模式下 `--voice` 只作为 defaults 缺省值，`--model` 若与多模型脚本冲突会退出 2；与 TEXT/`--stdin`/`--file` 同时出现会退出 2；显式传入的 `--language/--instruct/--max-new-tokens/--pause-ms` 覆盖脚本 `defaults` 并体现在 plan/recipe 中。
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

### `qtts script`

```text
qtts script lint FILE [--model MODEL] [--lexicon FILE.json]
qtts script compile FILE -o PLAN.json [--model MODEL] [--lexicon FILE.json]
```

- 脚本 schema：`schema_version: 1`；`defaults` 支持 `voice/language/format/seed/max_new_tokens/pause_ms/instruct`；`segments[]` 支持 `id/text/role/voice/language/instruct/pause_after_ms/seed/max_new_tokens`。
- 缺省 `id` 自动生成 `s001...`，重复 id、未知字段、字段类型/整数错误与顶层未知字段（包括未支持的 `project`）退出 2；YAML 额外拒绝重复 key、多文档、anchor/alias、超大文件，JSON 拒绝重复 key；缺 PyYAML 退出 3。
- 每段独立解析 voice/model 并在任何 `TtsEngine.load()` 前完成语言、instruct、ICL ref_text 能力校验。一个 source segment 产出多个 piece（多句、marker 分块或预算切分）时，全部 piece 使用稳定后缀 id（`s001.p1`、`s001.p2`），并保存 `source_segment_id`；单 piece 保留原 id。
- `compile` 输出按 key 排序、无时间戳、无 `cache`/绝对 `script_source` 的计划；同一环境重复编译字节稳定。`lint` 语法/schema 错退出 2，未知 voice/能力冲突退出 4。

### `qtts batch`

```text
qtts batch plan MANIFEST.jsonl --out-dir DIR [--model MODEL] [--lexicon FILE.json]
qtts batch run MANIFEST.jsonl --out-dir DIR
  [--jobs N] [--resume] [--retry N] [--fail-fast] [--watch]
  [--model MODEL] [--lexicon FILE.json]
```

- manifest 每行字段与 `say --stdin --jsonl` 对齐：`id`（唯一必填）、`text`、`voice`、`language`、`instruct`、`out`、`format`、`subtitle`、`seed`、`max_new_tokens`、`pause_ms`。默认输出为 `<out-dir>/<safe id>.<format>`。
- `plan` 对所有行先解析、校验、解析 voice/能力并编译，再写 `plan.json` 与初始全 `pending` 的 `status.jsonl`；不加载模型。坏 JSON/重复 key 退出 2，重复 id/未知 voice/能力冲突退出 4。`out` 只要有后缀就必须是 `.wav`/`.flac`，且与显式 `format` 一致。
- `run` 复用单条 `say` 执行器与错误结构；默认 `--jobs 1`。任务按模型/音色/语言/估算时长分桶排序，调度可并发，但模型推理由全局锁串行（MPS 同样只保留一个模型槽）。状态为 `pending/running/done/failed/skipped`，由单线程调度器原子重写；`job_key` 包含 `lexicon_sha256`。
- `--resume` 要求 `status=done`、`job_key` 相同、输出存在且 sha256 匹配、recipe 存在且其 `job_key` 匹配；`running` 恢复为 `pending`，`skipped` 下次重新调度。`--retry N` 只重试执行期 transient 失败（ModelError/OSError/RuntimeError），普通异常立即失败；status 记录 `attempts`、`last_error` 与 `transient`，`--resume --retry` 只重新调度 `transient=true` 的失败；`--fail-fast` 只停止调度新任务，未启动项标 `skipped`。
- 进度与 `--watch` 只写 stderr；stdout 只有最终 summary（`--json` 为单个对象）。任一任务失败退出 6。

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
  [--dry-run] [--report FILE] [--json]
```

- 输入只支持 `.wav`/`.flac`（后缀大小写不敏感）；mp3/m4a 输入、无后缀、未知后缀或目录退出 2。`--out` 只接受一个输入；`--out` 与 `--out-dir` 互斥且必须给一个。
- 输出格式解析：`--out` 有已知后缀时 `--format` 若给出必须一致，否则用后缀；`--out` 无后缀时必须给 `--format` 并追加后缀；`--out` 未知后缀即使给了 `--format` 也退出 2；`--out-dir` 必须给 `--format`，文件名为 `<input stem>.<format>`。
- `--out -`、`--out-dir -`、`--report -` 和 `--dry-run --report` 均退出 2。输入/输出/report 在写文件前做规范化碰撞检查（`resolve(strict=False)`、NFC、大小写折叠，含 symlink），任一碰撞退出 2；多输入还校验规范化 basename 唯一。
- 全局错误（参数范围、路径/格式、ffmpeg/encoder 缺失）退出 2/3 且 0 写入；逐文件错误（解码、指标不可用、trim 后为空、DSP/编码/IO）写入 `files[].error`。全部成功退出 0；部分失败退出 6；全部失败退出输入顺序第一个错误码；单输入失败返回该错误码。
- `--lufs` 范围 [-70, 0]，`--true-peak` ≤ 0，`--sample-rate` 范围 [8000, 192000]；`--bit-depth` 只用于 wav/flac，`--bitrate` 只用于 mp3/m4a，trim 细项必须与 `--trim-silence` 同用，违反退出 2。静音且未请求响度/true-peak 时可透传；显式请求的指标不可用、或 trim 后为空时该文件退出 4。
- 报告 `schema_version` 为 1，顶层含 `request`（含 `true_peak_ceiling_scope: "encoded"`）与 `files[]`；每项含 `input_format/output_format/codec/encoder/dsp_version`、`before/after_dsp/after_encoded`、`applied`（`gain_db/limiter_reduction_db/safety_attenuation_db/limiter_algorithm/encode_retries/encoder_margin_db`）与稳定 `warnings`。`--json` 与 `--report` 逐字节一致；进程 `exit_code` 不写入 canonical 文档。
- `--dry-run` 仍做路径/碰撞/ffmpeg 预检和内存 DSP，但不创建目录、不编码、不写报告；`output` 为计划路径，`output_sha256` 与 `after_encoded` 为 `null`。`--report` 写失败退出 4，但已成功写出的音频保留，stdout/JSON 仍包含同一 `files[]` 并在顶层给出 `error`。

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

### `qtts voice`

```text
qtts voice add NAME --kind clone|design ...
qtts voice list
qtts voice inspect NAME
qtts voice preview NAME [--text TEXT] [--out FILE] [--format wav|flac]
qtts voice rm NAME [--yes]
qtts voice export NAME -o FILE.qttsvoice [--force]
qtts voice import FILE.qttsvoice [--name NAME] [--force]
```

`.qttsvoice` 是 zip 容器，只允许 `manifest.json`、`voice.json`、`prompt.safetensors`、`ref.wav`、`preview.wav`；导入逐项校验路径、大小、压缩比、sha256 与 safetensors 结构，不使用 `extractall`，替换已有音色失败时回滚。

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

---

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
- `serve` 与 OpenAI 兼容 `/v1/audio/speech`
- 可插拔后端（DashScope / vLLM-Omni batch）
- 有声书 / dub
- 微调工作流包装
- 插件 SDK

---

## 相关文档

- [PRD.md](PRD.md) — 产品需求、功能清单、命令规范、数据模型、验收标准
- [WORKLOG.md](WORKLOG.md) — 从需求讨论到 P0 交付的工作进程记录
- [CHANGELOG.md](CHANGELOG.md) — 版本变更记录与已知限制

---

## License

Apache-2.0
