# qwen-tts-cli

基于 Qwen3-TTS 的本地语音制作命令行工具：一条命令完成**文本 → 音频 + 字幕时间轴 + 可复现配方**，并能对参考音频、模型能力和运行环境做前置校验。

> **原文是真值。** 字幕内容来自输入原文，ForcedAligner 只负责时间轴，ASR 只用于质检而不是改写内容。

当前状态：**P0 / M0 已实现并通过审查；P1 Wave 1 已提交，Wave 2 已实现待审查**（多角色脚本、词典/导演标记、批量清单；Wave 1 包含字幕分段与质量门禁、JSONL stdin、音色归档、项目锁文件）。ASR 质检、takes、serve 等 P2 能力尚未实现，见[路线图](#路线图)。

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
| **音色归档** | `voice export/import` 单文件 `.qttsvoice`，白名单成员、流式校验、原子替换 |
| **项目锁文件** | `project init/lock` 固定模型全量 sha256、音色、采样参数、字幕策略与运行时版本 |
| **计划与诊断** | `--dry-run` 不加载模型即可输出分段与预算；`doctor --bench` 产出本机性能基线 |
| **机器可读** | 所有命令支持 `--json`；稳定退出码；错误走 stderr |

---

## 环境要求

- Python 3.10–3.12
- macOS 或 Linux（Windows 未验证）
- 本地推理需要 `qwen-tts` 与 `qwen-asr`（ForcedAligner）
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

# 1) 安装 CLI 本体（numpy + soundfile + PyYAML）
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
--ref-audio FILE --ref-text TEXT [--mode icl|xvec]
--i-have-rights --license TEXT --consent-note TEXT --tags a,b
```

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
  [--recipe-out FILE | -]
```

- 单样本对齐上限约 5 分钟；超过会退出 4（长音频分窗属 P1）。
- 默认 `--granularity sentence`，按句子出 cue；`char` / `word` 可显式选择。
- `--srt` 输入不会重新分组：输入 cue 边界优先于 TTS 计划段和原句边界，只重算时间；cue 文本保持逐字原文。
- `--lexicon` 时对齐器仍读 `spoken` 文本，时间轴再按字符映射回 display；`--text/--text-file/--srt` 都支持，SRT 模式下词典项不能跨越 cue 边界。
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
| 4 | 输入校验失败（音频不合格、能力不支持、对齐覆盖率过低） |
| 5 | 质量门禁未通过 |
| 6 | 批处理部分失败（`say --stdin --jsonl`、`batch run`） |
| 130 | 用户中断 |

---

## 开发与测试

```bash
# 单元测试：不需要 GPU / 模型 / torch
PYTHONPATH=. pytest tests -q

# 集成测试：需要本地模型与 MPS/CUDA
RUN_INTEGRATION=1 PYTHONPATH=. pytest tests/test_integration_smoke.py -q

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
├── doctor.py              # 环境诊断与 bench
├── models_ops.py          # models list/pull/prune
└── commands/              # say / script / batch / voice / align / project / models / doctor

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
- **对齐器单样本约 5 分钟**：`say` 按段对齐，限制按段计算；超长音频的分窗策略属 P1。
- **对齐器不能替代 ASR**：CER/WER 与自动 `ref_text` 需要 Qwen3-ASR，属 P1。
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

### P1 Wave 2（已实现，待审查）

- 多角色脚本 `script lint/compile` 与 `say --script`（逐段音色/模型/语言/instruct/seed）
- 发音词典 `--lexicon` 与 `[[pause:MS]]` 导演标记（spoken/display 映射）
- JSONL 批处理 `batch plan/run`（确定性计划、原子状态、`--jobs/--resume/--retry/--fail-fast`）

### M1（剩余）

- ASR `check`（CER/WER）、自动 `ref_text`
- 多 take 资产库与自动选优
- 音频 QC
- 长音频对齐分窗

### M2

- `serve` 与 OpenAI 兼容 `/v1/audio/speech`
- 可插拔后端（DashScope / vLLM-Omni batch）
- 有声书 / dub / 后处理 / bench
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
