# qwen-tts-cli

基于 Qwen3-TTS 的本地语音制作命令行工具：一条命令完成**文本 → 音频 + 字幕时间轴 + 可复现配方**，并能对参考音频、模型能力和运行环境做前置校验。

> **原文是真值。** 字幕内容来自输入原文，ForcedAligner 只负责时间轴，ASR 只用于质检而不是改写内容。

当前状态：**P0 / M0 已实现并通过审查**。P1/P2（批量、多角色脚本、ASR 质检、takes、serve 等）尚未实现，见[路线图](#路线图)。

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
| **可复现配方** | 每次生成写 `*.recipe.json`：seed、模型、dtype、采样参数、文本/音频/prompt 哈希 |
| **缓存** | 内容哈希缓存；命中时不加载模型；缓存键包含模型 fingerprint 与实际 dtype |
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

# 1) 安装 CLI 本体（仅 numpy + soundfile）
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
export QTTS_MODELS=/path/to/Qwen3-TTS/models
qtts models list --model-dir "$QTTS_MODELS"
```

从远端下载：

```bash
qtts models pull Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --source hf --revision main --model-dir "$QTTS_MODELS"
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
  --model-dir "$QTTS_MODELS" \
  --out hello.wav \
  --subtitle srt

# 产出：hello.wav、hello.srt、hello.recipe.json
```

### 2. 先看计划，不加载模型

```bash
qtts say "第一段。第二段。" \
  --voice builtin:Vivian \
  --model-dir "$QTTS_MODELS" \
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
  --model-dir "$QTTS_MODELS"

qtts voice preview alice --model-dir "$QTTS_MODELS" --out alice-preview.wav
qtts say "你好，我是 Alice。" --voice voice:alice \
  --model-dir "$QTTS_MODELS" --out alice.wav
```

### 4. 设计音色并固化

```bash
qtts voice add narrator \
  --kind design \
  --description "低沉、沙哑的中年男声，语速偏慢" \
  --sample-text "这是试音句。" \
  --language Chinese \
  --materialize-clone \
  --model-dir "$QTTS_MODELS"
```

### 5. 对已有音频做对齐

```bash
qtts align hello.wav \
  --text-file script.txt \
  --language Chinese \
  --format srt,vtt,ass,json \
  --model-dir "$QTTS_MODELS"
```

### 6. 诊断与性能基线

```bash
qtts doctor --bench --model-dir "$QTTS_MODELS"
```

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
qtts say [TEXT | - | --file FILE | --script FILE]
  --voice builtin:NAME | voice:NAME | design:DESCRIPTION
  [--language auto|Chinese|English|...]
  [--instruct TEXT]
  [--out FILE | -]
  [--format wav|flac]
  [--subtitle srt,vtt,ass,json]
  [--subtitle-out-prefix PREFIX | -]
  [--recipe-out FILE | -]
  [--granularity char|word]
  [--max-new-tokens N]
  [--pause-ms N]
  [--model MODEL]
  [--dry-run] [--yes]
```

要点：

- `--instruct` 只支持 1.7B CustomVoice / VoiceDesign；Base、0.6B CustomVoice 收到会直接退出 4。
- `--language auto` 按段判定并逐段校验。
- stdout 一次只能承载一个产物：`--out -`、`--json`、`--subtitle-out-prefix -`、`--recipe-out -` 互斥。
- `--out -` 时 stdout 是纯音频，摘要写 stderr。

### `qtts voice`

```text
qtts voice add NAME --kind clone|design ...
qtts voice list
qtts voice inspect NAME
qtts voice preview NAME [--text TEXT] [--out FILE] [--format wav|flac]
qtts voice rm NAME [--yes]
```

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
  [--language Chinese]
  [--granularity char|word]
  [--format srt,vtt,ass,json]
  [--out-prefix PREFIX]
  [--recipe-out FILE | -]
```

- 单样本对齐上限约 5 分钟；超过会退出 4（长音频分窗属 P1）。
- `--srt` 输入会保留原 cue 的编号和边界，只重算时间。
- 字幕文本是该 cue 的规范化原文，覆盖率过低会告警或退出 4。

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
| 6 | 批处理部分失败（P1） |
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
├── plan.py                # GenerationPlan IR、语言与 prompt 预检
├── cache.py               # 缓存键与命中
├── voice_store.py         # 音色资产与原子提交
├── admit.py               # 参考音频准入检查
├── tts_engine.py          # qwen-tts 懒加载
├── align_engine.py        # qwen-asr ForcedAligner 懒加载
├── subtitles.py           # 时间轴清洗与 SRT/VTT/ASS/JSON
├── sidecar.py             # recipe 读写
├── doctor.py              # 环境诊断与 bench
├── models_ops.py          # models list/pull/prune
└── commands/              # say / voice / align / models / doctor

tests/
├── test_capabilities.py
├── test_text_pipeline.py
├── test_subtitles.py
├── test_store_cache.py
├── test_cli.py
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
- **MPS 不保证 bit-perfect**：同 seed 可复现趋势，但跨设备/后端结果会漂移。
- **依赖钉版冲突**：见[环境要求](#环境要求)；M0 采用单环境混装，桥接方案保留为 fallback。

---

## 路线图

### M0（已完成）

`say`、`voice add/list/inspect/preview/rm`、`align`、`models list/pull/prune`、`doctor [--bench]`、`--dry-run`、sidecar、缓存、能力矩阵、字幕四格式。

### M1

- JSONL 批处理（分桶、`--resume`、`--retry`、失败隔离）
- 多角色脚本与 `script compile`
- ASR `check`（CER/WER）
- 多 take 资产库与自动选优
- 字幕 CPS / 行宽门禁
- `voice export/import`
- `project init/lock`
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
