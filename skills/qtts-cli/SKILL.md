---
name: qtts-cli
description: >-
  Operate the local `qtts` CLI (qwen-tts-cli 0.4.x) for Qwen3-TTS speech
  production: say, subtitles, multi-role scripts, batch, voice clone/design,
  forced alignment, audiobook, video dubbing, ASR quality checks, audio QC and
  post-processing, benchmarks, HTTP serve, and finetuning. Use for qtts,
  qwen-tts, Qwen3-TTS, local TTS, 语音合成, 配音, 字幕, 克隆音色, 音色设计,
  有声书, 视频配音, 批量合成, 强制对齐, 语音质检, or reproducible local audio production.
---

# qtts-cli

`qtts` 是本地 Qwen3-TTS 语音制作 CLI：文本 → 音频 + 字幕时间轴 + 可复现配方。
**原文是真值**：字幕内容来自输入，ForcedAligner 只负责时间轴；ASR 主要用于质检，
长音频对齐与 `--auto-ref-text` 也会用到 ASR。

命令在项目根目录执行。若 `qtts` 不在 PATH，改用 `uv run qtts` 或 `.venv/bin/qtts`，
选定一个并全程复用。本文按 0.4.1 编写，先 `qtts --version` 核对。

## 最小可用路径

```bash
qtts doctor --json
qtts say "你好，世界。" --voice builtin:Vivian --dry-run --json
qtts say "你好，世界。" --voice builtin:Vivian --out hello.wav --subtitle srt --json
```

- `doctor` 的 `ok` 只表示命令执行成功；`healthy` 只表示没有 incomplete model
  （`--bench` 失败也会拉低它）。本地生成还要求 `model_dir` 存在、`models[]` 非空、
  所需模型 `weights_ok=true`、`torch.installed=true`；m4a/mp3/mux 还要求
  `tools.ffmpeg`。空模型目录也可能 `ok=true`、`healthy=true`，必须看 `models[]`
  与 `advice`。
- `--dry-run` 不加载权重，但仍解析模型/音色；缺模型会退出 3/4，不会自动下载。
- `say --out`、`voice preview --out` **静默覆盖**同名音频与 recipe；`audiobook`/`dub`
  的 `--out` 已存在时默认拒绝，需要 `--force`。写前确认目标不存在，或换新路径。

## 先读这三条

1. **下载、外发与代码执行都由用户决定**：`models pull`、插件、`--backend dashscope`、
   `finetune train`、`uv run` 可能下载、把文本发到远端或执行第三方代码。先说明，再等确认。
2. **授权与删除不能代签**：`--i-have-rights`（clone add / import-btts）、`--force`、
   `--yes`、`voice rm`、`models prune`、覆盖已有输出，都必须由用户明确同意。
3. **指标不是评委**：`check`/`qc`/`bench` 的门禁只决定是否放行；最终音色与 take
   必须由人试听。

## 心智模型

| 音色引用 | 路由 | 说明 |
|---|---|---|
| `builtin:Vivian` | CustomVoice | 9 个内置音色；`voice list` 查看 |
| `voice:alice` | Base + clone prompt | 本地克隆、固化或 finetune profile |
| `design:低沉沙哑的中年男声` | VoiceDesign | 即时生成，不落盘；可用 `voice add --kind design --materialize-clone` 固化 |

- 文本流水线：规范化 → 分段/预算 → 逐段语言 → 合成 → 字幕对齐与 QC。
- 产物三件套：`<out>.wav|flac`、`<stem>.recipe.json`、可选 `.srt/.vtt/.ass/.json`。
- `--instruct` 只支持 1.7B CustomVoice / VoiceDesign；Base、0.6B 收到退出 4。
- 一次生成的身份由音色 + seed + 采样 + 语言 + instruct + 模型 fingerprint 决定；
  recipe 记录这些，不要手改。

## 退出码

| code | 含义 | 下一步 |
|---|---|---|
| 0 | 成功 | — |
| 2 | 参数/用法 | 看 stderr；修参数或互斥关系 |
| 3 | 模型/后端/编码器 | `doctor --json`；查模型、torch、ffmpeg、API key |
| 4 | 输入/能力/路径 | 音频不合格、音色不存在、路径冲突、锁冲突 |
| 5 | 质量门禁 | **产物通常已写出**；读报告，调阈值或修音色/文本 |
| 6 | 批处理部分失败 | 读 `status.jsonl`；`--resume --retry N` 只重试 transient |
| 130 | 用户中断 | 长任务用 `--resume`；检查中间产物 |

`1` 是内部兜底错误，不要依赖；`5` 不等于没有产物。

## 任务路由

| 任务 | 最小命令 | 下一步 | 参考 |
|---|---|---|---|
| 单条语音 + 字幕 | `qtts say TEXT --voice ... --out X.wav --subtitle srt --json` | `check` / `qc` | workflows.md |
| 多角色 | `script lint` → `say --script` | 先 `lint`，再 `compile` | workflows.md |
| 批量 | `batch plan` → `batch run` | 新 `--out-dir`；`--resume` | workflows.md |
| 音色克隆 | `voice add NAME --kind clone ...` | 用户确认授权；`preview` | voice-and-assets.md |
| 音色设计 | `voice add NAME --kind design ...` | `preview`；要稳定用 `--materialize-clone` | voice-and-assets.md |
| 结构化设计/盲听 | `design init` → `lint` → `explore` → `report` | 人耳试听后再 `freeze` | commands-core.md |
| 音色归档 | `voice export` / `import` / `import-btts` | 校验 manifest 与 hash | voice-and-assets.md |
| 对齐已有音频 | `align AUDIO --text-file ...` | >300s 自动切窗 | workflows.md |
| 有声书 | `audiobook build book.yaml` | 先 `--dry-run` | workflows.md |
| 视频配音 | `dub build subs.srt --voice ...` | 先 `--dry-run`；SRT/EDL | workflows.md |
| 质检 | `check` / `qc` | 阈值按需；null 不要编造 | commands-production.md |
| 后处理 | `post raw.wav --out final.flac --lufs -16` | true peak 以编码后为准 | commands-production.md |
| 多 take | `say --takes N --select lufs` | `takes compare/promote` | voice-and-assets.md |
| 复现 | `project init/lock` | 锁文件记录全量 sha256 | voice-and-assets.md |
| HTTP 服务 | `serve --port 0 --json` | 默认回环；非回环必须 token | commands-production.md |
| 微调 | `finetune prepare/train/eval` | 高风险；先 `--dry-run` | commands-production.md |

## 输出与破坏性操作

写文件前先看 `references/safety.md` 的输出目标矩阵。最容易踩的几条：

- 所有输出旗标都可能覆盖：`--out`、`--out-dir`、`--out-prefix`、`--recipe-out`、
  `--report`、`-o`、`--audio-dir`、`--subtitle-out-prefix`、`--stems-out`、
  `--output-jsonl`、`--output-dir`。其中 `say --out`、
  `voice preview --out`、`script compile -o`、`project lock --out`、
  `check/qc/bench --report`、`post --out` 会**静默覆盖**已有文件；
  `audiobook`/`dub` 的 `--out` 默认拒绝覆盖，需要 `--force`。
- `batch plan` **不是只读**：它创建 `--out-dir` 并写 `plan.json`/`status.jsonl`。
  复用旧目录前先 diff；默认换新目录。
- `say --stdin --jsonl` 在读取 stdin 前就创建 `--out-dir`，且不支持 `--dry-run`。
- `audiobook` 用 journal 发布，可回滚；`dub` 没有跨文件 journal，中断可能留下
  新旧混合文件，重跑（必要时 `--force`）重建。
- `post` 会做路径碰撞检查，但不拒绝已存在输出；`takes promote --out` 可覆盖任意
  文件，只保护 job 自身文件。
- `voice export` 的 full 包可能含参考音频、逐字转写与 consent，按个人数据处理。

## 硬性禁止

- 不自动 `models pull`，不静默接受许可证；下载前说明体积与来源，建议 pin revision。
- 不替用户传 `--i-have-rights`；不在用户未明确要求时 `voice rm`、`models prune`、
  `--force`、`--yes`。
- 任何输出旗标（`--out`、`--out-dir`、`--out-prefix`、`--recipe-out`、`--report`、
  `-o`、`--audio-dir`、`--subtitle-out-prefix`、`--stems-out`、`--output-jsonl`、
  `--output-dir`）都不得指向输入、参考音频、recipe、项目/锁文件、job/take 资产或
  其他源文件。`check/qc/bench --report` 可以直接覆盖被分析的音频；
  `script compile -o`、`project lock --out`、`align --out-prefix` 同样危险。
- `batch` manifest 与 `say --stdin --jsonl` 每行的 `out` 字段同样是输出路径，遵守
  同一规则；batch 的相对 `out` 相对 cwd，不是 `--out-dir`。
- 不手改 `voice.json`、`*.recipe.json`、`qtts.lock.json`、`job.json`、`status.jsonl`。
- 不把 `serve` 绑到 `0.0.0.0`/公网；非回环必须 `--token`，优先 `QTTS_SERVE_TOKEN`
  避免 token 出现在 argv；服务没有 TLS。
- 不运行未审计的插件、`--backend` 插件或 `finetune --python`；它们执行第三方代码。
- 不编造 `null`/`unavailable` 的 QC 指标；不把门禁通过当审美结论。
- 不在用户确认前跑 `finetune train` 或长时间 batch。
- 不声称 `dub` 整组原子；不把不同模型/音色的 take 混进同一 job。

## 参考

- `references/safety.md`：输出目标矩阵、破坏性语义、执行/外联/授权门。
- `references/commands-core.md`：say/script/batch/align/voice/takes/project。
- `references/commands-production.md`：audiobook/dub/check/qc/post/bench/serve/finetune/models/doctor/plugins/completion。
- `references/workflows.md`：按任务的完整配方。
- `references/voice-and-assets.md`：音色路由、克隆/设计、归档、数据目录、复现。
- `references/troubleshooting.md`：doctor 读法、按退出码排障。
