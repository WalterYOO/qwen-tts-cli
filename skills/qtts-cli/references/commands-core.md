# 核心命令参考：say / script / batch / align / voice / takes / project

## 全局约定

```bash
qtts [--config FILE] [--data-dir DIR] [--model-dir DIR] [--cache-dir DIR]
     [--device auto|cpu|mps|cuda|cuda:0|cuda:1]
     [--dtype auto|bf16|fp16|fp32]
     [--seed 1234] [--offline] [--json] [--quiet] [--no-cache]
     [--log-level debug|info|warning|error] <command>
```

- 模型目录：`--model-dir` > `$QTTS_MODEL_DIR` > `$QWEN_TTS_MODEL_DIR`。
- 数据目录：macOS `~/Library/Application Support/qtts`，Linux
  `~/.local/share/qtts`；`--data-dir` 覆盖。布局：
  `voices/<name>/`、`cache/`、`recipes/`、`takes/<job_id>/`、`serve/`。
- `--config` 接受 JSON（`data_dir/model_dir/cache_dir/device/dtype/offline/seed/no_cache`），
  CLI 显式值优先。
- stdout 一次只能承载一个产物：`--out -`、`--json`、`--subtitle-out-prefix -`、
  `--recipe-out -` 互斥。
- 所有命令支持 `--json`；错误写 stderr，退出码稳定。

## say

```bash
qtts say [TEXT | - | --file FILE | --script FILE | --stdin [--jsonl]]
  [--voice builtin:NAME | voice:NAME | design:DESCRIPTION]
  [--language auto|Chinese|English|...] [--instruct TEXT]
  [--model MODEL] [--backend transformers|dashscope|vllm-omni|PLUGIN]
  [--backend-model ID]
  [--out FILE | -] [--out-dir DIR] [--format wav|flac]
  [--subtitle srt,vtt,ass,json]
  [--subtitle-out-prefix PREFIX | -] [--recipe-out FILE | -]
  [--granularity sentence|char|word]
  [--max-cue-chars 42] [--max-line-width 42] [--max-cps 20]
  [--min-cue-seconds 0.8] [--max-cue-seconds 7.0] [--max-merge-gap-ms 300]
  [--no-subtitle-qc]
  [--max-new-tokens 2048] [--pause-ms 300] [--lexicon FILE.json]
  [--qc] [--reference-voice NAME]
  [--max-clipping-ratio N] [--target-lufs N --lufs-tolerance N]
  [--max-true-peak-dbtp N] [--max-dc-offset N] [--max-silence-ratio N]
  [--min-speech-rate N] [--max-speech-rate N]
  [--takes N --select cer,speaker-sim,lufs] [--asr-model PATH|ID]
  [--dry-run] [--yes]
```

- 默认输出 `out.wav`；`--out` 的已知后缀决定格式，`--format` 可显式指定。
  默认同时写 `<stem>.recipe.json`。`--recipe-out`/`--subtitle-out-prefix` 不得指向
  输入文本或已有字幕；这些路径没有碰撞检查。
- **非 `--script` 输入必须提供 `--voice`**；只有脚本模式能从 `defaults`/`segments`
  取音色。`--voice` 三选一：`builtin:`（9 个内置音色）、`voice:`（本地 profile）、
  `design:`（即时设计，不落盘）。
- `--subtitle` 可多选；字幕内容来自原文，`[[pause:MS]]` 只影响音频/时间轴。
- 字幕默认策略：单 cue 42 显示宽、单行 42、CPS 20、最短 0.8s、最长 7s、
  合并最大静音 300ms。可修复的格式问题会修复；无法修复或阈值越界退出 5。
- `--qc` 对最终音频测量；不给阈值只测量。`--reference-voice` 需要 `--qc` **或**
  `--takes N --select speaker-sim`；它要求本地 transformers backend 与 clone profile，
  并会额外加载 Base 模型算说话人相似度。
- `--takes N` 与 `--select` 必须成对；`N` 为 2–16；准则优先级按给定顺序。
  需要本地 transformers；`cer` 需要本地 ASR。
- `--dry-run` 输出计划、`will_write`、依赖与内存估算，不加载权重；但仍解析
  模型/音色，缺模型会退出 3/4。
- `--stdin --jsonl`：每行 `{"id","text","voice"?,"out"?}`，每行结果
  `{"id","status","output","recipe","metrics"}` 或错误对象；全部成功 0，
  任一失败 6，空 stdin/启动参数错误 2。它在读取 stdin 前创建 `--out-dir`，
  且不支持 `--dry-run`；行间只拒绝同一路径，磁盘上已有文件不检查。
- 退出：参数 2；模型/后端 3；音色/能力/路径 4；字幕 QC 或 `--qc` 阈值 5；
  JSONL 部分失败 6；中断 130。

## script

```bash
qtts script lint FILE [--model MODEL] [--lexicon FILE.json]
qtts script compile FILE -o PLAN.json [--model MODEL] [--lexicon FILE.json]
  [--backend ...] [--backend-model ID]
```

- `lint` 只校验，不写计划、不加载权重；但仍会扫描并解析模型/音色，模型目录为空
  时退出 3。
- `compile` 输出排序、无时间戳、无缓存的确定 `GenerationPlan` JSON；同一环境
  重复编译字节稳定。`-o -` 写 stdout。`-o` 不得指向输入脚本；没有碰撞检查。
- 脚本 schema：`schema_version: 1`；`defaults` 支持
  `voice/language/format/seed/max_new_tokens/pause_ms/instruct`；
  `segments[]` 支持 `id/text/role/voice/language/instruct/pause_after_ms/seed/max_new_tokens`。
- `format` 是作业级参数，不能按段覆盖。缺省 `id` 生成 `s001...`。
- 语法/schema/重复 id/未知字段退出 2；未知音色、能力冲突、ICL 缺 `ref_text`
  退出 4；缺 PyYAML 退出 3。
- 多句/多 marker/预算切分时 piece 使用 `s001.p1` 等稳定后缀，
  `source_segment_id` 保持 `s001`。

## batch

```bash
qtts batch plan MANIFEST.jsonl --out-dir DIR [--model MODEL] [--lexicon FILE]
  [--backend ...] [--backend-model ID]
qtts batch run MANIFEST.jsonl --out-dir DIR
  [--jobs 1] [--resume] [--retry N] [--fail-fast] [--watch]
  [--model MODEL] [--lexicon FILE] [--backend ...] [--backend-model ID]
```

- manifest 每行字段与 `say --stdin --jsonl` 对齐：`id` 必填且唯一；其余
  `text/voice/language/instruct/out/format/subtitle/seed/max_new_tokens/pause_ms`。
  默认输出 `<out-dir>/<safe id>.<format>`。`out` 是输出路径，可写绝对路径；
  相对路径相对 cwd，不是 `--out-dir`；不得指向输入/资产。
- `plan` 完整解析、校验、编译全部行，写 `plan.json` 与 `status.jsonl`；不加载模型。
  **它不是只读**：会创建目录并覆盖同名计划/状态。
- `run` 复用 `say` 执行器；`--jobs` 只是调度并发，模型推理仍全局串行。
  非 `--resume` 不读旧 status，重写 `plan.json`/`status.jsonl`，按 manifest 全部
  任务重跑并覆盖输出；只有 `--resume` 才按 `job_key`/sha256 跳过已完成任务。
- 状态 `pending/running/done/failed/skipped`，单线程原子重写。
- `--resume` 仅在 `status=done`、`job_key`、输出 sha256、recipe 的 `job_key`
  全部匹配时跳过；`running` 恢复为 `pending`，`skipped` 重新调度。
- `--retry N` 只重试 transient（ModelError/OSError/RuntimeError）；`--fail-fast`
  只停止调度新任务，未启动项标 `skipped`。
- 部分失败退出 6；进度与 `--watch` 只写 stderr。
- 默认换新的 `--out-dir`；复用旧目录前先 diff `plan.json`/`status.jsonl`。

## align

```bash
qtts align AUDIO
  (--text TEXT | --text-file FILE | --srt FILE | --script FILE)
  [--lexicon FILE.json] [--language auto|Chinese|English|...]
  [--granularity sentence|char|word]
  [--format srt,vtt,ass,json]          # 默认 srt
  [--out-prefix PREFIX]                # 默认 audio stem
  [--max-cue-chars 42] [--max-line-width 42] [--max-cps 20]
  [--min-cue-seconds 0.8] [--max-cue-seconds 7.0] [--max-merge-gap-ms 300]
  [--no-subtitle-qc]
  [--window-seconds 270] [--window-overlap 3]
  [--approximate] [--asr-model MODEL]
  [--recipe-out FILE | -]
```

- 四种文本输入互斥且必须给一个，否则退出 2。
- 音频 ≤300s 走单次 forced alignment，不加载 ASR。
- 音频 >300s 默认按静音切窗：每窗 240–285s（默认 270），重叠 ≥2s（默认 3），
  单窗 ≤300s；需要 `Qwen3-ASR-*` + `Qwen3-ForcedAligner-*`。
- `--approximate` 跳过 ASR，按窗口比例分配文本；边界不保证，会 warning；
  不能与 `--srt` 同用。
- `--srt` 的 cue 边界是硬锚点，只重算时间；不改写 cue 文本。
- `--lexicon` 时 aligner 读 `spoken`，时间轴映射回显示文本。
- 覆盖率门禁：低于 30% 拒绝，低于 80% warning；结构性错误退出 4；
  字幕 QC 退出 5。

## voice

```bash
qtts voice add NAME --kind clone --ref-audio FILE
  (--ref-text TEXT | --auto-ref-text) [--mode icl|xvec]
  [--language LANG] [--asr-model PATH|ID]
  [--i-have-rights] [--license TEXT] [--consent-note TEXT] [--tags a,b]
  [--model MODEL] [--yes]

qtts voice add NAME --kind design --description TEXT --sample-text TEXT
  [--language Chinese] [--materialize-clone] [--model MODEL]

qtts voice list [--json]
qtts voice inspect NAME [--json]
qtts voice preview NAME [--text TEXT] [--out FILE] [--format wav|flac]
  [--language LANG]
qtts voice rm NAME [--yes]
qtts voice export NAME -o FILE.qttsvoice [--force]
qtts voice import FILE.qttsvoice [--name NAME] [--force]
qtts voice import-btts BUNDLE (--voice NAME ... | --all)
  [--name NAME] [--i-have-rights] [--auto-ref-text] [--ref-text TEXT]
  [--asr-model PATH|ID] [--mode icl|xvec] [--model MODEL] [--force]
```

- 内置音色：Serena、Vivian、Uncle_Fu、Ryan、Aiden、Ono_Anna、Sohee、Eric、Dylan。
- clone 参考音频准入：<2s 硬失败；2–3s warning；3–30s 推荐；>60s warning；
  <16kHz warning；另有削波、静音比例、有效语音占比检查。没有 `--allow-long-ref`。
- ICL 模式必须有 `ref_text`；`--auto-ref-text` 用 Qwen3-ASR 转写（需要 ASR 模型），
  不能与 `--ref-text` 或 `--mode xvec` 同用。xvec 不需要 `ref_text`，但跨文本稳定性
  通常不如 ICL。
- `--kind clone` 必须 `--i-have-rights`（非 TTY），否则退出 4；design 自动记录
  `rights_confirmed: true`。
- `--materialize-clone` 把 design 生成参考音频并固化为 clone prompt，跨文本更稳定。
- `voice preview` 默认文本为内置试听句；`--out` 与 `<out-stem>.recipe.json`
  静默覆盖。`--out` 不得指向 `ref.wav`/`prompt.safetensors`；没有碰撞检查。
- `voice export` 生成 `.qttsvoice`（白名单：`manifest.json`、`voice.json`、
  `prompt.safetensors`、`ref.wav`、`preview.wav`），已有目标需 `--force`。
- `voice import` 逐项校验路径、大小、压缩比、sha256 与 safetensors 结构，不使用
  `extractall`；替换失败回滚。**没有同意旗标**，导入前由用户确认授权。
- `voice import-btts` 读取 btts `voice export` 的目录或 `.zip`；有 `manifest.json`
  时校验 SHA-256、大小与完整文件集，拒绝路径穿越/绝对路径/反斜杠/symlink/
  特殊文件/大小写冲突/未知文件。必须 `--i-have-rights`；已有音色需 `--force`。
- `voice rm` 只删除 store 根目录内单个音色；非 TTY 必须 `--yes`。

## takes

```bash
qtts takes list JOB_ID [--json]
qtts takes compare JOB_ID [--ids 1,2,3] [--json]
qtts takes promote JOB_ID --id N (--out FILE | --out -) [--format wav|flac] [--json]
```

- `JOB_ID` 匹配 `[A-Za-z0-9._-]{1,64}`，只解析 `<data-dir>/takes/` 下的目录；
  绝对路径、`..`、symlink 退出 2；不存在退出 4。
- 三个子命令都不加载模型。`list`/`compare` 会校验音频 sha256 与 recipe 身份，
  损坏退出 4。
- `promote` 复制所选 take 的原始字节，不转码；目标后缀必须与 job 格式一致。
  会保护 `job.json`、take 音频与 recipe，但**可以覆盖其他任意 `--out` 目标**。
- `--out -` 不能与 `--json` 同用。

## project

```bash
qtts project init [DIR] --voice REF [--language LANG] [--format wav|flac]
  [--max-new-tokens N] [--subtitle srt,vtt,ass,json]
  [--granularity ...] [--max-cue-chars N] [--max-line-width N] [--max-cps N]
  [--min-cue-seconds N] [--max-cue-seconds N] [--max-merge-gap-ms N]
  [--no-subtitle-qc] [--force]

qtts project lock [--project FILE] [--out FILE]
```

- `init` 写 `qtts.project.json`（默认当前目录）；已存在需 `--force`。
- `lock` 解析相对 `model_dir`，为所有引用模型/音色记录全量文件 sha256 与大小，
  并记录采样参数、seed、字幕策略、dtype/device 与 Python/torch/transformers 版本。
  输出排序后的 JSON，不含时间戳；原子写入。`--out` 不得指向 `qtts.project.json`
  或输入；没有碰撞检查。
- 需要字幕对齐但本机没有 aligner 时退出 3。
