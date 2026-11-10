# 按任务的工作流

所有示例假设 `qtts` 在 PATH，且 `QTTS_MODEL_DIR` 已指向本地模型目录。
真实合成会加载模型；`--dry-run` 不加载权重，但仍解析模型/音色。

## 1. 单条语音 + 字幕 + 质检

```bash
# 先看计划，不加载权重
qtts say "第一段。第二段。" --voice builtin:Vivian --out hello.wav \
  --subtitle srt --dry-run --json

# 确认 will_write，且 hello.wav/hello.srt/hello.recipe.json 不存在后合成
qtts say "第一段。第二段。" --voice builtin:Vivian --out hello.wav \
  --subtitle srt --json

# 转写比对（默认 CER/WER 自动选择，阈值 0.03）
qtts check hello.wav --text-file script.txt --report check.json --json

# 测量音频；只测量时永远 0 退出
qtts qc hello.wav --json --report qc.json

# 归一化到 -16 LUFS / -1 dBTP，输出 FLAC
qtts post hello.wav --out final.flac --lufs -16 --true-peak -1
```

要点：

- `--out` 静默覆盖；先查目标。
- 字幕来自原文；`[[pause:MS]]` 不进入字幕。
- `check` 超过 300s 退出 4；`qc` 的 `null` 是正常测量缺失，不要填数。
- `post` 的 true peak 以编码后文件为准。

## 2. 多角色脚本

```yaml
# roles.yaml
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
qtts script lint roles.yaml --json        # 只校验；不写计划、不加载权重，但仍扫描模型
qtts script compile roles.yaml -o build.plan.json
qtts say --script roles.yaml --out roles.wav --subtitle srt --json
```

- `format` 只能作业级设置，不能按段覆盖。
- 每段可覆盖 `voice/language/instruct/seed/max_new_tokens/pause_after_ms`。
- 语法/schema 错误在任何模型扫描前退出 2；未知音色/能力冲突退出 4。
- 多句/预算切分产生 `s001.p1` 等 piece id；`source_segment_id` 仍是 `s001`。

## 3. 批量清单

```jsonl
{"id":"a","text":"第一段。","voice":"builtin:Vivian","language":"Chinese"}
{"id":"b","text":"第二段。","voice":"design:温柔女声","subtitle":"srt"}
```

```bash
# 新目录；plan 会写 plan.json/status.jsonl，不是只读
qtts batch plan manifest.jsonl --out-dir build-01
qtts batch run manifest.jsonl --out-dir build-01 --jobs 2 --json

# 中断后续跑；只重试 transient 失败
qtts batch run manifest.jsonl --out-dir build-01 --resume --retry 2 --watch
```

- 默认 `--jobs 1`；只是调度并发，模型推理全局串行。
- 复用旧 `--out-dir` 前先 diff `plan.json`/`status.jsonl`；`--resume` 在
  `job_key` 漂移时也会重跑并覆盖。
- 部分失败退出 6；读 `status.jsonl` 的 `last_error/transient`。
- `--fail-fast` 只停止调度新任务，未启动项标 `skipped`。

## 4. 长音频对齐

```bash
# ≤300s：单次 forced alignment，不加载 ASR
qtts align narration.wav --text-file script.txt \
  --format srt,vtt --out-prefix narration --json

# >300s：默认静音切窗 + ASR 锚定；需要 ASR 与 Aligner 两个模型
qtts align long.wav --text-file book.txt \
  --window-seconds 270 --window-overlap 3 \
  --format srt --out-prefix long --recipe-out long.align.json

# 没有 ASR 时按比例近似；边界不保证，不能与 --srt 同用
qtts align long.wav --text-file book.txt --approximate --format srt
```

- `--srt` 输入的 cue 边界是硬锚点，只重算时间，不改文本。
- `--lexicon` 让 aligner 读 `spoken`，时间轴再映射回显示文本。
- 覆盖率 <30% 退出 4，<80% warning；结构性错误退出 4；字幕 QC 退出 5。

## 5. 有声书

```yaml
# book.yaml
schema_version: 1
title: 夜航
author: 某人
voice: builtin:Vivian
chapters:
  - title: 第一章
    source: chapters/01.txt
  - title: 第二章
    text: |
      第二章的正文直接写在项目里。

      第二段。
```

```bash
qtts audiobook build book.yaml --dry-run --json
qtts audiobook build book.yaml --out-dir build --m4b --lufs -18
```

- 默认输出在项目文件旁 `build/<project-stem>.<fmt>`。
- 每章必须且只能有一个 `source` 或 `text`；段落按空行切分。
- 逐段缓存复用；`--keep-stems` 保留中间文件；已有输出需 `--force`。
- 发布有 journal，可自动恢复被中断的事务。

## 6. 视频配音

```bash
qtts dub build subs.srt --voice builtin:Vivian --dry-run --json
qtts dub build subs.srt --voice builtin:Vivian --out dub.wav --json

# 项目定义逐角色音色；EDL 25fps；背景 bed + mux + 分轨
qtts dub build timeline.edl --project dub.yaml --fps 25 \
  --video movie.mp4 --bed-db -12 --mux --stems-out stems --out dub.wav
```

- `--out` 省略时为 `dub.wav`；已有输出未给 `--force` 退出 4。
- 音色优先级：`--speaker-voice` → 项目 `speakers` → `--voice` → 项目 `voice`。
- 时长适配 0.85–1.25 倍；超出目标帧默认 `warn`，`--overflow fail` 退出 5。
- `--bed-db`/`--mux` 需要 `--video`；stems 已存在需 `--force`。
- 没有跨文件 journal；中断后重跑重建。

## 7. 多 take 选优

```bash
qtts say "你好。" --voice builtin:Vivian --takes 4 --select lufs,cer \
  --out hello.wav --json

qtts takes list tk_0123456789abcdef --json
qtts takes compare tk_0123456789abcdef --ids 1,2,3 --json
qtts takes promote tk_0123456789abcdef --id 2 --out chosen.wav
```

- `--takes` 与 `--select` 必须成对，N 2–16；准则顺序即优先级。
- `cer` 需要本地 ASR；`speaker-sim` 需要 clone/profile 参考。
- take 在基准 seed 上依次 +1，可复现。
- `promote` 不转码；目标后缀必须与 job 格式一致；可以覆盖任意 `--out`。

## 8. 可复现项目锁

```bash
qtts project init . --voice builtin:Vivian --language Chinese \
  --subtitle srt --format wav
qtts project lock --project qtts.project.json --out qtts.lock.json
```

- 锁文件记录所有引用模型/音色的全量 sha256 与大小、采样参数、seed、字幕策略、
  dtype/device 与运行时版本；排序后写入，不含时间戳。
- 提交前先 `project lock`，把 `qtts.project.json` 与 `qtts.lock.json` 一起保留。
- 不要手改锁文件；模型或音色变化后重新 lock。
