# 生产命令参考：audiobook / dub / check / qc / post / bench / serve / finetune / models / doctor / plugins / completion

## audiobook

```bash
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

- 默认输出目录是**项目文件旁的 `build/`**，文件名 `<project-stem>.<fmt>`；
  `--out` 与 `--out-dir` 互斥，`--out -` 退出 2。
- 项目文件严格拒绝未知字段、重复键、alias/anchor、多文档；每个 chapter 必须且
  只能有一个 `source` 或 `text`。段落按空行切分，Markdown 不解释。
- 默认格式 wav。m4a/m4b/mp3 需要 ffmpeg；`--bitrate` 32–320（AAC 默认 128，
  MP3 默认 192）。
- 响度默认对 m4a/m4b/mp3 与含 BGM 的项目启用；wav/flac 默认关闭。
  `--lufs` 开启、`--no-loudness` 关闭，二者互斥；`--true-peak` 需要响度归一化。
- 逐段缓存指纹只覆盖合成输入；改采样率/暂停/BGM/格式不会重新合成。
  `--force`/`--no-cache` 绕过复用；损坏缓存视为 miss。
- 发布使用 journal（`prepared` → `published`），下次构建自动恢复被中断的事务；
  已有输出需 `--force`。
- 退出：参数 2；项目/输入/BGM/路径/锁 4；模型/后端/ffmpeg/编码 3；中断 130。

## dub

```bash
qtts dub build TIMELINE.srt|TIMELINE.edl
  --out FILE [--format wav|flac|m4a|mp3]
  [--voice REF] [--speaker-voice SPEAKER=REF ...] [--project FILE]
  [--language LANG] [--instruct TEXT]
  [--video FILE] [--bed-db DB] [--mux]
  [--stems-out DIR]
  [--fps 25] [--sample-rate 24000]
  [--min-speed 0.85] [--max-speed 1.25] [--overflow warn|fail]
  [--lufs -18 | --no-loudness] [--true-peak -1.5]
  [--model ID] [--keep-work] [--force] [--dry-run] [--json] [--quiet]
```

- `--out` 省略时为 cwd 的 `dub.wav`；后缀必须与 `--format` 一致；`--out -` 退出 2。
  manifest 为 `<out-stem>.dub.json`。
- 时间轴只接受 `.srt` 或 CMX3600 子集 `.edl`；EDL 用 `--fps` 1–120（默认 25），
  仅 `C` 事件、轨道含 `V`，非 drop-frame。
- SRT 每个 block 必须是索引行、时间行、非空文本；结构不合法退出 4。
- cue 按半开区间 `[start, end)` 放置；文本可写 `speaker: text`。
  音色优先级：`--speaker-voice` → 项目 `speakers` → `--voice` → 项目 `voice`；
  未映射且无默认音色退出 4。`__anonymous__` 是保留键，出现即退出 4。
- 时长适配：速度比夹到 0.85–1.25，差 >0.005 才调用一次 `ffmpeg atempo`，再精确
  重采样补静音；超出目标帧按 `--overflow warn|fail`（默认 warn）。
- 默认响度开启（-18 LUFS / -1.5 dBTP）；`--no-loudness` 仍测量并 warning。
- `--bed-db` 必须与 `--video` 同用；`--mux` 也要求视频含视频流。
- stems 目录可复用；已存在的 stem 文件或 `timeline.json` 需 `--force`。
- 发布逐文件原子，但**没有跨文件 journal**；SIGKILL/掉电可能留下新旧混合文件，
  重跑（必要时 `--force`）重建。
- 退出：参数 2；时间轴/路径/锁 4；模型/ffmpeg/DSP 3；overflow fail 5；中断 130。

## check

```bash
qtts check AUDIO (--text TEXT | --text-file FILE | --srt FILE)
  [--language auto|Chinese|English|...]
  [--asr-model PATH|ID] [--metric auto|cer|wer]
  [--threshold 0.03] [--report FILE] [--json]
```

- 参考输入三者互斥且必须给一个；音频不存在/不可读、超过 300s 退出 4（加载模型前）。
- `auto`：中日文用 CER，空格分词语言用 WER；CJK 的 CER 去掉所有空白按字符比较。
- `ratio = distance / max(1, 参考单元数)`；`ratio <= threshold` 通过，默认 0.03，
  超阈值退出 5。
- 报告含 `schema_version/ok/audio/metric/threshold/ratio/distance/reference_units/
  units/reference/hypothesis/errors/asr/comparison_language/normalization`；
  `--json` 与 `--report` 逐字节一致。`--report` 不得指向被分析音频；没有碰撞检查。
- 语言非法或 ASR 模型不支持时在加载权重前退出 2；ASR 缺失 3、歧义 2。

## qc

```bash
qtts qc AUDIO [--text TEXT | --text-file FILE] [--language auto]
  [--reference-voice NAME] [--model MODEL]
  [--max-clipping-ratio N] [--target-lufs N --lufs-tolerance N]
  [--max-true-peak-dbtp N] [--max-dc-offset N] [--max-silence-ratio N]
  [--min-speech-rate N] [--max-speech-rate N]
  [--report FILE] [--json]
```

- 只测量时退出 0；任何显式阈值越界：完整指标照常输出、`ok=false`、退出 5。
- 指标：`peak_dbfs`、`clipping_ratio/count`、`dc_offset`、`silence_ratio`、
  `longest_silence_s`、`lufs_integrated`、`true_peak_dbtp`、`speech_rate`、
  `speaker_similarity`。
- 数字静音、音频过短或 scipy 不可用时指标为 `null`，原因在 `unavailable`；
  **不要编造数值**。`--report` 不得指向被分析音频；没有碰撞检查。
- 语速阈值需要非空文本；`--target-lufs` 与 `--lufs-tolerance` 必须成对。
- `--reference-voice` 必须是 clone profile，需要 Base 模型提供说话人 embedding。
- 超过 2 声道退出 4；模型/能力问题退出 3。

## post

```bash
qtts post INPUT [INPUT ...] (--out FILE | --out-dir DIR)
  [--lufs N] [--true-peak DBTP]
  [--sample-rate HZ] [--channels keep|mono|stereo]
  [--format wav|flac|mp3|m4a] [--bitrate 192k] [--bit-depth 16|24]
  [--trim-silence] [--trim-threshold-dbfs -50] [--trim-min-silence-s 0.2]
  [--trim-padding-ms 50] [--trim-mode leading|trailing|both]
  [--plugin NAME] [--plugin-param KEY=VALUE ...]
  [--dry-run] [--report FILE] [--json]
```

- 输入只支持 `.wav`/`.flac`；`--out` 只接受一个输入；`--out` 与 `--out-dir` 互斥
  且必须给一个；`--out-dir` 必须给 `--format`。
- 固定 DSP 顺序：读取 → 静音修剪 → 声道转换 → 重采样 → 响度增益 →
  true-peak 限幅 → 最终测量 → 编码。
- `--lufs` 范围 [-70, 0]；未给 `--true-peak` 时默认 -1.0 dBTP；`--sample-rate`
  [8000, 192000]；`--bit-depth` 只用于 wav/flac；`--bitrate` 只用于 mp3/m4a。
- true peak 以**编码后的文件**解码测量为准（WAV/FLAC 写临时文件读回；mp3/m4a
  编码后解码）。超出上限最多重编码 2 次。
- 写前做路径/大小写/NFC/symlink 碰撞检查；但不拒绝已存在输出。
- `--dry-run` 做预检与内存 DSP，不创建目录、不编码、不写报告。
- 全部成功 0；部分失败 6；全部失败返回第一个错误码；单输入失败返回该错误码。

## bench

```bash
qtts bench --suite smoke|batch|longform
  [--model ID] [--voice builtin:NAME|voice:NAME]
  [--language auto|Chinese|English]
  [--text TEXT | --text-file FILE]
  [--count 8] [--runs 1]
  [--cer] [--asr-model ID]
  [--speaker-sim --reference-voice NAME]
  [--max-cer R] [--min-speaker-similarity S]
  [--dry-run] [--report FILE] [--json]
```

- `--suite` 必填。`--count` 1–64（默认 8，仅 batch）；`--runs` 1–10（默认 1）。
- `--voice` 只接受 `builtin:`/`voice:`；`design:` 退出 2。
- `--cer` 需要本地 ASR；`--speaker-sim` 必须配 `--reference-voice`。
- 指标：`load_ms`、cold/warm first segment、`wall_s`、`generate_s`、`audio_s`、
  `rtf`、`wall_rtf`、RSS、可选 CER/说话人相似度。
- `--dry-run` 不加载模型、不 import torch/qwen、不写文件。
- 门禁失败仍输出完整 payload，`ok=false`、`gate_failures[]`，退出 5；
  报告写失败退出 4。`--report` 不得指向输入/音频；没有碰撞检查。

## serve

```bash
qtts serve [--backend transformers|dashscope|vllm-omni|PLUGIN]
  [--backend-model ID] [--model NAME]
  [--api native|openai|both]
  [--host 127.0.0.1] [--port 8000] [--allowed-host HOST ...]
  [--max-concurrency 1] [--queue-size 32] [--max-connections 32]
  [--keep-jobs 100] [--max-body-bytes 1048576]
  [--token TOKEN] [--shutdown-timeout 10.0] [--dry-run]
```

- 默认 `127.0.0.1:8000`，`--port 0` 用临时端口；本地 backend 的
  `effective_max_concurrency` 恒为 1。
- 绑定非 loopback（含 `0.0.0.0`/`::`）必须给 token，否则绑定前退出 2；
  通配绑定还必须给 `--allowed-host`。没有 TLS。
- token 用 `--token` 或 `QTTS_SERVE_TOKEN`；优先环境变量，避免 argv 泄漏。
- 原生 API：`POST /v1/jobs`、`GET /v1/jobs/{id}`、`GET …/audio`、
  `GET …/subtitles/{fmt}`、`DELETE /v1/jobs/{id}`、`GET /health`。
- OpenAI 兼容：`POST /v1/audio/speech`、`GET /v1/models`；voice 别名映射到内置
  音色；只支持 wav/flac，`speed` 只接受 1，`stream_format` 只接受 `audio`。
- 作业状态 `queued/running/cancel_requested/done/failed/cancelled`；队列满 429；
  连接超限 503。
- 作业产物在 `<data-dir>/serve/staging|jobs/`；`--keep-jobs N` 控制 jobs 保留数，
  `0` 关闭该清理。每次启动仍会清空残留的 `staging/`。
- `--dry-run` 只解析后端元数据，不 bind、不加载模型、不写数据目录。

## finetune

```bash
qtts finetune prepare --project FILE
  [--tokenizer-model-path DIR] [--raw-jsonl FILE] [--output-jsonl FILE]
  [--device auto|cpu|mps|cuda|cuda:0|cuda:1] [--force] [--dry-run] [--offline] [--quiet] [--json]

qtts finetune train --project FILE
  [--train-jsonl FILE] [--init-model-path DIR] [--output-dir DIR]
  [--finetuning-dir DIR] [--python CMD_OR_PATH] [--speaker-name NAME]
  [--batch-size 2] [--lr 2e-05] [--epochs 3]
  [--checkpoint last|N|DIR] [--register NAME] [--register-only]
  [--force] [--dry-run] [--offline] [--json]

qtts finetune eval --project FILE
  [--voice NAME | --checkpoint last|N|DIR]
  [--asr-model SPEC] [--base-model SPEC]
  [--max-cer R] [--min-speaker-similarity S] [--audio-dir DIR]
  [--dry-run] [--json]
```

- 项目文件为严格 YAML/JSON（`schema_version: 1`），含 `dataset`、可选 `prepare`、
  `train`、`register`、`eval`；路径相对项目文件解析。
- `prepare` 校验每条音频为 24kHz 单声道 WAV、0.5–30s（参考 2–30s）、文本非空
  ≤2000 字符，再调用官方 `prepare_data.py` 生成训练 JSONL。
- `train` 以固定 argv 调用官方 `sft_12hz.py`；成功的 checkpoint 为
  `checkpoint-epoch-<N>`，必须包含 `config.json`、非空 `model.safetensors`、
  `speech_tokenizer/`。
- `--register` 只写 profile 元数据（权重/config 哈希、数据集哈希、训练参数、
  consent），不复制/移动/链接 checkpoint；`--register-only` 不启动训练。
- `--force` 只用于 `prepare`/`train`：`prepare` 覆盖 JSONL 并清理 `<output>.tmp`；
  `train` 删除 `.partial` 并替换输出目录。`eval` 没有 `--force`，`--audio-dir` 会
  创建目录并静默替换 `0001.wav` 等同名文件。
- 并发保护：项目锁 `.<project>.finetune.lock` 与 artifact 锁，非阻塞 flock；
  冲突退出 4。
- `--offline` 只向子进程加入 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`；
  `--dry-run` 不 import torch/qwen、不取锁、不建目录、不写文件。
- 退出：参数 2；子进程/checkpoint/哈希/IO 3；项目/路径/锁/选择器/profile 冲突 4；
  评测门禁 5；中断 130。

## models

```bash
qtts models list [--model-dir DIR]
qtts models pull MODEL_ID [--source hf|modelscope] [--revision REV]
qtts models prune [--yes]
```

- `pull` 联网下载，体积大；先问用户，建议 `--revision` 固定版本。
- `prune` 删除本地模型，必须 `--yes`；默认交互确认。
- 模型目录需包含 `Qwen3-TTS-12Hz-1.7B-*` 与 `Qwen3-ForcedAligner-0.6B` 等子目录。

## doctor

```bash
qtts doctor [--bench] [--json]
```

- `ok` 只表示命令执行成功；`healthy` 只表示没有 incomplete model（以及 `--bench`
  是否成功），**不反映模型是否存在**。空模型目录也可能 `ok=true`、`healthy=true`。
- 关键字段：`incomplete_models`、`models[]`（是否非空、`weights_ok`）、
  `torch.installed`、`torch.mps`、`tools.ffmpeg`、`model_dir`、`advice`、
  `warnings`、`capabilities`。本地生成要求 `models[]` 非空且所需 kind 的
  `weights_ok=true`；m4a/mp3/mux 还要求 `tools.ffmpeg`。
- `--bench` 真实加载模型，输出 RTF、加载耗时、峰值 RSS；失败会拉低 `healthy`。

## plugins

```bash
qtts plugins list [--strict] [--json]
```

- 通过 entry point 发现 `qtts.backends` 与 `qtts.postprocess`；
  `entry_point.load()` 会执行插件模块顶层代码，**安装插件等于允许其代码运行**。
- 条目的 `status`：`ok`、`unsupported_api`、`broken`、`invalid`、`duplicate_name`。
  发现异常只产生 warning，列表仍退出 0；`--strict` 在任何插件不可用时退出 3。

## completion

```bash
qtts completion [auto|bash|zsh|fish] [--install|--uninstall|--status]
  [--dir DIRECTORY] [--rc-file PATH] [--follow-symlinks]
  [--dry-run] [--json] [--quiet]
```

- 支持 bash/zsh/fish；PowerShell 未实现。
- 不带动作时把补全脚本写 stdout。
- `--install` 幂等写独立补全文件；只有 shell 不能自动加载时才在 rc 加 marker block。
- `--uninstall` 删除补全文件与 block；`--status` 报告
  `current/stale/not_installed/malformed/unsupported_version/symlink_refused`，
  非 `current` 退出 4。
- 默认拒绝 symlink rc；dotfiles 场景显式 `--follow-symlinks`。
- `--dir` 只写补全文件，不改 rc。
