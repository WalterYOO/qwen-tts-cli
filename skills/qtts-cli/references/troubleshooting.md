# 排障

先跑 `qtts doctor --json`。再按退出码与症状定位。

## doctor 怎么读

```bash
qtts doctor --json
qtts doctor --bench --json
```

| 字段 | 含义 |
|---|---|
| `ok` | 只表示命令执行成功，**不是**环境健康 |
| `healthy` | 只表示没有 incomplete model（以及 `--bench` 是否成功）；**空模型目录时仍可能为 true**，必须结合 `models[]`、torch、tools、advice |
| `incomplete_models` | 权重不完整的模型列表 |
| `models[].weights_ok` | 单个模型权重是否完整 |
| `torch.installed` / `torch.mps` | torch 与 Apple MPS 可用性 |
| `tools.ffmpeg` | ffmpeg 路径；m4a/mp3/mux/atempo 需要 |
| `model_dir` | 当前解析到的模型目录 |
| `advice` | 按环境给出的下一步建议 |
| `warnings` | 版本组合/钉版告警 |

模型缺失或目录不对通常**不会**让 doctor 非零退出；必须读 JSON。

## 按退出码

### 2 参数/用法

- 互斥旗标同时出现：`--out -`/`--json`/`--subtitle-out-prefix -`/`--recipe-out -`；
  `--out` 与 `--out-dir`；`--lufs` 与 `--no-loudness`；`--text`/`--text-file`/`--srt`。
- `--out` 后缀与 `--format` 不一致；`post --out-dir` 缺 `--format`。
- 脚本 schema/未知字段/重复 id；manifest 坏 JSON/重复 key。
- `serve` 非 loopback 缺 token；`--allowed-host` 带端口/scheme。

### 3 模型/后端/编码器

- 模型目录缺失或不完整：`doctor --json` 看 `incomplete_models`、`weights_ok`；
  确认 `QTTS_MODEL_DIR`/`--model-dir`。
- torch/依赖缺失：`torch.installed=false`，按 `advice` 安装。
- ffmpeg 缺失：m4a/mp3/m4b/mux/atempo/bed 都需要；`tools.ffmpeg=null`。
- dashscope 缺 `DASHSCOPE_API_KEY`；`--offline` 与远程 backend 冲突。
- `qwen-tts` 与 `qwen-asr` 的 transformers pin 冲突是已知现象；实测 4.57.6
  可同时运行，`pip check` 报错不必修。

### 4 输入/能力/路径

- 参考音频不合格：<2s、采样率过低、削波、静音过多；按提示重录/裁剪。
- 音色不存在、`prompt.safetensors` 缺失、ICL clone 缺 `ref_text`。
- `--instruct` 用在 Base/0.6B；`design:` 用在只支持 builtin 的后端。
- 对齐覆盖率 <30%；SRT 结构不合法；cue 超过 285s 或边界不在静音内。
- 路径碰撞：`post` 的输入/输出/report 相同，或大小写/NFC/symlink 冲突。
- `dub` 输出已存在未给 `--force`；`audiobook` 已有输出未给 `--force`。
- finetune 项目锁/artifact 锁冲突；checkpoint 目录结构不合法。
- `serve` 绑定失败。

### 5 质量门禁

- `check` 的 `ratio > threshold`（默认 0.03）。
- `qc`/`say --qc` 的显式阈值越界。
- 字幕 QC：CPS/行宽/时长/重叠/标点悬挂等；可修复的会修复，仍超限才退出 5。
- `bench` 的 `--max-cer`/`--min-speaker-similarity` 门禁。
- `dub --overflow fail`。
- `finetune eval` 的 CER/相似度门禁。

**退出 5 时产物通常已经写出**；读报告，不要重跑前先确认是否只需要调阈值。

### 6 批处理部分失败

- `say --stdin --jsonl` 或 `batch run` 有任务失败。
- 读 `status.jsonl`：`status`、`attempts`、`last_error`、`transient`。
- `--resume` 只复用 `done` 且 `job_key`/sha256/recipe 全部匹配的任务；
  `--resume --retry N` 只重试 `transient=true`。
- `post` 多输入部分失败也返回 6。

### 130 中断

- 长任务（batch/audiobook/dub/align）被 Ctrl-C。
- `audiobook` 在提交点前回滚，提交点后保留完整新文件。
- `dub` 没有跨文件 journal；可能留下新旧混合文件，重跑重建。
- `batch` 用 `--resume`；`align` 没有 resume，重跑即可。

## 常见症状

| 症状 | 处理 |
|---|---|
| `qtts: command not found` | 用 `uv run qtts` 或 `.venv/bin/qtts`；确认已安装 |
| 合成很慢 / 内存高 | 本地模型大；关闭其他应用；用 `--dry-run` 先规划 |
| 输出被截断 | 增大 `--max-new-tokens`；检查文本是否超预算 |
| 字幕时间轴为空/过短 | 确认 aligner 模型存在；长音频需要 ASR |
| `voice preview` 失败 | clone 缺 prompt/ref_text；design 缺 description |
| `voice import` 失败 | 归档缺白名单成员、hash 不符或路径非法；不要手改 zip |
| `import-btts` 拒绝 | manifest 校验/文件集/路径安全失败；先解压检查，不要绕过 |
| `serve` 启动即退出 | 端口占用、token/Host 规则、backend 未配置 |
| `serve` 请求 429/503 | 队列满/连接超限；调 `--queue-size`/`--max-connections` |
| `finetune train` 立即失败 | 项目路径/锁/官方脚本/解释器问题；先 `--dry-run` |
| `project lock` 退出 3 | 本机缺 aligner 或模型；先补齐模型 |
| `doctor --bench` 失败 | 看 `bench.cases[].error`；通常是内存/设备/模型 |

## 正常的“吓人”状态

- `pip check` 报 `qwen-tts requires transformers==4.57.3`：已知冲突，不是故障。
- `doctor.ok=true` 但 `healthy=false`：命令跑成功，但模型不完整。
- `ok=true` 且 `healthy=true` 但 `models=[]`：模型目录为空，仍不能生成。
- `qc` 的 `lufs_integrated`/`true_peak_dbtp` 为 `null`：数字静音、音频过短或
  scipy 缺失；看 `unavailable`，不要编造。
- `takes` 的某些指标为 `null`：没有请求该准则或评分不可用。
- 字幕 QC warning 但退出 0：已修复或仅提示，不等于失败。
- `align` 长音频 warning “boundaries are approximate”：`--approximate` 的预期行为。
