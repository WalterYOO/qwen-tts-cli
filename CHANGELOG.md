# Changelog

本项目的所有重要变更记录在此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- **可插拔后端（FR-P2-02）**：新增 `--backend transformers|dashscope|vllm-omni|PLUGIN` 与 `--backend-model`，覆盖 `say`、`say --stdin --jsonl`、`script compile`、`batch plan/run`。`BackendSpec`/`BackendCapabilities` 统一描述 modes、instruct、sampling、语言与远程属性；后端身份（`dashscope:<model>:<region>:<workspace>` / `plugin:<name>@<version>:<model>` / `transformers` / `vllm-omni`）写入 voice identity、cache key、batch `job_key`、recipe 与 JSON payload。`dashscope` 仅支持 `builtin:` 与实时会话 API（`dashscope>=2.0.1`，key 只从 `DASHSCOPE_API_KEY` 读取，永不进入报告），`vllm-omni` 为离线 batch 且 clone 只需 `ref.wav`。后端解析、语音 capability、instruction、offline/key、ref 路径等检查全部发生在 create/load/写文件之前；`--dry-run` 不 import `dashscope`/`vllm_omni`/`torch`/`qwen_tts`。TTS 模型来自后端，aligner/QC/takes 仍使用本地扫描模型。`batch --resume` 检测 `plan.json` 的后端身份不一致并在任何写入前退出 2。
- **插件 SDK（FR-P2-08）**：通过 `qtts.backends` / `qtts.postprocess` entry point 发现插件；后端插件实现 `api_version`、`name`、`capabilities()`、`models(cfg, backend_model=None)`、`create(cfg, model)` 与可选 `preflight`，后处理插件实现 `process(samples, sr, params)`。新增 `qtts plugins list [--strict] [--json]`，状态为 `ok/unsupported_api/broken/invalid/duplicate_name`，发现阶段异常只产生 warning，`--strict` 在任一插件不可用时退出 3。`post --plugin` 在重采样后、响度增益前执行，严格校验返回的声道/采样率/finite/时长（0.5x–2.0x），失败不写输出；`--plugin-param KEY=VALUE` 拒绝重复 key 与 `NaN/Infinity`。
- **音频后处理（FR-P2-05）**：`qtts post INPUT ... (--out FILE | --out-dir DIR)` 按固定顺序完成静音修剪、声道转换、重采样、BS.1770-4 响度归一化、`lookahead-min-4x` true-peak 限幅与 wav/flac/mp3/m4a 编码。`--lufs` 未配 `--true-peak` 时默认 -1.0 dBTP；最终 true peak 以发布文件 PCM 解码测量为准（mp3/m4a 编码后解码回 24-bit WAV），必要时最多重编码 2 次，仍超限则该文件失败且不替换输出。输入仅接受 wav/flac；输出后缀与 `--format` 必须一致，输入/输出/report 在写入前做 symlink、大小写与 NFC 规范化碰撞检查。多输入逐文件报告，部分失败退出 6；`--dry-run` 不创建目录、不编码、不写报告；`--json` 与 `--report` 逐字节一致。
- **性能基准（FR-P2-06）**：新增 `qtts bench --suite smoke|batch|longform`，支持 `--model/--voice/--language/--text/--text-file/--count/--runs/--cer/--asr-model/--speaker-sim/--reference-voice/--max-cer/--min-speaker-similarity/--dry-run/--report/--json`。计时用外部 `perf_counter` 未舍入值聚合，`rtf` 取每 run 中位数、`wall_rtf=全场 wall/全场 audio`；TTS → 释放 → ASR（可选）→ 释放 → Base（可选）单驻留，`finally` 保证释放。CER 逐 case 按该段语言转写并累计字符编辑距离；speaker-sim 优先 prompt `ref_spk_embedding`、否则用同一 Base 从 `ref.wav` 提取，任一段空/零/维度不符时整体为 `null` 并记录已评分段数。`--dry-run` 只做元数据/文件/计划检查，不 import torch/qwen、不加载权重、不写文件；`--json` 与 `--report` 逐字节一致，门禁失败输出完整 payload 并退出 5，报告写失败退出 4。

## [0.2.0] - 2026-11-04

### Added

- **多 take 资产库（FR-P1-04）**：`say --takes N --select cer,speaker-sim,lufs` 一次生成 2–16 个可复现 take，按准则优先级排序并自动选择；take 1 保留原 seed，后续 take 在 plan 与每个 segment 上依次加 `t−1`，溢出在生成前退出 2。新增 `takes list/compare/promote`（不加载模型），资产写入 `<data-dir>/takes/<job_id>/`，`job_id` 由规范化请求哈希得到；相同请求复用已完成 job，`scoring_failed` 只重试评分；job 锁、staging 发布、损坏隔离、sha256 校验与符号链接/路径穿越防护齐备。`--takes` 与 JSONL/batch 互斥；promote 只复制原始字节、不转码。
- **长音频对齐（FR-P1-11）**：`align` 对 >300s 音频按合格静音切窗（core 240–285s、目标 270s、至少 2s 重叠、单窗 ≤300s），默认逐窗用 Qwen3-ASR 无时间戳转写做单调文本锚定，再逐窗调用 forced aligner，最后按全局字符归属合并并只构建一次 cue；新增 `--window-seconds`、`--window-overlap`、`--approximate`、`--asr-model`。SRT 输入以 cue 边界为硬锚点且不跑 ASR，`--srt --approximate` 在加载模型前退出 2；`--approximate` 按实际 audio 区间（含重叠）分配文本，静音计划无解时退化为固定间隔并 warning，时间倒退做确定性单调裁剪；末窗允许有界尾部溢出并裁剪。长路径报告/recipe 增加真实覆盖率门禁与 `windowing`，短音频保持原字节且 `windowing.enabled=false`。字幕与 recipe 先 staging 后原子提交且在失败时回滚，ASR 完全释放后才加载 aligner。
- **多角色脚本**：`script lint FILE`、`script compile FILE -o PLAN.json`、`say --script FILE`，接受 YAML/JSON；逐段 `voice/language/instruct/seed/max_new_tokens/pause_after_ms`，缺省 id 自动生成，重复 id/未知字段/能力冲突在加载模型前失败；编译计划稳定可复现。
- **词典与导演标记**：`say/align --lexicon` 让 TTS 读 `spoken`、字幕保留原文 `term`，显式 spoken→display 字符映射；`[[pause:MS]]` 支持句首/句间/句尾/连续出现，负数和 `speed/rate/pitch` 等不支持标记退出 2。
- **批量清单**：`batch plan/run`，`--jobs`、`--resume`、`--retry`、`--fail-fast`、`--watch`；确定性 `plan.json`、原子 `status.jsonl`、job_key/输出 sha256/recipe 校验、部分失败退出 6。
- **ASR 质检**：`check AUDIO --text/--text-file/--srt`，Qwen3-ASR 转写后按语言自动选 CER/WER，输出编辑距离、错误片段与 `schema_version:1` 报告；`ratio > threshold` 退出 5，参考输入互斥、音频上限 300s、`errors[].start/end` 在 Wave 3a 固定为 `null`。
- **音频 QC**：`qc AUDIO` 测量 peak/clipping/DC/silence/LUFS/true peak/speech rate/speaker similarity；所有阈值可选且默认 `None`，只测量时退出 0，越界退出 5；`--json` 与 `--report` 逐字节一致。`say --qc` 复用同一实现并把指标写入 payload 与 recipe。
- **自动参考文本**：`voice add --kind clone --auto-ref-text`（可配 `--language`/`--asr-model`）在 staging 内用 ASR 转写参考音频，成功后释放 ASR 再加载 TTS；`voice.json` 记录 `ref_text_source=asr` 与 `asr_model`。
- ASR 模型发现：`catalog.scan()` 区分 `Qwen3-ASR-*` 与 `Qwen3-ForcedAligner-*`（两者可能同为 `model_type=qwen3_asr`），新增独立 `select_asr()`，缺失退出 3、歧义退出 2；`models list`/`doctor` 显示 ASR 能力。
- `scipy>=1.10` 基础依赖，用于 BS.1770-4 LUFS 与 true peak 重采样；scipy 不可用时对应指标为 `null` 并给出 `unavailable` 原因。
- `TtsEngine.extract_speaker_embedding()`：对非 24kHz 音频重采样后调用底层说话人编码器，返回扁平 CPU float32。
- `PyYAML>=6.0` 基础依赖；YAML 严格拒绝重复 key、多文档、alias/anchor 与超大输入。

### Changed

- `SegmentPlan` 保存完整 voice/model 快照与 `source_segment_id`；预算切分 id 形如 `s001.p2`。多处执行器只保留一个 TTS 模型槽，切换模型前释放旧 engine 与 clone prompt。
- 文本投影顺序固定为 raw → marker → 词典 display → spoken → 分句/预算；缓存键包含词典内容哈希，recipe 记录逐段 voice/model 与 `lexicon_sha256`。`PreparedText` 额外携带 display 字符的原始 `source_spans`，SRT 长音频只投影一次即可把原始 cue 边界映射到 display 坐标。

### Fixed

- Wave 3c 返修：speaker-sim 的参考音色身份（名称/kind/prompt 与 ref 哈希/embedding 来源/评分版本）进入 canonical request、job_id 与每 take recipe，切换 `--reference-voice` 不再复用旧 job；`takes promote` 在 `scoring_failed` 上先校验音频/recipe 与更新后的 job，再写 job 和目标文件，失败不留半写产物；damaged job 重建失败会恢复备份；`--select cer` 在 TTS 前预检 `qwen_asr`；`takes list/compare` 校验音频哈希与 recipe；复用 job 与下游输出前校验 selected recipe；`validate_job` 补齐 segment 结构/seed/时间边界校验；`--out`/`--recipe-out`/字幕前缀禁止覆盖 job 资产。
- 报告类命令的 canonical JSON 不再把进程 `exit_code` 写入文档，`--json` 与 `--report` 保持逐字节一致。
- Wave 3a 返修：`check` 在任何音频/模型检查前校验并 canonicalize ASR 语言（非法语言恒退出 2）；小写语言名与规范名产生相同的数字/日期/货币归一化；`asr.language` 不再用参考语言回填，新增 `comparison_language`。
- `voice add` 补齐 `--asr-model`，与 `check` 共用 `select_asr` 消歧；`say --qc --reference-voice` 在生成前校验 profile，失败不产生音频/recipe。
- 说话人 embedding 能力改为类级探测、加载权重前失败；qc 的 Base 模型选择错误统一退出 3；空文本不再触发语速门禁（退出 2）。
- ASR 缓存键包含 aligner；ASR `model_type` 只接受精确 `qwen3_asr`；Wave 3a 集成测试改从 `QTTS_SMOKE_MODEL_DIR` 读取模型目录。
- `say --qc --reference-voice` 在生成前解析 Base 并做无权重能力探测；参数/输出组合错误优先于 profile/模型预检。`voice add` 在没有 `--auto-ref-text` 时拒绝 `--asr-model`。
- 多角色脚本：一个 segment 含多句、marker 分块或预算切分时，所有 piece 使用唯一稳定 id（`s001.p1`），不再重复 `s001`。
- `say --script` 不再静默忽略 TEXT/`--stdin`/`--file`（退出 2）；显式 `--language/--instruct/--max-new-tokens/--pause-ms` 覆盖脚本 defaults 并进入 plan/recipe。
- 脚本 schema 错误统一退出 2（重复 id、未知字段、类型/整数错误、未闭合 marker、顶层 `project`），并先校验脚本再扫描模型；YAML 纯 anchor 也会被拒绝。
- 脚本 segment 首尾 `[[pause:MS]]` 进入计划与音频；中间段首尾停顿合并到相邻 `pause_after_ms`。
- `align --srt --lexicon` 的 coverage 按 aligner 实际读取的 spoken 文本计算；顺序投影只跳过标点/空白，不再跳过内容字符。
- batch `job_key` 纳入 `lexicon_sha256`；显式 `format` 不再绕过输出后缀校验；仅 transient 失败重试，永久失败不会因 `--resume --retry` 复活；status 使用 `last_error`/`transient`，manifest 重复 key 退出 2。
- ICL clone 的 `ref_text` 进入 voice identity/cache key；`--out -` 的 fallback recipe 使用完整请求 hash 命名。

## [0.1.1] - 2026-11-03

### Added

- `--granularity sentence` for `say` and `align`：按句子分段生成 cue；`char` / `word` 保留。
- 中英文逐句字幕与 `join(cue.text) == normalized_text` 的回归测试。

### Changed

- 字幕粒度默认从 `char` 改为 `sentence`，一句一 cue，修复中文 SRT 默认逐字、不可读的问题。
- `GenerationPlan.granularity` 与 `build_plan()` 默认值同步为 `sentence`；`--dry-run` 与 recipe 报告同一值。

### Fixed

- 默认 SRT 输出不再把 CJK 文本拆成单字 cue。

### Known Limitations

- `sentence` 模式暂不切分超长句、不合并过短句；CPS 与行宽门禁属 P1。

## [0.1.0] - 2026-11-03

首个 P0 / M0 版本。定位是「以原文为内容真值的本地语音制作工具」：统一音色资产，生成音频与时间轴，产出可复现配方。

### Added

- **统一生成入口 `qtts say`**
  - `builtin:`、`voice:`、`design:` 三种音色引用，自动路由到 CustomVoice、Base 克隆或 VoiceDesign
  - 输入支持位置参数、stdin（`-`）、`--file`、`--script`
  - 输出支持 wav/flac、文件或 stdout（`--out -`）
  - `--subtitle srt,vtt,ass,json` 与 `--subtitle-out-prefix`
  - `--dry-run` 输出分段、模型、语言、token/时长预算、缓存命中与依赖，不加载模型
- **音色资产库 `qtts voice`**
  - `add`（clone / design）、`list`、`inspect`、`preview`、`rm`
  - 克隆支持 ICL 与 xvec 模式；design 支持 `--materialize-clone` 固化为克隆 profile
  - profile 保存 `voice.json`、`prompt.safetensors`、`ref.wav`、`preview.wav`
  - 原子提交：临时目录写完再 `os.replace`，失败不留半成品
- **参考音频准入检查**
  - 时长、采样率、声道、削波、静音比例、有效语音占比等
  - consent / license 元数据与 `--i-have-rights`
- **文本流水线**
  - 数字、日期、金额、单位、URL、emoji、全半角规范化
  - 语义分段与 token/时长预算（`max_new_tokens`，12Hz ≈ 12.5 token/秒）
  - `--language auto` 逐段判定并校验
  - 中文日期紧邻与长引号分段处理
- **字幕与对齐**
  - `qtts align` 支持 `--text` / `--text-file` / `--srt` / `--script`
  - SRT / VTT / ASS / JSON 四格式；`--srt` 输入保留原 cue 编号与边界
  - `say --subtitle` 逐段对齐并累加时间偏移
  - 未匹配区间填充、时间裁剪到 `[0, audio_duration]`、强制单调且 `end > start`
  - 字符覆盖率检查：低于 30% 退出 4，30–80% 告警
- **可复现与缓存**
  - 每次生成写 `<stem>.recipe.json`：seed、模型 revision、dtype、采样参数、文本/音频/prompt 哈希、逐段语言与 instruct
  - 内容哈希缓存；命中时不加载模型
  - 缓存键包含模型 fingerprint 与实际 dtype
  - `--no-cache`、缓存探测、并发安全的原子写
- **模型管理**
  - `qtts models list/pull/prune`
  - HF / ModelScope 源、revision pin、`--offline`
- **诊断与基线**
  - `qtts doctor`：Python/torch/设备、ffmpeg/SoX、磁盘、模型完整性、依赖版本与钉版告警、能力矩阵
  - `doctor --bench`：加载耗时、RTF、峰值 RSS
- **CLI 基础设施**
  - 全部命令支持 `--json`
  - 稳定退出码：0/2/3/4/5/6/130
  - stdout 单消费者约束；`--out -` 时 stdout 为纯音频，摘要写 stderr
  - argparse 级错误也遵守 `--json` 单对象输出
- **测试**
  - 119 个测试：能力矩阵、分段、预算、缓存、字幕、voice store、CLI、审查回归
  - 5 个集成 smoke：中文 / 英文 / 同进程 TTS→align
  - 集成测试通过 `RUN_INTEGRATION=1` 启用

### Changed

- 二进制命令名使用 `qtts`，PyPI distribution 为 `qwen-tts-cli`，import package 为 `qtts_cli`，避免与官方 `qwen-tts` 冲突。
- M0 采用**单环境混装**：`qwen-tts 0.1.1` + `qwen-asr 0.0.6` + `transformers 4.57.6`。两者元数据钉版不一致，但端到端实测可用；`pip check` 告警属预期。子进程桥接保留为 fallback。
- sidecar 命名从 `<audio>.recipe.json` 统一为 `<stem>.recipe.json`（`a.wav` → `a.recipe.json`）。
- `say` 的对齐按段执行，5 分钟限制按单段而非整篇计算。
- `serve` 定位为 P2；Agent 场景优先 `say --stdin --jsonl`。

### Fixed

预发布审查中发现并修复：

- `--subtitle` 的 aligner 选择与 5 分钟校验移到模型加载前；`--dry-run` 同步报告缺失依赖
- stdout 同时承载音频、字幕、JSON 的冲突；`--out -` 二进制流不再混入摘要
- `ValueError` / `FileNotFoundError` 统一包装为退出码 2/4，不再输出 traceback
- `--json` 错误路径只输出单个可解析 JSON
- sidecar 路径、顶层 seed、stdout 输出落点与 preview 字段补全
- 字幕部分匹配时的重叠、零时长、越界与单调性问题
- `voice add` / `preview` 绕过能力矩阵；clone preview 在加载前检查 prompt 与 ICL 元数据
- 逐段语言未校验、clone prompt 缺失发现过晚
- 未知输出后缀静默回退 WAV
- `--offline` 未传递到模型加载器
- profile 非原子写入与 design→clone 期间的双模型驻留
- SRT 输入丢失原 cue 边界
- 缓存 probe 忽略 `--no-cache`、空文件误命中；缓存键缺 model fingerprint 与实际 dtype
- 取消删除命令返回非 0 退出码
- 缓存/profile 固定 `.tmp` 文件名导致的并发覆盖

### Known Limitations

- 没有本地 token 级流式生成；`non_streaming_mode` 只模拟流式文本输入。
- Base 与 0.6B CustomVoice 不支持 instruct；0.6B 的 instruct 在底层被静默丢弃，CLI 在参数层报错。
- ForcedAligner 单样本约 5 分钟；超长音频分窗尚未实现。
- 字幕低覆盖率阈值（0.3 / 0.8）为经验值，极短文本可能误报。
- 缓存 fingerprint 对大权重只取文件名与大小，同尺寸替换无法发现。
- SRT 输入会先规范化再对齐：cue 边界保留，但文本是规范化后的写法。
- MPS 不保证 bit-perfect；同 seed 只保证趋势一致。
- ASR 转写侧在 `transformers==4.57.3` 下未验证；P1 实现 `check` 时需补测。
