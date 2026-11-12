# qtts design / metrics / voice 改造方案（定稿）

- 联合方案：Claude + Codex（Orca 终端评审）
- 基线：qwen-tts-cli 0.3.2
- 评审记录：Codex 独立评审指出了 4 个 blocker（design instruction 不生效、clone prompt 未生成、archive 丢 provenance、export --all 契约不完整），本文已全部修正。
- 已拍板：
  1. `design freeze` 的 `--as` **必填**，不设默认。
  2. 多音色 `voice export --all` **本期后置**；先做稳单音色 `.qttsvoice`。
  3. HNR **praat 为 canonical**；缺 `praat-parselmouth` 时 `null + unavailable`，不用自相关近似参与排序/门禁。
  4. design YAML 采用 **btts 结构化字段 + qtts 自研 compiler**，保留 `description` / `instruction_override` 逃生口。

## 0. 原则

- 不伪造引擎能力：qtts 没有 cfg/guidance，就用 `seed × sampling profile`。
- 缺失指标一律 `null + unavailable`，禁止 0/-1/NaN/Infinity。
- 旧入口永久保留：`say --voice design:DESC`、`voice add --kind design`、`voice inspect`、`voice rm`、`.qttsvoice` schema 1。
- 容错读、严格写：list/show 跳过坏 profile；rename/tag/delete 拒绝 symlink/损坏/目录名不一致。
- 先离线层（指标、YAML、lint、report），再接模型路径。

## 1. design 命令组

```text
qtts design init FILE [--template T] [--name NAME] [--language zh|en] [--force]
qtts design lint FILE [--show-rendered] [--strict] [--json]
qtts design explore FILE --out RUN_DIR [--resume] [--wide]
                      [--seeds 19,97] [--sampling default|stable|expressive]...
                      [--max-new-tokens N] [--retry-failed]
                      [--model DIR] [--device ...] [--dtype ...]
                      [--yes] [--dry-run]
qtts design report RUN_DIR [--metrics basic|full]
                         [--csv/--no-csv] [--html/--no-html] [--reel/--no-reel]
                         [--gap 1.0] [--require-full-metrics] [--fail-on-degraded]
qtts design freeze RUN_DIR --take TAKE_ID --name NAME --as clone|design
                        [--confirm-transcript] [--ref-text TEXT]
                        [--allow-long-ref] [--allow-incomplete] [--force]
                        [--tag TAG]... [--license TEXT] [--consent-note TEXT]
qtts say ... [--voice builtin:NAME | voice:NAME | design:DESC | --design FILE]
```

### 1.1 design init

- 默认模板按语言：`zh-late-night-radio/v1` / `en-podcast-host/v1`。
- `--template` 与 `--language` 冲突 → exit 2；`FILE` 已存在且无 `--force` → 2。
- `--name` 默认文件 stem，必须通过 voice 名规则。

### 1.2 design lint

- 离线、确定性；YAML 解析错误 → 4；block → 1；warn/info → 0；`--strict` warn → 1。
- `EXIT_LINT = 1` 语义只用于 lint；`cli.main` 现有 `ok=False → 1` 的行为要用显式 `exit_code` 区分。
- 五维评分：specificity / executability / consistency / language_match / testability，0–100，总分平均；≥85 green，≥70 yellow，否则 red。
- block：empty-instruction、inline-event、language-mismatch、dialogue-in-instruction、no-anchors、invalid-template、instruction-too-long（>1000 字符）、invalid-cover（空/全空白/>200 字符）。
- unsupported-language：**有 override 时 warning，无 override 才 block**。
- warn/info：missing-anchors、vague-term、conflict、no-covers、cover-length、long-instruction（>300）、name-format、unknown-vocab、no-avoid、override-ignores-structure、unknown-field。
- `description` 与 `instruction_override` 同时非空 → warn `both-overrides`，优先级 `instruction_override > description > compiled`。

### 1.3 YAML schema 1

复用 btts 字段；qtts 加 `description` / `instruction_override` 逃生口；模板 ID 加 qtts 命名空间，避免与 btts 同 ID 不同内容。

```yaml
schema_version: 1
name: dusk
language: zh
template: qtts/zh-late-night-radio/v1
identity:
  age: 30s
  gender_presentation: female
  timbre: [warm, clean, slightly_husky]
  pitch: low_mid
  articulation: clear, relaxed
  persona: late_night_host
delivery_baseline:
  pace: medium_slow
  energy: medium_low
  tone: [calm, intimate]
scene:
  proximity: close_mic
  room: dry_studio
avoid: [childlike, shouting]
covers:
  - "晚上好，欢迎收听今天的节目。"
  - "外面下着雨，我们慢慢聊。"
description: null
instruction_override: null
```

- 未知字段保留 `raw`，lint info，不失败。
- YAML 用 `safe_load`，拒绝重复 key、anchor/alias、>256 KiB、非 UTF-8。
- covers 1–2 条；`--wide` 需要至少 2 条，否则 exit 2。

### 1.4 compiler

- `qtts_cli/design_compile.py`，纯 Python，不 import torch。
- `CompiledDesign(structured, rendered_instruction, rendered_sha256, template, template_sha256, compiler_version, language, warnings)`。
- 渲染顺序：identity → delivery_baseline → scene → avoid。
- 未识别 vocab 下划线转空格回退 + warning。
- run/profile 记录 `compiler_version` 与 `template_sha256`；resume 不一致拒绝。

### 1.5 design explore

- `--out` 必填；已存在且非空且无 `--resume` → 2；lint block → 2。
- 默认矩阵：2 个 100 以内随机质数 seed（由全局 `--seed` 驱动的 RNG 抽样）× 2 个 sampling profile（`default`、`stable`）= 4 个候选。
- sampling profile 必须写全 `do_sample/top_k/top_p/temperature/repetition_penalty`：
  - `stable`: temperature 0.7
  - `default`: temperature 0.9
  - `expressive`: temperature 1.1
  - 其余 top_k=50, top_p=1.0, repetition_penalty=1.05
- `--wide`：叠加 cover #2，候选数翻倍；covers < 2 → 2。
- `--seeds/--sampling` 显式覆盖；候选上限 16，超过需显式矩阵 + `--yes`。
- 候选 ID：`s<seed>-c<cover>-<profile>`，只含 `[a-z0-9_-]`。
- `--dry-run` 不加载模型，打印与真实 run 相同的矩阵。
- 一次模型加载；候选原子写 WAV/recipe 并 fsync；最后原子写 `run.json`。
- `.lock` 用 `fcntl.flock(LOCK_EX|LOCK_NB)`；拿不到 → 2；Windows 本期明确报不支持，不静默无锁。
- 第一次 Ctrl-C 当前 take 完成后停止，exit 130；第二次立即中断。
- `--resume`：矩阵/采样/模板/instruction/model fingerprint 任一变化 → 2；校验 WAV sha256；`failed` 默认不重跑，`--retry-failed` 才重跑。

### 1.6 RUN_DIR 与 run.json 契约

```text
RUN_DIR/
  run.json                 # schema 1，唯一索引
  design.yaml              # 输入副本
  design.compiled.json     # compiled 快照
  .lock
  candidates/<take_id>.wav
  candidates/<take_id>.recipe.json
  report/report.json
  report/metrics.csv
  report/report.md
  report/all_candidates.wav
  report/audition.html
```

`run.json` 至少：`schema_version/kind/run_id/created_at/updated_at/qtts_version/source{path,sha256}/design{name,language,template,template_sha256,schema_version,structured,rendered_instruction,rendered_sha256,compiler_version}/model{id,revision,fingerprint,path,dtype,backend}/matrix{seeds,covers,sampling_profiles,sampling_order,wide,max_new_tokens}/candidates[{take_id,seed,cover_index,sampling,status,wav,recipe,sha256,duration_s,completion_status,error}]`。

候选 recipe 记录 `text/text_sha256/language/instruct/instruct_sha256/seed/sampling/max_new_tokens/codec_rate/model/duration_s/sample_rate/finish_reason(null)/generated_tokens(null)`。

### 1.7 design report

- 只读，不加载模型；输出固定 `RUN_DIR/report/`。
- `report.json`：`schema_version/metrics_version/run_id/design/analysis{metrics,f0_backend,hnr_backend,versions}/candidates[]/summary{count,complete,degraded,best_take_id}/warnings`。
- 候选：`take_id/rank/recommended/status/reasons/audio{path,sha256,duration_s,sample_rate,channels}/metrics{basic,full}/completion/unavailable/failure_count/degraded`。
- 不写 `generated_at` 等 volatile 字段；时间用 `run.json.created_at`；路径相对 RUN_DIR。
- `metrics.csv` 固定表头、按 rank/take_id 排序、null 空字段、`csv.writer(lineterminator="\n")`、UTF-8。
- `all_candidates.wav`：只拼 complete/unknown；同采样率/声道；候选间插 `--gap`（0–30，默认 1.0）；不一致跳过 + warning。
- `audition.html`：单文件、内联 CSS/JS、无网络请求；HTML 转义；shell quote；盲听模式隐藏推荐；不自动播放。
- 缺 full 指标 → warning + null，exit 0；`--require-full-metrics` → 任一缺失 exit 5。
- 无 complete 候选 → `degraded=true`，默认 0；`--fail-on-degraded` → 5。
- 排序：`failure_count` ↑ → completion 严重度 ↑（complete 0 / unknown 1 / suspect 2 / too_short 3 / failed 4）→ clipping ↑ → silence ↑ → hnr ↓ → voiced_ratio ↓ → take_id ↑；缺失值排最后，不计 failure。
- failure_count：suspect/too_short/failed、clipping > 默认阈值、silence > 默认阈值、数字静音/无音频；不含 unknown、null 指标。

### 1.8 design freeze（--as 必填）

**`--as design` 必须先修 instruction 路径：**

- 同时写 `source.description = rendered_instruction` 与 `source.design = {schema_version,template,template_sha256,structured,rendered_instruction,rendered_sha256,compiler_version,yaml_sha256,yaml}`。
- 当前 `plan.py:380`、`plan.py:614`、`commands/voice.py:538` 读 `source.description`；只写 `default_instruct` 不生效。
- 明确 label：design freeze 保存的是**可重生成配方**，不是选中 take 的声学冻结；`say` 目前不会回放 profile 的 seed/sampling，文档必须写清。
- 可选后续：让 `say` 读取 `source.design.rendered_instruction` 优先；本期先写 `source.description` 保证兼容。

**`--as clone` 完整流程：**

- 校验 take 存在、sha256 与 run.json 一致、admission 通过。
- 参考文本 = 该候选 cover text；`--ref-text` 可覆盖但仍要求 `--confirm-transcript`（未提供 → 打印原文 + exit 2）。
- 时长 <1s → 5；<6s warning；>30s 需用户明确同意后加 `--allow-long-ref`，否则 2。
- 加载 Base 模型，`create_clone_prompt_items()` 生成 prompt，`save_prompt_tensors()` 写 `prompt.safetensors`；`source.mode="icl"`、`source.ref_text`、`source.ref_text_source="cover|user"`、`source.origin="design-freeze"`。
- 失败清理 staging；同名 voice 无 `--force` → 2；`--force` 走 staging + 备份 + 失败恢复。
- 拒绝 `suspect_truncated/too_short/failed`；`unknown` 默认拒绝，`--allow-incomplete` 才允许并 warning。

### 1.9 say --design FILE

- 与 `--voice`、`--instruct`、`--script`、`--jsonl` 的互斥关系写进 help 和测试。
- 解析/编译 YAML；block → 2。
- recipe 记录 `design_yaml_sha256`、`rendered_instruction_sha256`、`compiler_version`、seed/sampling。
- `--voice design:DESC` 逐字节保持不变，并加回归测试。

## 2. 指标层

### 2.1 依赖

- 核心：numpy/scipy/soundfile/PyYAML。
- optional extra：`metrics = ["praat-parselmouth>=0.4"]`。
- F0：numpy **YIN/CMNDF** 自实现，40–50ms 窗，不用 librosa。
- HNR：praat canonical；缺失 → `hnr_db=null` + `unavailable["hnr_db"]="dependency_missing:praat-parselmouth"`。
- spectral centroid / LTAS / HF-LF：numpy/scipy 自实现。

### 2.2 模块

```text
qtts_cli/audio_features.py    # F0 / voiced ratio / HNR / spectral / LTAS / HF-LF
qtts_cli/completeness.py        # 截断/不完整判定
qtts_cli/design_report.py     # 候选指标、排序、CSV/HTML/MD/reel
```

### 2.3 full 指标定义

- **F0**：mono downmix；窗长 `max(0.04s, 3/f0_min)`，hop 10ms；YIN CMNDF 阈值 0.15 + 能量门限；搜索 60–500 Hz；输出 median/mean/std/p10/p90。至少 3 个 voiced 帧才出 median/mean；至少 10 个才出百分位，否则 `f0_p10/p90=null` + `insufficient_voiced_frames_for_percentiles`。`f0_backend="yin"`。
- **voiced_ratio**：voiced 帧 / 非静音帧；分母 0 → null + `no_non_silent_audio`。
- **HNR**：`praat-parselmouth` 的 `to_harmonicity_cc(time_step=0.01, minimum_pitch=60, silence_threshold=0.1, periods_per_window=4.5)`；取 finite 且 > -100 dB 的帧中位数；无有效帧 → `no_voiced_frames`；缺依赖 → `dependency_missing:praat-parselmouth`。
- **spectral_centroid**：幅度谱 `sum(f*|X|)/sum(|X|)`，Hann n_fft=1024 hop=256；只在非静音帧上取 median。
- **LTAS slope**：Welch（Hann, nperseg=min(4096,len), noverlap=nperseg//2, detrend=constant, scaling=density），50 Hz–Nyquist 且 ≤8 kHz；`polyfit(log2(f), 10*log10(Pxx), 1)`，dB/octave；<0.5s 或频点 <10 → `too_short_for_ltas`。
- **HF/LF**：LF 500–4000，HF 4000–8000；`10*log10(E_hf/E_lf)`；频带不足 → `band_unavailable`。
- 报告记录 `analysis_channels="downmix"`、`f0_backend`、`hnr_backend`、numpy/scipy/praat 版本。

### 2.4 completion

判定顺序：

1. recipe `finish_reason=length/max_new_tokens` → suspect；`stop/eos` → complete。
2. duration < 0.5s → too_short；clone admission 另用 1s 硬门槛。
3. `max_audio_s = max_new_tokens / codec_rate`；`duration >= 0.98*max_audio_s` → suspect（token_budget_exhausted）。
4. expected 时长按语言字符速率估算；`duration_ratio < 0.5` 且 expected ≥2s → unknown（shorter_than_expected），不直接判截断。
5. 尾部 100ms RMS 与 voiced 中位 RMS 差 > -6 dB 且无自然静音尾 → unknown（abrupt_tail）；接近 token 预算则升级 suspect。
6. 无 recipe → unknown（no_recipe），不默认 complete。
7. 否则 complete。

`failed` 由上层在 WAV 缺失/损坏/生成异常时注入，纯函数不猜。

### 2.5 null / unavailable

- 固定 reason code：digital_silence、too_short_for_lufs、too_short_for_ltas、no_voiced_frames、no_non_silent_audio、insufficient_voiced_frames、insufficient_voiced_frames_for_percentiles、band_unavailable、dependency_missing:praat-parselmouth、analysis_failed、no_text/no_words/cjk_no_word_boundaries、no_recipe。
- `json.dumps(allow_nan=False)`；round 统一。
- 缺失值排序排最后，不计 failure；显式阈值门禁时缺失 → `op=unavailable` violation。

### 2.6 CLI 接入

`qc` 新增：`--metrics basic|full`、`--recipe FILE`、`--fail-on-incomplete`、`--require-full-metrics`、`--hnr-backend praat`，以及 opt-in 阈值：duration、voiced_ratio、hnr、f0 min/max/std、spectral centroid min/max、ltas slope min/max、hf_lf min/max。

`check` 本期**不接** acoustic metrics；`--qc-metrics` 后置。`bench` 本期不接。

报告 schema 保持 1，新增 `metrics_version: 1` 与 additive 字段。

## 3. voice 重新设计

### 3.1 命令面

```text
qtts voice list [QUERY] [--match fuzzy|prefix|exact]
                [--tag TAG]... [--kind clone|design|finetune]
                [--language LANG] [--names-only] [--include-builtin]
qtts voice filter ...                       # list 别名
qtts voice show NAME
qtts voice inspect NAME                     # show 兼容别名
qtts voice rename OLD NEW [--dry-run]
qtts voice tag NAME [--add TAG]... [--remove TAG]... [--set TAG]... [--clear] [--dry-run]
qtts voice delete NAME [--yes] [--dry-run]
qtts voice rm NAME [--yes]                  # delete 兼容别名，保持旧语义
qtts voice preview NAME [--text TEXT] [--out FILE] [--format wav|flac]
                        [--language LANG] [--no-save]
qtts voice export NAME -o FILE.qttsvoice [--force] [--dry-run]
qtts voice import ARCHIVE [--name NAME] [--force]
qtts voice import-btts ...
```

- `--json` 是全局参数，不在子命令重复注册；`--names-only` 与全局 `--json` 在运行时互斥检查 → 2。

### 3.2 list/filter

- QUERY strip + 去一个前导 `@`；空 query → 2。
- fuzzy（默认，连续子串）/prefix/exact，case-insensitive。
- `--tag` 可重复，AND；`--kind` 支持 clone/design/finetune；`--language` 规范化。
- 无 query/filter：保持 builtin + user voices。
- 纯名字 query：默认仍包含 builtin（避免 `voice list Vivian` 失效）。
- `--kind/--tag` 隐式 user-only；`--include-builtin` 只影响 name/language 匹配；`--names-only` 永不输出 builtin。
- 无匹配 → 0；表格 `no matching voices.`；JSON `count=0`。
- JSON 保留 `ok/builtin/voices`，新增 `schema_version/kind/query/match/filters/count/warnings`；voices 新增 `tags/source_kind/design_template`（不加 voice_id）。
- 坏 profile 跳过 + warning；tombstone 跳过 + warning；目录名/name 不一致用目录名 + 修复提示。

### 3.3 show/inspect

- show 主命令，inspect 永久别名，不打印 deprecation。
- 支持 builtin 与 user voice；输出 profile + 文件清单 + prompt_sha256/preview/design_yaml/warnings。
- 目录名/name 不一致、clone 参考缺失/逃逸、profile 损坏 → **exit 4**（与现有 qtts InputError 语义一致；本期不引入新的 5 语义）。
- 只读路径允许跟随 symlink（保持现有行为），但 warning。

### 3.4 rename

- NEW 通过 validate_name；已存在（含 symlink 目标、仅大小写不同）→ 2。
- 源不存在 → 2；源目录/profile 为 symlink → 4。
- 写入顺序：先原子改写 `voice.json.name=NEW`，再 `os.replace(OLD_DIR, NEW_DIR)`。
- **crash 修复**：若 `voices/OLD` 存在但 `voice.json.name==NEW`，再次执行 `rename OLD NEW` 直接完成目录改名；不要先 `require(OLD)` 而失败。
- show OLD 检测 mismatch → 4 + 重跑提示；list 用目录名 + warning。
- tags/provenance/consent/license/source 相对路径保留；不比较 ref.wav 内容哈希。

### 3.5 tag

- 无操作 → 列 tags，0，不写文件。
- `--set` 替换全部，每个 tag 一个 flag；`--clear` 可与 `--add` 组合；`--set` 与 add/remove 互斥。
- add/remove 可组合；remove 不存在幂等。
- 规范化 trim/去重保序/空/纯空白/>64/控制字符 → 2；匹配删除用 casefold。
- 原子写；变更校验 clone 参考存在/未逃逸（缺失 → 4）；list 模式不校验。

### 3.6 delete/rm

- delete 主命令；rm 兼容别名，**保持旧语义**：非 TTY 无 --yes → 4；用户取消 → `ok=True, cancelled=True`。
- delete 新命令：默认交互确认；非 TTY 无 --yes → 2；`--dry-run` 只打印。
- 只删 store 根目录内真实音色目录；symlink 目录/profile → 4。
- 先 rename 到 `.deleting-<name>-<token>` tombstone，再 rmtree；失败 → 4 + tombstone hint。
- 允许按目录名删除损坏 profile（确认后）；finetune 只删 profile，不删 checkpoint。
- list 跳过 `.deleting-*` 并 warning。

### 3.7 preview

- 保持 `--text/--out/--format/--language`。
- 默认写 `<profile>/preview.wav`（兼容）；`--format flac` 必须配 `--out *.flac`，否则 2（避免 FLAC 字节写进 .wav）。
- `--no-save` 必须配 `--out`，且不得指向 profile 的 preview.wav。
- 写入先临时文件再原子 replace；覆盖 preview 时 stderr 提示。

### 3.8 export/import

- 单音色 `.qttsvoice`，archive schema 1 不变；`-o` 始终是文件。
- `--force` 语义不变；`--dry-run` 新增。
- **design provenance 自包含**：design profile 的 `voice.json.source.design` 内嵌 YAML（≤256 KiB）与 rendered instruction/hash/template/compiler_version；`design.yaml` 只是 profile 目录里的人类可读副本，不进入 archive。
- 多音色 `--all` 本期不做；后续另设 `voice export --all --out-dir DIR` + `voice import-bundle DIR` + staging 原子发布。

### 3.9 profile schema

- 不 bump `schema_version`；不加 `voice_id/updated_at`（本期无消费者）。
- 新增可选 `source.design` 子对象，字段见 3.8；旧 profile 读取忽略。
- design freeze 写 `source.description` 保证 `say`/`preview` 兼容。

## 4. 分阶段

- **Phase 0 指标地基**：audio_features、completion、audio_qc metrics 模式、quality 阈值、qc 新参数、单测。
- **Phase 1 voice store 硬化**：tags 规范化、rename/tag/delete/tombstone 原语、mismatch 检测、单测。
- **Phase 2 voice CLI 管理面**：list/filter、show、rename、tag、delete/rm、preview 约束、单音色 export dry-run、CLI 测试。
- **Phase 3 design 离线层**：design_yaml/compile/lint、init/lint、design_run（矩阵/resume/lock）、design_report（CSV/MD/JSON/HTML/reel）、fixture 测试。
- **Phase 4 design 模型路径**：explore、report 接真实候选、freeze design/clone、say --design、voice add --kind design 适配、模型集成测试。
- **Phase 5 后置**：多音色 export/import-bundle、archive schema 2、check/bench metrics、Windows 锁、多语言模板、design prune。

> 实施状态（2026-11-12）：Phase 0–4 已实现，design 测试 55 passed，全量测试仅剩两项既有的缺失 `qwen-asr` 失败；Phase 5 按计划后置。

## 5. 测试策略（P0）

1. freeze --as design 后 `say --voice voice:NAME --dry-run --json` 的 instruction == rendered_instruction；preview design 分支同样。
2. freeze --as clone 后 profile 有 `prompt.safetensors`，`say --voice voice:NAME` 可解析；缺 Base/缺 --confirm-transcript 的 exit code 正确。
3. design profile export → import → say 仍可用，`source.design` 完整；archive 1 不丢 provenance。
4. `voice export NAME` 单文件行为/退出码/--force/--dry-run 不变；输出在 `data-dir/voices/` 内被拒绝。
5. delete 能删损坏 profile（按目录名），拒绝 symlink；finetune 只删 profile。
6. rename 正常/目标存在/大小写-only/源 symlink/目标 symlink/crash 中间态修复；--dry-run 不写盘。
7. list 对 mismatch/corrupt/tombstone/symlink 的 warning 和 JSON 形状；--names-only + 全局 --json → 2；旧键不删。
8. preview --no-save 不写 preview.wav；默认覆盖原子写；--format flac 无 --out → 2。
9. YAML：safe_load、超长、alias/锚点炸弹、重复 key、非 UTF-8、description+override 同时非空、unsupported language+override。
10. report：排序 tie-breaker、缺失值、degraded、--fail-on-degraded、--require-full-metrics、op=unavailable、CSV 空值/表头/行序、JSON allow_nan=False、两次生成字节稳定（排除 volatile 字段）。
11. 指标：220Hz/60Hz 正弦、白噪声、数字静音、0.3s、立体声反相 downmix、8k/16k/48k、>2 声道 4、NaN/Inf 4、LTAS 斜率、HF/LF 频带不足、praat 缺失、YIN 八度误差。
12. completion：finish_reason length/stop、token 预算、expected 边界、tail energy、recipe 缺失、unknown 不计 failure。
13. HTML/shell 注入：script/引号/&/</script>、freeze 命令的 ;/换行/空格/引号/$()。
14. 回归：现有 pytest 全绿；`say --voice design:DESC` instruction 不变；`voice list --json` 旧键不删；`.qttsvoice` v1 fixture 永久保留。

## 6. 主要风险

- 指令语义不匹配：compiler 自研 + 固定 cover A/B；不承诺 btts 1:1。
- 无 cfg：文档明确；用 seed × sampling。
- 截断检测启发式：finish_reason 优先；unknown 不直接拒绝；--allow-incomplete。
- F0/HNR 数值不可直接横比 btts：报告记录 backend；阈值按 qtts 自身分布调。
- 并发 rename/delete：symlink 拒绝、tombstone、store 锁、O_NOFOLLOW（POSIX）。
- 磁盘占用：design run 保留候选；--no-reel、文档、后续 design prune。
- 双入口：文档明确“即时路由 = 快速试音；design 流程 = 可复现资产”。
- HTML/shell 注入：转义 + quote + 无网络请求。

## 7. 已拍板 / 已修正

- freeze `--as` 必填（拍板）。
- 多音色 export 后置（拍板）。
- HNR praat canonical（拍板）。
- YAML btts 结构化 + compiler（拍板）。
- design freeze 写 `source.description`（修正 blocker 1）。
- clone freeze 生成 prompt.safetensors（修正 blocker 2）。
- design provenance 内嵌 `source.design`，archive 1 不变（修正 blocker 3）。
- export --all 后置（修正 blocker 4）。
- voice show/inspect/delete/rm 退出码沿用 qtts 现有 4/2 语义，不引入新的 5 语义（修正 review 2.2/2.3）。
- preview --format flac 约束（修正 review 2.4）。
- list builtin 行为（修正 review 2.5）。
- report 不写 volatile 字段（修正 review 1.2 #19）。
- completion too_short 相对化、tail 能量相对化（修正 review 5.4）。
