# 安全、覆盖与执行边界

本文件是 `qtts` 的破坏性操作清单。**任何写文件、下载、删除、外发或执行第三方代码
之前，先在这里确认语义。**

## 1. 输出目标矩阵

“覆盖”指目标已存在时直接替换；“拒绝”指已存在时要求 `--force` 或直接报错；
“回滚”指写入失败时会恢复旧文件。

| 命令 | 旗标 | 对已有目标 | 说明 |
|---|---|---|---|
| `say` | `--out`, `--out-dir`(jsonl), `--recipe-out`, `--subtitle-out-prefix` | **静默覆盖** | 原子写入，无提示、无 `--force`；recipe/字幕同样覆盖；**无输入碰撞检查**，可指向输入 |
| `say --stdin --jsonl` | 每行 `out` + `--out-dir` | **静默覆盖** | 每行 `out` 是输出路径，可指向任意文件；读取 stdin 前创建 `--out-dir`；不支持 `--dry-run`；行间拒绝同一路径，磁盘已有文件不检查 |
| `script compile` | `-o/--out` | **静默覆盖** | 原子写计划 JSON；`-o` 可等于输入脚本，**无碰撞检查**；`-` 写 stdout |
| `batch plan` | `--out-dir` | **写/覆盖** | 创建目录并写 `plan.json`、`status.jsonl`；不是只读 |
| `batch run` | `--out-dir` + manifest 每行 `out` | **覆盖** | 非 `--resume` 不读旧 status，重写 `plan.json`/`status.jsonl`，按全部 manifest 任务重跑并覆盖输出；manifest 的 `out` 可写任意路径，相对 cwd，不是 `--out-dir` |
| `voice add` | 音色目录 | **拒绝** | 同名音色已存在退出 4；没有 `--force` |
| `voice preview` | `--out` | **静默覆盖** | 默认写 `preview.wav` 或 `<speaker>.preview.wav`，并同时写 `<out-stem>.recipe.json`；没有 `--dry-run`；**无输入碰撞检查**，可指向 `ref.wav` |
| `voice rm` | 音色目录 | **删除** | 非 TTY 必须 `--yes`；只删 store 根目录内的单个音色 |
| `voice export` | `-o` | **拒绝** | 已有归档需 `--force`；full 包含参考音频/转写/consent |
| `voice import` | 音色目录 | **拒绝/回滚** | 已有音色需 `--force`；替换失败回滚；`.qttsvoice` 无同意旗标 |
| `voice import-btts` | 音色目录 | **拒绝/回滚** | 必须 `--i-have-rights`；已有音色需 `--force`；staging 失败不留半成品 |
| `align` | `--out-prefix`, `--recipe-out` | **覆盖但回滚** | 先备份旧目标，提交失败时恢复；**无输入碰撞检查**，`--out-prefix` 可覆盖 `--srt` 输入 |
| `check` / `qc` / `bench` | `--report` | **静默覆盖** | 原子写 canonical JSON；**`--report` 可等于被分析的音频**，无碰撞检查 |
| `post` | `--out`, `--out-dir`, `--report` | **覆盖** | 写前做路径/大小写/NFC/symlink 碰撞检查；不拒绝已有输出 |
| `audiobook build` | `--out`, `--out-dir` | **拒绝** | 已有输出需 `--force`；还写 `<out-dir>/<project-stem>.json`、`.audiobook.lock`、`.audiobook.publish.json`、`segments/`、`chapters/`、`recipes/`；journal（prepared→published）主要恢复 final/manifest |
| `dub build` | `--out`, `--stems-out` | **拒绝** | 已有输出未给 `--force` 退出 4；还写 `<out-stem>.dub.json`、`<out-stem>.dub-work/`、`--mux` 的 `<out-stem>.mp4`、stems 与 `timeline.json`；已存在的 stem/timeline 需 `--force` |
| `takes promote` | `--out` | **覆盖** | 可覆盖任意目标，只保护 `job.json`、take 音频与 recipe；不转码 |
| `project init` | `qtts.project.json` | **拒绝** | 已存在需 `--force` |
| `project lock` | `--out` | **静默覆盖** | 原子写排序后的锁 JSON；**`--out` 可等于 `qtts.project.json`**，无碰撞检查 |
| `models pull` | 模型目录 | 下载/续传 | 联网；建议 `--revision` 固定版本 |
| `models prune` | 模型目录 | **删除** | 必须 `--yes`；默认交互确认 |
| `completion --install` | 补全文件 + rc block | 幂等替换 | 可能修改 `.zshrc`/`.bashrc`；先 `--dry-run` |
| `completion --uninstall` | 补全文件 + rc block | **删除** | 删除补全文件与 marker block；幂等 |
| `serve` | `<data-dir>/serve/` | 自动清理 + **删除** | `--keep-jobs N` 保留最近 N 个 job 目录，超出者删除；`0` 关闭 jobs 保留数/orphan 清理，但启动时仍清空 `staging/` |
| `finetune prepare` | `--output-jsonl` | **拒绝** | 已有 JSONL 需 `--force`；写 `<output>.tmp`，失败时清理 |
| `finetune train` | `--output-dir`, profile | **拒绝/回滚** | 已有输出目录需 `--force`；子进程失败保留 `.partial`；发布用 `.old` 备份；`--register` 写 profile |
| `finetune eval` | `--audio-dir` | **静默覆盖** | 创建目录并写 `0001.wav` 等；同名 WAV 原子替换；没有 `--force` |

### 写前检查清单

1. 列出本次命令的所有输出旗标：`--out`、`--out-dir`、`--out-prefix`、
   `--recipe-out`、`--report`、`-o`、`--audio-dir`、`--subtitle-out-prefix`、
   `--stems-out`、`--output-jsonl`、`--output-dir`；以及 batch manifest 与
   JSONL 每行的 `out` 字段。
2. 目标是否已存在？存在就换新路径，或先得到用户明确同意。
3. 目标是否是输入文件、参考音频、recipe、`qtts.project.json`、`qtts.lock.json`、
   `job.json` 或 `status.jsonl`？是就停止。**这些路径大多没有输入碰撞检查**：
   - `check/qc/bench --report` 可以直接覆盖被分析的音频；
   - `script compile -o` 可以覆盖输入脚本；
   - `project lock --out` 可以覆盖 `qtts.project.json`；
   - `say --recipe-out`/`--subtitle-out-prefix`、`align --out-prefix` 可以覆盖
     输入文本/字幕；
   - `voice preview --out` 可以覆盖 `voices/<name>/ref.wav`；
   - `batch` manifest 与 JSONL 每行的 `out` 没有输入碰撞检查，可指向任意
     `.wav`/`.flac`；batch 的相对路径相对 cwd。
4. `--out-dir` 是否是非空旧目录？`batch plan`/`batch run` 先 diff；`say --stdin
   --jsonl` 不提供 `--dry-run`。
5. 输出是否需要跨文件一致性？`audiobook` 有 journal（主要覆盖 final/manifest）；
   `dub` 没有跨文件 journal，不能承诺整组原子。

## 2. 下载、外发与代码执行

| 入口 | 风险 | 规则 |
|---|---|---|
| `models pull` | 大体积下载、供应链 | 先说明来源/体积；`--revision` 固定；`--source hf|modelscope`；不要自动执行 |
| `--backend dashscope` | 文本发往远端 | 需要 `DASHSCOPE_API_KEY`；先说明；`--offline` 与它冲突 |
| `plugins list` | `entry_point.load()` 执行插件模块顶层代码 | 只列已安装插件；不要安装来源不明的插件 |
| `--backend PLUGIN` / `post --plugin` | 在 qtts 进程内运行插件 | 需要用户明确同意并知道插件来源 |
| `finetune prepare/train` | 调用官方脚本子进程 | `--python`/`--finetuning-dir` 必须是用户信任的路径 |
| `uv run qtts` | 可能同步/安装依赖 | 优先使用已安装的 `qtts`；离线环境加 `--offline` |
| `completion --install` | 修改 shell rc | 先 `--dry-run`；`--dir` 只写文件不改 rc |

## 3. 授权与同意的真实位置

- `--i-have-rights` **只**存在于：
  - `voice add NAME --kind clone`
  - `voice import-btts`
- `voice add --kind design` 会自动记录 `rights_confirmed: true`（没有旗标）。
- `voice import FILE.qttsvoice` **没有**同意旗标；导入前由用户确认授权。
- `finetune` 的授权来自项目文件 `register.consent.rights_confirmed`；`--register`
  之前先确认项目内容。
- `say --yes` 与 `voice add --yes` 目前是 no-op；真正需要 `--yes` 的是
  `voice rm` 和 `models prune`。

## 4. serve 的默认安全边界

- 默认绑定 `127.0.0.1:8000`；`--port 0` 让系统分配端口。
- 绑定非 loopback（含 `0.0.0.0`/`::`）必须提供 token，否则退出 2。
- token 优先用 `QTTS_SERVE_TOKEN`，避免出现在进程 argv/历史记录。
- 所有请求校验 `Host`；通配绑定必须给 `--allowed-host`。
- 没有 TLS；不要直接暴露到公网，需要时放在反向代理后面。
- 本地 backend 的 `effective_max_concurrency` 恒为 1；`--max-concurrency` 设得更大
  也不会超过 1，并发控制用 `--queue-size`/`--max-connections`。

## 5. 隐私

- `voice export` 的 full 包可能含 `ref.wav`、逐字转写与 consent 元数据；按个人
  数据处理，不要当匿名样本分享。
- `voice import-btts` 读取的 bundle 是外部输入；manifest/hash 校验通过不代表授权
  成立，授权仍由用户确认。
- recipe、锁文件与 `job.json` 含模型/音色身份与哈希，不含原始参考音频；但可能
  暴露本地路径（报告类输出会尽量避免绝对路径，仍应检查再分享）。

## 6. 维护者核对入口

行为以 0.3.2 源码为准，关键路径：

- `qtts_cli/commands/say.py`（输出/recipe/字幕写入）
- `qtts_cli/audio_io.py`（`write_audio` 原子覆盖）
- `qtts_cli/commands/script.py`、`qtts_cli/commands/batch.py`
- `qtts_cli/commands/voice.py`、`qtts_cli/voice_archive.py`、`qtts_cli/voice_btts.py`
- `qtts_cli/commands/align.py`（备份/回滚）
- `qtts_cli/commands/post.py`（路径碰撞）、`qtts_cli/quality.py`（报告写入）
- `qtts_cli/audiobook/manifest.py`（journal）、`qtts_cli/dub*.py`（无跨文件 journal）
- `qtts_cli/commands/takes.py`、`qtts_cli/project.py`
- `qtts_cli/finetune/`、`qtts_cli/plugins.py`、`qtts_cli/commands/completion.py`
