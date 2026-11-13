# qtts `voice import-btts` 兼容导入方案 v3

> v3 已吸收 Codex 对 v2 的复审：默认行为、recipe/model-free 边界、模板 canonicalization、`source.btts` 完整性、崩溃原子性、preview provenance。

## 0. 兼容目标

| 层 | 兼容程度 | 做法 |
|---|---|---|
| 元数据 / provenance / tags / license / consent | 全量保留（有大小上限） | 写入 `source.btts`，不丢 btts 原字段 |
| 配方（instruction / template / structured / seed / cfg / max_new_tokens） | 完整保留 | design 默认导入为 qtts `kind=design`，`say` 直接复用 btts `design.instruction` |
| 模板身份 | 身份兼容，不承诺逐字重编译 | btts 原始 ID / `btts/...` 都 canonicalize 到 qtts 模板 ID；原始 ID 留在 `source.btts.template` |
| 声学结果 | 不可兼容 | Breeze TTS 2 ≠ Qwen3-TTS VoiceDesign；需要近似音色时用 clone 模式 |

本次的实质变化：**design 音色不再默认降级为 clone；完整设计配方、模板来源和 preview 来源都保留；不可等价的部分明确输出。**

## 1. 默认行为与模式

新增：

```text
--design-mode auto|recipe|clone
```

未显式传时：

```text
若出现任一 clone-only 参数（--auto-ref-text / --ref-text / --asr-model / --mode / --model）
    -> design_mode = clone
否则
    -> design_mode = auto
```

这样旧命令 `--auto-ref-text`、`--ref-text`、`--mode xvec` 保持原行为，不会因为新增默认值而回归。

模式语义：

| 模式 | design 音色 | clone 音色 | 模型加载 |
|---|---|---|---|
| `auto` | recipe | clone | 只有选中的 clone 音色需要模型 |
| `recipe` | recipe | clone（recipe 只约束 design） | 只有选中的 clone 音色需要模型 |
| `clone` | clone（旧行为） | clone | 需要 clone 模型；design 另需 ref text/ASR/xvec |

补充规则：

- `--mode` 默认改成 `None`；clone 路径内部再解析为 `icl`。这样 `--design-mode recipe` 且没显式传 `--mode` 不会误报。
- `--design-mode recipe` + 显式 clone-only 参数（含 `--model`）→ UsageError。
- `--design-mode auto` + clone-only 参数（含 `--model`）→ UsageError，要求显式 `clone`。
- `--design-mode clone` + design 音色但没有 `--auto-ref-text`/`--ref-text`/`--mode xvec` → 保持现有 InputError。
- `--model` 只对 clone 路径有效；`--design-mode recipe` + `--model` → UsageError；未显式 `--design-mode` 且带 `--model` 时推断为 clone。
- 全 design + recipe：不 `scan(cfg.model_dir)`、不加载 TTS/ASR，空模型目录也能成功。
- mixed bundle + recipe/auto：design 走 recipe，clone 走 clone；只要有 clone 音色，就按需扫描并加载 clone 模型。

## 2. instruction 的真值

btts `say` 对 design 使用 `profile.design.instruction`，所以导入时：

```text
replay_instruction = design.instruction.strip() or identity.rendered_instruction.strip()
```

- 两者都为空 → staging 前 `InputError`，不允许生成 `say` 时才失败的 profile。
- `source.description = replay_instruction`
- `source.design.rendered_instruction = replay_instruction`
- `default_instruct = replay_instruction`
- 两者不一致：保留 `source.design.identity_rendered_instruction`，warning `instruction-mismatch`，继续导入。

## 3. 模板：输入别名，存储 canonical

新增唯一解析入口：

```python
resolve_template_id(value: str) -> Optional[str]
```

接受：

```text
qtts/zh-late-night-radio/v1
zh-late-night-radio/v1
btts/zh-late-night-radio/v1
```

统一返回 canonical：

```text
qtts/zh-late-night-radio/v1
```

别名表：

```text
zh-late-night-radio/v1      -> qtts/zh-late-night-radio/v1
zh-documentary-narrator/v1  -> qtts/zh-documentary-narrator/v1
zh-anime-character/v1       -> qtts/zh-anime-character/v1
en-podcast-host/v1          -> qtts/en-podcast-host/v1
en-audiobook-narrator/v1    -> qtts/en-audiobook-narrator/v1
en-game-character/v1        -> qtts/en-game-character/v1
```

同时注册 `btts/<raw>` 为同一别名。`btts/...` **不是一等模板 ID，不进入 `TEMPLATES` 的 canonical 键**；它只是输入别名，避免：

- `template_sha256` 相同但 template 字符串不同，导致 `design_run.check_resume` 假 mismatch；
- `lint`/`compile`/`explore` 对同一内容出现两个身份。

存储规则：

```text
source.design.template      = canonical qtts ID（已知模板）
source.design.template_id   = 原始 btts ID（仅 provenance）
source.btts.template        = 原始 btts ID
source.design.template_sha256 = sha256(canonical TEMPLATES[ID])
```

未知 btts 模板：

```text
source.design.template = None
source.design.template_id = "<raw>"
source.design.template_sha256 = None
source.design.compiler_family = "btts"
source.design.compiler_version = "<raw>"
warning: unsupported-btts-template
```

不猜映射。recipe 仍能 `say`，但不能重新编译/lint 成 qtts 模板。

新增 4 个 qtts canonical 模板：

```text
qtts/zh-documentary-narrator/v1
qtts/zh-anime-character/v1
qtts/en-audiobook-narrator/v1
qtts/en-game-character/v1
```

模板数据加入 `starter`、`default_covers`、`label`、`description`；`example_document()` 必须按模板的 `starter` 生成，而不是对 zh/en 硬编码 late-night/podcast。`design init --template btts/...` canonicalize 后写 `qtts/...`。

**边界**：qtts 编译器当前按语言选择 `_render_zh/_render_en`，模板本身不影响渲染。导入 recipe 直接使用 btts 的 rendered instruction，因此播放层是忠实的；重新编译不保证逐字相同。若未来要求逐字重编译，再给模板加 render profile，不在本阶段伪装。

## 4. design.yaml 与 preview

### design.yaml

不另写独立文件。qtts archive 的 `ALLOWED_MEMBERS` 不包含 `design.yaml`，独立文件会在 `voice export`/`import` 中丢失。

统一存进 `source.design`：

```json
{
  "yaml": "<raw UTF-8 text>",
  "yaml_sha256": "..."
}
```

- 存在且合法：保存原文和 hash；`sample_text` 取第一个 cover。
- 不存在：`sample_text` 取 `structured.covers[0]` / `identity.structured.covers[0]` / `DEFAULT_DESIGN_SAMPLE`。
- 解析失败 / name/language/template 与 profile 不一致：warning `invalid-design-yaml` 或 `design-yaml-mismatch`，profile 优先，recipe 继续。
- YAML 的 template 在保存前用 `resolve_template_id()` 归一化，避免以后 lint/compile 失败。

### preview

- btts `preview.wav` 复制为 qtts profile 的 `preview.wav`。
- `source.preview = {origin: "btts", file: "preview.wav", sha256: ...}`
- `source.btts.preview_sha256` 永久保留原始 btts preview hash。
- `voice preview` 成功覆盖 `preview.wav` 后，必须更新 `source.preview = {origin: "qtts", file: "preview.wav", sha256: <new>, updated_at: ...}`；不能继续声称 btts 来源。
- `voice inspect/show` JSON 增加 `preview_origin`；文本输出只在有来源时显示。
- 文档说明：btts preview 是 Breeze 的声音，不代表 qtts 输出；`voice preview` 会用 qtts 重新生成。

## 5. seed / cfg / max_new_tokens

只保存，不自动回放：

```text
source.design.seed
source.design.cfg_scale
source.design.max_new_tokens
source.btts.defaults.seed
source.btts.defaults.cfg_scale
source.btts.defaults.direction
source.btts.design
```

- `seed_replayed=false`、`cfg_replayed=false` 写进导入结果。
- `cfg_scale` 永远不映射到 qtts 参数（qtts 没有 cfg/guidance）。
- `max_new_tokens` 只作为生成提示，不默认覆盖 qtts 的 2048。
- 未来可加显式 `--replay-seed`，但不在本阶段默认开启。

## 6. `source.btts` 完整 schema

字段必须有界，不能把 1 MiB profile 无上限塞进 `voice.json`。建议上限：字符串 4 KiB，结构化 JSON 16 KiB；超限写入 `{present: true, truncated: true, sha256: <canonical-json-sha256>}`。

```json
{
  "type": "design",
  "origin": "btts",
  "description": "...",
  "sample_text": "...",
  "language": "Chinese",
  "materialized": false,
  "preview": {"origin": "btts", "file": "preview.wav", "sha256": "..."},
  "design": {
    "schema_version": 1,
    "template": "qtts/zh-late-night-radio/v1",
    "template_id": "zh-late-night-radio/v1",
    "template_sha256": "...",
    "structured": {"...": "btts identity.structured"},
    "rendered_instruction": "...",
    "rendered_sha256": "...",
    "identity_rendered_instruction": "...",
    "compiler_family": "btts",
    "compiler_version": "1",
    "yaml": "...",
    "yaml_sha256": "...",
    "seed": 19,
    "cfg_scale": 1.0,
    "max_new_tokens": 300
  },
  "btts": {
    "bundle": "btts-voices-2026...",
    "bundle_format": "directory",
    "bundle_root_prefix": "",
    "manifest": {"present": true, "verified": true, "sha256": "..."},
    "voice_id": "v_...",
    "name": "narrator",
    "kind": "design",
    "language": "zh",
    "profile_path": "narrator/profile.json",
    "profile_sha256": "...",
    "tags": ["design", "narration"],
    "identity": {
      "structured": {"...": "..."},
      "rendered_instruction": "...",
      "template": "zh-late-night-radio/v1",
      "compiler_family": "btts",
      "compiler_version": "1"
    },
    "design": {"instruction": "...", "seed": 19, "cfg_scale": 1.0, "max_new_tokens": 300},
    "defaults": {"seed": 19, "cfg_scale": 1.0, "direction": null},
    "provenance": {"...": "source take/run/model/cli/api"},
    "license": {"model": "breezeblue-research-non-commercial"},
    "consent": {"present": true, "...": "bounded"},
    "files": {"profile.json": "...", "preview.wav": "..."},
    "preview_sha256": "..."
  }
}
```

- 不保存 bundle 绝对路径；只保存 basename。命令行输出仍可显示完整路径，但不进 `voice.json`。
- `tags`：qtts `tags` = btts 原 tags + 去重后的 `btts` 派生标签；原文保留在 `source.btts.tags`。
- `manifest.verified` 按实际值写；manifest 缺失时是 `false`，不是示例里的硬编码 `true`。
- `compiler_family` 区分 `btts` 和 `qtts`；不要写 `compiler_version: "btts:1"`。

## 7. consent / license

- `--i-have-rights` 仍然强制；不因为 recipe 不需要参考音频就放宽。
- `consent.rights_confirmed = true` 表示用户本次确认，不表示采纳 btts 的 consent。
- `consent.license` 写 `"btts-import"`，不直接把 btts 模型许可证冒充为 qtts consent。
- btts 的完整 license/consent 放 `source.btts.license` / `source.btts.consent`，note 里说明“btts consent 仅为 provenance”。

## 8. 批量导入的原子性

不要宣称“天然全有或全无”。现有 `_commit_imported` 只在 Python 异常路径回滚，SIGKILL/断电会在多次 `os.replace` 之间留下部分 profile 和 `.backup.*`。

采用与 `import-bundle` 相同的 journal 状态机，并覆盖每个崩溃窗口：

```text
<store>/.btts-import.json       # journal
<store>/.btts-import.lock       # 独占锁，避免两个 import 互相覆盖 journal
```

每个 entry 至少记录：

```json
{
  "target": "voices/alice",
  "existing_path": "voices/Alice",
  "staging": "voices/.alice.abc123",
  "backup": "voices/.alice.backup.def456",
  "had_existing": true,
  "backup_done": false,
  "installing": false,
  "installed": false,
  "replacing": false,
  "restoring": false,
  "restored": false,
  "staging_removed": false
}
```

- `target` 是本次要安装的名字；`existing_path` 是磁盘上实际旧目录（大小写可能不同，例如 `Alice`）。
- `had_existing=false` 时 `existing_path` 必须为 `null`；`had_existing=true` 时它必须是一个已验证的 voice 目录名（恢复期间允许暂时不存在）。
- 备份/恢复都使用 `existing_path`，不是 `target`。这样 case-only rename 在大小写敏感文件系统上也能正确回滚。
- 路径比较不能用原始字符串。新增 `same_path_identity(a, b)`：先 `a == b`；两者都存在时用 `os.path.samefile()`；只有一方存在时按“不同”处理。这样大小写不敏感文件系统上 `alice` / `Alice` 指向同一目录时不会被误判。
- 全局 `committed: false`。所有路径必须 resolve 到 `store.root` 内；`target` 是已验证 voice 名；`existing_path` 是合法的 voice 子目录（存在时不得为 symlink），且必须满足 `existing_path.casefold() == target.casefold()`（只允许大小写差异，禁止指向另一个真实音色）；`backup`/`staging` 是 `voices/` 下的隐藏子项，且不是 symlink；否则 fail-closed。
- `schema_version` 必须是整数 `1`；`committed` 和所有 entry flag 必须是布尔值；`committed=true` 时所有 entry 必须 `installed=true`，否则 fail-closed，防止在未安装状态下清理 backup。
- 反向不变量同样 fail-closed：`restored/staging_removed` 为真但 backup 仍在磁盘上，或 cleanup 阶段发现 backup 仍存在，都保留 journal 与数据，绝不删除 backup。

### 正常提交阶段

1. 所有 staging 完成后，写 journal（`committed=false`，所有 entry 未安装）。
2. 逐音色：
   - 若 `had_existing`：`os.replace(existing_path, backup)`，然后写 `backup_done=true`。
   - 写 `installing=true`。
   - `os.replace(staging, target)`。
   - 写 `installed=true, installing=false`。
3. 全部完成：写 `committed=true`。
4. 清理 staging、backup、journal。

### 恢复规则（必须按状态组合判断）

先看全局：

- `committed=true`：**只清理，绝不回滚**。删除 staging、backup、journal；若某个 `installed=true` 的 target 不存在，fail-closed；`had_existing=true` 的 backup 此时是旧版本，直接删除，不恢复。
- `committed=false`：按 entry 逆序回滚。

单 entry 的规则：

| 状态 | 磁盘事实 | 动作 |
|---|---|---|
| `restored=true` | backup 存在 | fail-closed：backup 是旧 profile 的唯一副本，不清理、不覆盖 journal |
| `restored=true` | backup 不存在 | 不再碰 target；若 staging 存在则删除；只清理 journal |
| `staging_removed=true` | backup 存在 | fail-closed（同上） |
| `staging_removed=true` | backup 不存在 | staging 已清理过；按已完成回滚处理，只补删残留 staging/journal，不再删 target |
| `restoring=true, had_existing=false` | target 存在 | 删除 target；写 `restored=true` |
| `restoring=true, had_existing=false` | target 不存在 | 直接写 `restored=true` |
| `restoring=true, had_existing=true` | backup 存在，target 存在 | 视为 target 是新安装：删除 target；`os.replace(backup, existing_path)`；写 `restored=true`。大小写不敏感时 target/existing_path 可能是同一路径，也按此处理，不因 `same_path_identity` 为真而 fail-closed |
| `restoring=true, had_existing=true` | backup 存在，target 不存在 | `os.replace(backup, existing_path)`；写 `restored=true` |
| `restoring=true, had_existing=true` | backup 不存在，existing_path 存在 | 上次 restore 已完成、只是没写 `restored`：写 `restored=true`，**不删 existing_path** |
| `restoring=true, had_existing=true` | backup 不存在，existing_path 不存在 | fail-closed |
| `installed=true` | target 存在，backup 存在 | 写 `restoring=true`；删 target；`os.replace(backup, existing_path)`；写 `restored=true` |
| `installed=true` | target 存在，backup 不存在，`had_existing=true` | **fail-closed**，不能把 target 当新音色删除 |
| `installed=true` | target 存在，backup 不存在，`had_existing=false` | 新音色：删 target；写 `restored=true` |
| `installed=true` | target 不存在，backup 存在 | 恢复 backup → existing_path；写 `restored=true` |
| `installed=true` | target 不存在，backup 不存在，existing_path 存在 | 视为已恢复：写 `restored=true` |
| `installing=true, installed=false` | staging 不存在 + target 存在 | replace 已成功但未写 installed：走 `installed=true` 的回滚分支 |
| `installing=true, installed=false` | staging 存在 + target 不存在 | replace 未发生：恢复 backup（若有），删除 staging，写 `restored=true, staging_removed=true` |
| `installing=true, installed=false` | staging 不存在 + target 不存在 + backup 存在 | 恢复 backup → existing_path；写 `restored=true, staging_removed=true` |
| `installing=true, installed=false` | staging 不存在 + target 不存在 + backup 不存在 + `had_existing=false` | 新音色从未安装：写 `restored=true, staging_removed=true` |
| `installing=true, installed=false` | staging 不存在 + target 不存在 + backup 不存在 + `had_existing=true` + existing_path 存在 | 旧 profile 已恢复：写 `restored=true, staging_removed=true` |
| `installing=true, installed=false` | staging 不存在 + target 不存在 + backup 不存在 + `had_existing=true` + existing_path 不存在 | fail-closed |
| `installing=true, installed=false` | staging 和 target 都在 | fail-closed |
| `backup_done=true, installing=false, installed=false` | existing_path 不存在 + backup 存在 | 恢复 backup → existing_path；写 `restored=true` |
| `backup_done=true, installing=false, installed=false` | existing_path 存在 + target 不存在 | 视为已恢复：写 `restored=true` |
| `backup_done=true, installing=false, installed=false` | 其它组合 | fail-closed（状态与磁盘不一致） |
| `backup_done=false, installed=false` | backup 存在 | 视为「已 move 但未写 backup_done」：target 必须不存在；恢复 backup → existing_path；写 `restored=true` |
| `backup_done=false, installed=false` | backup 不存在 | 旧目录未被移动：只清理 staging；若 target 存在且 `same_path_identity(target, existing_path)` 为假，fail-closed；为真则按同一目录处理 |
| 任意状态 | target/existing_path/backup/staging 不在 store.root 内或为 symlink | fail-closed |

关键点：

- `had_existing=true` 但 backup 缺失、且没有 `restoring/restored` 证据时，**绝不删除 target**，一律 fail-closed。
- 每次 `delete target` / `os.replace(backup, existing_path)` 之前先写 `restoring=true`，之后写 `restored=true`；恢复过程本身可重入。
- `restoring=true` 且 `had_existing=false` 时，恢复目标就是“没有旧 profile”，因此 target 存在就删、不存在就直接标记 restored；不会落到 “都不在” 的 fail-closed。
- `restored=true` 的 entry 不会再被回滚，避免「恢复成功后再次崩溃，下一次把恢复的旧音色删掉」；但如果此时 backup 仍存在，说明 journal 与磁盘不一致，必须 fail-closed，不能把 backup 当垃圾清掉。
- `staging_removed=true` 表示该 entry 的回滚和清理已经完成，后续恢复只补删 journal，不再碰 target/backup（backup 仍存在时同样 fail-closed）。
- case-insensitive 文件系统上 `alice`/`Alice` 用 `same_path_identity` 判定为同一目录，不会因为字符串不同而错误 fail-closed；但 `restoring=true` 时只要 backup 存在，target 就按“新安装”处理，不因路径相同而拒绝恢复。

清理顺序：

1. 先处理 target、existing_path 和 backup。
2. 删除 staging **之后**写 `staging_removed=true`，再写 `restored=true`；如果删除后、写标记前崩溃，按 `staging 不存在 + target 不存在` 的组合表恢复。
3. 全部 entry 成功回滚后才删除 journal。
4. 任意一步失败：保留 journal 和 backup，抛 `InputError`，提示人工恢复。

### 并发与 dry-run

- 整个 import/recovery 持有 `.btts-import.lock`（POSIX `flock` / Windows `msvcrt.locking`，与 `RunLock` 同模式）。
- 锁冲突 → 退出 4，不开始写。
- `--dry-run` 若发现 journal：不写入，报 `pending recovery`，要求先跑一次非 dry-run 恢复。
- `--dry-run` 不创建 staging/store 目录，不加载模型，不跑 ASR。

这样承诺改为：**进程内异常原子；崩溃后下次运行按状态机恢复；已 committed 的导入不会被回滚；恢复过程可重入；无法判定时 fail-closed 不继续覆盖。**

## 9. CLI / 输出契约

```bash
qtts voice import-btts BUNDLE
  (--all | --voice NAME ...)
  [--name NEW_NAME]
  [--design-mode auto|recipe|clone]
  [--mode icl|xvec]                     # clone only
  [--auto-ref-text] [--ref-text TEXT] [--asr-model ID]  # clone only
  [--model ID]                          # clone only
  [--force] [--dry-run] [--i-have-rights]
```

每个音色输出：

```json
{
  "name": "narrator",
  "kind": "design",
  "btts_kind": "design",
  "design_mode": "recipe",
  "template": "qtts/zh-late-night-radio/v1",
  "btts_template": "zh-late-night-radio/v1",
  "template_mapped": true,
  "instruction_sha256": "...",
  "preview_origin": "btts",
  "metadata_compatible": true,
  "acoustic_equivalence": false,
  "seed_replayed": false,
  "cfg_replayed": false,
  "replaced": false,
  "warnings": []
}
```

全局 warning：

```text
acoustic-equivalence-not-possible: btts uses Breeze TTS 2; qtts uses Qwen3-TTS VoiceDesign
```

文本 formatter 必须按 `design_mode` 分支：

```text
narrator: design <- btts design (zh -> Chinese, recipe, template qtts/zh-late-night-radio/v1)
clone-a:  clone  <- btts clone  (zh -> Chinese, ref_text_source=btts)
```

不能继续硬编码 `clone <- btts`。

`--dry-run`：

- 不调用 `store.staging_dir()`，不创建 store/staging 目录。
- 不 `scan(cfg.model_dir)`（除非 clone admission 确实需要？不需要）。
- 不加载 TTS/ASR。
- clone + `--auto-ref-text`：跳过 ASR，warning `asr-skipped-in-dry-run`。
- 做 bundle 解析、选择、命名/大小写冲突、参数互斥、clone ref admission 的离线检查。
- 输出 `dry_run=true` 和计划动作。

## 10. 代码改动点

| 文件 | 改动 |
|---|---|
| `qtts_cli/voice_btts.py` | 增加 design profile 字段访问/边界校验辅助；保持 bundle 安全边界 |
| `qtts_cli/commands/voice.py` | `import_btts` 模式分流；recipe 离线 staging；clone 模型延迟加载；journal 提交/恢复；输出 formatter |
| `qtts_cli/cli.py` | `--design-mode`、`--dry-run`；`--mode` 默认 None；互斥校验 |
| `qtts_cli/design_yaml.py` | `resolve_template_id`、别名表、4 个 canonical 模板、starter 驱动 `example_document` |
| `qtts_cli/design_compile.py` | 词表补 btts 全量 token；`template_sha256` 走 resolver |
| `qtts_cli/design_lint.py` | template 校验走 resolver |
| `qtts_cli/commands/design.py` | `init` canonicalize 别名；4 个模板 starter 生效 |
| `qtts_cli/design_run.py` | resume 比较使用 canonical template；避免别名假 mismatch |
| `qtts_cli/commands/voice.py`（preview） | 覆盖 preview 后更新 `source.preview` 和 hash |
| `qtts_cli/voice_archive.py` | 确认 `source.design.yaml` 随 `voice.json` round-trip；补 design source 透传测试 |
| `tests/test_voice_import_btts.py` | 新增 recipe/clone/auto/mixed/dry-run/journal/未知模板/casefold/formatter |
| `tests/test_design.py` | 别名 canonicalization、4 模板 starter、词表覆盖、init |
| `tests/test_voice_import_btts.py` / `test_p1_voice_archive.py` | archive round-trip、preview provenance |
| README / CHANGELOG / docs / skill references | 兼容级别、默认行为、不可等价声明、dry-run/journal |

## 11. 测试矩阵

### 兼容回归

1. 不带 `--design-mode` 的旧 `--auto-ref-text` / `--ref-text` / `--mode xvec` / `--model` 调用仍走 clone。
2. `--design-mode recipe` 且未显式传 `--mode` 成功。
3. `--design-mode recipe` + `--mode` / `--auto-ref-text` / `--ref-text` / `--asr-model` / `--model` → UsageError。
4. `--design-mode auto` + clone-only 参数（含 `--model`）→ UsageError。
5. `--design-mode clone` + design 无 ref 选项 → 原 InputError。

### recipe 正确性

6. design recipe 不加载模型、不跑 ASR；空 model 目录成功。
7. instruction 取 `design.instruction`；为空时取 `identity.rendered_instruction`；都为空时 InputError。
8. `identity.rendered_instruction` 不同 → warning + 保留。
9. 已知 6 个模板：canonical 写入、raw 保留、`template_sha256` 与 canonical 一致。
10. 未知模板：`template=None`、`template_id` 保留、warning；`say` 仍可用；export/import 不崩。
11. design.yaml 非法/名称/语言/template 不一致 → warning，recipe 成功。
12. design.yaml 原文 + yaml_sha256 随 archive round-trip。
13. btts preview 复制 + provenance；`voice preview` 覆盖后 `preview_origin=qtts`，btts hash 仍保留。

### 元数据

14. `source.btts` 保留 identity.structured、defaults.direction、tags、consent、完整 license、files、manifest verified=false 场景。
15. 超长字段触发 `truncated` + sha256，不写爆 `voice.json`。
16. `source.btts.bundle` 只存 basename，不泄漏绝对路径。
17. qtts tags = btts tags + `btts` 去重；原 tags 保留在 source。

### 原子性 / 路径

18. mixed `--all`：design recipe + clone 一起导入，失败时全部回滚。
19. journal：安装第 2 个音色时崩溃，下次运行恢复第 1 个并清理 staging。
20. journal 每个崩溃窗口：backup 已 move 未写 `backup_done`；replace 已成功未写 `installed`；`committed=true` 后清理前崩溃；新音色无 backup；case-only rename（`existing_path` 与 `target` 大小写不同）。
21. 恢复过程再次崩溃：`restoring=true` 后、restore 前；restore 完成但未写 `restored`；staging 删除后、写 `staging_removed` 前；下一次恢复不得删除已恢复的旧 profile。`had_existing=true` 且 backup 缺失时必须 fail-closed。
22. 新音色 `had_existing=false` 时 `existing_path=null` 可通过 journal 校验；`restoring=true` 后删 target 前/后崩溃都能幂等恢复为“无目标”。
23. `same_path_identity` 在 case-insensitive 文件系统上把 `alice`/`Alice` 视为同一目录；`restoring=true` 且 backup 存在时，target 按新安装删除后恢复 backup，不 fail-closed。
24. journal 状态损坏 / target、existing_path、backup、staging 越界或 symlink → fail-closed，保留 journal。
    - `schema_version` 非整数 1、`committed/flag` 非布尔、`committed=true` 但存在 `installed=false` 的 entry。
    - `existing_path` 指向另一个真实音色（casefold 与 target 不同）。
    - `restored/staging_removed=true` 但 backup 仍存在；`backup_done=true` 且 target 与 backup 同时存在。
    - 以上都必须保留 journal、backup、target 原状，不删除任何真实音色目录。
25. 两个 import 并发时锁冲突退出 4，后到者不覆盖 journal。
26. `--force` 替换、`--name`、大小写 `Alice`/`alice`、多目标冲突；失败后不留下第二个大小写目录。
27. `--dry-run` 不创建 staging/store 目录、不加载模型、不跑 ASR；存在 pending journal 时拒绝开始。
28. 失败/中断后的 `.backup.*` 残留可被恢复或明确报错。

### 模板 / 词表

29. `resolve_template_id` 对 raw、`btts/...`、canonical 都返回同一 canonical。
30. `design lint/compile/run --resume` 使用 canonical，别名不产生假 mismatch。
31. 4 个新模板 `design init` 生成各自 starter，不是 late-night/podcast。
32. btts 全部 vocab token 在 qtts 中不产生 `unknown-vocab`；`far_field`/`far_mic` 均可解析。
33. `design init --template btts/...` 写出 canonical `qtts/...`。

## 12. 非目标

- 不承诺跨模型声学等价。
- 不把 btts cfg 伪装成 qtts 参数。
- 不默认应用 btts seed。
- 不把未知模板猜成已知模板。
- 不改变 `voice.json` 的 `schema_version` 主版本；新增字段都在 `source` 下。
- 不改变 `voice import-bundle` 的 bundle schema 2。
- 不自动下载模型、不自动接受 license/consent。

## 13. 实施顺序

1. **模板/词表层**：resolver、4 个 canonical 模板、starter、词表补齐。
2. **recipe 导入层**：模式分流、profile 映射、instruction/design.yaml/preview、离线路径。
3. **原子性与 CLI**：journal、dry-run、参数互斥、formatter/JSON。
4. **测试/文档**：上面的矩阵 + README/CHANGELOG/skill references。
