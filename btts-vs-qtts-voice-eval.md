# btts vs qtts：音色评估能力对比

> 整理自 `claude-video-f5` 会话中对 3 个角色 × 12 个设计音色候选的实测与讨论。
> 版本口径：qtts 0.3.2 / btts（Breeze TTS 2 本地工作台）。

## 结论

- **btts 是「音色设计工作台」**：强在整批候选的技术指标排序、`Metric pick`、盲听页，以及对**设计描述本身**的就绪度评分。
- **qtts 是「生产流水线」**：强在台词保真（ASR CER/WER）、响度/语速门禁、对 clone 参考的说话人相似度，以及 take 自动选择。
- **两边都没有**「这个音色像不像 7–9 岁小女孩 / 够不够狡黠」这类语义或审美评估。

## 评估能力逐项对比

- **设计描述质量**
  - btts：✅ `design lint`，0–100 分，五个分项（specificity / executability / consistency / language_match / testability），分 block / warn / info 三级；评的是 instruction，不是音频。
  - qtts：❌ 无对应能力。

- **基础技术指标**
  - btts：✅ `design report --metrics basic`，含时长、采样率、峰值、RMS、削波、静音、过短/过长。
  - qtts：✅ `qc`，含峰值、削波、DC 偏移、静音比例、最长静音、LUFS、true peak、语速。
  - 差异：qtts 多响度与语速，btts 多采样率与长短异常。

- **声学/音色特征**
  - btts：✅ `--metrics full`，含 F0 中位数 / p10–p90 / 标准差、voiced ratio、HNR、频谱质心、LTAS 斜率、HF/LF 比；需 librosa + praat-parselmouth。
  - qtts：❌ 无 F0 / HNR / 频谱类特征。

- **台词保真（CER/WER）**
  - btts：❌ 无 ASR 校验。
  - qtts：✅ `check`，用 ASR 算 CER/WER，可设阈值；能抓读错、漏字、复读、幻觉。

- **说话人相似度**
  - btts：❌ 无。
  - qtts：✅ `qc --reference-voice` / `say --takes --select speaker-sim` / `bench --speaker-sim`；必须有 clone profile 当参考并加载 Base 模型，无参考时为 `null`。

- **自动排序 / 推荐**
  - btts：✅ report 做确定性排序（失败项 → 削波 → 静音 → HNR → voiced ratio → take id），标出 `Metric pick` 并默认勾选；面向一批设计候选。
  - qtts：⚠️ `say --takes N --select` 可在单次生成的 2–16 个 take 里按 cer / speaker-sim / lufs 选；`batch` 无排序，面向单句多 take。

- **盲听页**
  - btts：✅ 内置 `audition.html`，可盲听（隐藏推荐），可复制 take ID / freeze 命令 / 试听文本。
  - qtts：❌ 无内置页面；这次 qtts 的试听页是手工写的。

- **拼接试听**
  - btts：✅ `all_candidates.wav`，可设间隔。
  - qtts：❌ 无内置；这次是手工 ffmpeg 拼的 `*_all.m4a`。

- **完整性 / 截断检查**
  - btts：✅ report 标 incomplete / truncated；`freeze` 拒绝截断或 <1s 的 take。
  - qtts：⚠️ 可测时长与静音，截断需看 recipe 的 `finish_reason` 或 `check` 结果。

- **自动 take 对比**
  - btts：⚠️ 只能在 report 里看整批指标。
  - qtts：✅ `takes list/compare` 显示已记录的 cer / speaker-sim / lufs；但只对 `say --takes` 产生的 job 有效，`batch run` 的产物不能直接 compare。

- **性能基准**
  - btts：❌ 无。
  - qtts：✅ `bench`，输出 RTF、加载时间、RSS，可选 CER / 相似度；明确不支持 `design:`（退出码 2）。

- **报告产物**
  - btts：`metrics.csv` + `report.md` + `audition.html` + 拼接 wav，适合人工评审。
  - qtts：`qc/check --report` JSON、batch 的 `status.jsonl`，适合脚本门禁。

## 对这次 36 条候选的具体用法

- **btts 侧**：已有 report 可看基础指标与 `Metric pick`；但当时用的是 `basic`，F0/HNR 为空，需重跑 `--metrics full` 才有音色声学特征。
- **qtts 侧**：能自动做的是 `check`（台词有没有读错）+ `qc`（削波/静音/响度/语速）；无参考 clone 时 `speaker_similarity` 为 `null`，不能给 12 个候选排序。
- **不能横向比**：btts 的 report 只认自己 run 目录里的 candidates，不能直接分析 qtts 的 wav；两边指标集也不同，F0/HNR 不能跨工具对照。
- **要「像某个目标音色」**：必须先有该目标的 clone profile（例如把选中的 btts c008 参考音冻结成 clone，需确认授权），再算相似度；这测的是「接近目标」，仍不测「贴合角色描述」。

## 共同限制

- **指标不是评委**：两边文档都明确，指标只用于排除技术性失败和排序。
- **最终选 take 必须人耳试听**：自动化最多筛掉读错台词和爆音/静音/语速异常的废片。
- **不编造 null/unavailable 指标**：无参考、无模型时的空值不能当 0 或拿来排序。

---

## `voice` 命令功能对比

### 子命令一览

- **btts**：`voice add` / `voice list`（别名 `voice filter`）/ `voice show` / `voice rename` / `voice tag` / `voice delete` / `voice export`
- **qtts**：`voice add` / `voice list` / `voice inspect` / `voice preview` / `voice rm` / `voice export` / `voice import` / `voice import-btts`
- **两边都有**：创建（add）、列出（list）、查看（show / inspect）、删除（delete / rm）、导出（export）
- **btts 独有**：`rename`、`tag`、强大的 `list/filter` 筛选、`export` 的 `full|wav` 双模式与多选/全量导出
- **qtts 独有**：`preview`、`import`、`import-btts`、内置音色、`add --kind design`、`--materialize-clone`、`--auto-ref-text`

### 创建音色：`voice add`

- **btts**
  - 只有克隆一种：`voice add NAME --ref REF.wav --text REF.txt --confirm-transcript`
  - `--text` 必须由用户提供逐字转写，CLI **不会自动转写**；先不带 `--confirm-transcript` 跑一次，把打印的转写交用户逐字确认后再加上。
  - 支持 `--language zh|en`（默认 zh）、`--allow-long-ref`、`--force`、`--tag`。
  - 设计音色**不走** `voice add`，而是 `design init → explore → report → freeze --as clone|design`。
- **qtts**
  - 克隆：`voice add NAME --kind clone --ref-audio FILE (--ref-text TEXT | --auto-ref-text) [--mode icl|xvec] --i-have-rights`
  - 设计：`voice add NAME --kind design --description TEXT --sample-text TEXT [--materialize-clone]`
  - 支持 `--license`、`--consent-note`、`--tags`、`--asr-model`、`--model`、`--yes`。
  - `--auto-ref-text` 用 Qwen3-ASR 自动转写；`--mode xvec` 不需要转写但跨文本稳定性通常不如 ICL。
  - `--materialize-clone` 可把 design 音色固化成 clone prompt，之后走 clone 路由。
- **核心差异**
  - qtts 在 `voice add` 层就能直接设计音色，btts 必须走完整的 design 流水线再 freeze。
  - qtts 支持自动转写，btts 强制人工提供并逐字确认转写。
  - qtts clone 必须显式 `--i-have-rights`；btts 没有这个旗标，授权由用户口头确认。

### 参考音频准入

- **btts**：<1s 失败；<6s 警告；6–20s 推荐；>30s 必须用户明确同意后加 `--allow-long-ref`。
- **qtts**：<2s 硬失败；2–3s 警告；3–30s 推荐；>60s 警告；<16kHz 警告；另检查削波、静音比例、有效语音占比；**没有** `--allow-long-ref`。
- **差异**：qtts 下限更宽松但校验项更多，且长参考没有放行旗标；btts 长参考需要显式同意。

### 列出与筛选：`list` / `filter`

- **btts**
  - `voice list [QUERY] [--match fuzzy|prefix|exact] [--tag TAG ...] [--kind design|clone] [--language zh|en] [--names-only | --json]`
  - `voice filter` 是完全相同的别名；QUERY 只匹配可寻址目录名，`fuzzy` 为不区分大小写的连续子串。
  - `--tag` 可重复且要求同时包含全部标签（AND）；`--tag` / `--kind` / `--language` 与 QUERY 之间全部 AND。
  - `--names-only` 每行一个名字，适合管道；与 `--json` 互斥。
  - 容错扫描：坏音色跳过，tombstone、目录名与 profile 名不一致、symlink 等在 stderr 警告。
  - 表格显示 name、kind、language、voice_id、cfg、tags 和 store 路径。
- **qtts**
  - `voice list [--json]`，JSON 同时返回内置音色与本地 profile。
  - 没有关键字筛选、标签、kind/language 过滤，也没有 `--names-only`。
  - 输出是给人看的表格或单个 JSON 文档。

### 查看详情：`show` / `inspect`

- **btts**：`voice show NAME`，输出 profile 详情（人类可读）。
- **qtts**：`voice inspect NAME [--json]`，支持结构化 JSON。
- **差异**：qtts 更适合脚本读取；btts 的 `--json` 主要体现在 `list` 和 `export`。

### 试听：`preview`

- **btts**：没有 `voice preview`。试听靠 `say --voice @NAME` 直接合成，或 `design` 流程里的 `audition.html`。
- **qtts**：`voice preview NAME [--text TEXT] [--out FILE] [--format wav|flac] [--language LANG]`，默认用内置试听句；`--out` 与 `<out-stem>.recipe.json` 会静默覆盖。
- **差异**：qtts 有专门的单音色试听入口；btts 更依赖 `say` 和设计阶段的试听页。

### 改名与标签

- **btts**
  - `voice rename OLD NEW`：同时改目录名和 `profile.json` 的 `name`，保留 `voice_id`、tags、provenance、consent、license；旧名立即失效；目标已存在或仅大小写不同时拒绝。
  - `voice tag NAME [--add ...] [--remove ...] [--set ...] [--clear]`：无参数只列出 tags；`--set` 每个标签一个 flag；`--clear` 可与 `--add` 组合。
- **qtts**：没有 rename，也没有 tag 子命令；音色名和标签在创建时确定。
- **差异**：btts 把音色库当可维护资产，qtts 的 `voice` 层更偏创建与生产，不提供库管理操作。

### 删除：`delete` / `rm`

- **btts**：`voice delete NAME [--yes]`，默认交互确认；非 TTY 无 `--yes` 直接拒绝；删除前先原子改名为 `.deleting-*` tombstone；拒绝 symlink；不校验 clone 参考文件；只删 store 根目录内单个音色，无批量。
- **qtts**：`voice rm NAME [--yes]`，默认交互确认；只删除 store 根目录内单个音色；不校验 clone 参考文件完整性。
- **差异**：功能接近；btts 的删除实现更防御（tombstone、symlink 拒绝、目录描述符操作）。

### 导出：`export`

- **btts**
  - `voice export [NAME ...] [--all] [--mode full|wav] [-o PATH] [--zip] [--json]`
  - 支持多个名字或 `--all`；`--mode full` 镜像完整音色目录，`--mode wav` 只导出 `<name>.wav` 试听样本。
  - 两种模式都带 `manifest.json`、`LICENSE.txt`、`README.txt`，manifest 记录每个文件的 `sha256:`。
  - **绝不覆盖**已有目标（没有 `--force`），目标位于音色库内也拒绝；`--all` 遇到坏音色整体失败。
- **qtts**
  - `voice export NAME -o FILE.qttsvoice [--force]`
  - `.qttsvoice` 是 zip 容器，白名单成员固定为 `manifest.json`、`voice.json`、`prompt.safetensors`、`ref.wav`、`preview.wav`。
  - 一次一个音色；已有目标需 `--force`。
- **差异**：btts 支持多选/全量导出并可选择只导试听样本；qtts 导出格式固定、单音色、带严格白名单。

### 导入与迁移：`import` / `import-btts`

- **btts**：没有 `import` 子命令，只能重新 `voice add` 或重跑 design。
- **qtts**
  - `voice import FILE.qttsvoice [--name NAME] [--force]`：逐项校验路径、大小、压缩比、sha256 与 safetensors 结构，不使用 `extractall`；替换失败回滚；没有同意旗标，导入前由用户确认授权。
  - `voice import-btts BUNDLE (--voice NAME ... | --all) --i-have-rights`：读取 btts `voice export` 的目录或 `.zip`，有 manifest 时校验每文件 SHA-256、大小与完整文件集，拒绝路径穿越/绝对路径/反斜杠/symlink/特殊文件/大小写冲突/未知文件；design preview 作为参考音频；ICL 需要转写，可用 `--auto-ref-text` 或逐个 `--ref-text`。
- **差异**：迁移是单向的——btts → qtts 有官方桥（`import-btts`），qtts → btts 没有对应路径。

### 内置音色

- **btts**：没有内置音色，必须设计或克隆。
- **qtts**：9 个内置音色 `builtin:`：Serena、Vivian、Uncle_Fu、Ryan、Aiden、Ono_Anna、Sohee、Eric、Dylan；`voice list` 可见。

### 存储格式

- **btts**：`voices/<name>/` 下有 `profile.json`、`ref.wav` + `ref.txt`（clone）或 `design.yaml`（design）、`preview.wav`、`LICENSE.txt`。
- **qtts**：`voices/<name>/` 下有 `voice.json`、`prompt.safetensors`、`ref.wav`、`preview.wav`。
- **差异**：btts 存的是「配方 + 参考」，qtts 存的是「克隆 prompt + 参考」，两者都不可手改。

### 授权与确认语义

- **btts**：`voice add` 用 `--confirm-transcript` 确认的是「转写逐字正确」，不是授权；`export --mode full` 可能含参考音频、逐字转写和 consent，按个人数据处理。
- **qtts**：clone 必须 `--i-have-rights`（非 TTY 必须）；design 自动记录 `rights_confirmed: true`；`import` 没有同意旗标，需人工确认；`import-btts` 必须 `--i-have-rights`。

### 一张清单看懂差异

- **建库**：btts = clone + design freeze；qtts = clone + design add（可选固化）。
- **转写**：btts 人工逐字确认；qtts 可 ASR 自动。
- **管理**：btts 有 rename / tag / 高级筛选；qtts 没有。
- **试听**：btts 靠 `say` 和 audition 页；qtts 有 `voice preview`。
- **打包**：btts 多选/全量、`full|wav`；qtts 单音色 `.qttsvoice`。
- **迁移**：btts → qtts 有 `import-btts`；反向没有。
- **内置**：btts 无；qtts 9 个。
