# 音色、资产与复现

## 三种音色路由

| 引用 | 路由 | 稳定性 | 落盘 |
|---|---|---|---|
| `builtin:Vivian` | CustomVoice 内置音色 | 高；同一 speaker 固定 | 否 |
| `voice:alice` | Base + clone prompt / finetune profile | 高；参考音频 + 逐字转写是锚点 | 是 |
| `design:低沉沙哑的中年男声` | VoiceDesign 即时生成 | 同 instruction 换文本可能漂移 | 否，除非固化 |

- 需要「跨文本稳定」时优先 clone；只需要复现配方时用 design。
- `--instruct` 只支持 1.7B CustomVoice / VoiceDesign；Base、0.6B 收到退出 4。
- `qtts voice list --json` 同时返回 builtin 与本地 profile。

## 克隆音色

```bash
qtts voice add alice --kind clone \
  --ref-audio ref.wav --ref-text "参考音频逐字转写" \
  --mode icl --language Chinese \
  --i-have-rights --license self-owned \
  --model-dir "$QTTS_MODEL_DIR"
```

- **参考音频准入**：<2s 硬失败；2–3s warning；3–30s 推荐；>60s warning；
  <16kHz warning；还会检查削波、静音比例、有效语音占比。没有 `--allow-long-ref`。
- `--mode icl` 必须有 `ref_text`；跨文本最稳。
- `--mode xvec` 不需要 `ref_text`，但稳定性通常不如 ICL。
- `--auto-ref-text` 用 Qwen3-ASR 转写参考音频；不能与 `--ref-text` 或 `xvec`
  同用。转写发生在 staging，失败不会留下半成品 profile。
- `--i-have-rights` 是授权确认，必须由用户明确给出；不要代签。
- `voice.json` 记录 `source.ref_text_source=asr` 与 `source.asr_model`（自动转写时）。

## 设计音色

```bash
qtts voice add narrator --kind design \
  --description "低沉、沙哑的中年男声，语速偏慢" \
  --sample-text "这是试音句。" --language Chinese

# 需要跨文本稳定时，固化为 clone
qtts voice add narrator --kind design \
  --description "低沉、沙哑的中年男声，语速偏慢" \
  --sample-text "这是试音句。" --materialize-clone
```

- design 会自动记录 `rights_confirmed: true`，不需要 `--i-have-rights`。
- `--materialize-clone` 生成参考音频并冻结为 clone prompt；之后 `voice:narrator`
  走 clone 路径。
- 固化前建议先 `voice preview` 试听；不满意就改 description 重新 add。

## 试听、检查与删除

```bash
qtts voice list --json
qtts voice inspect alice --json
qtts voice preview alice --text "你好，我是 Alice。" --out alice-preview.wav
qtts voice rm alice            # 交互确认
qtts voice rm alice --yes      # 只在用户明确同意后
```

- `voice preview` 没有 `--dry-run`；`--out` 与 `<out-stem>.recipe.json` 静默覆盖。
  内置音色默认写 cwd 的 `<speaker>.preview.wav`；本地 clone/design 默认写
  `<data-dir>/voices/<name>/preview.wav`。
- `voice rm` 只删除 store 根目录内的单个音色；不会校验 clone 参考文件完整性。

## 归档：`.qttsvoice`

```bash
qtts voice export alice -o alice.qttsvoice
qtts voice import alice.qttsvoice --name alice-copy
```

- `.qttsvoice` 是 zip 容器，白名单成员：
  `manifest.json`、`voice.json`、`prompt.safetensors`、`ref.wav`、`preview.wav`。
- 导入逐项校验路径、大小、压缩比、sha256 与 safetensors 结构，不使用
  `extractall`；替换已有音色失败时回滚。
- `export` 已有目标需 `--force`；`import` 已有音色需 `--force`。
- **`voice import` 没有同意旗标**；导入前由用户确认授权。
- full 包可能含参考音频、逐字转写与 consent，按个人数据处理。

## 多音色 bundle

```bash
qtts voice export --all --out-dir ./voices-bundle
qtts voice import-bundle ./voices-bundle --dry-run --json
qtts voice import-bundle ./voices-bundle
```

- 目录含 `bundle.json`（`format="qtts-voice-bundle"`、`bundle_schema_version: 2`）
  与每个单音色 `.qttsvoice`（archive schema 1 不变）。
- 导出/导入都先 staging 全部归档再一次性发布；失败回滚，崩溃后由
  `.bundle-export.json` / `.bundle-import.json` 自动恢复。
- 导出拒绝大小写冲突、symlink 输出与非托管文件；导入拒绝 symlink、大小/哈希
  不符与大小写冲突；`--force` 替换，`--dry-run` 只校验不落盘。

## 从 btts 导入

```bash
# 默认 auto：clone 走 ICL，design 离线导入 recipe
qtts voice import-btts ./btts-export --voice dusk --i-have-rights --json
qtts voice import-btts ./btts-export.zip --all --i-have-rights --json

# 只规划；不写 profile、不 scan 模型
qtts voice import-btts ./btts-export --all --i-have-rights --dry-run --json

# 只有确实需要旧 clone 行为时才显式选择
qtts voice import-btts ./btts-export --voice narrator --i-have-rights \
  --design-mode clone --auto-ref-text --json
```

- bundle 是 `btts voice export --mode full` 的目录或 `.zip`。
- 有 `manifest.json` 时校验每文件 SHA-256、大小与完整文件集；拒绝路径穿越、
  绝对路径、盘符、反斜杠、symlink、特殊文件、大小写冲突、未知文件与超大 bundle。
- `--design-mode auto`（默认）：clone voice 走 clone；design voice 保存为 qtts
  `kind=design` recipe，保留 instruction/structured/template/seed/cfg 与内嵌
  `design.yaml`，不加载模型、不跑 ASR、不重放 btts seed。跨模型声学不等价。
- 传 `--auto-ref-text`（Qwen3-ASR）、`--ref-text`、`--asr-model`、`--mode` 或
  `--model` 时自动按 clone 处理；也可显式 `--design-mode clone`。recipe/auto 与
  这些 clone-only 选项互斥。
- 6 个 btts 模板 ID 和 `btts/...` 会 canonicalize 为 `qtts/...`；原始 ID 保留在
  `source.design.template_id`。未知模板不猜测，保留 raw ID 并 warning。
- 必须 `--i-have-rights`；已有音色需 `--force`；多音色安装走
  `.btts-import.json` journal + lock，失败/崩溃后自动回滚。
- `--voice NAME` 可重复；`--all` 与 `--voice` 二选一；`--name` 只用于单个 voice。

## 数据目录与 takes

```text
<data-dir>/
├── voices/<name>/          # voice.json、prompt.safetensors、ref.wav、preview.wav
├── cache/                  # 内容哈希音频缓存
├── recipes/                # --out - 时的 recipe 落点
├── takes/<job_id>/         # job.json、take-NN.wav、take-NN.recipe.json
│   └── .locks/<job_id>.lock
└── serve/                  # staging/<job_id>、jobs/<job_id>
```

- `job_id` 是规范化请求哈希（`tk_` + 16 hex）；请求完全一致时复用已完成 job。
- `job.json` 损坏或缺文件时旧目录隔离为 `.damaged-*` 后重建。
- `takes list/compare` 输出前校验音频 sha256 与 recipe 身份，损坏退出 4。
- `takes promote` 不转码，只复制原始字节；目标后缀必须与 job 格式一致。
- 不要手改 `job.json`、take WAV 或 recipe；改了会校验失败。

## 项目锁与模型

```bash
qtts project init . --voice builtin:Vivian --subtitle srt
qtts project lock --project qtts.project.json
qtts models list --json
qtts models pull Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --source hf --revision main
qtts models prune --yes
```

- 模型目录需要 1.7B Base / CustomVoice / VoiceDesign 与 ForcedAligner 等子目录；
  ASR 模型用于 `check`、长音频对齐、`--auto-ref-text`、`--select cer`。
- `models pull` 联网且体积大；先问用户，建议 `--revision` 固定。
- `models prune` 删除本地模型，必须 `--yes`。
- `project lock` 记录全量 sha256/大小与运行时版本；不要手改。

## 微调 profile

`finetune train --register NAME` 会把选中的 checkpoint 注册为 `voice:NAME`
（kind=`finetune`），不复制权重。之后 `say --voice voice:NAME` 走该 profile；
`voice preview` 也支持。详见 `commands-production.md` 的 finetune 一节。
