# qwen-tts-cli 工作进程文档

| 项目 | 内容 |
|---|---|
| 文档性质 | 过程记录，不是需求文档 |
| 记录范围 | 2026-11-02 需求讨论 → 2026-11-03 P0 交付 |
| 对应需求 | [PRD.md](PRD.md) v0.4 |
| 对应代码 | `qtts_cli/` P0 实现 |
| 当前状态 | P0 已完成，经两轮独立审查并修复，119 个测试通过 |

本文记录「为什么做成现在这样」，包括走过的弯路、被推翻的判断和关键取舍。面向后续维护者，避免重复踩坑。

---

## 1. 参与方与分工

| 角色 | 位置 | 职责 |
|---|---|---|
| 用户 | — | 提出原始需求、四项核心能力、设备与模型约束、最终决策 |
| t5（Claude Code） | Herdr 早期 pane | 第一轮功能构思，产出音色资产/文本流水线/批量/字幕质检四层设计 |
| t11（Claude Code） | Herdr 当前会话 | 需求澄清、PRD 编写、与 codex 讨论、任务编排、独立验证 |
| t15（Claude Code） | Herdr `qwen-tts-cli / Claude Code` | P0 实现与返修 |
| t10（codex） | Herdr `qwen-tts-cli / Codex` | 技术核对、PRD 评审、两轮代码审查 |

Herdr 的 `hc` CLI 用于跨 pane 发送任务、读取输出与等待状态；模型与代码始终留在本机。

---

## 2. 时间线总览

| 阶段 | 时间 | 产出 |
|---|---|---|
| 1. 需求构思 | 11-02 下午 | 四项核心能力；第一版功能地图 |
| 2. 三方讨论 | 11-02 晚 | codex 核对源码，纠正 8 处技术判断 |
| 3. PRD v0.1 | 11-02 晚 | 合并稿落成 `PRD.md` |
| 4. 决策更新 v0.2 | 11-02 晚 | Mac 为首个目标、中英双语、9 个内置音色 |
| 5. 依赖实测 v0.3 | 11-02 深夜 | 推翻「必须子进程桥接」的判断，确认混装可用 |
| 6. P0 实现 | 11-02 深夜 → 11-03 凌晨 | `qtts_cli/`、`tests/`、114 个测试 |
| 7. 第一轮审查 | 11-03 凌晨 | 10 条 P0 阻断 + 10 条 P1 |
| 8. 返修与复核 | 11-03 凌晨 | 3 条 PARTIAL 收尾，119 个测试 |
| 9. 文档与仓库初始化 | 11-03 | `README.md`、`WORKLOG.md`、`.gitignore`、git 提交 |

---

## 3. 阶段记录

### 3.1 需求构思

用户提出 `qwen-tts-cli` 的四项核心能力：

1. 语音复刻（Base）
2. 音色设计（VoiceDesign）
3. 内置音色生成（CustomVoice）
4. 字幕（Qwen3-ForcedAligner）

第一轮构思的关键结论是：**不要做三个 `generate_*` 的薄壳**。真正的价值在四层：

- 音色资产（voice profile，而不是每次传参）
- 文本流水线（规范化、分段、预算）
- 批量与长文本
- 字幕/质检闭环

这四层后来成为 PRD 的主干。

### 3.2 与 codex 的技术核对

codex 实际读了 `../Qwen3-TTS` 源码，纠正了上一版的 8 处判断：

| 原判断 | 核对结果 |
|---|---|
| 无流式，只有 `non_streaming_mode` | 方向对，但该参数只模拟流式**文本输入**，不是流式生成 |
| batch 只是 CLI 的 manifest 问题 | 底层原生接受 list，但没有 `max_inference_batch_size`，必须自行分桶限批 |
| 克隆必须 `ref_text` | 只有 ICL 模式需要；xvec 模式不需要 |
| design→clone 必须顺序加载 | 是工程选择，不是 API 约束 |
| 可复用 `scripts/align_srt.py` | 它调用了内部接口，不建议复用 |
| ASR 可用于质检 | 对，但对齐器不能替代 ASR |
| prompt 可以 pickle | 建议 `profile.json + prompt.safetensors` |
| 长文本直接送模型 | `max_new_tokens` 默认 2048，是 codec token，12Hz 约 12.5 token/秒 |

这些结论直接写进了 PRD §3「技术事实与约束」。

### 3.3 PRD 版本演进

| 版本 | 变更 |
|---|---|
| v0.1 | 合并三方讨论，形成 P0/P1/P2 功能清单、命令面、数据模型、里程碑 |
| v0.2 | 首个目标设备定为 Mac；验收语言增加英文；M0 支持全部 9 个内置音色；新增依赖门禁 |
| v0.3 | 依赖实测后改写 §3.3：从「硬冲突、必须桥接」改为「钉版不一致、混装可用」 |
| v0.4 | P0 实现完成；sidecar 路径澄清为 `<stem>.recipe.json` |

### 3.4 依赖兼容性：一次被推翻的判断

这是过程中最重要的一次纠错。

**初始判断**：`qwen-tts 0.1.1` 要求 `transformers==4.57.3`，PyPI `qwen-asr 0.0.6` 要求 `transformers==4.57.6`，两者不能共存，必须独立环境 + 子进程桥接。

**用户质疑**：4.57.3 和 4.57.6 能有什么差别？

**实测**（Herdr 新开 Claude Code，使用 `qwen3-asr` 环境）：

| 测试 | 结果 | 耗时 |
|---|---|---|
| TTS 1.7B CustomVoice 加载（MPS） | OK | 5.89s |
| 生成 4.00s 中文音频 | OK | 7.60s |
| ForcedAligner 加载 | OK | 1.86s |
| 对齐同一段音频 | OK，11 个 token | 0.51s |
| 同进程先 TTS 后 align | OK，无注册冲突 | — |
| 4.57.3 对照组 TTS | OK | — |

**结论修正**：这是元数据钉版不一致，不是运行时不兼容。M0 改为单环境混装，锁定实测版本组合；子进程桥接保留为 fallback。

**教训**：`pip check` 的版本告警不等于运行时不兼容。元数据是各团队测试范围的声明，不能替代端到端验证。

### 3.5 P0 实现

实现任务交给 t15，约束如下：

- 只做 FR-P0-01 ~ FR-P0-08，不碰 P1/P2
- 使用 `qwen3-asr` 解释器做集成测试，但不修改该环境
- 模型只读 `../Qwen3-TTS/models`
- 9 个内置音色从模型配置动态枚举，不硬编码
- `--dry-run` 不得 import torch/qwen_tts
- 能力校验必须先于模型加载

交付结果：

| 项 | 数值 |
|---|---|
| Python 文件 | 34 个（含测试） |
| 代码量 | 约 7,600 行 |
| 单测 | 66 passed + 3 skipped |
| 集成 | 3 passed |
| 中文字幕 smoke | 4.38s 音频 / 12.69s 端到端 |
| 英文 smoke | 4.70s 音频 / 4.49s 端到端 |
| `doctor --bench` | RTF 0.886（中）/ 0.875（英），峰值 RSS 约 1.2 GiB |

### 3.6 第一轮代码审查

codex 做只读审查，结论是「P0 不可验收」，提出：

**10 条 P0 阻断**

1. `--subtitle` 的 aligner 校验发生在 TTS 生成之后
2. stdout 可能同时承载音频、字幕、JSON
3. `--out -` 的二进制流尾部混入人类摘要
4. `ValueError`/`FileNotFoundError` 未包装，出现 traceback + exit 1
5. `--json` 错误路径输出了非单一 JSON
6. sidecar 路径为 `a.wav.recipe.json`，且缺 seed；stdout 不写 recipe
7. 字幕部分匹配时可能重叠、零时长或越界
8. `voice add`/`preview` 绕过能力矩阵
9. `--language auto` 的逐段语言未校验；clone prompt 缺失发现太晚
10. 未知扩展名静默回退 WAV

**10 条 P1**：offline 未传加载器、profile 非原子、SRT 边界丢失、sidecar 字段不全、缓存 fingerprint/dtype、cache probe、取消退出码、并发原子写、测试盲区、中文日期/引号分段。

返修后新增 45 个回归测试，总数从 66 增至 114。

### 3.7 第二轮复核

codex 只核对 10 条 P0：

- PASS：1、2、3、4、6、8、10
- PARTIAL：5（argparse 级错误未走 JSON）、7（尾部饱和仍产生零时长 cue）、9（clone preview 仍先加载模型）

三条返修：

| 问题 | 修法 |
|---|---|
| argparse JSON | 自定义 `QttsArgumentParser.error()` 抛 `UsageError`；`main()` 在 `parse_args()` 前探测 `--json` |
| 零时长 cue | 尾部饱和时整体前移；仍放不下则按比例压缩，每 cue 保底 0.001s；空间不足报错 |
| preview 缺 prompt | 在 `engine.load()` 前检查 `prompt.safetensors` 与 ICL 元数据 |

最终 119 个测试，ruff 通过。

### 3.8 独立验证

编排方（t11）没有只看实现者报告，独立复跑：

```text
pytest tests -q                     # 119 passed（5 个集成默认 skip）
RUN_INTEGRATION=1 pytest ...        # 5 passed
ruff check qtts_cli tests           # All checks passed
```

额外探针：

- `--json` + 非法参数：rc=2、stdout 0 字节、stderr 单个 JSON
- `build_cues("AB", ..., audio_duration=2.0)`：`[(1.8,1.9,'A'),(1.9,2.0,'B')]`，全部 `end > start`

---

## 4. 关键决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 包名 | dist `qwen-tts-cli`、import `qtts_cli`、命令 `qtts` | PyPI 的 `qwen-tts` 已被官方占用 |
| CLI 框架 | argparse | 依赖少，P0 不需要复杂插件体系 |
| 生成入口 | 统一 `say` + 三种 voice 引用 | 用户不必理解三个底层 API |
| 字幕来源 | 原文，不用 ASR 结果 | 原文是内容真值，ASR 只做质检 |
| `say` 对齐粒度 | 按段对齐、按段检查 5 分钟 | 逐段边界已知，比整段对齐更稳；5 分钟是单样本限制 |
| 依赖形态 | 单环境混装 | 实测 4.57.6 下可用；桥接复杂度高 |
| 缓存键 | 含文本、voice prompt、模型 + fingerprint、实际 dtype、采样参数、seed | 避免错误复用；fingerprint 弥补本地模型无 revision |
| sidecar | `<stem>.recipe.json` | 与验收标准一致，避免 `a.wav.recipe.json` |
| `serve` | 降到 P2 | 单用户本地 CLI 边际价值低；Agent 用 `--stdin --jsonl` 更合适 |
| `--takes` | P1，且显式开启 | 成本线性增长，默认不选优 |
| speaker similarity | 复用 Base embedding | 不引入独立说话人模型 |

---

## 5. 产物清单

| 路径 | 说明 |
|---|---|
| `PRD.md` | 需求、功能清单、命令规范、数据模型、验收标准 |
| `README.md` | 面向使用者的安装、命令与示例 |
| `WORKLOG.md` | 本文 |
| `pyproject.toml` | 打包、extras、入口点、pytest 配置 |
| `qtts_cli/` | P0 实现 |
| `tests/` | 119 个测试，含审查回归与集成 smoke |
| `.gitignore` | 忽略缓存、虚拟环境、生成产物 |

不在仓库内：

- 模型权重：`../Qwen3-TTS/models`
- 集成测试用的隔离 venv：`/tmp/qtts-test-venv`
- 用户的 conda 环境：未修改

---

## 6. 复现方式

```bash
cd qwen-tts-cli

# 单元测试
PYTHONPATH=. /tmp/qtts-test-venv/bin/python -m pytest tests -q

# 集成测试
RUN_INTEGRATION=1 PYTHONPATH=. /tmp/qtts-test-venv/bin/python -m pytest tests -q

# 真实生成
PYTHONPATH=. $HOME/miniconda3/envs/qwen3-asr/bin/python -m qtts_cli say \
  "你好，这是一次测试。" --voice builtin:Vivian \
  --model-dir $HOME/workspace/QwenLM/Qwen3-TTS/models \
  --out /tmp/check.wav --subtitle srt

# 性能基线
PYTHONPATH=. $HOME/miniconda3/envs/qwen3-asr/bin/python -m qtts_cli doctor --bench \
  --model-dir $HOME/workspace/QwenLM/Qwen3-TTS/models
```

如果 `/tmp/qtts-test-venv` 已不存在，可用：

```bash
$HOME/miniconda3/envs/qwen3-asr/bin/python -m venv --system-site-packages /tmp/qtts-test-venv
/tmp/qtts-test-venv/bin/python -m pip install pytest
```

---

## 7. 遗留问题与下一步

### 已知限制

- 缓存 fingerprint 对大权重只取「文件名 + 大小」，同尺寸替换无法发现。
- SRT 输入会先规范化再对齐：cue 边界保留，但文本是规范化后的写法。
- 字幕低覆盖率阈值 0.3 / 0.8 是经验值，极短文本可能误报。
- 单条超长且内部无断点的引文，仍可能从引号内切开。
- MPS 不保证 bit-perfect；同 seed 只保证趋势一致。
- ASR 转写侧在 `transformers==4.57.3` 下未验证；P1 做 `check` 时需补测。

### 下一步（M1）

1. JSONL 批处理：分桶、`--resume`、`--retry`、失败隔离
2. 多角色脚本与 `script compile`
3. ASR `check` 与 CER/WER 门禁
4. 多 take 资产库
5. 字幕 CPS / 行宽门禁
6. `voice export/import`
7. `project init/lock`
8. 长音频对齐分窗（>5 分钟）

---

## 8. 复盘

**做对的：**

- 先写 PRD 再写代码，把模型能力差异、依赖冲突、对齐限制提前暴露。
- 关键技术判断要求实现者/审查者读源码并给行号，不接受二手结论。
- 审查与实现分离：codex 不写代码，只报可执行问题；t15 负责修复与回归。
- 编排方独立复跑测试，避免「报告说通过」代替验证。
- 依赖冲突用端到端实测推翻元数据推断，而不是绕开问题。

**可以更好的：**

- 一开始把 `pip check` 的告警当成硬冲突，PRD 走了半天弯路。
- 逐文件新建时的 Fact-Forcing Gate 消耗了不少实现时间，后续可对 `qtts_cli/**` 配置路径豁免。
- t15 单轮任务过长（约 1 小时、多次 context 压缩），后续可把 P0 拆成「骨架 + say/voice + align/doctor」三个子任务。
