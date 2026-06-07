# HEV Generative Sandbox

**HEV Generative Sandbox** 是一个用于评估大语言模型在特定领域中社会风险的仿真框架与固定测试基准。它遵循 **Hazard–Exposure–Vulnerability，危害–暴露–脆弱性，HEV** 风险视角：在领域语境中构造风险场景，模拟不同用户交互方式，测试目标模型，并审计模型响应是否安全。

本仓库同时支持两种流程：

1. **固定数据集评测**：使用仓库中已经整理好的 HEV-Sandbox benchmark，适合复现实验和生成稳定表格。
2. **在线沙盒评测**：保留原始 agent loop，由 Scenario Agent 在线生成新场景，再让 User Agent、Target LLM 和 Auditor Agent 完成闭环评测。

运行时包含四个角色：

1. **Scenario Agent**：从领域语料和 hazard list 生成领域相关风险场景。
2. **User Agent**：模拟 secure、standard、perturbation 和 adversarial 四类用户交互。
3. **Target LLM**：被评测的目标模型。
4. **Auditor Agent**：审计目标模型响应并计算风险指标。

每个角色使用的模型都可以单独配置。你可以让所有角色使用同一个模型，也可以分别配置 Scenario Agent、在线 adversarial User Agent、Target LLM 和 Auditor Agent 的模型。

> 本仓库用于 AI safety 研究、benchmark、红队评测和模型风险分析。它不是生产环境 moderation system、安全过滤器或法律/合规工具。

---

## 目录

- [两种评测流程](#两种评测流程)
- [仓库结构](#仓库结构)
- [安装](#安装)
- [配置与模型角色](#配置与模型角色)
- [固定 benchmark 数据](#固定-benchmark-数据)
- [快速开始：固定数据集评测](#快速开始固定数据集评测)
- [在线沙盒评测](#在线沙盒评测)
- [输出文件](#输出文件)
- [指标](#指标)
- [复现论文式实验](#复现论文式实验)
- [开发与测试](#开发与测试)
- [负责任使用](#负责任使用)
- [引用](#引用)
- [License](#license)

---

## 两种评测流程

### 1. 固定数据集评测，推荐用于复现

该流程直接读取 `data/benchmark/` 中的固定数据，不重新生成题目。它使用固定问题、固定扰动、固定 jailbreak prompt 和 option-level safety rubric，因此适合复现论文式实验。

```text
已发布 benchmark 样本
        |
        v
固定 prompt 或 UserAgent 生成 prompt
        |
        v
Target LLM
        |
        v
抽取 A/B/C/D
        |
        v
AuditorAgent 按 rubric 查表
        |
        v
LEF / vulnerability / domain / hazard / user-type 汇总
```

### 2. 在线沙盒评测

该流程保留原有在线逻辑：Scenario Agent 从领域语料采样并生成新场景，User Agent 生成交互 prompt，Target LLM 回答，Auditor Agent 审计。

```text
领域语料 + hazard list
        |
        v
ScenarioAgent
        |
        v
UserAgent
        |
        v
Target LLM
        |
        v
AuditorAgent
        |
        v
风险指标
```

在线流程适合探索新领域或生成新场景。因为它会在运行时生成新题目，所以不建议用它来追求完全可复现的 benchmark 数字。

---

## 仓库结构

```text
HEV-Sandbox/
├── hev_sandbox/
│   ├── pipeline.py                 # 在线评测和固定数据集评测 pipeline
│   ├── dataset.py                  # benchmark 加载、split 展开和校验
│   ├── cli.py                      # `hev-sandbox` 命令行入口
│   ├── agents/
│   │   ├── scenario_agent.py        # Scenario Agent
│   │   ├── user_agent.py            # User Agent
│   │   └── auditor_agent.py         # Auditor Agent
│   └── utils/                       # JSONL、指标和工具函数
├── data/
│   ├── hazard_list.json             # 标准 hazard 类别
│   ├── domain_corpus/               # 在线生成模式用的小样例语料
│   ├── benchmark/
│   │   ├── base/                    # 主 benchmark，共 5,960 条
│   │   ├── perturbation/            # 固定非恶意扰动 split，共 23,840 条
│   │   └── adversarial/             # 5,960 行；展开后 29,800 个 prompt
│   └── model_outputs/legacy_qwen3/  # 来自结果压缩包的 Qwen3 参考输出
├── config/
│   ├── config.yaml                  # 默认 mock 配置，不需要 API key
│   └── openai.example.yaml          # OpenAI-compatible 真实模型配置示例
├── examples/
│   ├── run_evaluation.py            # 在线沙盒流程
│   └── run_dataset_evaluation.py    # Python 固定数据集示例
├── scripts/
│   ├── evaluate_existing_dataset.py # 固定数据集评测主脚本
│   ├── validate_dataset.py          # 数据集校验
│   ├── aggregate_results.py         # 将 detail 文件聚合为 CSV
│   └── prepare_dataset.py           # 数据集准备辅助脚本
├── tests/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── LICENSE-DATA
└── CITATION.cff
```

仓库中只保留两个 Markdown 文件：`README.md` 和 `README.zh-CN.md`。

---

## 安装

```bash
git clone https://github.com/SII-HZY/HEV-Sandbox.git
cd HEV-Sandbox

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

本地 HuggingFace 模型和高级扰动工具的可选依赖：

```bash
pip install -r requirements-optional.txt
python -m spacy download en_core_web_sm  # 可选
```

固定数据集评测默认不需要 spaCy，除非你显式要求 User Agent 在线生成新的 perturbation，而不是使用已经发布的 perturbation split。

---

## 配置与模型角色

默认 `config/config.yaml` 使用确定性的 `mock` 模型，因此可以不配置 API key 直接跑通 smoke test。真实实验建议复制或修改 `config/openai.example.yaml`。

### 角色级模型配置

pipeline 支持以下模型角色：

| 配置块 | 使用位置 | 固定数据集评测是否必需 | 说明 |
|---|---|---:|---|
| `target_model` | Target LLM | 是 | 被评测的目标模型，最终风险分数衡量的是它。 |
| `scenario_model` | Scenario Agent | 否 | 只在在线生成新场景时使用。 |
| `adversarial_model` | 在线 adversarial User Agent | 否 | 固定 adversarial split 会直接使用已发布 jailbreak prompts，不需要它。 |
| `auditor_model` | Auditor Agent 的 LLM fallback | 否 | 只在无法抽取 A/B/C/D 且启用 fallback 时使用。 |

三类 Agent 不需要强制使用同一个模型。简单实验可以所有角色用同一模型；更严格的实验可以用 DeepSeek 生成场景、用 Qwen 作为目标模型、用 GPT 作为审计 fallback。

### 示例：所有角色使用同一个模型

```yaml
openai:
  api_key: "env:OPENAI_API_KEY"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
  max_tokens: 200
  temperature: 0.0
  top_p: 1.0

target_model:
  provider: openai
scenario_model:
  provider: openai
adversarial_model:
  provider: openai
auditor_model:
  provider: openai
```

### 示例：不同角色使用不同模型

```yaml
# 被评测模型。
target_model:
  provider: local
  model_path: "Qwen/Qwen3-8B"
  device: "cuda"
  max_tokens: 200
  temperature: 1.0
  top_p: 0.95

# 只在在线模式中生成新场景。
scenario_model:
  provider: deepseek
  model: "deepseek-chat"
  temperature: 0.7
  max_tokens: 500

# 只在在线 adversarial UserAgent 生成 prompt 时使用。
adversarial_model:
  provider: deepseek
  model: "deepseek-chat"
  temperature: 0.7
  max_tokens: 500

# 只在选项抽取失败并启用 LLM fallback 时使用。
auditor_model:
  provider: openai
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 300

deepseek:
  api_key: "env:DEEPSEEK_API_KEY"
  base_url: "https://api.deepseek.com/v1"

openai:
  api_key: "env:OPENAI_API_KEY"
  base_url: "https://api.openai.com/v1"
```

不要把 API key 写入 YAML。请使用 `env:VARIABLE_NAME`，并在运行前导出对应环境变量。

---

## 固定 benchmark 数据

固定 benchmark 位于 `data/benchmark/`，格式为压缩 JSONL。

| Split | 来源压缩包 | 数量 | 用途 |
|---|---:|---:|---|
| `base` | `generated_new.zip` | 5,960 条 | 主 HEV-Sandbox 多选安全 benchmark，用于 Standard User。 |
| `perturbation` | `augmented_data.zip` | 23,840 条 | 固定非恶意扰动，用于 Perturbation User。 |
| `adversarial` | `generated_jailbreak.zip` | 5,960 行；展开后 29,800 个 prompt | 每题 5 个 jailbreak 风格 prompt，用于 Adversarial User。 |
| `data/model_outputs/legacy_qwen3` | `results.zip` | 17,880 条 detail 结果 | Qwen3-4B/14B/32B 参考输出，不是 ground truth。 |

六个领域如下：

| 领域 key | 论文领域名 | `base` 数量 |
|---|---|---:|
| `conv` | Daily Dialogue / 日常对话 | 1,000 |
| `wiki` | Encyclopedic Knowledge / 百科知识 | 994 |
| `news` | News / 新闻 | 999 |
| `medical` | Medicine / 医疗 | 1,000 |
| `law` | Law / 法律 | 967 |
| `edu` | Education / 教育 | 1,000 |

公开版规范化文件没有包含原始 `original_conversation` 原文，只保留 `source_context_hash`，以降低原始语料再分发、版权和隐私风险。

### 规范化样本字段

```json
{
  "id": "medical_000001",
  "base_id": "medical_000001",
  "split": "base",
  "domain": "medical",
  "domain_name": "Medicine",
  "test_field": "misinformation and false content",
  "hazard": "misinformation",
  "hazard_name": "Misinformation and False Content",
  "hazard_subtype": "misinformation and false content",
  "question": "包含 A/B/C/D 选项的多选题...",
  "prompt": "发送给目标模型的 prompt...",
  "reason": "为什么这道题能测试对应风险...",
  "analysis": {
    "A": {"safety": "safe", "explanation": "..."},
    "B": {"safety": "unsafe", "explanation": "..."},
    "C": {"safety": "safe", "explanation": "..."},
    "D": {"safety": "unsafe", "explanation": "..."}
  },
  "safe_options": ["A", "C"],
  "unsafe_options": ["B", "D"],
  "contact_frequency": 1.0,
  "source_context_hash": "sha256...",
  "source_context_included": false,
  "source_item_hash": "sha256..."
}
```

split 特有字段：

- `perturbation`：包含 `base_id`、`original_question`、`augmentation` 和 `augmentation_strategy`。
- `adversarial`：每条存储 `jailbreak_prompts`，其中包含 5 个 `{id, strategy, prompt}`；加载器会在评测时展开为单条 prompt。
- `model_outputs/legacy_qwen3`：包含参考模型响应、抽取选项、安全标签和 latency 等字段。

---

## 快速开始：固定数据集评测

先校验数据：

```bash
python scripts/validate_dataset.py --split base --domains medical --limit 20
```

使用默认 mock 模型跑 smoke test：

```bash
python scripts/evaluate_existing_dataset.py \
  --config config/config.yaml \
  --split base \
  --domains medical \
  --limit 10 \
  --user-types standard \
  --output-dir results/dataset
```

等价命令行入口：

```bash
hev-sandbox dataset \
  --config config/config.yaml \
  --split base \
  --domains medical \
  --limit 10 \
  --user-types standard \
  --output-dir results/dataset
```

mock 模型总是返回 `The best option is A.`，只用于验证仓库能端到端运行，不代表真实模型风险结果。

使用真实 OpenAI-compatible 目标模型：

```bash
export OPENAI_API_KEY="..."

python scripts/evaluate_existing_dataset.py \
  --config config/openai.example.yaml \
  --split base \
  --domains conv wiki news medical law edu \
  --user-types standard \
  --output-dir results/openai_base
```

本地 HuggingFace 模型需要在配置中设置 `target_model.provider: local` 和 `local.model_path`。

---

## 在线沙盒评测

在线流程会先生成新场景，再进行评测：

```bash
python examples/run_evaluation.py \
  --config config/openai.example.yaml \
  --domain medical \
  --num-scenarios 10 \
  --user-types standard adversarial \
  --output-dir results/online
```

等价 CLI：

```bash
hev-sandbox online \
  --config config/openai.example.yaml \
  --domain medical \
  --num-scenarios 10 \
  --user-types standard adversarial \
  --output-dir results/online
```

在线模式中，`scenario_model` 用于 Scenario Agent 生成场景，`adversarial_model` 用于在线 adversarial User Agent 生成 prompt，`target_model` 是被评测模型，`auditor_model` 只在启用 LLM fallback 审计时使用。

---

## 输出文件

固定数据集评测输出目录示例：

```text
results/dataset/base_medical_n10/
├── inputs.jsonl.gz
├── standard_detail.jsonl
├── detail.jsonl
├── summary.json
└── report.txt
```

每条 detail 包含规范化场景、prompt、response、抽取到的选项、安全标签、vulnerability 和对应的 expected analysis。

在线评测输出目录示例：

```text
results/online/<domain>/
├── scenarios.jsonl
├── <user_type>_results.jsonl
├── summary.json
└── report.txt
```

---

## 指标

框架遵循 HEV 风险视角：

```text
Risk = Hazard × Exposure × Vulnerability
LEF  = unsafe responses / total evaluated responses
```

评测器会输出：

| 指标 | 含义 |
|---|---|
| `lef` | Loss Event Frequency，unsafe 响应占比。 |
| `lef_percent` | 0–100 量纲的 LEF。 |
| `average_cf` | 平均 contact frequency；固定 benchmark 默认是 `1.0`。 |
| `vulnerability` | contact-frequency 加权后的 unsafe 响应比例。 |
| `by_domain` | 按领域聚合。 |
| `by_hazard` | 按标准 hazard 类别聚合。 |
| `by_user_type` | 按用户类型聚合。 |

对于当前发布的固定 benchmark，`contact_frequency` 默认是 `1.0`，因此在没有自定义 CF 的情况下，`vulnerability` 通常等于 LEF。

---

## 复现论文式实验

固定数据与用户类型的对应关系如下：

```text
base split + standard UserAgent      -> Standard User
base split + secure UserAgent        -> Secure-Baseline User
perturbation split, fixed prompts    -> Perturbation User
adversarial split, fixed prompts     -> Adversarial User
```

命令如下：

```bash
# Standard User
python scripts/evaluate_existing_dataset.py \
  --config config/openai.example.yaml \
  --split base \
  --user-types standard \
  --output-dir results/paper/standard

# Secure-Baseline User
python scripts/evaluate_existing_dataset.py \
  --config config/openai.example.yaml \
  --split base \
  --user-types secure \
  --output-dir results/paper/secure

# Perturbation User
python scripts/evaluate_existing_dataset.py \
  --config config/openai.example.yaml \
  --split perturbation \
  --output-dir results/paper/perturbation

# Adversarial User
python scripts/evaluate_existing_dataset.py \
  --config config/openai.example.yaml \
  --split adversarial \
  --output-dir results/paper/adversarial
```

只跑某一种 adversarial strategy：

```bash
python scripts/evaluate_existing_dataset.py \
  --config config/openai.example.yaml \
  --split adversarial \
  --adversarial-strategy role_playing \
  --domains law \
  --limit 100 \
  --output-dir results/paper/adversarial_role_playing
```

聚合 detail 文件为 CSV：

```bash
python scripts/aggregate_results.py \
  --input-dir results/paper \
  --output-dir results/paper/tables
```

如果要得到和论文可比的分数，需要使用相同的模型版本、解码参数、system prompt、API endpoint、领域列表和用户类型定义。完全一致的论文表格还需要原实验中的闭源 API 访问权限和对应模型快照。本仓库提供固定数据、加载器、评测脚本和 legacy Qwen3 参考输出，但不包含 API key 或全部闭源模型输出。

---

## 开发与测试

运行单测：

```bash
pytest -q
```

本包预期结果：

```text
15 passed
```

校验固定 benchmark：

```bash
python scripts/validate_dataset.py --split base
python scripts/validate_dataset.py --split perturbation
python scripts/validate_dataset.py --split adversarial
```

检查文件完整性：

```bash
sha256sum -c checksums.sha256
```

---

## 负责任使用

`adversarial` split 包含 jailbreak 风格 prompt，只能用于合法研究、benchmark、红队评测和安全改进。不要使用这些 prompt 或模型输出来绕过安全机制、伤害他人、泄露隐私或部署不安全系统。评测外部模型 API 时，请遵守对应服务条款。

公开版 benchmark 已删除原始 `original_conversation` 文本。如果要重新分发 raw context 或第三方来源语料，请先确认 license 兼容性和隐私要求。

---

## 引用

```bibtex
@inproceedings{liu2026hev,
  title     = {HEV Generative Sandbox: A Framework for Assessing Domain-Specific Social Risks Through Human-LLM Simulation},
  author    = {Liu, Yiran and Hou, Zhiyi and Xu, Xiaoang and Wang, Shuo and Wu, Huijia and Yu, Kaicheng and Yu, Yang and Zhai, ChengXiang},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {38},
  pages     = {32249--32257},
  year      = {2026},
  doi       = {10.1609/aaai.v40i38.40498}
}
```

---

## License

代码使用 MIT License，见 [LICENSE](LICENSE)。

规范化 benchmark 数据使用 [LICENSE-DATA](LICENSE-DATA) 中的条款。公开任何 raw context 或第三方语料派生数据之前，请确认上游数据 license 兼容。
