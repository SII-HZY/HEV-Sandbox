# HEV Generative Sandbox

**A Framework for Assessing Domain-Specific Social Risks through Human-LLM Simulation**

[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-blue)](https://github.com/SII-HZY/HEV-Sandbox)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 Overview

HEV Generative Sandbox is a simulation-based framework for systematically quantifying social risks in Large Language Model (LLM) deployments. The framework decomposes risk into three key interdependent dimensions:

- **Hazard (H)**: Domain-specific threats inherent to a given context
- **Exposure (E)**: The extent to which the LLM and its users are subjected to hazardous scenarios
- **Vulnerability (V)**: The susceptibility of the system to risk due to human interaction or model weaknesses

### Key Features

✨ **Three Core Agents**:
1. **Scenario Agent**: Generates domain-rooted test scenarios from corpus
2. **User Agent**: Simulates 4 types of user interactions (Secure-Baseline, Standard, Perturbation, Adversarial)
3. **Auditor Agent**: Evaluates model responses for safety risks

🔬 **Comprehensive Evaluation**: Supports multiple domains (Medicine, Law, Education, News, etc.)

🚀 **Scalable**: Automated end-to-end pipeline for large-scale testing

🎯 **Actionable Insights**: Provides detailed risk attribution and mitigation guidance

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from source

```bash
# Clone the repository
git clone https://github.com/SII-HZY/HEV-Sandbox.git
cd HEV-Sandbox

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🚀 Quick Start

### 1. Prepare Your Data

Place your domain corpus files in `data/domain_corpus/`:

```
data/domain_corpus/
├── medicine.txt
├── law.txt
├── education.txt
└── ...
```

### 2. Configure API Keys

Edit `config/config.yaml`:

```yaml
# API Configuration
openai:
  api_key: "your-api-key-here"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4"

# Or use other providers (DeepSeek, etc.)
deepseek:
  api_key: "your-deepseek-key"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
```

### 3. Run Evaluation

```python
from hev_sandbox import HEVPipeline

# Initialize pipeline
pipeline = HEVPipeline(config_path="config/config.yaml")

# Run full evaluation
results = pipeline.evaluate(
    domain="medicine",
    num_scenarios=100,
    user_types=["secure", "standard", "perturbation", "adversarial"]
)

# Analyze results
pipeline.generate_report(results, output_dir="results/")
```

Or use the command-line interface:

```bash
python examples/run_evaluation.py \
    --domain medicine \
    --num_scenarios 100 \
    --user_types all \
    --output_dir results/
```

## 📚 Core Components

### Scenario Agent

Generates domain-specific test scenarios with potential risks:

```python
from hev_sandbox.agents import ScenarioAgent

agent = ScenarioAgent(
    domain_corpus="data/domain_corpus/medicine.txt",
    hazard_list="data/hazard_list.json"
)

scenarios = agent.generate_scenarios(num_samples=100)
```

### User Agent

Simulates different types of user interactions:

```python
from hev_sandbox.agents import UserAgent

# Initialize with specific user type
user_agent = UserAgent(user_type="adversarial")

# Generate prompts
prompts = user_agent.generate_prompts(scenarios)
```

**Supported User Types**:
- `secure`: Security-conscious user with safety guardrails
- `standard`: Average user without modifications
- `perturbation`: Non-malicious perturbations (typos, synonyms, etc.)
- `adversarial`: Jailbreak-style attacks (role-playing, privilege escalation, etc.)

### Auditor Agent

Evaluates model responses for safety risks:

```python
from hev_sandbox.agents import AuditorAgent

auditor = AuditorAgent(
    hazard_list="data/hazard_list.json",
    model="gpt-4"
)

# Evaluate responses
evaluations = auditor.evaluate_responses(
    scenarios=scenarios,
    prompts=prompts,
    responses=model_responses
)
```

## 📊 Output Format

The framework generates comprehensive evaluation results:

```json
{
  "summary": {
    "total_tests": 1000,
    "safe_responses": 850,
    "unsafe_responses": 120,
    "undetermined": 30,
    "loss_event_frequency": 0.12
  },
  "by_domain": {
    "medicine": {"safe": 200, "unsafe": 45, "LEF": 0.18},
    "law": {"safe": 150, "unsafe": 75, "LEF": 0.33}
  },
  "by_user_type": {
    "secure": {"safe": 250, "unsafe": 10, "LEF": 0.04},
    "adversarial": {"safe": 100, "unsafe": 80, "LEF": 0.44}
  },
  "by_hazard": {
    "content_safety": {"count": 45, "percentage": 0.38},
    "privacy_breach": {"count": 30, "percentage": 0.25}
  }
}
```

## 🔧 Advanced Usage

### Custom Hazard List

Define your own hazard categories in `data/hazard_list.json`:

```json
{
  "hazards": [
    {
      "name": "Content Safety",
      "description": "Harmful content including violence, hate speech, etc.",
      "examples": ["violence", "hate_speech", "adult_content"]
    },
    {
      "name": "Privacy Breach",
      "description": "Exposure of personal or confidential information",
      "examples": ["pii_exposure", "confidential_data"]
    }
  ]
}
```

### Batch Evaluation

Evaluate multiple models simultaneously:

```python
models = ["gpt-4", "claude-3", "gemini-pro"]

for model in models:
    pipeline = HEVPipeline(model=model)
    results = pipeline.evaluate(domain="law", num_scenarios=100)
    pipeline.save_results(results, f"results/{model}/")
```

## 📈 Metrics

The framework calculates the following key metrics:

- **Loss Event Frequency (LEF)**: Probability of generating harmful content
- **Contact Frequency (CF)**: Domain-specific exposure probability
- **Probability of Action (PoA)**: Likelihood of hazard activation given scenario
- **Vulnerability (Vuln)**: Conditional probability of harm given scenario and prompt

Formula: `LEF = E[CF(s) × PoA(p,r|s) × Vuln(h|s,p,r)]`

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{liu2026hev,
  title={HEV Generative Sandbox: A Framework for Assessing Domain-Specific Social Risks through Human-LLM Simulation},
  author={Liu, Yiran and Hou, Zhiyi and Xu, Xiaoang and Wang, Shuo and Wu, Huijia and Yu, Kaicheng and Yu, Yang and Zhai, ChengXiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 📞 Contact

- **Project Lead**: Zhiyi Hou (houzhiyi@westlake.edu.cn)
- **Corresponding Author**: Yang Yu (yangyu@cup.edu.cn)
- **GitHub Issues**: [https://github.com/SII-HZY/HEV-Sandbox/issues](https://github.com/SII-HZY/HEV-Sandbox/issues)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This work was supported by Westlake University, Zhejiang University, and University of Illinois at Urbana-Champaign.
