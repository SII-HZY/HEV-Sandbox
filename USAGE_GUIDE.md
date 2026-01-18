# HEV Sandbox - Complete Usage Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Core Components](#core-components)
5. [Running Evaluations](#running-evaluations)
6. [Understanding Results](#understanding-results)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The HEV Generative Sandbox implements a three-agent architecture for systematic LLM risk assessment:

```
┌──────────────────────────────────────────────────────────────────┐
│                     HEV Pipeline Orchestrator                     │
└──────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Scenario   │      │     User     │      │   Auditor    │
│    Agent     │ ───► │    Agent     │ ───► │    Agent     │
│  (Generate)  │      │  (Simulate)  │      │  (Evaluate)  │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
   Scenarios             Prompts              Evaluations
   + CF                + Responses           + Vulnerability
```

### Three Core Agents

1. **Scenario Agent** (Exposure - E)
   - Samples from domain-specific corpus
   - Generates test scenarios with hazards
   - Estimates Contact Frequency (CF)

2. **User Agent** (Probability of Action - PoA)
   - Simulates 4 user types:
     * Secure-Baseline: Safety-conscious
     * Standard: Average user
     * Perturbation: Non-malicious noise
     * Adversarial: Jailbreak attempts
   - Generates varied prompts

3. **Auditor Agent** (Vulnerability - V)
   - Evaluates model responses
   - Identifies hazards and impacts
   - Calculates vulnerability metrics

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/SII-HZY/HEV-Sandbox.git
cd HEV-Sandbox
```

### Step 2: Create Virtual Environment

```bash
# Using conda
conda create -n hev-sandbox python=3.9
conda activate hev-sandbox

# Or using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Download spacy model for linguistic perturbations
python -m spacy download en_core_web_sm
```

### Step 4: Install Package

```bash
pip install -e .
```

---

## Configuration

### 1. API Keys

Edit `config/config.yaml` and add your API keys:

```yaml
# For OpenAI
openai:
  api_key: "sk-your-key-here"
  base_url: "https://api.openai.com/v1"
  model: "gpt-4"

# For DeepSeek
deepseek:
  api_key: "your-deepseek-key"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
```

### 2. Select Target Model

Choose which model to evaluate:

```yaml
target_model:
  provider: "openai"  # or "deepseek", "anthropic", "local_model"
```

### 3. Domain Corpus

Place your domain corpus files in `data/domain_corpus/`:

- Each file should be plain text
- One context per line
- At least 20-50 contexts recommended

---

## Core Components

### Scenario Agent

**Purpose**: Generate domain-rooted test scenarios

**Code Location**: `hev_sandbox/agents/scenario_agent.py`

**Key Functions**:
- `generate_scenario()`: Create single scenario
- `generate_scenarios()`: Batch generation
- `_estimate_contact_frequency()`: Calculate CF

**Example Usage**:

```python
from hev_sandbox.agents import ScenarioAgent
from hev_sandbox import LLMClient

# Initialize
llm = LLMClient("openai", {"api_key": "...", "model": "gpt-4"})
agent = ScenarioAgent(
    domain_corpus_path="data/domain_corpus/medicine.txt",
    hazard_list_path="data/hazard_list.json",
    llm_client=llm
)

# Generate scenarios
scenarios = agent.generate_scenarios(num_scenarios=10)
agent.save_scenarios(scenarios, "output/scenarios.jsonl")
```

**Output Format**:
```json
{
  "id": 1,
  "background": "Patient context...",
  "user_intent": "Test intent...",
  "reason": "Testing rationale...",
  "hazard": "Privacy Breaches",
  "test_question": "Question text...",
  "analysis": {...},
  "contact_frequency": 0.75
}
```

---

### User Agent

**Purpose**: Simulate diverse user interactions

**Code Location**: `hev_sandbox/agents/user_agent.py`

**User Types**:

1. **Secure** (`user_type="secure"`):
   - Adds safety guidelines
   - Promotes responsible use
   
2. **Standard** (`user_type="standard"`):
   - No modifications
   - Direct questions

3. **Perturbation** (`user_type="perturbation"`):
   - Typos and misspellings
   - Synonym substitution
   - Word deletion/insertion

4. **Adversarial** (`user_type="adversarial"`):
   - Role playing attacks
   - Privilege escalation
   - Prompt obfuscation

**Example Usage**:

```python
from hev_sandbox.agents import UserAgent

# Standard user
user = UserAgent(user_type="standard")
prompt = user.generate_prompt(scenario)

# Adversarial user
adv_user = UserAgent(
    user_type="adversarial",
    llm_client=llm  # For sophisticated attacks
)
adv_prompt = adv_user.generate_prompt(scenario)
```

---

### Auditor Agent

**Purpose**: Evaluate response safety

**Code Location**: `hev_sandbox/agents/auditor_agent.py`

**Key Functions**:
- `evaluate_response()`: Single evaluation
- `evaluate_batch()`: Batch evaluation
- `calculate_metrics()`: Aggregate statistics
- `generate_report()`: Create report

**Example Usage**:

```python
from hev_sandbox.agents import AuditorAgent

auditor = AuditorAgent(
    hazard_list_path="data/hazard_list.json",
    llm_client=llm
)

# Evaluate single response
evaluation = auditor.evaluate_response(
    scenario=scenario,
    prompt=prompt,
    response=model_response
)

# Batch evaluation
evaluations = auditor.evaluate_batch(
    scenarios=scenarios,
    prompts=prompts,
    responses=responses
)

# Generate report
report = auditor.generate_report(evaluations, "output/report.txt")
```

**Evaluation Output**:
```json
{
  "hazard": "Privacy Breaches",
  "impact": "high",
  "safety": "unsafe",
  "explanation": "Response discloses PII",
  "vulnerability": 1,
  "chosen_option": "B"
}
```

---

## Running Evaluations

### Method 1: Using Pipeline (Recommended)

```python
from hev_sandbox import HEVPipeline

# Initialize
pipeline = HEVPipeline(config_path="config/config.yaml")

# Run evaluation
results = pipeline.evaluate(
    domain="medicine",
    num_scenarios=100,
    user_types=["secure", "standard", "adversarial"],
    output_dir="results/"
)
```

### Method 2: Command Line

```bash
python examples/run_evaluation.py \
    --domain medicine \
    --num_scenarios 100 \
    --user_types all \
    --output_dir results/medicine/
```

### Method 3: Step-by-Step

```python
# 1. Generate scenarios
scenario_agent = ScenarioAgent(...)
scenarios = scenario_agent.generate_scenarios(100)

# 2. Simulate users and query model
user_agent = UserAgent(user_type="adversarial")
prompts = user_agent.generate_prompts(scenarios)
responses = [query_model(p) for p in prompts]

# 3. Evaluate
auditor = AuditorAgent(...)
evaluations = auditor.evaluate_batch(scenarios, prompts, responses)

# 4. Analyze
metrics = calculate_metrics(evaluations)
```

---

## Understanding Results

### Output Structure

```
results/
└── medicine/
    ├── scenarios.jsonl              # Generated scenarios
    ├── secure_results.jsonl         # Secure user results
    ├── standard_results.jsonl       # Standard user results
    ├── perturbation_results.jsonl   # Perturbation results
    ├── adversarial_results.jsonl    # Adversarial results
    ├── summary.json                 # Aggregate metrics
    └── report.txt                   # Human-readable report
```

### Key Metrics

**Loss Event Frequency (LEF)**:
- `LEF = Unsafe responses / Total responses`
- Primary safety metric
- Range: 0.0 (safest) to 1.0 (most unsafe)

**Contact Frequency (CF)**:
- Domain-specific hazard exposure
- Estimated per scenario
- Higher CF = more hazard-relevant content

**Vulnerability**:
- System susceptibility to risks
- Weighted by CF
- Indicates alignment robustness

### Example Summary

```json
{
  "overall": {
    "total_tests": 400,
    "safe_count": 320,
    "unsafe_count": 80,
    "lef": 0.20,
    "average_cf": 0.65,
    "vulnerability": 0.18
  },
  "by_user_type": {
    "secure": {"lef": 0.05},
    "standard": {"lef": 0.15},
    "adversarial": {"lef": 0.40}
  },
  "by_hazard": {
    "Privacy Breaches": {"count": 30, "percentage": 37.5},
    "Content Safety": {"count": 25, "percentage": 31.3}
  }
}
```

---

## Advanced Usage

### Custom Hazard List

Create your own `hazard_list.json`:

```json
{
  "hazards": [
    {
      "name": "Custom Hazard",
      "description": "...",
      "keywords": [...],
      "severity": "high"
    }
  ]
}
```

### Batch Multiple Models

```python
models = ["gpt-4", "claude-3-opus", "gemini-pro"]

for model in models:
    # Update config
    config["openai"]["model"] = model
    
    # Run evaluation
    pipeline = HEVPipeline(config)
    results = pipeline.evaluate(domain="law")
```

### Custom User Agent

```python
class CustomUserAgent(UserAgent):
    def _generate_custom_prompt(self, question):
        # Your custom logic
        return modified_question

user_agent = CustomUserAgent(user_type="custom")
```

---

## Troubleshooting

### Common Issues

**1. API Rate Limits**
- Add delays: `time.sleep(1)` between requests
- Reduce `num_scenarios`
- Use batch processing

**2. Spacy Model Missing**
```bash
python -m spacy download en_core_web_sm
```

**3. Out of Memory**
- Reduce batch sizes
- Use smaller models
- Process in chunks

**4. Config Not Found**
```python
# Ensure correct path
pipeline = HEVPipeline(config_path="./config/config.yaml")
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

- GitHub Issues: https://github.com/SII-HZY/HEV-Sandbox/issues
- Email: houzhiyi@westlake.edu.cn
- Documentation: See README.md

---

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{liu2026hev,
  title={HEV Generative Sandbox: A Framework for Assessing Domain-Specific Social Risks through Human-LLM Simulation},
  author={Liu, Yiran and Hou, Zhiyi and Xu, Xiaoang and Wang, Shuo and Wu, Huijia and Yu, Kaicheng and Yu, Yang and Zhai, ChengXiang},
  booktitle={AAAI 2026},
  year={2026}
}
```
