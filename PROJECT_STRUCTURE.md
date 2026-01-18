# HEV Sandbox Project Structure

This document provides a comprehensive overview of the project structure and explains the purpose of each component.

## Directory Tree

```
HEV-Sandbox/
│
├── README.md                    # Main project documentation
├── LICENSE                      # MIT License
├── USAGE_GUIDE.md              # Detailed usage instructions
├── setup.py                     # Package installation script
├── requirements.txt             # Python dependencies
│
├── config/
│   └── config.yaml             # Configuration file for API keys, settings
│
├── data/
│   ├── hazard_list.json        # Hazard categories and descriptions
│   └── domain_corpus/          # Domain-specific text corpora
│       ├── medicine.txt        # Medical domain corpus
│       ├── law.txt             # Legal domain corpus
│       ├── education.txt       # Education domain corpus (add your own)
│       └── ...                 # Add more domains as needed
│
├── hev_sandbox/                # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── pipeline.py             # Main evaluation pipeline
│   │
│   ├── agents/                 # Three core agents
│   │   ├── __init__.py
│   │   ├── scenario_agent.py  # Scenario generation (Exposure)
│   │   ├── user_agent.py      # User simulation (PoA)
│   │   └── auditor_agent.py   # Safety evaluation (Vulnerability)
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── data_loader.py     # Data loading utilities
│       └── metrics.py         # Metrics calculation
│
├── examples/
│   └── run_evaluation.py      # Example evaluation script
│
├── tests/
│   └── test_agents.py         # Unit tests
│
└── results/                    # Output directory (created at runtime)
    └── [domain]/
        ├── scenarios.jsonl
        ├── [user_type]_results.jsonl
        ├── summary.json
        └── report.txt
```

## Component Descriptions

### Core Package (`hev_sandbox/`)

#### `pipeline.py`
**Main orchestrator** that integrates all three agents:
- `HEVPipeline`: Main class coordinating evaluation flow
- `LLMClient`: Wrapper for different LLM API providers
- Handles end-to-end evaluation workflow
- Generates comprehensive reports

**Key Classes**:
- `HEVPipeline`: Orchestrates scenario generation → user simulation → evaluation
- `LLMClient`: Unified interface for OpenAI, DeepSeek, Anthropic, etc.

#### Agents Module (`agents/`)

##### `scenario_agent.py`
**Purpose**: Generate domain-specific test scenarios

**Corresponds to Paper**: Section "Scenario Agent: Domain-Adaptive Scenario Generation for Risk Exposure"

**Key Functions**:
- `generate_scenario()`: Creates single test scenario from corpus
- `generate_scenarios()`: Batch generation
- `_estimate_contact_frequency()`: Calculates CF metric
- `sample_context()`: Samples from domain corpus

**Implements**: Contact Frequency (CF) estimation, Exposure (E) factor

##### `user_agent.py`
**Purpose**: Simulate diverse user interactions

**Corresponds to Paper**: Section "User Agent: Simulating Diverse User Interactions to Assess Model Vulnerabilities"

**Key Classes**:
- `UserAgent`: Main class with 4 user types
- `LinguisticMutator`: Advanced syntactic transformations

**User Types**:
1. **Secure-Baseline**: Safety-conscious prompts
2. **Standard**: Unmodified queries
3. **Perturbation**: Non-malicious variations
4. **Adversarial**: Jailbreak attempts

**Implements**: Probability of Action (PoA) estimation

##### `auditor_agent.py`
**Purpose**: Evaluate model responses for safety

**Corresponds to Paper**: Section "Auditor Agent for Hazard Analysis"

**Key Functions**:
- `evaluate_response()`: Single response evaluation
- `evaluate_batch()`: Batch evaluation
- `calculate_metrics()`: Aggregate statistics
- `generate_report()`: Create detailed report

**Implements**: Vulnerability (V) measurement, Loss Event Frequency (LEF)

#### Utils Module (`utils/`)

##### `data_loader.py`
Utilities for loading and saving data:
- `load_jsonl()`: Load JSONL files
- `save_jsonl()`: Save to JSONL
- `load_json()`: Load JSON configuration
- `load_text_corpus()`: Load domain corpus

##### `metrics.py`
Calculate HEV framework metrics:
- `calculate_lef()`: Loss Event Frequency
- `calculate_contact_frequency()`: CF metric
- `calculate_poa()`: Probability of Action
- `calculate_vulnerability()`: Vulnerability score
- `generate_summary_statistics()`: Comprehensive metrics

### Configuration (`config/`)

#### `config.yaml`
Central configuration file:
- API keys and endpoints
- Model selection
- Agent parameters
- Domain configurations
- Hazard definitions
- Output settings

### Data (`data/`)

#### `hazard_list.json`
Defines risk categories:
```json
{
  "hazards": [
    {
      "name": "Content Safety",
      "description": "...",
      "keywords": [...],
      "severity": "high"
    }
  ]
}
```

#### `domain_corpus/`
Text files containing domain-specific contexts:
- One context per line
- Used for scenario generation
- Minimum 20-50 contexts recommended

### Examples (`examples/`)

#### `run_evaluation.py`
Complete example showing:
- Command-line interface
- Pipeline initialization
- Running evaluations
- Result interpretation

**Usage**:
```bash
python examples/run_evaluation.py \
    --domain medicine \
    --num_scenarios 100 \
    --user_types all
```

### Tests (`tests/`)

#### `test_agents.py`
Unit tests for:
- Agent initialization
- Prompt generation
- Answer extraction
- Metrics calculation

**Run tests**:
```bash
pytest tests/test_agents.py -v
```

## Data Flow

```
1. SCENARIO GENERATION
   Domain Corpus → ScenarioAgent → Test Scenarios (with CF)
   
2. USER SIMULATION
   Test Scenarios → UserAgent → Modified Prompts
   
3. MODEL QUERYING
   Modified Prompts → Target LLM → Responses
   
4. SAFETY EVALUATION
   (Scenarios, Prompts, Responses) → AuditorAgent → Evaluations (with V)
   
5. METRICS CALCULATION
   Evaluations → Metrics Module → LEF, CF, PoA, V
   
6. REPORT GENERATION
   Metrics → Pipeline → Summary & Report
```

## Key Formulas

As implemented in the code:

**LEF (Loss Event Frequency)**:
```python
LEF = E[CF(s) × PoA(p,r|s) × Vuln(h|s,p,r)]
```

**Contact Frequency**:
```python
CF(s) = keyword_matches / total_keywords
```

**Vulnerability**:
```python
Vuln = 1 if unsafe else 0
```

## Adding New Components

### New Domain
1. Create corpus file in `data/domain_corpus/[domain].txt`
2. Add entry to `config.yaml` under `domains`
3. Run evaluation: `python examples/run_evaluation.py --domain [domain]`

### New Hazard
1. Add hazard definition to `data/hazard_list.json`
2. Include keywords and description
3. Scenarios will automatically incorporate new hazard

### New User Type
1. Extend `UserAgent` class in `user_agent.py`
2. Implement `_generate_[type]_prompt()` method
3. Add to `USER_TYPES` list

### New Model Provider
1. Add configuration to `config.yaml`
2. Implement adapter in `LLMClient` class
3. Set as target in config

## Output Files

### `scenarios.jsonl`
Generated test scenarios with metadata:
- Scenario ID
- Background context
- Test question
- Expected analysis
- Contact Frequency

### `[user_type]_results.jsonl`
Results for each user type:
- Scenario
- Generated prompt
- Model response
- Evaluation

### `summary.json`
Aggregate metrics:
- Overall statistics
- By domain breakdown
- By user type breakdown
- By hazard breakdown

### `report.txt`
Human-readable report:
- Executive summary
- Detailed breakdowns
- Recommendations

## Best Practices

1. **Corpus Quality**: Ensure domain corpus is representative and diverse
2. **Scenario Volume**: Use at least 100 scenarios per domain for statistical significance
3. **User Type Coverage**: Test all 4 user types to understand vulnerability surface
4. **Multiple Runs**: Run multiple times with different random seeds for robustness
5. **Version Control**: Track config changes and corpus updates

## Troubleshooting

See `USAGE_GUIDE.md` section "Troubleshooting" for common issues and solutions.

## Contributing

When contributing:
1. Follow existing code structure
2. Add tests for new features
3. Update documentation
4. Maintain backward compatibility

## Questions?

- GitHub Issues: https://github.com/SII-HZY/HEV-Sandbox/issues
- Email: houzhiyi@westlake.edu.cn
