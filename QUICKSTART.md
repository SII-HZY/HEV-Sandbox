# HEV Sandbox 快速开始指南

## 5分钟快速开始

### 1. 安装 (1分钟)

```bash
# 克隆代码
cd HEV-Sandbox

# 安装依赖
pip install -r requirements.txt

# 下载spacy模型
python -m spacy download en_core_web_sm

# 安装包
pip install -e .
```

### 2. 配置API密钥 (1分钟)

编辑 `config/config.yaml`:

```yaml
# 第14行: 填入你的OpenAI API Key
openai:
  api_key: "sk-your-api-key-here"  # 改这里!
  
# 或者使用DeepSeek (更便宜)
deepseek:
  api_key: "your-deepseek-key-here"  # 改这里!

# 第57行: 选择provider
target_model:
  provider: "openai"  # 或 "deepseek"
```

### 3. 运行评估 (3分钟)

```bash
python examples/run_evaluation.py \
    --domain medicine \
    --num_scenarios 10 \
    --user_types standard adversarial \
    --output_dir results/
```

### 4. 查看结果

```bash
# 查看报告
cat results/medicine/report.txt

# 查看详细数据
cat results/medicine/summary.json
```

---

## 完整示例

### Python代码方式

```python
from hev_sandbox import HEVPipeline

# 1. 初始化
pipeline = HEVPipeline("config/config.yaml")

# 2. 运行评估
results = pipeline.evaluate(
    domain="medicine",
    num_scenarios=100,
    user_types=["secure", "standard", "adversarial"],
    output_dir="results/"
)

# 3. 查看结果
print(f"LEF: {results['summary']['overall']['lef']:.4f}")
print(f"Safe: {results['summary']['overall']['safe_percentage']:.1f}%")
print(f"Unsafe: {results['summary']['overall']['unsafe_percentage']:.1f}%")
```

### 输出示例

```
=========================================================
HEV GENERATIVE SANDBOX - EVALUATION REPORT
=========================================================

Domain: medicine
Date: 2026-01-19 10:30:00

---------------------------------------------------------
OVERALL STATISTICS
---------------------------------------------------------

Total Tests: 300
Safe Responses: 255 (85.0%)
Unsafe Responses: 45 (15.0%)
Undetermined: 0

**Loss Event Frequency (LEF): 0.1500**
Average Contact Frequency: 0.6200
System Vulnerability: 0.1350

---------------------------------------------------------
RESULTS BY USER TYPE
---------------------------------------------------------

User Type: secure
  Total: 100
  Safe: 96 | Unsafe: 4
  LEF: 0.0400

User Type: standard
  Total: 100
  Safe: 88 | Unsafe: 12
  LEF: 0.1200

User Type: adversarial
  Total: 100
  Safe: 71 | Unsafe: 29
  LEF: 0.2900

---------------------------------------------------------
HAZARD DISTRIBUTION
---------------------------------------------------------

Privacy Breaches: 18 (40.0%)
Misinformation: 15 (33.3%)
Content Safety: 8 (17.8%)
Bias & Discrimination: 4 (8.9%)

=========================================================
```

---

## 常见问题

### Q1: 如何更换评估的LLM?

在 `config/config.yaml` 中修改:

```yaml
target_model:
  provider: "openai"  # 改成 "deepseek", "anthropic" 等

openai:
  model: "gpt-4"  # 改成 "gpt-3.5-turbo", "gpt-4-turbo" 等
```

### Q2: 如何添加新的domain?

1. 创建语料文件 `data/domain_corpus/your_domain.txt`
2. 在 `config/config.yaml` 添加配置:
```yaml
domains:
  - name: "your_domain"
    corpus_path: "data/domain_corpus/your_domain.txt"
```
3. 运行评估:
```bash
python examples/run_evaluation.py --domain your_domain
```

### Q3: 如何减少API开销?

- 减少场景数: `--num_scenarios 10`
- 只测试部分用户类型: `--user_types standard`
- 使用更便宜的模型: DeepSeek比OpenAI便宜很多

### Q4: 如何并行评估多个模型?

```python
models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]

for model in models:
    # 更新配置
    config["openai"]["model"] = model
    
    # 运行评估
    pipeline = HEVPipeline(config)
    results = pipeline.evaluate(
        domain="medicine",
        output_dir=f"results/{model}/"
    )
```

---

## 下一步

- 阅读 [CODE_EXPLANATION.md](CODE_EXPLANATION.md) 了解代码详情
- 阅读 [USAGE_GUIDE.md](USAGE_GUIDE.md) 了解高级用法
- 阅读 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解项目结构
- 查看 [论文](HEV_Generative_Sandbox___Camera_Ready__Copy_-5.pdf) 了解理论基础

## 获取帮助

- GitHub Issues: https://github.com/SII-HZY/HEV-Sandbox/issues
- Email: houzhiyi@westlake.edu.cn

## 引用

```bibtex
@inproceedings{liu2026hev,
  title={HEV Generative Sandbox: A Framework for Assessing Domain-Specific Social Risks through Human-LLM Simulation},
  author={Liu, Yiran and Hou, Zhiyi and Xu, Xiaoang and Wang, Shuo and Wu, Huijia and Yu, Kaicheng and Yu, Yang and Zhai, ChengXiang},
  booktitle={AAAI 2026},
  year={2026}
}
```
