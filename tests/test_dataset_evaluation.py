from pathlib import Path

from hev_sandbox import HEVPipeline, load_benchmark_records, load_evaluation_items
from hev_sandbox.agents.auditor_agent import AuditorAgent


def test_load_base_records():
    records = load_benchmark_records(
        dataset_path="data/benchmark",
        split="base",
        domains=["conv"],
        limit=2,
    )
    assert len(records) == 2
    assert records[0]["domain"] == "conv"
    assert "question" in records[0]
    assert "analysis" in records[0]


def test_expand_adversarial_records():
    items, records = load_evaluation_items(
        dataset_path="data/benchmark",
        split="adversarial",
        domains=["conv"],
        limit=1,
    )
    assert len(records) == 1
    assert len(items) == 5
    assert {item.user_type for item in items} == {"adversarial"}
    assert all(item.prompt for item in items)


def test_auditor_extracts_markdown_and_chinese_answers():
    class MockLLM:
        model = "mock"

    auditor = AuditorAgent("data/hazard_list.json", MockLLM())
    assert auditor._extract_answer("The best option is **C** because...") == "C"
    assert auditor._extract_answer("答案是 A。") == "A"
    assert auditor._extract_answer("[D] is preferable") == "D"


def test_mock_dataset_evaluation(tmp_path):
    pipeline = HEVPipeline(config_path="config/config.yaml")
    result = pipeline.evaluate_dataset(
        dataset_path="data/benchmark",
        split="base",
        domains=["conv"],
        limit=3,
        user_types=["standard"],
        output_dir=str(tmp_path),
    )
    out = Path(result["output_dir"])
    assert (out / "detail.jsonl").exists()
    assert (out / "summary.json").exists()
    assert result["num_prompts"] == 3
    assert result["summary"]["overall"]["total_tests"] == 3
