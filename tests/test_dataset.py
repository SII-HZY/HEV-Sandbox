from pathlib import Path

from hev_sandbox.dataset import load_benchmark_split, validate_rows
from hev_sandbox.agents.auditor_agent import AuditorAgent


class MockLLM:
    model = "mock"


def test_load_base_split_smoke():
    rows = load_benchmark_split("data/benchmark", split="base", domains=["medical"], limit=5)
    assert len(rows) == 5
    row = rows[0]
    assert row["domain"] == "medical"
    assert row["split"] == "base"
    assert set("ABCD").issubset(row["analysis"].keys())


def test_validate_base_rows_smoke():
    rows = load_benchmark_split("data/benchmark", split="base", domains=["law"], limit=10)
    report = validate_rows(rows)
    assert report["valid"]


def test_extract_markdown_and_chinese_answers():
    auditor = AuditorAgent(hazard_list_path="data/hazard_list.json", llm_client=MockLLM(), use_llm_fallback=False)
    cases = {
        "The best option is **C** because ...": "C",
        "The best option is [D].": "D",
        "答案是 A。": "A",
        "选择（B）更安全。": "B",
    }
    for text, expected in cases.items():
        assert auditor._extract_answer(text) == expected
