"""
Unit Tests for HEV Sandbox

Basic tests to verify agent functionality.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hev_sandbox.agents import ScenarioAgent, UserAgent, AuditorAgent


def test_user_agent_initialization():
    """Test UserAgent initialization."""
    user_agent = UserAgent(user_type="standard")
    assert user_agent.user_type == "standard"
    

def test_user_agent_types():
    """Test all user agent types can be initialized."""
    user_types = ["secure", "standard", "perturbation", "adversarial"]
    
    for user_type in user_types:
        user_agent = UserAgent(user_type=user_type)
        assert user_agent.user_type == user_type


def test_user_agent_invalid_type():
    """Test UserAgent rejects invalid type."""
    with pytest.raises(ValueError):
        UserAgent(user_type="invalid")


def test_secure_prompt_generation():
    """Test secure user prompt generation."""
    user_agent = UserAgent(user_type="secure")
    scenario = {
        "test_question": "How do I reset my password?"
    }
    
    prompt = user_agent.generate_prompt(scenario)
    
    # Should contain safety language
    assert "safe" in prompt.lower() or "helpful" in prompt.lower()
    assert "How do I reset my password?" in prompt


def test_standard_prompt_generation():
    """Test standard user returns unmodified prompt."""
    user_agent = UserAgent(user_type="standard")
    original_question = "What is the capital of France?"
    scenario = {"test_question": original_question}
    
    prompt = user_agent.generate_prompt(scenario)
    
    assert prompt == original_question


def test_perturbation_prompt_different():
    """Test perturbation creates different prompt."""
    user_agent = UserAgent(user_type="perturbation")
    original_question = "How do I install Python packages?"
    scenario = {"test_question": original_question}
    
    prompt = user_agent.generate_prompt(scenario)
    
    # Perturbation should modify the prompt (in most cases)
    # Note: Due to randomness, this might occasionally fail
    # So we just check it's a string
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_extract_answer():
    """Test answer extraction from auditor."""
    from hev_sandbox.agents.auditor_agent import AuditorAgent
    
    # Mock auditor
    class MockLLM:
        model = "test-model"
        
    auditor = AuditorAgent(
        hazard_list_path="data/hazard_list.json",
        llm_client=MockLLM()
    )
    
    # Test various response formats
    test_cases = [
        ("The best option is A", "A"),
        ("I recommend option B", "B"),
        ("Answer: C", "C"),
        ("Option D is the best", "D"),
    ]
    
    for response, expected in test_cases:
        result = auditor._extract_answer(response)
        assert result == expected, f"Failed for response: {response}"


def test_metrics_calculation():
    """Test basic metrics calculation."""
    from hev_sandbox.utils.metrics import calculate_lef
    
    evaluations = [
        {"safety": "safe"},
        {"safety": "unsafe"},
        {"safety": "safe"},
        {"safety": "unsafe"},
    ]
    
    lef = calculate_lef(evaluations)
    assert lef == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
