"""
HEV Generative Sandbox

A Framework for Assessing Domain-Specific Social Risks through Human-LLM Simulation
"""

__version__ = "1.0.0"
__author__ = "Yiran Liu, Zhiyi Hou, et al."
__email__ = "houzhiyi@westlake.edu.cn"

from .pipeline import HEVPipeline, LLMClient
from .agents import ScenarioAgent, UserAgent, AuditorAgent

__all__ = [
    "HEVPipeline",
    "LLMClient",
    "ScenarioAgent",
    "UserAgent",
    "AuditorAgent"
]
