"""Core module for EMSE-Babel LLM-as-Judge evaluation pipeline.

Components:
- judge: OpenRouter/Ollama API wrapper
- config: JudgeConfig dataclass
- manager: Orchestration logic
- prompts: Reusable prompt templates
- parsing: Response parsing and normalization
- data: Data loading and export utilities
"""

from .judge import LLMJudge
from .config import JudgeConfig
from .manager import JudgeManager
from .prompting.prompt_base import PromptBase
from .prompting.prompt_english import PromptLibrary
from .parsing import (
    ResponseParser, 
    ParseResult, 
    ModelEvaluation, 
    extract_evaluations_from_judge_output,
    extract_tokens_from_judge_output
)
from .data import DataLoader, extract_tokens

__all__ = [
    "LLMJudge",
    "JudgeConfig", 
    "JudgeManager",
    "PromptBase",
    "PromptLibrary",
    "ResponseParser",
    "ParseResult",
    "ModelEvaluation",
    "extract_evaluations_from_judge_output",
    "extract_tokens_from_judge_output",
    "DataLoader",
    "extract_tokens",
]
