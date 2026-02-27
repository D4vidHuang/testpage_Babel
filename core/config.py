"""Configuration dataclass for LLM-as-a-Judge setups."""

from dataclasses import dataclass
from typing import List, Optional

from .judge import LLMJudge


@dataclass
class JudgeConfig:
    """Configuration for a single LLM-as-a-Judge setup.

    Attributes:
        name: Unique identifier for this configuration
        models: List of model identifiers for OpenRouter
        temperature: Sampling temperature for the model
        system_message: Base system prompt describing the role
        assignment_message: Task-specific instructions
        structured_output: Whether to return structured JSON output
        provider: 'openrouter' (default) or 'ollama'
        ollama_url: Optional custom URL for Ollama
    """

    name: str
    models: List[str]
    temperature: float
    system_message: str
    assignment_message: str
    structured_output: bool = False
    provider: str = "openrouter"
    ollama_url: Optional[str] = None

    def build_assignment(self) -> str:
        """Return the final assignment message."""
        return self.assignment_message

    def create_judge(self) -> LLMJudge:
        """Instantiate an LLMJudge using this configuration."""
        system_message = self.system_message
        if self.structured_output:
            system_message += "\n\nYou must return structured JSON output as specified in the instructions."
        else:
            system_message += "\n\nReturn a plain text answer."

        kwargs = {
            "models": self.models,
            "temperature": self.temperature,
            "system_message": system_message,
            "assignment_description": self.build_assignment(),
            "provider": self.provider,
        }
        if self.provider == "ollama" and self.ollama_url:
            kwargs["ollama_url"] = self.ollama_url

        return LLMJudge(**kwargs)
