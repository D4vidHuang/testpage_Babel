"""Prompting module for EMSE-Babel.

Contains language-specific prompt libraries for multilingual LLM-as-Judge evaluation.
"""

from .prompt_base import PromptBase
from .prompt_english import PromptLibrary

__all__ = ["PromptBase", "PromptLibrary"]
