"""Public surface of the LLM layer: role-based router + versioned prompt registry."""

from . import prompts
from .router import ROLES, LLMRouter, LLMUnavailable

__all__ = ["LLMRouter", "LLMUnavailable", "ROLES", "prompts"]
