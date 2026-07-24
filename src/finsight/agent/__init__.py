"""Public surface of the agent layer."""

from . import guards
from .brief import BRIEF_CHECKLIST, compose_markdown, is_brief_request, run_brief
from .calculator import CalcError, extract_expression, is_math_query, safe_eval
from .graph import AgentEngine
from .state import Deps, RAGState
from .verify import verify_numbers

__all__ = ["AgentEngine", "Deps", "RAGState", "guards",
           "CalcError", "extract_expression", "is_math_query", "safe_eval",
           "verify_numbers",
           "BRIEF_CHECKLIST", "compose_markdown", "is_brief_request", "run_brief"]
