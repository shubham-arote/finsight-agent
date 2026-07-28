"""Public surface of the agent layer."""

from . import guards
from .brief import BRIEF_CHECKLIST, compose_markdown, is_brief_request, run_brief
from .calculator import CalcError, extract_expression, is_math_query, safe_eval
from .compare import compute_delta, extract_verified_figure, run_compare
from .graph import AgentEngine
from .state import Deps, RAGState
from .verify import verify_numbers

__all__ = [
           "BRIEF_CHECKLIST",
           "AgentEngine",
           "CalcError",
           "Deps",
           "RAGState",
           "compose_markdown",
           "compute_delta",
           "extract_expression",
           "extract_verified_figure",
           "guards",
           "is_brief_request",
           "is_math_query",
           "run_brief",
           "run_compare",
           "safe_eval",
           "verify_numbers",
]
