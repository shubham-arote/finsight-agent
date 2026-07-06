"""calculate — exact figures: the LLM emits ONE arithmetic expression over the retrieved
numbers; `safe_eval` (AST whitelist, never eval) computes it deterministically. Offline
or on any failure -> skip; generate falls back to its normal path."""

from __future__ import annotations

from ...llm import LLMUnavailable, prompts
from ..calculator import CalcError, extract_expression, safe_eval
from ..citations import tag_evidence
from ..state import Deps, RAGState


def calculate(state: RAGState, deps: Deps) -> dict:
    retrieved = state.get("retrieved", [])
    if not deps.router.available("fast") or not retrieved:
        return {}
    p = prompts.get("calculate_expression")
    try:
        reply = deps.router.complete(
            "fast", p.render(context=tag_evidence(retrieved),
                             question=state["original_question"]),
            max_tokens=40)
    except LLMUnavailable:
        return {}
    expr = extract_expression(reply)
    if not expr:
        return {}
    try:
        return {"computation": {"expr": expr, "result": safe_eval(expr)}}
    except CalcError:
        return {}
