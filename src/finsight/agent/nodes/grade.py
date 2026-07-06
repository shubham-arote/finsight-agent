"""grade — can this context answer the question? `relevant` or `weak`.

calc/compare lanes: the answer is DERIVED (never stated verbatim), so don't ask an LLM
whether the answer is present — accept the context as long as it holds figures, and let
the calculator + cite_check handle correctness. QA lane: cloud one-word verdict when
keyed, keyword-overlap heuristic otherwise.
"""

from __future__ import annotations

from ...llm import LLMUnavailable, prompts
from ...retrieval.text import _STOP, tok
from ..citations import tag_evidence
from ..state import Deps, RAGState


def grade(state: RAGState, deps: Deps) -> dict:
    retrieved = state.get("retrieved", [])
    ctx = tag_evidence(retrieved)
    if state.get("task") in ("calc", "compare"):
        return {"grade": "relevant" if any(c.isdigit() for c in ctx) else "weak"}
    if deps.router.available("fast") and ctx:
        p = prompts.get("grade")
        try:
            verdict = deps.router.complete(
                "fast", p.render(question=state["original_question"], context=ctx),
                max_tokens=4).lower()
            return {"grade": "relevant" if "relevant" in verdict else "weak"}
        except LLMUnavailable:
            pass
    return {"grade": _heuristic(state["original_question"], ctx)}


def _heuristic(question: str, ctx: str) -> str:
    if not ctx:
        return "weak"
    terms = {w for w in tok(question) if w not in _STOP and len(w) > 2}
    if not terms:
        return "relevant"
    ctx_low = ctx.lower()
    hit = sum(1 for w in terms if w in ctx_low)
    return "relevant" if hit / len(terms) >= 0.5 else "weak"
