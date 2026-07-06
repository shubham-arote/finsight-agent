"""rewrite — weak retrieval: turn the question into a keyword query and retry."""

from __future__ import annotations

from ...llm import LLMUnavailable, prompts
from ...retrieval.text import _STOP, tok
from ..state import Deps, RAGState


def rewrite(state: RAGState, deps: Deps) -> dict:
    if deps.router.available("fast"):
        try:
            better = deps.router.complete(
                "fast", prompts.get("rewrite_query").render(question=state["original_question"]),
                max_tokens=40).strip()
            if better:
                return {"question": better}
        except LLMUnavailable:
            pass
    stripped = " ".join(w for w in tok(state["original_question"]) if w not in _STOP)
    return {"question": stripped or state["question"]}
