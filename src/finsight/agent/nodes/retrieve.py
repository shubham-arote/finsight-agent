"""retrieve — hit the Retriever seam; count attempts for the rewrite budget."""

from __future__ import annotations

from ..state import Deps, RAGState


def retrieve(state: RAGState, deps: Deps) -> dict:
    hits = deps.retriever.retrieve(state["question"], deps.k)
    return {"retrieved": [dict(h) for h in hits],
            "attempts": state.get("attempts", 0) + 1}
