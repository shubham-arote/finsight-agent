"""supervise — the supervisor picks the lane: `calc` when the answer needs arithmetic
over retrieved figures (growth/margin/ratio/comparison), `compare` when that spans
multiple documents, else `qa`."""

from __future__ import annotations

from ...retrieval import classify
from ..calculator import is_math_query
from ..state import Deps, RAGState


def supervise(state: RAGState, deps: Deps) -> dict:
    q = state.get("original_question") or state["question"]
    needs_math = is_math_query(q) or classify(q) == "comparison"
    task = ("compare" if deps.multi_doc and needs_math
            else "calc" if needs_math else "qa")
    return {"task": task}
