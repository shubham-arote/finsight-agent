"""state.py — the agent's typed state (JSON-serializable: it gets checkpointed).

Claims are the structured-citation contract:
    claim    = {"text": str, "citations": [{"page": int, "block_id": int|None}],
                "verified": bool}
Every figure a claim asserts must trace to its *cited* blocks (or the computation) —
checked deterministically by the cite_check node.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from ..llm import LLMRouter
from ..retrieval.base import Retriever

MAX_ATTEMPTS = 3


class RAGState(TypedDict, total=False):
    user_question: str                      # raw text the user typed (for history)
    question: str                           # working query (contextualized -> retrieval)
    original_question: str                  # resolved standalone question (grade/generate)
    retrieved: list[dict]
    grade: str                              # relevant | weak
    attempts: int
    task: str                               # supervisor lane: qa | calc | compare
    computation: dict | None                # {expr, result} from the calculator
    answer: str
    claims: list[dict]                      # structured claim -> citation mapping
    sources: list[dict]                     # top evidence summaries (UI/highlighting)
    unverified: list[str]                   # figures not traceable to citations/computation
    injection_flags: list[str]
    history: Annotated[list, operator.add]  # [{q, a}, ...] accumulates across turns


@dataclass
class Deps:
    """What nodes need besides state — injected once at graph build."""
    router: LLMRouter
    retriever: Retriever
    k: int = 6
    multi_doc: bool = False      # retriever spans >1 document (cross-doc compare lane)
