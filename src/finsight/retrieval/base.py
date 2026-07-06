"""base.py — the retrieval seam: `Retriever` protocol + `Evidence` shape.

The agent depends on this protocol, never on a concrete retriever. `Evidence` is a
citation-ready hit: page + block_id + bbox anchor the exact spot on the page, `content`
is the matched text, `parent_text`/`section_heading` carry the small-to-big context.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class Evidence(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    page: int
    block_id: int | None
    section_id: int | None
    type: str                  # text | title | table | ...
    heading: str
    text: str                  # heading-tagged matching text
    content: str               # raw block content
    bbox: list[int]
    context: str               # contextual-retrieval prefix
    parent_text: str           # the parent section (small-to-big context for the LLM)
    section_heading: str
    score: float
    exact: bool                # True = deterministic exact-value/period match, not fuzzy


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 6) -> list[Evidence]: ...
