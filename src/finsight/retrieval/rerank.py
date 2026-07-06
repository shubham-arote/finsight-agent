"""rerank.py — cross-encoder reranking (Cohere rerank-v3.5), key-optional.

With a COHERE_API_KEY the fused candidate window is reordered by the cross-encoder;
without one (or on any API failure) a lexical heuristic ported from the old repo takes
over: exact-term overlap, numeric-value hits, heading match, and a table boost for
number questions. Both return the same (candidates, scores) shape.
"""

from __future__ import annotations

import re

import litellm

from ..config import settings
from .text import _STOP, tok


def rerank(query: str, cands: list[dict], k: int) -> list[dict]:
    """Reorder candidates, best first; sets `score` on each. Never raises."""
    if settings.cohere_api_key and cands:
        try:
            return _rerank_cohere(query, cands, k)
        except Exception:
            pass                                   # API failure -> lexical fallback
    return _rerank_heuristic(query, cands)[:k]


def _rerank_cohere(query: str, cands: list[dict], k: int) -> list[dict]:
    docs = [(c.get("content") or c.get("text") or "") for c in cands]
    resp = litellm.rerank(model=settings.rerank_model, query=query, documents=docs,
                          top_n=k, api_key=settings.cohere_api_key)
    out = []
    for r in resp.results:
        c = dict(cands[r["index"]])
        c["score"] = float(r["relevance_score"])
        out.append(c)
    return out


def _rerank_heuristic(query: str, cands: list[dict]) -> list[dict]:
    """Lexical rerank: base score + exact-term / numeric / heading / table boosts."""
    qterms = {w for w in tok(query) if w not in _STOP and len(w) > 2}
    qnums = set(re.findall(r"\d[\d.,]{2,}", query))    # real values (6,303 / 5.9)

    def score(c: dict) -> float:
        s = float(c.get("score", 0.0))
        text = (c.get("content") or c.get("text") or "").lower()
        s += 0.4 * len(qterms & set(tok(text)))
        s += 1.2 * sum(1 for n in qnums if n in text)
        if any(w in (c.get("heading") or "").lower() for w in qterms):
            s += 0.6
        if qnums and c.get("type") == "table":
            s += 0.4
        return s

    ranked = [dict(c) for c in cands]
    for c in ranked:
        c["score"] = score(c)
    return sorted(ranked, key=lambda c: c["score"], reverse=True)
