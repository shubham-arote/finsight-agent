"""hybrid.py — HybridRetriever: query-typed RRF fusion + rerank + deterministic lookup.

The full retrieval stack over a QdrantIndex, behind the `Retriever` protocol:

    sparse candidates (BM25-style, always)      ┐
    dense candidates (Cohere, when keyed)       ├─ weighted RRF by query type
                                                ┘        │
                              cross-encoder rerank (Cohere / lexical fallback)
                                                         │
              deterministic exact-value/period lookup floated on top (`exact: True`)
                                                         │
                                     Evidence (page + block + bbox + parent context)

Ported from financial_analyst_agent's hybrid.py, restated over the stateless Qdrant
index. Key-optional throughout: keyless = sparse + lexical rerank + exact lookup.
"""

from __future__ import annotations

import re

from .base import Evidence
from .qdrant_index import QdrantIndex
from .rerank import rerank

# query-type -> (dense_weight, sparse_weight)
_TYPE_WEIGHTS = {
    "lookup":     (0.4, 0.6),   # exact number / line-item      -> favour keyword
    "procedure":  (0.5, 0.5),
    "diagnostic": (0.7, 0.3),   # why / cause / impact          -> favour semantic
    "comparison": (0.6, 0.4),
    "general":    (0.5, 0.5),
}
_RRF_K = 60

_LOOKUP = re.compile(r"\b(capacity|spec|specification|value|amount|rate|ratio|margin|"
                     r"revenue|profit|eps|figure|total|how much|how many)\b", re.I)
_PROC = re.compile(r"\b(how to|how do|steps?|procedure|process|calculate|compute)\b", re.I)
_DIAG = re.compile(r"\b(why|cause|reason|fault|error|fail|decline|drop|fell|impact|risk)\b", re.I)
_COMP = re.compile(r"\b(vs|versus|compare|comparison|difference|between|year-?over-?year|"
                   r"yoy|prior year)\b", re.I)

_VALUE = re.compile(r"\d[\d.,]{2,}")                            # 6,303 · 12.1 · 23.8
_PERIOD = re.compile(r"\b(?:FY\d{2,4}|Q[1-4]|H[12])\b", re.I)   # fiscal periods


def classify(query: str) -> str:
    """Zero-LLM query routing -> RRF weights (and, later, the supervisor's lanes)."""
    if _COMP.search(query):
        return "comparison"
    if _DIAG.search(query):
        return "diagnostic"
    if _PROC.search(query):
        return "procedure"
    if _LOOKUP.search(query):
        return "lookup"
    return "general"


def lookup_terms(query: str) -> list[str]:
    """Exact tokens a deterministic lookup must match: explicit values + fiscal periods.
    Empty -> no exact anchor, lookup is skipped."""
    terms = set(_VALUE.findall(query))
    terms |= {m.group(0).upper() for m in _PERIOD.finditer(query)}
    return sorted(terms, key=len, reverse=True)


class HybridRetriever:
    """Satisfies the `Retriever` protocol; optionally scoped to one doc_id."""

    def __init__(self, index: QdrantIndex, doc_id: str | None = None, candidates: int = 30):
        self.index = index
        self.doc_id = doc_id
        self.candidates = candidates

    def retrieve(self, query: str, k: int = 6) -> list[Evidence]:
        n = max(self.candidates, k)
        sparse = self.index.sparse_candidates(query, n=n, doc_id=self.doc_id)
        dense: list[dict] = []
        if self.index.embedder:
            try:
                dense = self.index.dense_candidates(query, n=n, doc_id=self.doc_id)
            except Exception:
                dense = []                          # embed failure -> sparse-only

        cands = self._fuse(query, sparse, dense) if dense else sparse
        ranked = rerank(query, cands[:n], k)

        exact = self._exact_hits(lookup_terms(query))
        seen = {c["chunk_id"] for c in exact}
        merged = exact + [c for c in ranked if c["chunk_id"] not in seen]

        out: list[Evidence] = []
        for c in merged[:max(k, len(exact))]:
            ev: Evidence = {**c}                    # payload already carries citation fields
            ev["exact"] = c.get("exact", False)
            ev.setdefault("score", 0.0)
            out.append(ev)
        return out

    def _fuse(self, query: str, sparse: list[dict], dense: list[dict]) -> list[dict]:
        """Client-side weighted Reciprocal Rank Fusion (weights by query type)."""
        w_dense, w_sparse = _TYPE_WEIGHTS[classify(query)]
        fused: dict[str, dict] = {}
        scores: dict[str, float] = {}
        for ranking, w in ((sparse, w_sparse), (dense, w_dense)):
            for r, c in enumerate(ranking):
                cid = c["chunk_id"]
                fused.setdefault(cid, c)
                scores[cid] = scores.get(cid, 0.0) + w / (_RRF_K + r)
        out = [dict(fused[cid], score=s) for cid, s in scores.items()]
        return sorted(out, key=lambda c: c["score"], reverse=True)

    def _exact_hits(self, terms: list[str], limit: int = 3) -> list[dict]:
        """Deterministic exact-match: pull candidates whose content contains the query's
        values/periods as exact substrings; prefer more terms, then tables. Floated above
        fuzzy ranking (high precision — the finance-critical exact path).

        Precision guard: a hit qualifies only with a *value* match (6,303) or >=2 exact
        terms. A lone period label ("FY26") matches half the document — floating on that
        would override the reranker with noise."""
        if not terms:
            return []
        values = [t for t in terms if not _PERIOD.fullmatch(t)]
        pool = self.index.sparse_candidates(" ".join(terms), n=12, doc_id=self.doc_id)
        scored = []
        for i, c in enumerate(pool):
            text = (c.get("content") or c.get("text") or "").lower()
            hits = sum(1 for t in terms if t.lower() in text)
            v_hits = sum(1 for t in values if t.lower() in text)
            if v_hits >= 1 or hits >= 2:
                scored.append((hits, c.get("type") == "table", -i, c))
        scored.sort(key=lambda t: t[:3], reverse=True)
        out = []
        for hits, _, _, c in scored[:limit]:
            out.append(dict(c, exact=True, score=float(hits)))
        return out


def make_retriever(index: QdrantIndex, doc_id: str | None = None) -> HybridRetriever:
    """The retrieval stack for a document (or the whole collection when doc_id is None)."""
    return HybridRetriever(index, doc_id=doc_id)
