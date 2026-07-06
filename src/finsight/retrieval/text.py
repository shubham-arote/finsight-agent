"""text.py — shared lexical primitives (tokenizer + stopwords), ported from the old repo.

Used by the sparse (BM25-style) indexer and, later, offline agent fallbacks.
"""

from __future__ import annotations

import re
import zlib

_STOP = {"the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "was", "were",
         "are", "what", "which", "how", "much", "many", "did", "do", "does", "tell",
         "me", "about", "could", "you", "please", "on", "at", "by", "with", "this",
         "that", "it", "its", "their", "from", "be", "as", "we", "i"}


def tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9$%.]+", s.lower())


def term_id(term: str) -> int:
    """Stable 31-bit id for a term — sparse-vector index dimension."""
    return zlib.crc32(term.encode()) & 0x7FFFFFFF


def sparse_counts(text: str) -> tuple[list[int], list[float]]:
    """Term-frequency sparse vector (indices, values). IDF weighting is applied
    server-side by Qdrant (Modifier.IDF), so raw counts are correct here."""
    counts: dict[int, float] = {}
    for t in tok(text):
        counts[term_id(t)] = counts.get(term_id(t), 0.0) + 1.0
    return list(counts.keys()), list(counts.values())
