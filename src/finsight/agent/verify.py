"""verify.py — numeric faithfulness check (ported).

Every figure the answer asserts must be traceable to a retrieved chunk or the verified
computation. Numbers matching neither are hallucination candidates — surfaced as a
transparent caveat, never silently trusted.
"""

from __future__ import annotations

import math
import re

from .citations import contains_number, is_year

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PAGE_CITE = re.compile(r"\[page\s+\d+\]", re.I)


def _norm(s: str) -> str:
    return (s or "").replace(",", "")


def verify_numbers(answer: str, retrieved: list[dict], computation: dict | None = None) -> list[str]:
    """Figures asserted in `answer` unsupported by context or computation. Page-citation
    markers (``[page N]``) are structural, not claims — ignored."""
    ans = _PAGE_CITE.sub("", answer or "")
    ctx = " ".join(_norm(c.get("content") or c.get("text") or "") for c in (retrieved or []))
    comp = computation.get("result") if computation else None

    seen: set[str] = set()
    unsupported: list[str] = []
    for tok in (_norm(m.group()) for m in _NUM.finditer(ans)):
        if tok in seen:
            continue
        seen.add(tok)
        if is_year(tok):
            continue                        # a period label, not a figure to verify
        if contains_number(ctx, tok):       # standalone match: "22" is not in "2022"
            continue
        try:
            val = float(tok)
        except ValueError:
            continue
        if comp is not None and math.isclose(val, comp, rel_tol=0.01, abs_tol=0.01):
            continue
        unsupported.append(tok)
    return unsupported
