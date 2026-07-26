"""citations.py — the structured claim→citation contract.

The generate node asks the LLM for JSON: an answer plus claims, each claim citing the
evidence blocks it came from (page + block_id, matching the [S# | page P | block B] tags
shown in the prompt). This module builds the tagged context, parses/validates the LLM's
JSON (untrusted!), and deterministically checks every claim's figures against its *cited*
blocks — that per-claim strictness is what FinRAGBench-V's citation precision measures.
"""

from __future__ import annotations

import json
import math
import re

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _norm(s: str) -> str:
    return (s or "").replace(",", "")


# ── tagged context for the prompt ───────────────────────────────────────────
def tag_evidence(retrieved: list[dict], max_chars: int = 4500) -> str:
    """Dedup parent sections small-to-big; tag every entry with its citation anchor."""
    seen, parts = set(), []
    for c in retrieved:
        sid, doc = c.get("section_id"), c.get("doc_id")
        key = ("s", doc, sid) if sid is not None else ("c", doc, c.get("block_id"), c.get("page"))
        if key in seen:
            continue
        seen.add(key)
        head = c.get("section_heading") or c.get("heading") or ""
        body = c.get("parent_text") or c.get("content") or c.get("text") or ""
        parts.append(f"[page {c['page']} | block {c.get('block_id')}] {head}\n{body}")
    return "\n\n".join(parts)[:max_chars]


# ── parsing (LLM output is untrusted — validate everything) ─────────────────
def parse_structured(raw: str) -> dict | None:
    """Parse the generate LLM's JSON reply -> {answer, claims, insufficient} or None."""
    text = _FENCE.sub("", (raw or "").strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        doc = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("answer"), str):
        return None
    claims = []
    for cl in doc.get("claims") or []:
        if not isinstance(cl, dict) or not isinstance(cl.get("text"), str) or not cl["text"].strip():
            continue
        cites = []
        for ct in cl.get("citations") or []:
            if isinstance(ct, dict) and isinstance(ct.get("page"), int):
                bid = ct.get("block_id")
                cites.append({"page": ct["page"],
                              "block_id": bid if isinstance(bid, int) else None})
        claims.append({"text": cl["text"].strip(), "citations": cites})
    return {"answer": doc["answer"].strip(), "claims": claims,
            "insufficient": bool(doc.get("insufficient"))}


def validate_citations(claims: list[dict], retrieved: list[dict]) -> list[dict]:
    """Drop citations that don't point at actually-retrieved evidence (the model may not
    invent sources). block_id None is allowed when the page itself was retrieved."""
    pages = {c["page"] for c in retrieved}
    blocks = {(c["page"], c.get("block_id")) for c in retrieved}
    out = []
    for cl in claims:
        kept = [ct for ct in cl["citations"]
                if (ct["page"], ct["block_id"]) in blocks
                or (ct["block_id"] is None and ct["page"] in pages)]
        out.append({**cl, "citations": kept})
    return out


# ── deterministic citation snapping ─────────────────────────────────────────
def _block_score(ev: dict, values: list[str], nums: list[str], cited_bid) -> tuple:
    """Rank a retrieved block as the true home of a claim's figures:
    (real values matched, any numbers matched, is the block the model named)."""
    content = _norm(ev.get("content") or ev.get("text") or "")
    return (sum(1 for n in values if n in content),
            sum(1 for n in nums if n in content),
            1 if ev.get("block_id") == cited_bid else 0)



def snap_citations(claims: list[dict], retrieved: list[dict]) -> list[dict]:
    """Re-anchor every citation to the retrieved block that actually CONTAINS the
    claim's figures. The prompt tags evidence per parent section but labels it with the
    one child block-id that matched retrieval — so the model may quote a figure from the
    section while citing a sibling block. The page is right; the block must be made
    right deterministically: snap to the best figure-bearing block on that page, or
    degrade to a page-level citation (block_id None) when no retrieved block carries
    the figure. The click-to-highlight promise depends on this."""
    def _is_year(tok: str) -> bool:
        return len(tok) == 4 and tok.isdigit() and tok[:2] in ("19", "20")

    def norm_nums(text: str) -> list[str]:
        return [_norm(m.group()) for m in _NUM.finditer(text or "")]

    by_page: dict[int, list[dict]] = {}
    for ev in retrieved:
        by_page.setdefault(ev["page"], []).append(ev)

    out = []
    for cl in claims:
        nums = norm_nums(cl["text"])
        values = [n for n in nums if not _is_year(n)]      # years anchor nothing
        snapped, seen = [], set()
        for ct in cl["citations"]:
            page, bid = ct["page"], ct["block_id"]
            cands = by_page.get(page, [])
            if not nums or not cands:
                best_bid = bid
            else:
                def score(ev, _values=values, _nums=nums, _bid=bid):
                    return _block_score(ev, _values, _nums, _bid)
                best = max(cands, key=score)
                v_hits, a_hits, _ = score(best)
                # a real value match wins; year-only matches keep the model's block;
                # nothing at all -> page-level citation (never point at the wrong block)
                best_bid = (best.get("block_id") if v_hits > 0
                            else bid if (a_hits > 0 and not values) else None)
            key = (page, best_bid)
            if key not in seen:
                seen.add(key)
                snapped.append({"page": page, "block_id": best_bid})
        out.append({**cl, "citations": snapped})
    return out


# ── deterministic per-claim verification ────────────────────────────────────
def check_claims(claims: list[dict], retrieved: list[dict],
                 computation: dict | None = None) -> tuple[list[dict], list[str]]:
    """Set `verified` per claim: every figure in the claim must appear in the *cited*
    blocks' content (not just anywhere in the context) or match the computation.
    Returns (claims, all unverified figures)."""
    by_block = {(c["page"], c.get("block_id")): _norm(c.get("content") or c.get("text") or "")
                for c in retrieved}
    by_page: dict[int, list[str]] = {}
    for (page, _), text in by_block.items():
        by_page.setdefault(page, []).append(text)
    comp = computation.get("result") if computation else None

    unverified_all: list[str] = []
    out = []
    for cl in claims:
        cited_texts = []
        for ct in cl["citations"]:
            if ct["block_id"] is not None:
                cited_texts.append(by_block.get((ct["page"], ct["block_id"]), ""))
            else:
                cited_texts.extend(by_page.get(ct["page"], []))
        cited = " ".join(cited_texts)
        bad = []
        for m in _NUM.finditer(cl["text"]):
            tok = _norm(m.group())
            if tok in cited:
                continue
            try:
                val = float(tok)
            except ValueError:
                continue
            if comp is not None and math.isclose(val, comp, rel_tol=0.01, abs_tol=0.01):
                continue
            bad.append(tok)
        verified = bool(cl["citations"]) and not bad
        out.append({**cl, "verified": verified})
        unverified_all.extend(b for b in bad if b not in unverified_all)
    return out, unverified_all
