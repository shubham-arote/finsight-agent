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


def contains_number(haystack_norm: str, token: str) -> bool:
    """Does `token` appear in the text as a STANDALONE number?

    Plain substring matching silently lies about provenance: "22" is inside "2022",
    "40.8" is inside "140.8". That let a claim look supported by a block that merely
    contained a year, and let citations snap to the wrong block. Both sides of the
    trust layer (snapping and verification) go through this.
    """
    return re.search(rf"(?<![\d.]){re.escape(token)}(?![\d.]?\d)", haystack_norm) is not None


def is_year(tok: str) -> bool:
    """Years are labels, not claim figures — they shouldn't need their own citation."""
    return len(tok) == 4 and tok.isdigit() and tok[:2] in ("19", "20")


# ── tagged context for the prompt ───────────────────────────────────────────
def tag_evidence(retrieved: list[dict], max_chars: int = 4500,
                 per_entry: int = 1400) -> str:
    """Small-to-big context: each entry is the MATCHED BLOCK plus its parent section
    for surrounding context, tagged with its citation anchor.

    The matched block is mandatory. `parent_text` is a section *excerpt* (capped at
    index time), so a block that matched can sit past that cutoff — emitting only the
    parent silently dropped the very sentence retrieval found. That caused confident
    'not disclosed' answers for facts that were in the document (seen live: a cash
    position on p3 retrieved at score 0.83 and then graded 'weak' three times).

    Budget is spent per entry so later hits still appear, instead of one long parent
    consuming the whole window.
    """
    seen, parts, used = set(), [], 0
    for c in retrieved:
        sid, doc = c.get("section_id"), c.get("doc_id")
        key = ("s", doc, sid) if sid is not None else ("c", doc, c.get("block_id"), c.get("page"))
        if key in seen:
            continue
        seen.add(key)
        head = c.get("section_heading") or c.get("heading") or ""
        content = (c.get("content") or c.get("text") or "").strip()
        parent = (c.get("parent_text") or "").strip()
        if content and parent and content not in parent:
            body = f"{parent[:per_entry]}\n…\n{content[:per_entry]}"
        else:
            body = (parent or content)[:per_entry]
        entry = f"[page {c['page']} | block {c.get('block_id')}] {head}\n{body}"
        if parts and used + len(entry) > max_chars:
            break
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


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
    return (sum(1 for n in values if contains_number(content, n)),
            sum(1 for n in nums if contains_number(content, n)),
            1 if ev.get("block_id") == cited_bid else 0)



def snap_citations(claims: list[dict], retrieved: list[dict]) -> list[dict]:
    """Re-anchor every citation to the retrieved block that actually CONTAINS the
    claim's figures. The prompt tags evidence per parent section but labels it with the
    one child block-id that matched retrieval — so the model may quote a figure from the
    section while citing a sibling block. The page is right; the block must be made
    right deterministically: snap to the best figure-bearing block on that page, or
    degrade to a page-level citation (block_id None) when no retrieved block carries
    the figure. The click-to-highlight promise depends on this."""
    def norm_nums(text: str) -> list[str]:
        return [_norm(m.group()) for m in _NUM.finditer(text or "")]

    by_page: dict[int, list[dict]] = {}
    for ev in retrieved:
        by_page.setdefault(ev["page"], []).append(ev)

    out = []
    for cl in claims:
        nums = norm_nums(cl["text"])
        values = [n for n in nums if not is_year(n)]       # years anchor nothing
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
    """Set `verified` per claim: every figure in the claim must appear in the evidence
    that claim cites, or match the computation. Returns (claims, all unverified figures).

    "Evidence" is what the model was actually SHOWN for that entry — the block's content
    *and* the parent section printed with it by `tag_evidence`. Checking only the block's
    own content flags correct answers whenever the figure sits elsewhere in the same
    section (seen live: a cited revenue figure reported as unverified). A verifier that
    cries wolf teaches people to ignore it, which costs more than it saves.
    """
    def shown(c: dict) -> str:
        return _norm(" ".join(filter(None, (c.get("content") or c.get("text") or "",
                                            c.get("parent_text") or ""))))

    by_block = {(c["page"], c.get("block_id")): shown(c) for c in retrieved}
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
            if is_year(tok):
                continue                    # a period label, not a figure to verify
            if contains_number(cited, tok):
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
