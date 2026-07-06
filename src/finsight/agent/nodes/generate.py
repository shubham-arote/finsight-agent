"""generate — the cited answer, in the structured claim→citation contract.

Cloud path: the model returns JSON {answer, claims:[{text, citations:[{page, block_id}]}]},
parsed and validated (invalid JSON → one retry → prose fallback; invented citations are
dropped). Offline path: extractive quotes, each carrying its real citation. Abstains
instead of confabulating when retrieval is empty or graded weak after retries.
"""

from __future__ import annotations

from ...llm import LLMUnavailable, prompts
from .. import guards
from ..citations import parse_structured, tag_evidence, validate_citations
from ..state import Deps, RAGState

ABSTAIN = "I couldn't find information to answer that in this document."


def generate(state: RAGState, deps: Deps) -> dict:
    retrieved = state.get("retrieved", [])
    q = state["original_question"]
    uq = state.get("user_question") or q
    if not retrieved or state.get("grade") == "weak":
        return {"answer": ABSTAIN, "claims": [], "sources": [],
                "history": [{"q": uq, "a": ABSTAIN}]}

    sources = [{"page": c["page"], "heading": c.get("section_heading") or c.get("heading"),
                "type": c.get("type"), "block_id": c.get("block_id"), "bbox": c.get("bbox"),
                "exact": c.get("exact", False),
                "snippet": (c.get("content") or c.get("text") or "").replace("\n", " ")[:200]}
               for c in retrieved[:3]]
    flags = guards.scan_context(retrieved)             # retrieval rail: flag injected text

    if deps.router.available("answer"):
        answer, claims = _cloud(state, deps, q, retrieved)
    else:
        answer, claims = _extractive(retrieved)

    return {"answer": answer, "claims": claims, "sources": sources,
            "injection_flags": flags, "history": [{"q": uq, "a": answer}]}


def _cloud(state: RAGState, deps: Deps, q: str, retrieved: list[dict]) -> tuple[str, list[dict]]:
    p = prompts.get("generate_answer")                 # latest = the structured contract
    comp = state.get("computation")
    calc_note = (f"\nA verified exact calculation (state this figure, rounded to at most two "
                 f"decimals; do not recompute): {comp['expr']} = {comp['result']:.4g}\n"
                 if comp else "")
    body = p.render(context=tag_evidence(retrieved), calc_note=calc_note, question=q)
    try:
        raw = deps.router.complete("answer", body, system=p.system, max_tokens=900)
        parsed = parse_structured(raw)
        if parsed is None:                             # one strict retry
            raw = deps.router.complete(
                "answer", body + "\n\nReturn ONLY the JSON object — no prose, no fences.",
                system=p.system, max_tokens=900)
            parsed = parse_structured(raw)
    except LLMUnavailable:
        return _extractive(retrieved)
    if parsed is None:                                 # unparseable twice -> prose fallback;
        return (raw or "").strip() or ABSTAIN, []      # cite_check then verifies the numbers
    if parsed["insufficient"]:
        return ABSTAIN, []
    return parsed["answer"], validate_citations(parsed["claims"], retrieved)


def _extractive(retrieved: list[dict]) -> tuple[str, list[dict]]:
    """Keyless: quote the top evidence blocks; each quote cites its own block."""
    seen, parts, claims = set(), [], []
    for c in retrieved:
        sid = c.get("section_id")
        if sid in seen:
            continue
        seen.add(sid)
        quote = (c.get("content") or c.get("text") or "").strip()
        if not quote:
            continue
        parts.append(c.get("parent_text") or quote)
        claims.append({"text": quote[:300],
                       "citations": [{"page": c["page"], "block_id": c.get("block_id")}]})
        if len(parts) >= 2:
            break
    answer = ("Based on the document (extractive — set a cloud key for synthesized "
              "answers):\n\n" + "\n\n".join(f"- {p}" for p in parts))
    return answer, claims
