"""brief.py — the autonomous analyst lane.

This is what makes finsight an *agent* rather than a chatbot: from ONE instruction
("analyze this filing"), the agent plans a standard first-pass analyst checklist and
runs each item through the full retrieve→grade→calculate→generate→verify loop on its
own — a dozen autonomous LLM+tool steps — then composes a single **cited, verified
one-page brief**. Every figure keeps its page+block citation and verified/unverified
status; items the document doesn't disclose are honestly marked "not disclosed" (the
abstain path), never fabricated.

The checklist is deterministic by default (a finance-standard first-read of a filing),
so the plan is reliable and free; `plan_brief` can adapt it to the document with the
`fast` role when a key is present.
"""

from __future__ import annotations

import re
import time
from typing import Iterator

# Standard analyst first-read of a filing. Order = how a brief reads top to bottom.
BRIEF_CHECKLIST: list[dict] = [
    {"heading": "Revenue", "question": "What was total revenue for the period?"},
    {"heading": "Revenue growth",
     "question": "By how much did revenue change versus the prior year, in percent?"},
    {"heading": "Profitability",
     "question": "What was operating profit or operating income for the period?"},
    {"heading": "Margin", "question": "What was the operating margin or gross margin?"},
    {"heading": "Bottom line",
     "question": "What was net income or profit after tax, and earnings per share?"},
    {"heading": "Cash", "question": "What was the cash position or cash flow for the period?"},
    {"heading": "Outlook",
     "question": "What guidance or outlook did management provide for the coming period?"},
]

_BRIEF_INTENT = re.compile(
    r"\b(brief|analy[sz]e|summar|overview|first[ -]?read|key (metrics|figures|numbers)|"
    r"break ?down|tl;?dr|walk me through|what are the highlights)\b", re.I)

ABSTAIN_MARK = "I couldn't find"


def is_brief_request(question: str) -> bool:
    """Does the user want a whole-document brief rather than one answer?"""
    q = question or ""
    return bool(_BRIEF_INTENT.search(q)) and len(q.split()) <= 12


def plan_brief(engine, doc_label: str = "") -> list[dict]:
    """The checklist for this document. Deterministic finance-standard set today;
    the hook for LLM adaptation (doc-type aware) lives here behind `fast`."""
    return list(BRIEF_CHECKLIST)


def run_brief(engine, doc_label: str = "", thread_id: str | None = None
              ) -> Iterator[dict]:
    """Plan → run each checklist item through the full agent → compose a cited brief.
    Streams progress events for the UI trace, then a final composed brief."""
    checklist = plan_brief(engine, doc_label)
    t0 = time.time()
    yield {"type": "brief_start", "doc_label": doc_label,
           "sections": [c["heading"] for c in checklist]}

    sections: list[dict] = []
    for i, item in enumerate(checklist):
        yield {"type": "brief_step", "i": i, "n": len(checklist),
               "heading": item["heading"], "question": item["question"]}
        out = engine.run(item["question"],
                         thread_id=f"{thread_id or 'brief'}:{i}")   # isolated per step
        answer = out.get("answer", "")
        abstained = answer.startswith(ABSTAIN_MARK)
        section = {
            "heading": item["heading"], "question": item["question"],
            "status": "not_disclosed" if abstained else "answered",
            "answer": "" if abstained else answer,
            "claims": [] if abstained else out.get("claims", []),
            "computed": bool(out.get("computation")),
            "verified": not out.get("unverified"),
            "task": out.get("task"),
        }
        sections.append(section)
        yield {"type": "brief_section", **section}

    markdown = compose_markdown(sections, doc_label)
    answered = sum(1 for s in sections if s["status"] == "answered")
    yield {"type": "brief_done", "markdown": markdown, "sections": sections,
           "answered": answered, "total": len(sections),
           "latency_s": round(time.time() - t0, 2)}


def compose_markdown(sections: list[dict], doc_label: str) -> str:
    """A one-page brief. Each line keeps its citations as [p{page}·b{block}] markers so
    the source is traceable even in the plain-text export."""
    title = doc_label or "Financial document"
    lines = [f"# Analyst brief — {title}", ""]
    for s in sections:
        lines.append(f"**{s['heading']}.**")
        if s["status"] == "not_disclosed":
            lines.append("_Not disclosed in this document._")
        else:
            cites = _cite_markers(s["claims"])
            flag = "" if s["verified"] else "  ⚠ contains an unverified figure"
            calc = " *(computed)*" if s["computed"] else ""
            lines.append(f"{s['answer']}{calc} {cites}{flag}".rstrip())
        lines.append("")
    verified = all(s["verified"] for s in sections if s["status"] == "answered")
    lines.append("---")
    lines.append(f"_{sum(1 for s in sections if s['status'] == 'answered')} of "
                 f"{len(sections)} items answered from the document; "
                 f"{'all figures verified against their citations' if verified else 'see ⚠ flags'}._")
    return "\n".join(lines)


def _cite_markers(claims: list[dict]) -> str:
    seen, marks = set(), []
    for cl in claims:
        for ct in cl.get("citations", []):
            key = (ct["page"], ct.get("block_id"))
            if key in seen:
                continue
            seen.add(key)
            b = ct.get("block_id")
            marks.append(f"[p{ct['page']}·b{b}]" if b is not None else f"[p{ct['page']}]")
    return " ".join(marks)
