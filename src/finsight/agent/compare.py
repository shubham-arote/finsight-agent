"""compare.py — period-over-period comparison across two documents.

The most repeated task in an analyst's week: "Q1 this year vs Q1 last year — what
moved, and by how much?" Doing it by hand means reading two filings, re-keying two
sets of figures, and computing deltas in a spreadsheet — three places to make the
mistake that actually costs you.

This runs the same metric checklist against BOTH documents through the full agent loop,
then, for each metric:

  1. extracts the headline figure from each side,
  2. **verifies that figure against the block its own answer cited** — a number that
     can't be traced back is dropped rather than compared, and
  3. computes the delta with the deterministic calculator (`safe_eval`), never the LLM.

So every row is two cited figures plus arithmetic that was calculated, not generated —
and a metric only appears if both sides survived verification.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

from .brief import BRIEF_CHECKLIST
from .calculator import CalcError, safe_eval
from .citations import contains_number, is_year

# Anchored so a trailing comma/period can't be swallowed: "31," must not parse as 31.
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
ABSTAIN_MARK = "I couldn't find"

# The unit a figure is expressed in. Two figures may only be compared when they share
# one: net income "in millions" against EPS "per diluted share" is not a delta, it's a
# category error — and a wrong delta in a briefing note is worse than a missing one.
_UNITS = (
    ("per_share", re.compile(r"per\s+(?:diluted\s+|basic\s+)?share", re.I)),
    ("percent", re.compile(r"%|percent|percentage|basis point|bps", re.I)),
    ("billion", re.compile(r"\bbillion\b|\bbn\b", re.I)),
    ("million", re.compile(r"\bmillion\b|\bm\b(?!\w)", re.I)),
    ("thousand", re.compile(r"\bthousand\b|\bk\b(?!\w)", re.I)),
)


def _unit_of(text: str, end: int) -> str:
    """Classify the unit from the words right after a figure ('$40.8 million' →
    million; '$0.01 per diluted share' → per_share). 'bare' when nothing qualifies."""
    tail = text[end:end + 40]
    for name, rx in _UNITS:
        if rx.search(tail):
            return name
    return "bare"

# The comparison asks for point-in-time values on each side; growth is what WE compute,
# so the checklist drops the brief's "change vs prior year" item (that would be asking
# each document to do the very arithmetic this lane exists to do deterministically).
# Each question must pin exactly ONE quantity, or the two sides can answer with
# different ones (net income in $m vs EPS per share) and the delta becomes a category
# error. The brief's "change vs prior year" item is dropped: computing that change IS
# this lane's job, deterministically.
COMPARE_CHECKLIST = [
    {"heading": "Revenue", "question": "What were total revenues for the period, in millions?"},
    {"heading": "Gross margin", "question": "What was the GAAP gross margin, as a percentage?"},
    {"heading": "Net income",
     "question": "What was GAAP net income or net loss for the period, in millions? "
                 "Give the dollar amount, not the per-share figure."},
    {"heading": "EPS",
     "question": "What was GAAP earnings or loss per diluted share, in dollars per share?"},
    {"heading": "Cash",
     "question": "What was the total of cash, cash equivalents and short-term "
                 "investments at period end, in millions?"},
]
_UNUSED_BRIEF = BRIEF_CHECKLIST      # kept imported for parity with the brief lane


def _norm(tok: str) -> str:
    return tok.replace(",", "")


def extract_verified_figure(out: dict) -> dict | None:
    """The headline figure from an agent answer, proven against its own citation.

    Takes the first non-year number in the first claim that actually appears in a block
    that claim cites. Anything unverifiable returns None — an unverifiable number must
    never enter a comparison, because the delta would inherit the error silently.
    """
    if not out or (out.get("answer") or "").startswith(ABSTAIN_MARK):
        return None
    by_block = {(c["page"], c.get("block_id")): _norm(c.get("content") or c.get("text") or "")
                for c in out.get("retrieved") or []}
    for claim in out.get("claims") or []:
        cites = claim.get("citations") or []
        cited_text = " ".join(by_block.get((ct["page"], ct.get("block_id")), "") for ct in cites)
        for m in _NUM.finditer(claim.get("text") or ""):
            tok = _norm(m.group())
            if is_year(tok) or not contains_number(cited_text, tok):
                continue
            try:
                value = float(tok)
            except ValueError:
                continue
            ct = cites[0]
            return {"value": value, "raw": m.group(),
                    "unit": _unit_of(claim.get("text") or "", m.end()),
                    "page": ct["page"], "block_id": ct.get("block_id"),
                    "text": claim["text"][:160]}
    return None


def compute_delta(current: float, prior: float) -> dict | None:
    """Absolute and percent change — both through the AST calculator, so the arithmetic
    in a comparison is as deterministic as the arithmetic in a single answer."""
    try:
        abs_change = safe_eval(f"{current}-{prior}")
        pct_change = safe_eval(f"({current}-{prior})/{prior}*100") if prior else None
    except (CalcError, ZeroDivisionError):
        return None
    return {"abs": abs_change, "pct": pct_change,
            "direction": "up" if abs_change > 0 else "down" if abs_change < 0 else "flat"}


def run_compare(engine_a, engine_b, label_a: str = "current", label_b: str = "prior",
                thread_id: str | None = None) -> Iterator[dict]:
    """Stream the comparison: plan → per-metric rows → summary."""
    t0 = time.time()
    yield {"type": "compare_start", "label_a": label_a, "label_b": label_b,
           "metrics": [c["heading"] for c in COMPARE_CHECKLIST]}

    rows = []
    for i, item in enumerate(COMPARE_CHECKLIST):
        yield {"type": "compare_step", "i": i, "n": len(COMPARE_CHECKLIST),
               "heading": item["heading"]}
        out_a = engine_a.run(item["question"], thread_id=f"{thread_id or 'cmp'}:a{i}")
        out_b = engine_b.run(item["question"], thread_id=f"{thread_id or 'cmp'}:b{i}")
        fig_a, fig_b = extract_verified_figure(out_a), extract_verified_figure(out_b)
        comparable = bool(fig_a and fig_b and fig_a["unit"] == fig_b["unit"])
        delta = compute_delta(fig_a["value"], fig_b["value"]) if comparable else None
        if fig_a and fig_b and not comparable:
            status = "unit_mismatch"      # e.g. EPS on one side, $m on the other
        elif delta:
            status = "compared"
        elif fig_a or fig_b:
            status = "partial"
        else:
            status = "not_disclosed"
        row = {"heading": item["heading"], "a": fig_a, "b": fig_b, "delta": delta,
               "status": status}
        rows.append(row)
        yield {"type": "compare_row", **row}

    compared = sum(1 for r in rows if r["status"] == "compared")
    yield {"type": "compare_done", "rows": rows, "compared": compared,
           "total": len(rows), "latency_s": round(time.time() - t0, 2),
           "markdown": compose_markdown(rows, label_a, label_b)}


def compose_markdown(rows: list[dict], label_a: str, label_b: str) -> str:
    """A delta table an analyst can paste into a note — every figure carries the page and
    block it came from, in whichever document it came from."""
    def cell(f):
        if not f:
            return "—"
        b = f"·b{f['block_id']}" if f.get("block_id") is not None else ""
        return f"{f['raw']} [p{f['page']}{b}]"

    lines = [f"# {label_a} vs {label_b}", "",
             f"| Metric | {label_a} | {label_b} | Change |", "|---|---|---|---|"]
    for r in rows:
        if r["status"] == "not_disclosed":
            lines.append(f"| {r['heading']} | — | — | _not disclosed_ |")
            continue
        if r["status"] == "unit_mismatch":
            lines.append(f"| {r['heading']} | {cell(r['a'])} | {cell(r['b'])} | "
                         "_not compared — the two filings state this in different units_ |")
            continue
        d = r["delta"]
        change = "—"
        if d:
            pct = f" ({d['pct']:+.1f}%)" if d["pct"] is not None else ""
            change = f"{d['abs']:+,.4g}{pct}"
        lines.append(f"| {r['heading']} | {cell(r['a'])} | {cell(r['b'])} | {change} |")
    n = sum(1 for r in rows if r["status"] == "compared")
    lines += ["", f"_{n} of {len(rows)} metrics compared. Figures are cited to their "
              "source block in each document; every change is computed by the "
              "calculator, not generated, and a figure that could not be verified "
              "against its own citation was excluded rather than compared._"]
    return "\n".join(lines)
