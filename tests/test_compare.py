"""Period-over-period compare: figure extraction must be citation-verified, deltas
must be computed deterministically, and unverifiable figures must be excluded."""

import pytest

from finsight.agent.compare import (
    compose_markdown,
    compute_delta,
    extract_verified_figure,
    run_compare,
)


def _out(claim_text, cited_content, page=3, block=6, answer="Revenue was $40.8 million."):
    return {"answer": answer,
            "retrieved": [{"page": page, "block_id": block, "content": cited_content}],
            "claims": [{"text": claim_text, "citations": [{"page": page, "block_id": block}]}]}


# ── extraction is only as good as its citation ──────────────────────────────
def test_extracts_figure_backed_by_its_citation():
    f = extract_verified_figure(_out("Total revenues were $40.8 million.",
                                     "Total revenues for the quarter were $40.8 million"))
    assert f["value"] == 40.8 and f["page"] == 3 and f["block_id"] == 6


def test_rejects_figure_absent_from_the_cited_block():
    """A number the cited block doesn't contain must never enter a comparison — the
    delta would silently inherit the error."""
    assert extract_verified_figure(_out("Revenue was $99.9 million.",
                                        "Total revenues were $40.8 million")) is None


def test_years_are_not_figures():
    f = extract_verified_figure(_out("In 2023 revenue was $40.8 million.",
                                     "2023 revenues were $40.8 million"))
    assert f["value"] == 40.8          # skips 2023, takes the real quantity


def test_abstained_answer_yields_nothing():
    out = _out("x", "y", answer="I couldn't find information to answer that in this document.")
    assert extract_verified_figure(out) is None


def test_trailing_comma_is_not_swallowed_into_a_figure():
    """'$31,' must not parse as 31 — a truncated token produced a nonsense comparison
    on a real filing."""
    f = extract_verified_figure(_out("Cash was 133.5 million at period end.",
                                     "cash and equivalents were 133.5 million"))
    assert f["value"] == 133.5 and f["unit"] == "million"


def test_units_are_captured():
    per_share = extract_verified_figure(_out("EPS was $0.01 per diluted share.",
                                             "or $0.01 per diluted share"))
    millions = extract_verified_figure(_out("Net income was $2.8 million.",
                                            "net income of $2.8 million"))
    assert per_share["unit"] == "per_share" and millions["unit"] == "million"


def test_unlike_units_are_not_compared():
    """EPS against net income is a category error, not a delta. Real failure: the agent
    reported -99.6% comparing $0.01/share to $2.8m."""
    class Eps:
        def run(self, q, thread_id=None):
            return _out("EPS was $0.01 per diluted share.", "or $0.01 per diluted share")

    class Millions:
        def run(self, q, thread_id=None):
            return _out("Net income was $2.8 million.", "net income of $2.8 million")

    done = list(run_compare(Eps(), Millions(), "A", "B"))[-1]
    assert done["compared"] == 0
    assert all(r["status"] == "unit_mismatch" for r in done["rows"])
    assert "different units" in done["markdown"]


# ── deltas are calculated, not generated ────────────────────────────────────
def test_delta_is_exact_and_signed():
    d = compute_delta(40.8, 33.4)
    assert d["abs"] == pytest.approx(7.4)          # binary float: 7.399999…
    assert round(d["pct"], 2) == 22.16 and d["direction"] == "up"
    assert compute_delta(985.0, 1052.0)["direction"] == "down"


def test_delta_handles_zero_prior_without_crashing():
    d = compute_delta(10.0, 0.0)
    assert d is None or d["pct"] is None


# ── the streamed run ────────────────────────────────────────────────────────
class StubEngine:
    """Answers every checklist question with one cited, verifiable figure."""

    def __init__(self, value):
        self.value = value

    def run(self, q, thread_id=None):
        return _out(f"The figure was {self.value} million.",
                    f"the reported figure was {self.value} million")


def test_run_compare_streams_rows_and_computes_every_delta():
    events = list(run_compare(StubEngine(40.8), StubEngine(33.4), "Q1 2023", "Q1 2022"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "compare_start" and kinds[-1] == "compare_done"
    done = events[-1]
    assert done["compared"] == done["total"]
    row = next(e for e in events if e["type"] == "compare_row")
    assert row["a"]["value"] == 40.8 and row["b"]["value"] == 33.4
    assert row["delta"]["direction"] == "up"


def test_one_sided_metric_is_partial_not_compared():
    class Silent(StubEngine):
        def run(self, q, thread_id=None):
            return _out("x", "y", answer="I couldn't find information to answer that in this document.")

    done = list(run_compare(StubEngine(40.8), Silent(0), "A", "B"))[-1]
    assert done["compared"] == 0
    assert all(r["status"] == "partial" for r in done["rows"])


def test_markdown_carries_both_sides_citations():
    rows = [{"heading": "Revenue", "status": "compared",
             "a": {"raw": "40.8", "page": 3, "block_id": 6, "value": 40.8},
             "b": {"raw": "33.4", "page": 2, "block_id": 4, "value": 33.4},
             "delta": {"abs": 7.4, "pct": 22.16, "direction": "up"}}]
    md = compose_markdown(rows, "Q1 2023", "Q1 2022")
    assert "[p3·b6]" in md and "[p2·b4]" in md      # cites into BOTH documents
    assert "+22.2%" in md or "+22.16" in md
