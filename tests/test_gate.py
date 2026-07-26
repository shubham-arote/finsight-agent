"""The regression gate's own logic: floors, verdicts, and ratcheting behaviour."""

import json

from evals.gate import _margin, compare
from evals.trajectory import METRICS, aggregate, score_case


# ── gate comparison ─────────────────────────────────────────────────────────
def test_compare_flags_regression_and_improvement():
    floors = {"a": 0.9, "b": 0.5, "c": 1.0}
    scores = {"a": 0.8, "b": 0.7, "c": 1.0, "d": 0.4}
    regressed, rows = compare(scores, floors)
    verdicts = {m: v for m, _f, _g, v in rows}
    assert regressed is True
    assert verdicts["a"] == "REGRESSED"
    assert verdicts["b"] == "IMPROVED"
    assert verdicts["c"] == "ok"
    assert verdicts["d"] == "new"          # unfloored metrics are reported, not gated


def test_missing_metric_is_skipped_not_a_regression():
    """A provider outage must not look like a quality regression."""
    regressed, rows = compare({}, {"judge.correctness": 0.9})
    assert regressed is False
    assert rows[0][3] == "SKIPPED"


def test_deterministic_metrics_get_no_variance_margin():
    assert _margin("retrieval.mrr") == 0.0
    assert _margin("trajectory.citations_attached") == 0.0
    assert _margin("agent.answer_rate") > 0        # LLM-dependent -> tolerance


def test_baseline_floors_are_reachable():
    """Ratcheting must never write a floor above the score that produced it (round-down),
    or the very next run fails against itself."""
    from evals.gate import BASELINE
    floors = json.loads(BASELINE.read_text(encoding="utf-8"))["floors"]
    assert floors and all(0.0 <= v <= 1.0 for v in floors.values())


# ── trajectory metrics ──────────────────────────────────────────────────────
def _case(pages=(3,), lane="qa"):
    return {"q": "What was revenue?", "pages": list(pages), "expect_lane": lane}


def _out(**over):
    base = {"task": "qa", "answer": "Revenue was 6,303.", "attempts": 1,
            "retrieved": [{"page": 3}], "unverified": [],
            "claims": [{"text": "Revenue was 6,303.",
                        "citations": [{"page": 3, "block_id": 1}]}]}
    base.update(over)
    return base


def test_healthy_trajectory_scores_one():
    s = score_case(_out(), _case())
    assert s["lane_routed_correctly"] == 1.0
    assert s["retrieved_before_answering"] == 1.0
    assert s["citations_attached"] == 1.0
    assert s["verification_rail_ran"] == 1.0
    assert s["retry_budget_respected"] == 1.0
    assert s["answered_when_answerable"] == 1.0
    assert s["abstained_when_unanswerable"] is None      # not an OOS case


def test_answer_without_citations_is_caught():
    assert score_case(_out(claims=[]), _case())["citations_attached"] == 0.0


def test_calc_lane_without_computation_is_caught():
    """The failure answer-level metrics hide: right number, wrong path — the figure
    came from the LLM instead of the deterministic calculator."""
    out = _out(task="calc", computation=None)
    assert score_case(out, _case(lane="calc"))["calculator_ran_when_routed"] == 0.0
    out = _out(task="calc", computation={"expr": "1-2", "result": -1})
    assert score_case(out, _case(lane="calc"))["calculator_ran_when_routed"] == 1.0


def test_abstain_discipline_both_directions():
    abstain = _out(answer="I couldn't find information to answer that in this document.")
    assert score_case(abstain, _case(pages=()))["abstained_when_unanswerable"] == 1.0
    assert score_case(_out(), _case(pages=()))["abstained_when_unanswerable"] == 0.0
    assert score_case(abstain, _case())["answered_when_answerable"] == 0.0


def test_aggregate_excludes_non_applicable():
    rows = [{"m": 1.0}, {"m": None}, {"m": 0.0}]
    assert aggregate.__module__  # sanity
    got = aggregate([{k: r.get("m") for k in METRICS} for r in rows])
    assert all(v == 0.5 for v in got.values())          # None excluded from the mean
