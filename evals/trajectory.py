"""trajectory.py — deterministic metrics over the agent's PATH, not just its answer.

A right answer reached the wrong way is a latent failure: the agent that stumbles onto
a figure without routing to the calculator, or that answers without citing, will break
on the next document. These metrics score the trajectory the way the reference course's
ADK `trajectory_metrics.py` scores tool calls and delegations — adapted to a LangGraph
state machine, where the equivalent signals are lane routing, node visits, the retry
budget, and whether the verification rail actually ran.

Every metric here is free and deterministic (no LLM), so the whole set can gate CI.
Each takes the agent's final state (+ the labelled case) and returns 1.0 / 0.0 / None,
where None means "not applicable to this case" and is excluded from the average.
"""

from __future__ import annotations

from finsight.agent.calculator import is_math_query
from finsight.agent.state import MAX_ATTEMPTS

ABSTAIN_MARK = "I couldn't find"


def _abstained(out: dict) -> bool:
    return (out.get("answer") or "").startswith(ABSTAIN_MARK)


# ── routing ─────────────────────────────────────────────────────────────────
def lane_routed_correctly(out: dict, case: dict) -> float | None:
    """The supervisor put a derivational question in the calc lane and a stated-figure
    lookup in the qa lane. `expect_lane` in the dataset is the label; when it's absent
    we fall back to the same intent heuristic the supervisor uses, which still catches
    a router that has stopped routing at all (everything one lane)."""
    expected = case.get("expect_lane") or ("calc" if is_math_query(case["q"]) else "qa")
    got = out.get("task")
    if got is None:
        return 0.0
    return 1.0 if got == expected else 0.0


def calculator_ran_when_routed(out: dict, case: dict) -> float | None:
    """A calc-lane answer must carry an actual computation — otherwise the lane routed
    but the deterministic arithmetic never happened and the figure came from the LLM."""
    if out.get("task") != "calc" or _abstained(out):
        return None
    return 1.0 if out.get("computation") else 0.0


# ── evidence handling ───────────────────────────────────────────────────────
def retrieved_before_answering(out: dict, case: dict) -> float | None:
    """No answer without evidence in state — the agent must never answer from memory."""
    if _abstained(out):
        return None
    return 1.0 if out.get("retrieved") else 0.0


def citations_attached(out: dict, case: dict) -> float | None:
    """Every answered question carries at least one claim with a citation."""
    if _abstained(out):
        return None
    claims = out.get("claims") or []
    return 1.0 if any(c.get("citations") for c in claims) else 0.0


def verification_rail_ran(out: dict, case: dict) -> float | None:
    """cite_check must have produced a verdict — `unverified` present (even empty) means
    the rail executed. A missing key means the answer skipped verification entirely."""
    if _abstained(out):
        return None
    return 1.0 if "unverified" in out else 0.0


# ── control flow ────────────────────────────────────────────────────────────
def retry_budget_respected(out: dict, case: dict) -> float | None:
    """The rewrite loop is bounded: a graph that exceeds MAX_ATTEMPTS is a runaway."""
    attempts = out.get("attempts")
    if attempts is None:
        return None
    return 1.0 if attempts <= MAX_ATTEMPTS else 0.0


def abstained_when_unanswerable(out: dict, case: dict) -> float | None:
    """Out-of-scope questions (no gold pages) must be refused, not answered."""
    if case.get("pages"):
        return None
    return 1.0 if _abstained(out) else 0.0


def answered_when_answerable(out: dict, case: dict) -> float | None:
    """In-scope questions must NOT be refused — over-abstention is a failure too."""
    if not case.get("pages"):
        return None
    return 0.0 if _abstained(out) else 1.0


METRICS = {
    "lane_routed_correctly": lane_routed_correctly,
    "calculator_ran_when_routed": calculator_ran_when_routed,
    "retrieved_before_answering": retrieved_before_answering,
    "citations_attached": citations_attached,
    "verification_rail_ran": verification_rail_ran,
    "retry_budget_respected": retry_budget_respected,
    "abstained_when_unanswerable": abstained_when_unanswerable,
    "answered_when_answerable": answered_when_answerable,
}


def score_case(out: dict, case: dict) -> dict[str, float | None]:
    return {name: fn(out, case) for name, fn in METRICS.items()}


def aggregate(rows: list[dict[str, float | None]]) -> dict[str, float]:
    """Mean per metric over the applicable cases (None excluded)."""
    scores: dict[str, float] = {}
    for name in METRICS:
        vals = [r[name] for r in rows if r.get(name) is not None]
        if vals:
            scores[name] = sum(vals) / len(vals)
    return scores
