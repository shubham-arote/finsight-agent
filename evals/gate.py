"""gate.py — the regression gate: run the eval suites, compare to committed floors.

`baseline.json` holds a **score floor** per metric. This runs the suites, prints a diff
table against those floors, and exits non-zero if anything dropped — so a change that
degrades retrieval, routing, citations, or verification is caught before it merges,
rather than discovered in a demo.

After a genuine improvement, **raise the floors** (`--ratchet` writes the observed
scores back) so the suite gets stricter over time and the agent can't silently regress
to yesterday's quality.

    uv run python -m evals.gate                # free tiers: retrieval + trajectory
    uv run python -m evals.gate --judge        # adds LLM-judge correctness (costs calls)
    uv run python -m evals.gate --ratchet      # accept current scores as the new floor

Tiers, following the reference course's split:
  retrieval  — hit@k / MRR over the labelled set (no LLM at all)
  trajectory — did the agent take the right PATH (lane, calculator, citations, rail,
               retry budget, abstain/answer discipline) — deterministic, free
  judge      — answer correctness by an independent model family (opt-in, costs money)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BASELINE = Path(__file__).parent / "baseline.json"
TOL = 1e-9        # floats: treat equal-within-epsilon as meeting the floor

# Metrics whose value depends on an LLM's judgement vary run to run, so ratcheting them
# to the exact observed score makes the gate cry wolf — and a gate that cries wolf gets
# ignored. Deterministic metrics (retrieval math, structural trajectory checks) keep a
# zero margin and are held exactly.
# 0.10 ≈ two flipped cases in a 20-case set, or one in a 10-case set. Metrics with few
# applicable cases (the calc lane has ~5) swing hard on a single LLM decision.
VARIANCE_MARGIN = 0.10
DETERMINISTIC = (
    "retrieval.",
    "trajectory.retrieved_before_answering",
    "trajectory.citations_attached",
    "trajectory.verification_rail_ran",
    "trajectory.retry_budget_respected",
    "trajectory.lane_routed_correctly",      # regex/heuristic router, no LLM
)


def _margin(metric: str) -> float:
    return 0.0 if metric.startswith(DETERMINISTIC) else VARIANCE_MARGIN


def _load_baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def run_suites(with_judge: bool) -> dict[str, float]:
    """Execute the suites and return {metric: score}. Imports are local so the module
    stays importable (and lintable) without building an agent."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "samples"))

    from evals.retrieval_baseline import build_retriever
    from evals.retrieval_baseline import evaluate as eval_retrieval
    from evals.run_agent_eval import build_engine
    from evals.run_agent_eval import evaluate as eval_agent

    scores: dict[str, float] = {}

    print("── retrieval tier")
    retriever, _mode = build_retriever()
    r = eval_retrieval(retriever, k=5)
    scores["retrieval.hit_at_5"] = r["hit_rate"]
    scores["retrieval.mrr"] = r["mrr"]
    print(f"   hit@5={r['hit_rate']:.0%} mrr={r['mrr']:.3f}")

    print("── agent + trajectory tier" + (" (with judge)" if with_judge else ""))
    engine = build_engine()
    m = eval_agent(engine, judge=with_judge)
    for key in ("answer_rate", "citation_hit", "verified_rate", "claim_coverage",
                "abstain_accuracy"):
        if m.get(key) is not None:
            scores[f"agent.{key}"] = m[key]
    for name, val in (m.get("trajectory") or {}).items():
        scores[f"trajectory.{name}"] = val
    if with_judge and m.get("correctness") is not None:
        scores["judge.correctness"] = m["correctness"]
    return scores


def compare(scores: dict[str, float], floors: dict[str, float]) -> tuple[bool, list[tuple]]:
    """Return (regressed, rows) where each row is (metric, floor, got, verdict)."""
    rows, regressed = [], False
    for metric, floor in sorted(floors.items()):
        got = scores.get(metric)
        if got is None:
            rows.append((metric, floor, None, "SKIPPED"))
            continue
        if got + TOL < floor:
            rows.append((metric, floor, got, "REGRESSED"))
            regressed = True
        else:
            rows.append((metric, floor, got, "ok" if got <= floor + TOL else "IMPROVED"))
    for metric in sorted(set(scores) - set(floors)):
        rows.append((metric, None, scores[metric], "new"))
    return regressed, rows


def print_table(rows: list[tuple]) -> None:
    print(f"\n{'metric':38} {'floor':>8} {'got':>8}  verdict")
    print("-" * 70)
    for metric, floor, got, verdict in rows:
        f = "—" if floor is None else f"{floor:.3f}"
        g = "—" if got is None else f"{got:.3f}"
        print(f"{metric:38} {f:>8} {g:>8}  {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", action="store_true", help="also score LLM-judge correctness")
    ap.add_argument("--ratchet", action="store_true",
                    help="write observed scores back as the new floors")
    args = ap.parse_args()

    baseline = _load_baseline()
    floors = dict(baseline.get("floors", {}))
    scores = run_suites(with_judge=args.judge)
    regressed, rows = compare(scores, floors)
    print_table(rows)

    if args.ratchet:
        # keep the higher of (old floor, observed) so ratcheting never loosens the gate
        merged = dict(floors)
        for metric, val in scores.items():
            observed_floor = max(0.0, val - _margin(metric))
            # round DOWN: round() can land above the observed score (0.95238 -> 0.9524),
            # which makes a floor the metric can never meet and fails the very next run
            merged[metric] = math.floor(
                max(observed_floor, floors.get(metric, 0.0)) * 10000) / 10000
        baseline["floors"] = dict(sorted(merged.items()))
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"\nratcheted {len(merged)} floors -> {BASELINE.name}")
        return 0

    if regressed:
        print("\nGATE FAILED — a metric dropped below its committed floor.")
        return 1
    print("\nGATE PASSED — every metric at or above its floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
