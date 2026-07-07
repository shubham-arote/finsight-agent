"""run_agent_eval.py — tier-1/2 eval of the FULL agent (not just retrieval).

Runs the labelled sample set end-to-end through the agent and scores:

  deterministic (always, offline):
    abstain_accuracy   out-of-scope questions refused, not confabulated
    citation_hit       an in-scope answer's cited sources include an expected page
    verified_rate      answers with zero unverified figures (the cite_check rail)
    claim_coverage     answers that carry >=1 structured claim with a citation

  LLM-as-judge (when a `judge` key/project is configured — Gemini locally,
  vertex_ai/gemini on GCP for large runs):
    correctness        answer conveys the reference fact (independent model family)

Writes a dated report to evals/reports/ (committed — measured, not claimed).

Run:  uv run python -m evals.run_agent_eval
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from qdrant_client import QdrantClient

from finsight.agent import AgentEngine
from finsight.ingestion import ArtifactStore, ingest
from finsight.llm import LLMRouter, LLMUnavailable, prompts
from finsight.retrieval import HybridRetriever, QdrantIndex

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(__file__).parent / "datasets" / "sample_report_qa.json"
REPORTS = Path(__file__).parent / "reports"
ABSTAIN_MARK = "I couldn't find"

sys.path.insert(0, str(ROOT / "samples"))


def build_engine(router: LLMRouter | None = None) -> AgentEngine:
    from make_sample_pdf import build
    router = router or LLMRouter()
    res = ingest(build(), doc_id="sample", router=router,
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="agent_eval")
    idx.index_chunks(res.chunks)
    return AgentEngine(HybridRetriever(idx, doc_id="sample"), router=router)


def judge_correct(router: LLMRouter, question: str, reference: str, answer: str) -> bool | None:
    """Independent-judge verdict; None when no judge is configured / reachable."""
    if not router.available("judge"):
        return None
    p = prompts.get("judge_correctness")
    try:
        verdict = router.complete("judge", p.render(
            question=question, reference=reference, answer=answer), max_tokens=4)
    except LLMUnavailable:
        return None
    return "incorrect" not in verdict.lower() and "correct" in verdict.lower()


def evaluate(engine: AgentEngine, judge_router: LLMRouter | None = None) -> dict:
    qa = json.loads(DATASET.read_text(encoding="utf-8"))["questions"]
    judge_router = judge_router or engine.deps.router
    rows = []
    for item in qa:
        out = engine.run(item["q"])
        answer = out.get("answer", "")
        abstained = answer.startswith(ABSTAIN_MARK)
        cited_pages = {s["page"] for s in out.get("sources", [])}
        row = {"q": item["q"], "oos": not item["pages"], "abstained": abstained,
               "cited_pages": sorted(cited_pages), "expected": item["pages"],
               "hit": bool(cited_pages & set(item["pages"])) if item["pages"] else None,
               "verified": not out.get("unverified"),
               "has_claims": bool(out.get("claims")),
               "task": out.get("task"), "judged": None}
        if item["pages"] and not abstained and item.get("a"):
            row["judged"] = judge_correct(judge_router, item["q"], item["a"], answer)
        rows.append(row)

    oos = [r for r in rows if r["oos"]]
    ins = [r for r in rows if not r["oos"]]
    answered = [r for r in ins if not r["abstained"]]
    judged = [r for r in rows if r["judged"] is not None]
    return {
        "n": len(rows), "n_in": len(ins), "n_oos": len(oos),
        "abstain_accuracy": sum(r["abstained"] for r in oos) / max(len(oos), 1),
        "answer_rate": len(answered) / max(len(ins), 1),
        "citation_hit": sum(bool(r["hit"]) for r in answered) / max(len(answered), 1),
        "verified_rate": sum(r["verified"] for r in answered) / max(len(answered), 1),
        "claim_coverage": sum(r["has_claims"] for r in answered) / max(len(answered), 1),
        "judged_n": len(judged),
        "correctness": (sum(r["judged"] for r in judged) / len(judged)) if judged else None,
        "rows": rows,
    }


def write_report(m: dict, mode: str, judge_label: str) -> Path:
    REPORTS.mkdir(exist_ok=True)
    pct = lambda v: f"{v:.0%}" if v is not None else "—"
    lines = [
        "# Agent evaluation (full loop)",
        "",
        f"Date: {date.today().isoformat()} · Agent mode: **{mode}** · "
        f"Judge: **{judge_label}** · Dataset: sample_report_qa.json "
        f"({m['n_in']} in-scope + {m['n_oos']} out-of-scope)",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Abstain accuracy (OOS refused) | **{pct(m['abstain_accuracy'])}** |",
        f"| Answer rate (in-scope answered) | **{pct(m['answer_rate'])}** |",
        f"| Citation hit (cited source on an expected page) | **{pct(m['citation_hit'])}** |",
        f"| Verified rate (zero unverified figures) | **{pct(m['verified_rate'])}** |",
        f"| Claim coverage (structured citations present) | **{pct(m['claim_coverage'])}** |",
        f"| Correctness (LLM-judge, n={m['judged_n']}) | **{pct(m['correctness'])}** |",
        "",
        "| Scope | Question | Task | Abstained | Cited p. | Hit | Verified | Judge |",
        "|---|---|---|---|---|---|---|---|",
    ]
    mark = lambda v: "—" if v is None else ("✓" if v else "✗")
    for r in m["rows"]:
        lines.append(f"| {'oos' if r['oos'] else 'in'} | {r['q'][:60]} | {r['task']} | "
                     f"{mark(r['abstained'])} | {r['cited_pages']} | {mark(r['hit'])} | "
                     f"{mark(r['verified'])} | {mark(r['judged'])} |")
    lines += ["", "_Synthetic 6-page document: a regression smoke set, not a generalization "
              "claim. The judge is an independent model family (Gemini/Vertex) from the "
              "answering models. External benchmark (FinRAGBench-V) is the Phase 5 target._"]
    out = REPORTS / f"{date.today().isoformat()}-agent-eval.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    engine = build_engine()
    m = evaluate(engine)
    path = write_report(m, engine.mode, engine.deps.router.label("judge"))
    print(f"abstain={m['abstain_accuracy']:.0%} citation_hit={m['citation_hit']:.0%} "
          f"verified={m['verified_rate']:.0%} claims={m['claim_coverage']:.0%} "
          f"correctness={m['correctness'] if m['correctness'] is not None else 'n/a'} "
          f"-> {path}")
