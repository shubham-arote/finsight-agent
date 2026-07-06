"""retrieval_baseline.py — tier-1 deterministic retrieval eval (hit@k + MRR).

Runs the labelled sample-report QA set through the full retrieval stack and reports:
  hit@k : expected page appears among the top-k retrieved pages
  MRR   : reciprocal rank of the first expected-page hit

Keyless it measures sparse + lexical rerank + exact lookup; with COHERE_API_KEY it also
measures the hybrid (dense + RRF + cross-encoder) stack. Writes a dated report to
evals/reports/ — those files are committed: every retrieval change is measured, not
claimed.

Run:  uv run python -m evals.retrieval_baseline
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from qdrant_client import QdrantClient

from finsight.config import settings
from finsight.ingestion import ArtifactStore, ingest
from finsight.llm import LLMRouter
from finsight.retrieval import HybridRetriever, QdrantIndex

ROOT = Path(__file__).resolve().parent.parent
DATASET = Path(__file__).parent / "datasets" / "sample_report_qa.json"
REPORTS = Path(__file__).parent / "reports"

sys.path.insert(0, str(ROOT / "samples"))


def build_retriever() -> tuple[HybridRetriever, dict]:
    from make_sample_pdf import build

    res = ingest(build(), doc_id="sample", router=LLMRouter(),
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="baseline")
    idx.index_chunks(res.chunks)
    mode = {"dense": bool(idx.embedder), "rerank": bool(settings.cohere_api_key)}
    return HybridRetriever(idx, doc_id="sample"), mode


def evaluate(retriever: HybridRetriever, k: int = 5) -> dict:
    qa = json.loads(DATASET.read_text(encoding="utf-8"))["questions"]
    in_scope = [q for q in qa if q["pages"]]
    rows, hits, rr_sum = [], 0, 0.0
    for item in in_scope:
        evs = retriever.retrieve(item["q"], k=k)
        pages = [e["page"] for e in evs]
        rank = next((i + 1 for i, p in enumerate(pages) if p in item["pages"]), None)
        hits += rank is not None
        rr_sum += 1.0 / rank if rank else 0.0
        rows.append({"q": item["q"], "expected": item["pages"], "got": pages, "rank": rank})
    n = len(in_scope)
    return {"k": k, "n": n, "hit_rate": hits / n, "mrr": rr_sum / n, "rows": rows}


def write_report(m: dict, mode: dict) -> Path:
    REPORTS.mkdir(exist_ok=True)
    stack = ("hybrid (sparse+dense RRF, Cohere rerank)" if mode["dense"]
             else "sparse + lexical rerank + exact lookup (keyless)")
    lines = [
        "# Retrieval baseline",
        "",
        f"Date: {date.today().isoformat()} · Stack: **{stack}** · "
        f"Dataset: sample_report_qa.json ({m['n']} in-scope questions) · k={m['k']}",
        "",
        f"| hit@{m['k']} | MRR |",
        "|---|---|",
        f"| **{m['hit_rate']:.0%}** | **{m['mrr']:.3f}** |",
        "",
        "| Question | Expected p. | Got (top-k) | Rank |",
        "|---|---|---|---|",
    ]
    for r in m["rows"]:
        lines.append(f"| {r['q']} | {r['expected']} | {r['got']} | {r['rank'] or '—'} |")
    lines += ["", "_Synthetic 6-page document — a smoke baseline for regressions, not a "
              "generalization claim. The external benchmark (FinRAGBench-V) lands in Phase 5._"]
    out = REPORTS / f"{date.today().isoformat()}-retrieval-baseline.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    retriever, mode = build_retriever()
    metrics = evaluate(retriever)
    path = write_report(metrics, mode)
    print(f"hit@{metrics['k']}={metrics['hit_rate']:.0%}  MRR={metrics['mrr']:.3f}  "
          f"(n={metrics['n']}, dense={mode['dense']}) -> {path}")
