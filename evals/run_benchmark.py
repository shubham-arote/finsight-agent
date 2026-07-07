"""run_benchmark.py — FinRAGBench-V (English, sampled) through the FULL agent.

Protocol (disclosed in every report):
  * seeded sample of N questions + ONLY their source PDFs (full corpus = 51k pages,
    beyond free-tier embedding/OCR budgets)
  * documents ingested by our own pipeline (text-layer/OCR routed, enrichment capped);
    retrieval runs over the WHOLE sampled corpus (not doc-scoped) — the harder setting
  * metrics:
      retrieval  : recall@k / MRR against gold `from_pages`
      generation : LLM-judge correctness vs the reference (judge role — Gemini/Vertex,
                   independent family), skipped keyless
      citation   : page-level precision/recall of the answer's claim citations vs gold
      verified   : answers whose figures all trace to cited blocks (our cite_check rail)

Run:  uv run python -m evals.run_benchmark --data-dir data/finragbench_v --sample 25
      (ingestion is cached in the ArtifactStore — reruns re-parse nothing)
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

from finsight.agent import AgentEngine
from finsight.ingestion import ArtifactStore, ingest
from finsight.llm import LLMRouter
from finsight.retrieval import HybridRetriever, QdrantIndex

from .benchmarks.finragbench_v import BenchCase, load_cases, pdf_bytes
from .run_agent_eval import judge_correct

REPORTS = Path(__file__).parent / "reports"
ABSTAIN_MARK = "I couldn't find"


def build_corpus(data_dir: str, cases: list[BenchCase], router: LLMRouter,
                 store: ArtifactStore, index: QdrantIndex,
                 contextual: bool, enrich: bool) -> dict:
    stats = {"docs": 0, "pages": 0, "chunks": 0, "enriched": 0, "skipped_pages": 0}
    for doc_name in sorted({c.doc_name for c in cases}):
        res = ingest(pdf_bytes(data_dir, doc_name), doc_id=doc_name, router=router,
                     store=store, contextual=contextual, enrich=enrich)
        index.index_chunks(res.chunks)
        stats["docs"] += 1
        stats["pages"] += res.page_count
        stats["chunks"] += len(res.chunks)
        stats["enriched"] += res.stats.get("enriched", 0)
        stats["skipped_pages"] += len(res.stats.get("skipped_pages", []))
        print(f"  ingested {doc_name}: {res.page_count}p {len(res.chunks)}c "
              f"(parser={res.parser}, cached={res.cached_pages})")
    return stats


def evaluate(cases: list[BenchCase], engine: AgentEngine, router: LLMRouter,
             k: int = 5) -> dict:
    rows = []
    for case in cases:
        t0 = time.time()
        out = engine.run(case.question)
        answer = out.get("answer", "")
        abstained = answer.startswith(ABSTAIN_MARK)
        gold = {(case.doc_name, p) for p in case.gold_pages}

        # retrieval: gold (doc, page) among the top-k retrieved
        got = [(c.get("doc_id"), c.get("page")) for c in out.get("retrieved", [])[:k]]
        rank = next((i + 1 for i, g in enumerate(got) if g in gold), None)
        recall = (len(gold & set(got)) / len(gold)) if gold else None

        # citation page-level P/R from the structured claims
        cited = set()
        for cl in out.get("claims", []):
            for ct in cl.get("citations", []):
                # resolve the claim's (page, block) back to its doc via retrieved evidence
                for ev in out.get("retrieved", []):
                    if ev.get("page") == ct["page"] and (
                            ct["block_id"] is None or ev.get("block_id") == ct["block_id"]):
                        cited.add((ev.get("doc_id"), ct["page"]))
        cite_p = (len(cited & gold) / len(cited)) if cited else None
        cite_r = (len(cited & gold) / len(gold)) if gold and cited else (0.0 if gold else None)

        judged = (judge_correct(router, case.question, case.reference, answer)
                  if not abstained and case.reference else None)
        rows.append({"qid": case.qid, "category": case.category, "abstained": abstained,
                     "rank": rank, "recall": recall, "cite_p": cite_p, "cite_r": cite_r,
                     "verified": not out.get("unverified"), "judged": judged,
                     "latency_s": round(time.time() - t0, 1)})
        print(f"  [{case.category[:24]:24}] rank={rank} recall={recall} "
              f"citeP={cite_p} judged={judged}")

    def avg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    answered = [r for r in rows if not r["abstained"]]
    judged = [r for r in rows if r["judged"] is not None]
    return {"n": len(rows), "k": k,
            "hit_at_k": sum(1 for r in rows if r["rank"]) / max(len(rows), 1),
            "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / max(len(rows), 1),
            "recall_at_k": avg("recall"),
            "citation_precision": avg("cite_p"), "citation_recall": avg("cite_r"),
            "verified_rate": (sum(r["verified"] for r in answered) / len(answered))
                             if answered else None,
            "answer_rate": len(answered) / max(len(rows), 1),
            "judged_n": len(judged),
            "correctness": (sum(r["judged"] for r in judged) / len(judged)) if judged else None,
            "rows": rows}


def write_report(m: dict, corpus: dict, args, mode: str, judge_label: str) -> Path:
    REPORTS.mkdir(exist_ok=True)
    pct = lambda v: f"{v:.0%}" if v is not None else "—"
    num = lambda v: f"{v:.3f}" if v is not None else "—"
    lines = [
        "# FinRAGBench-V (English, sampled subset) — full-agent run",
        "",
        f"Date: {date.today().isoformat()} · Agent: **{mode}** · Judge: **{judge_label}** · "
        f"Sample: **{m['n']} questions (seed {args.seed})** over {corpus['docs']} source PDFs "
        f"({corpus['pages']} pages → {corpus['chunks']} chunks, {corpus['enriched']} VLM-enriched, "
        f"{corpus['skipped_pages']} pages skipped) · k={m['k']} · corpus-wide retrieval",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Retrieval hit@{m['k']} / MRR | **{pct(m['hit_at_k'])}** / **{num(m['mrr'])}** |",
        f"| Retrieval recall@{m['k']} (gold pages) | **{pct(m['recall_at_k'])}** |",
        f"| Citation precision / recall (page-level) | **{pct(m['citation_precision'])}** / **{pct(m['citation_recall'])}** |",
        f"| Verified rate (figures trace to citations) | **{pct(m['verified_rate'])}** |",
        f"| Answer rate | **{pct(m['answer_rate'])}** |",
        f"| Correctness (LLM-judge, n={m['judged_n']}) | **{pct(m['correctness'])}** |",
        "",
        "## Per-category",
        "",
        "| Category | n | hit@k | judge |",
        "|---|---|---|---|",
    ]
    cats: dict[str, list] = {}
    for r in m["rows"]:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in sorted(cats.items()):
        hits = sum(1 for r in rs if r["rank"]) / len(rs)
        js = [r["judged"] for r in rs if r["judged"] is not None]
        j = f"{sum(js) / len(js):.0%}" if js else "—"
        lines.append(f"| {cat} | {len(rs)} | {hits:.0%} | {j} |")
    lines += ["", "_Sampled-subset protocol (full EN corpus is 51k pages). Our system is "
              "text-first with block-level citations; the paper's RGenCite baseline is a "
              "page-image pipeline — numbers are indicative, not strictly comparable. "
              "Citation P/R here is page-level; FinRAGBench-V's box-level protocol is a "
              "planned addition using its citation_labels set._"]
    out = REPORTS / f"{date.today().isoformat()}-finragbench-v.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/finragbench_v")
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--contextual", action="store_true",
                    help="contextual chunk prefixes (1 LLM call/chunk — budget!)")
    ap.add_argument("--no-enrich", action="store_true")
    args = ap.parse_args()

    cases = load_cases(args.data_dir, sample=args.sample, seed=args.seed,
                       categories=args.categories)
    if not cases:
        raise SystemExit(f"no cases found under {args.data_dir} — download queries_en.json "
                         "and the pdfs (see evals/benchmarks/finragbench_v.py docstring)")
    print(f"{len(cases)} cases over {len({c.doc_name for c in cases})} documents")

    router = LLMRouter()
    store = ArtifactStore()                       # disk cache -> interrupted runs resume
    index = QdrantIndex()
    corpus = build_corpus(args.data_dir, cases, router, store, index,
                          contextual=args.contextual, enrich=not args.no_enrich)
    engine = AgentEngine(HybridRetriever(index), router=router)   # corpus-wide retrieval
    m = evaluate(cases, engine, router, k=args.k)
    path = write_report(m, corpus, args, engine.mode, router.label("judge"))
    print(f"\nhit@{args.k}={m['hit_at_k']:.0%} MRR={m['mrr']:.3f} "
          f"citeP={m['citation_precision']} correctness={m['correctness']} -> {path}")


if __name__ == "__main__":
    main()
