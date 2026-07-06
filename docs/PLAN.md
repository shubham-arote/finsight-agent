# finsight — migration & build plan (source of truth)

Goal: portfolio-grade, deployable, explainable finance-domain agent. LangGraph multi-agent
core, MCP retrieval sidecar, async ingestion, Qdrant hybrid RAG, Langfuse observability +
prompt versioning, honest eval vs **FinRAGBench-V** (EMNLP 2025), Cloud Run + docker-compose.

Prior art: `E:\PROJECTS\ocr pipeline` (financial_analyst_agent). Reference infra:
`D:\Grokking AI Agents` weeks 1–5 (containerized RAG → Cloud Run → GCS/Pub-Sub ingestion →
MCP sidecar + Qdrant → Langfuse/eval labs).

## Port vs rewrite (from the old repo)

| Verdict | Component |
|---|---|
| Port | calculator.py (AST safe_eval + security tests), verify.py, guards.py, textlayer parser, deterministic number lookup, parent-child chunking concept, abstain path, DocStore seam, web UI |
| Rewrite | cloud.py → LiteLLM role-router (DONE P0) · graph.py → nodes/ modules + prompt registry · in-memory BM25 → Qdrant hybrid · eval harness · server split (api / ingest worker) |
| Drop | SRR single-image streaming demo, XY-cut detector, stub/easyocr recognizers |

## Key decisions

1. **Rate limits** → `llm/router.py`: role-based chains (`fast/answer/vision/judge`),
   fallback on 429/5xx with cooldown, keys optional. Judge = Gemini only (independent
   family; also the cloud-deploy judge — user decision 2026-07-06).
2. **Parsing** → born-digital: PyMuPDF text layer; scanned: cloud OCR via `vision` chain,
   per-page queue + backoff + persisted page artifacts (OCR runs once, resumable).
   Docling = optional local tier, never default, never in the lean image.
3. **Chunking** → structure-aware (heading/table boundaries) · parent-child (400–800-tok
   children / ~2k parents) · **contextual retrieval** (LLM 1–2 sentence context prepended
   per chunk at index time, cached) · table-aware (tables whole + NL summary) · metadata:
   doc_id, page, bbox, section_path, fiscal_period, units.
4. **Retrieval** → Qdrant server (container local / Qdrant Cloud free deployed): dense
   (Cohere embed-v4) + sparse hybrid fusion + Cohere rerank-3.5; deterministic exact-number
   lookup floated on top; behind the `Retriever` protocol.
5. **Agent** → modular LangGraph: `agent/state.py`, one file per node (supervise, retrieve,
   grade, calculate, generate, verify, cite-check), Postgres checkpointer. Retrieval via
   MCP tool (Streamable HTTP sidecar, week-4 pattern) with in-process fallback.
6. **Citations are structured**: generate emits JSON `{claims:[{text, citations:[{page,
   block_id}]}]}` schema-validated (retry on invalid); **citation post-check node** verifies
   each cited block contains/entails its claim. Matches FinRAGBench-V page/block citation P/R.
7. **Prompts** → versioned YAML registry (DONE P0), synced to Langfuse Prompt Management;
   traces link the exact prompt version per answer.
8. **Observability** → Langfuse cloud free tier; span per node with tokens/cost/latency;
   JSONL fallback offline.
9. **Eval** → three tiers: (1) deterministic CI gate — retrieval hit@k/MRR, calculator
   exactness, verifier catches planted errors, abstain on OOS; (2) LLM-judge (Gemini) —
   faithfulness, correctness, citation accuracy; (3) benchmark — **FinRAGBench-V English
   sampled subset** (~150–300 Qs + their source docs; full corpus 51k pages exceeds free
   embeddings — disclose sampling), metrics mirrored from the paper (retrieval recall/MRR,
   generation accuracy, citation P/R page+block). FinanceBench optional; ObliQA later for
   the regulatory-compliance lens. Reports versioned in `evals/reports/`.
10. **Reliability adds**: LLM+embedding cache (content-hash), calibrated abstain threshold,
    golden-trace regression tests, health/readiness endpoints. Optional: cropped-image
    answer path (bbox crop → vision model) for chart/figure questions.
11. **Deploy** → local: docker-compose (agent+mcp, qdrant, postgres, ingest); GCP: Cloud Run
    multi-container sidecar, Pub/Sub push ingestion, GCS, Cloud SQL, Secret Manager,
    GitHub Actions CI/CD. Pub/Sub is a deploy-phase feature, not on the critical path.

## Phases (each: tests green → eval no-regression → commit → README updated)

| # | Phase | Definition of done | Status |
|---|---|---|---|
| 0 | Scaffold + LLM router + prompt registry + CI | 429-fallback proven by test; prompts load by name@version | ✅ |
| 1 | Ingestion + chunking (parsers port, tiered OCR w/ resume, contextual+table chunks, Qdrant indexing) | sample 10-K indexed; golden-file chunk tests; rerun skips cached OCR | ✅ e2e: 425-page report → 11.6k chunks/425 tables in 735s; rerun 1.1s (all cached); sparse search page-accurate |
| 2 | Retrieval (hybrid + rerank + lookup behind protocol) | tier-1 retrieval baseline (hit@k/MRR) recorded in first eval report | ✅ keyless baseline: hit@5 100%, MRR 0.902 (21 Qs, sample report); gate in test_retrieval floors it at 90%/0.70 |
| 3 | Agent port (modular nodes, structured citations + cite-check from day one, checkpointer, guards, API+UI) | node unit tests; answer parity spot-check vs old repo | ✅ core (nodes/graph/citation contract/guards/calculator, 63 tests, CLI e2e) · ⏳ 3b: FastAPI + UI port |
| 4 | Observability (Langfuse traces + prompt links + cost) | full node-by-node trace visible for a YoY-margin question | ⏳ |
| 5 | Eval harness (3 tiers + FinRAGBench-V subset adapter) | honest baseline report committed; CI fails on faithfulness/citation regression | ⏳ |
| 6 | MCP sidecar + ingest worker + docker-compose | `docker compose up` → upload → ask → cited verified answer | ⏳ |
| 7 | GCP deploy + polish (Cloud Run sidecar, Pub/Sub, GCS, Cloud SQL, CD, README/architecture/demo) | live URL; README leads with eval numbers + trace screenshot | ⏳ |

## Out of scope (portfolio discipline)

A2A protocol, fine-tuning, local models (2 GB GPU), Terraform, multi-tenant auth,
>2 benchmarks, frameworks beyond LangGraph.
