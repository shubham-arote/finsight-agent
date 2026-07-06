# finsight

Citation-grounded **financial document agent**: upload filings (born-digital or scanned),
ask questions, get answers where **every claim cites the exact page + block** and every
figure is **computed deterministically and verified** — never LLM mental math.

LangGraph multi-agent core · hybrid RAG (Qdrant + Cohere rerank) · MCP retrieval sidecar ·
async ingestion · Langfuse observability + versioned prompts · evaluated against
**FinRAGBench-V** (EMNLP 2025) with an independent Gemini judge.

> **Status: Phase 2** — retrieval stack done: query-typed RRF fusion (sparse + optional
> Cohere dense), cross-encoder rerank with lexical fallback, deterministic exact-value
> lookup floated on top, small-to-big parent context — behind the `Retriever` protocol.
> Baseline (keyless): **hit@5 100% · MRR 0.902** on the labelled sample set, gated in CI
> ([evals/reports/](evals/reports/)). Phase 1: text-layer parser (exact figures),
> cloud-OCR with per-page resume cache (425-page report: ~12 min once, 1 s reruns),
> parent-child + contextual chunking, Qdrant index.
> Roadmap: [docs/PLAN.md](docs/PLAN.md). Prior art being migrated:
> the `financial_analyst_agent` repo (grounded doc-QA with calculator/verifier).

## Setup

```powershell
uv sync                                  # creates .venv, installs deps
copy .env.example .env                   # add any free-tier key (all optional)
uv run pytest                            # offline gate — must be green
```

## Design rules

- **Key-optional:** every provider key is optional; features degrade, never crash.
- **Roles, not providers:** code asks for `fast | answer | vision | judge`; the router
  maps each role to a fallback chain (Groq → Gemini → OpenRouter / Cohere) with
  rate-limit cooldowns. Judge is Gemini-only — independent from answering models.
- **No inline prompts:** all prompts are versioned YAML in `src/finsight/prompts/`,
  loaded by `name@version`, traceable per answer.
- **Config through `finsight.config.settings` only** — no `os.getenv` elsewhere.
- **No local ML models** — cloud-first (dev machine constraint); BM25 is the only
  local compute.
