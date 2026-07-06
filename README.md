# finsight

Citation-grounded **financial document agent**: upload filings (born-digital or scanned),
ask questions, get answers where **every claim cites the exact page + block** and every
figure is **computed deterministically and verified** — never LLM mental math.

LangGraph multi-agent core · hybrid RAG (Qdrant + Cohere rerank) · MCP retrieval sidecar ·
async ingestion · Langfuse observability + versioned prompts · evaluated against
**FinRAGBench-V** (EMNLP 2025) with an independent Gemini judge.

> **Status: Phase 1** — ingestion + chunking done: text-layer parser (exact figures),
> cloud-OCR path with per-page resume cache (a 425-page report parses once in ~12 min,
> reruns in 1 s), structure-aware parent-child + contextual chunking, Qdrant index
> (sparse works keyless; dense auto-added with a Cohere key).
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
