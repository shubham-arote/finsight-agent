# finsight — a citation-grounded financial document agent

Upload a filing (born-digital **or** scanned) and either ask questions or say
**"analyze this filing"** — the agent then works on its own: it plans an analyst
checklist, runs each item through its full reasoning loop, and returns a **one-page
brief where every figure carries the page + block it came from** and **every computed
number is calculated deterministically and verified against its own citation**.
Never LLM mental math; never an untraceable number; an honest "not disclosed" when the
document doesn't say.

**LangGraph multi-agent core · hybrid RAG (Qdrant) · MCP retrieval sidecar · Pub/Sub async
ingestion · Langfuse observability · versioned prompts · evaluated on FinRAGBench-V (EMNLP 2025).**

## The autonomous lane (what makes it an agent, not a chatbot)

One instruction → a dozen autonomous steps → a work product:

```
"analyze this filing"
   └─▶ plan       finance-standard first-read checklist (revenue, growth, margin,
   │              bottom line, cash, outlook — 7 items)
   └─▶ execute    each item runs the FULL agent loop by itself:
   │              retrieve → grade → (rewrite ↺ | calculate) → generate → cite_check
   └─▶ compose    one-page brief; every line cited [p·b] and click-to-highlight,
                  computed figures flagged, gaps marked "not disclosed" (never invented)
```

Everything else below is the machinery that makes those figures trustworthy.

```
 PDF ──▶ parse (routed PER PAGE)          text layer (exact figures, free)  |  cloud VLM OCR (scanned)
   │                                      + targeted Gemini enrichment: figures & borderless tables
   ▼
 parent-child chunks ──▶ Qdrant           sparse (keyless) + Cohere dense · query-typed RRF
   │                                      · rerank-v3.5 · deterministic exact-value lookup on top
   ▼
 LangGraph agent                          contextualize → supervise → retrieve → grade ─┬─ qa ──────▶ generate
   │                                              rewrite ↺ (budget)                    └─ calc ▶ AST calculator
   ▼                                                                                        │
 cited, VERIFIED answer                   generate → cite_check: JSON claims {text, citations:[{page, block}]},
   │                                      invented citations dropped, every figure traced to its cited block
   ▼
 web UI                                   click a citation → jumps to the page, highlights the exact box
```

## Why this design

- **Verified arithmetic.** Number questions route to a calculator lane: the LLM emits one
  arithmetic expression over retrieved figures; an **AST-whitelist evaluator** (never `eval`,
  security-tested) computes it exactly, and the verifier confirms every input traces to a citation.
- **Citations are a contract, not decoration.** Generation returns structured JSON claims;
  citations pointing at evidence that wasn't retrieved are deleted; a deterministic `cite_check`
  node validates each claim's figures against the *specific blocks it cites* and surfaces
  failures as a visible caveat. Unverifiable ≠ silently trusted.
- **Key-optional everything.** No API keys → sparse retrieval + extractive answers; each key
  (Groq/Gemini/Cohere) upgrades one capability. The full test suite runs offline.
- **Role-based model routing.** Code asks for `fast | answer | vision | judge`; a LiteLLM
  router maps each role to a fallback chain (Groq → Gemini → OpenRouter/Cohere) with rate-limit
  cooldowns. The judge is a **different model family** (Gemini / Vertex AI) from the answering
  models — no self-grading. On GCP, `vertex_ai/*` models join the same chains via ADC (no keys).
- **Cost-aware ingestion.** Every parsed page and VLM call is cached by content hash: a 425-page
  report parses once (~12 min) and reruns in ~1 s; interrupted OCR resumes at the failed page.
  Vision enrichment targets only figure/borderless-table crops (capped per doc) instead of
  paying page-image OCR for entire documents.

## Measured, not claimed

Every retrieval/agent change is gated in CI against committed eval reports ([evals/reports/](evals/reports/)).

| Eval (labelled sample set, offline/keyless) | Result |
|---|---|
| Retrieval hit@5 / MRR | **100% / 0.902** |
| Abstain accuracy (out-of-scope refused) | 75% (100% with cloud grading) |
| Citation hit · verified figures · claim coverage | **90% · 100% · 100%** |

**FinRAGBench-V** (EMNLP 2025, real filings — 539 EN questions / 105 PDFs): the harness
([evals/run_benchmark.py](evals/run_benchmark.py)) ingests the real source PDFs, scores retrieval
recall/MRR vs gold pages, **page-level citation precision/recall**, and judge correctness, with
per-category breakdowns. The committed keyless floor (sparse-only, 25-question sample) is
**hit@5 17% doc-scoped / 0% corpus-wide** — an honest baseline demonstrating exactly why the
hybrid stack (dense + rerank) and VLM enrichment exist; keyed runs measure their lift.

## Run it

```powershell
uv sync && copy .env.example .env        # keys optional — everything degrades gracefully
uv run pytest                            # offline test suite
uv run uvicorn finsight.server:app --port 8000    # → http://localhost:8000
```

**Containerised (sidecar-MCP architecture):**

```bash
docker compose up --build               # agent + MCP retrieval sidecar + Qdrant
uv run python scripts/verify_stack.py   # e2e proof: ingest → ask → cited, verified answer
```

The agent talks to retrieval over **MCP (Streamable HTTP)** when `MCP_SERVER_URL` is set and
in-process otherwise — same `Retriever` protocol, so the agent can't tell the difference.

**GCP** (Cloud Run multi-container + GCS→Pub/Sub async ingestion + Secret Manager + Vertex):
scripted end-to-end in [deploy/gcp/](deploy/gcp/) — see [docs/deploy.md](docs/deploy.md).

### Scaling past free-tier rate limits

Free keys cap request rate, which shows up during *ingest* (contextual chunking and VLM
enrichment fire one call per chunk/crop), not during questions. Three levers, in order:

1. **Demo profile** (default in compose/Cloud Run): `CONTEXTUAL_CHUNKS=0 ENRICH_BLOCKS=0`
   → ingest makes **zero** LLM calls; the agentic query path is untouched.
2. **Spread roles across providers** — a second free key (Gemini) moves `vision` and
   `judge` off the answer provider, so one quota isn't serving every role.
3. **Vertex AI for production quotas** — same role chains, no API keys, service-account
   auth (ADC); this is the real fix at volume:

```bash
GOOGLE_CLOUD_PROJECT=my-project
LLM_ANSWER=vertex_ai/gemini-2.5-flash,groq/llama-3.3-70b-versatile
LLM_FAST=vertex_ai/gemini-2.5-flash
LLM_VISION=vertex_ai/gemini-2.5-flash
LLM_JUDGE=vertex_ai/gemini-2.5-flash
```

Nothing else changes: providers are configuration, and every content-hash cache means a
rerun costs zero calls. A self-hosted vLLM cluster slots into the same chains via
`hosted_vllm/*` + `VLLM_BASE_URL` — see [docs/selfhosted-ocr-integration.md](docs/selfhosted-ocr-integration.md)
for when that's actually worth it (sovereignty / >10k pages a day).

## Repo map

```
src/finsight/
  config.py        single Settings object — the only place env is read
  llm/             role-based router (fallback chains, cooldowns) · versioned prompt registry
  prompts/         every prompt as append-only versioned YAML (name@version, changelogs)
  ingestion/       per-page routed parsers · VLM enrichment · parent-child+contextual chunking
                   · content-hash artifact cache (parse once, resume free)
  retrieval/       Retriever protocol · Qdrant hybrid (sparse+dense RRF) · rerank · exact lookup
                   · MCP client (the same seam, served remotely)
  agent/           LangGraph nodes (one file each) · AST calculator · verifier · guards
                   · structured-citation contract (citations.py)
                   · brief.py — the autonomous lane: plan → run each item → compose
  mcp_server/      retrieval as an MCP tool (Streamable HTTP sidecar)
  services/, server.py, web/   doc lifecycle · thin FastAPI · split-view UI w/ click-to-highlight
  ingest_worker.py Pub/Sub push worker (GCS upload → parse → index)
  obs.py           JSONL traces (+ optional Langfuse: cost, span-per-node, prompt links)
evals/             agent eval + retrieval baseline + FinRAGBench-V benchmark harness + reports
tests/             ~100 offline tests: security suite, golden files, live MCP round-trip, e2e
deploy/gcp/        Cloud Run multi-container spec + idempotent setup/deploy scripts
```

## Honest limitations

- Scanned documents get **page-level** citation granularity (OCR yields no per-block geometry);
  born-digital documents are block-precise.
- Offline (keyless) mode is extractive and lexical: it can't refuse topic-adjacent unanswerable
  questions (measured: 75% abstain) or rank long analytical queries well (the FinRAGBench-V floor).
- The keyless FinRAGBench-V numbers above are the *before* — dense+rerank and judge-scored runs
  require (free-tier) API keys and are the next committed report.
- Cloud Run demo config uses an in-service Qdrant (ephemeral); production points `QDRANT_URL`
  at Qdrant Cloud, and durable conversation state needs the Cloud SQL checkpointer (planned).
