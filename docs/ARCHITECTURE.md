# How finsight actually works — in plain language

*Written to be explainable by a human in an interview, without a backend background.
Every section answers one question you might get asked.*

## The 60-second story

> "A PDF goes in. We split every page into **blocks** — paragraphs, headings, tables —
> each with its exact position on the page. Blocks become **chunks** in a vector
> database (Qdrant). When you ask a question, we find the most relevant chunks, an
> LLM writes an answer **quoting only those chunks**, and a deterministic checker
> verifies every number in the answer actually appears in the cited block. If the
> evidence is weak, the agent says 'I can't answer that' instead of guessing.
> Math is never done by the LLM — a calculator does it."

That last paragraph is the whole product. Everything else is plumbing for it.

## The data journey (follow one PDF through)

```
1. UPLOAD      you POST a PDF → we hash its bytes (same file twice = no re-work)
2. PARSE       page by page:
                 • page has real text → PyMuPDF reads it directly (exact, free)
                 • page is a scan     → we render it to an image and a vision
                   model (Gemini) transcribes it   ← this is our OCR
3. BLOCKS      each page becomes a list of blocks: {type, text, bbox, page}
                 bbox = the rectangle on the page → this is what makes
                 click-to-highlight possible later
4. ENRICH      (optional, "quality profile") blocks the text layer can't read —
               charts, borderless tables — get their rectangle CROPPED and sent
               to Gemini vision; the description/markdown becomes the block text
5. CHUNK       blocks → children + parents (see "Chunking" below)
6. INDEX       every chunk goes into Qdrant with TWO vectors:
                 sparse  (keyword statistics — works with zero API keys)
                 dense   (Cohere embedding — semantic meaning, needs a key)
7. ASK         the LangGraph agent runs (see "The agent" below)
8. ANSWER      JSON claims, each citing {page, block} → UI draws the highlight box
```

Every parse and every vision call is **cached by content hash** in a small SQLite file
(`data/artifacts.db`). Cache is why a 425-page report parses once in ~12 minutes and
re-ingests in ~1 second, and why an interrupted OCR run resumes instead of restarting.

## Chunking — what we do and WHY (it is not random)

**Strategy: parent–child, also known in the industry as "small-to-big".**

- **Child = one block** (a paragraph, one table). Small and precise.
  Children are what we *search over* and what citations *point at*.
- **Parent = the section** (a heading plus everything under it). Big and coherent.
  Parents are what the *LLM reads* when writing the answer.

Why this shape — the one-liner for interviews:
> "Search wants small units (a query about revenue should match the revenue
> paragraph, not a 3-page section). But LLMs want context (the paragraph alone may
> not say which company or which year). Parent-child gives you both: **match small,
> read big** — and because the child is a real block with a bbox, the citation is
> pixel-precise for free."

Two deliberate rules on top:
1. **Tables are never split.** Half a table is worse than useless — rows lose their
   column headers. A table is always exactly one chunk.
2. **Contextual prefixes (optional):** before embedding, an LLM can prepend one
   sentence — "This is from PDF Solutions' Q1-2023 earnings release" — so the word
   "revenue" embeds differently for every document. This is Anthropic's "contextual
   retrieval" technique. It costs one LLM call per chunk, so it's OFF in the demo
   profile and ON for building evaluation corpora.

## Where is Qdrant running?

One codebase, three modes — chosen ONLY by the `QDRANT_URL` environment variable:

| Mode | Where Qdrant lives | When |
|---|---|---|
| `QDRANT_URL` unset | **inside the Python process** (qdrant-client's local mode — no server at all) | tests, `make demo`, keyless laptop use |
| `http://qdrant:6333` | the **qdrant container** in docker-compose (also a sidecar container on Cloud Run) | `make up`, the demo deploy |
| `https://xyz.qdrant.io` | **Qdrant Cloud** (managed, persistent) | production — lets the app and the ingest worker share one index |

The collection holds both vector types side by side (named vectors `sparse` + `dense`),
so keyless and keyed modes use the same schema.

## Tables and charts — are we "only PyMuPDF"? No. Three tiers:

1. **Ruled tables** (visible grid lines): PyMuPDF's `find_tables` reads them directly
   into markdown. Free, exact.
2. **Borderless tables** (whitespace-aligned — most financial statements): PyMuPDF
   sees only loose text. A numeric-density heuristic detects "this text block is
   secretly a table", crops its rectangle, and **Gemini vision** transcribes it into a
   real markdown table. Structure recovered, ~1 vision call per table, cached forever.
3. **Charts/figures** (numbers drawn as pixels): the figure's rectangle is cropped and
   Gemini describes it including the printed values, so chart content becomes
   searchable text.

Why crops instead of OCR-ing every page as an image (what the course's week-3 does)?
> "A 200-page filing needs ~200 page-OCR calls their way; ours needs 10–30 targeted
> crop calls, keeps prose figures *exact* from the text layer (no OCR digit errors),
> and keeps citations block-precise."

So: **yes we use OCR/vision — but surgically**: whole-page OCR only for scanned pages,
crop-level vision only for the blocks the text layer can't represent.

## The agent (why it's not "just a chatbot")

A LangGraph state machine — each node is a small function, in its own file:

```
contextualize → supervise → retrieve → grade ──(weak)── rewrite ↺ (max 3, then ABSTAIN)
                   │                      │(ok)
                   │ lane: qa ────────────┴────────────→ generate → cite_check → END
                   │ lane: calc → calculator (AST math, never the LLM) ↗
```

Decision points that a chatbot doesn't have: route by intent (lookup vs derivation),
retry retrieval with a rewritten query under a budget, refuse on weak evidence,
compute deterministically, and **audit its own citations** — `cite_check` verifies
every figure in every claim appears in the exact block that claim cites (or in the
calculator result), else it appends a visible warning.

### The brief lane — one instruction, a dozen autonomous steps

Say *"analyze this filing"* (or press 📋 Brief) and the agent stops answering and
starts **working**:

1. **Plan** — a finance-standard first-read checklist: revenue, growth, profitability,
   margin, bottom line, cash, outlook. (Deterministic today; the hook to have an LLM
   adapt it per document type lives in `plan_brief`.)
2. **Execute** — each checklist item is run through the **whole graph above**, on its
   own: retrieve → grade → maybe rewrite-and-retry → maybe calculate → generate →
   cite_check. Seven items ≈ 7 full agent runs ≈ 20+ LLM and tool calls, no human input.
3. **Compose** — a one-page brief. Every line keeps its `[p·b]` citations (clickable →
   highlights the block on the page), computed figures are labelled, and any item the
   document doesn't disclose is printed as **"not disclosed"** — the abstain path
   surfacing as an honest gap rather than a fabricated number.

Why this is the interesting part: *"a chatbot answers a question; this produces an
analyst's first hour of work from one click, and every number in it is traceable."*

Implementation note worth knowing: the lane lives in `agent/brief.py` at the **engine**
level, not as a graph node — it *orchestrates* whole graph runs, so making it a node
would mean a graph invoking itself. Each item also gets its own thread id, so the seven
runs stay independent instead of polluting each other's conversation memory.

## Rate limits — why they happen and the strategy

Where the calls go, per activity:

| Activity | LLM calls | Notes |
|---|---|---|
| One question | 2–4 (grade, maybe calculate, generate, maybe contextualize) | fine on any free tier |
| Ingest, demo profile | **0** | this is why demo mode exists |
| Ingest, quality profile | ~1 per chunk + 1 per enriched crop | a 100-page doc = hundreds of calls → THIS is what hits free-tier limits |

The strategy (each layer already built):
1. **Demo profile** for interactive use — ingest makes zero LLM calls.
2. **Role-based fallback chains** — `fast/answer/vision/judge` each map to a list of
   providers; a 429 puts that model on a 60s cooldown and the next provider answers.
3. **Spread roles across providers** — with only a Groq key, everything queues on
   Groq. Adding a (free) Gemini API key moves `vision` + spare capacity off Groq.
4. **Content-hash caching** — nothing is ever paid for twice.
5. **At scale: Vertex AI** — same chains, `vertex_ai/gemini-…` models, authenticated
   by the service account, production quotas. Scale is a config change, not code.

## What each folder is (the 6-piece mental model)

| Piece | Folder | One sentence |
|---|---|---|
| Config | `config.py` | every setting and key, read in exactly one place |
| Models | `llm/` | the role-router (which provider answers) + versioned prompts |
| Ingestion | `ingestion/` | PDF → blocks → chunks, with the cache |
| Retrieval | `retrieval/` | Qdrant index + hybrid search + the `Retriever` seam (also served over MCP) |
| Agent | `agent/` | the graph above + calculator + citation checker + guards |
| Serving | `server.py`, `web/`, `services/`, `ingest_worker.py` | FastAPI + UI + the Pub/Sub worker |

Two seams to remember: the agent only knows `Retriever.retrieve()` (in-process or MCP
sidecar — one env var), and all model access goes through the role-router (providers
are configuration). Everything else can change without touching the agent.
