# Demo runbook

Everything you need is in this folder. Follow it top to bottom. Don't change anything else.

---

## Before the demo (do this once, the night before)

**1. Start the server** — open a terminal, paste this, leave it running:

```
cd E:\PROJECTS\finsight-agent
make demo
```

**2. Check it's ready** — open a SECOND terminal:

```
cd E:\PROJECTS\finsight-agent
make demo-check
```

Wait for it to print **`READY`**. It loads all the documents and asks a real question.

- If it prints `READY` → you're good. Leave both terminals open.
- If it prints `[FAIL]` → read the line, it says what's wrong. Worst case: use the recording.

**3. Record a 2-minute backup video** of you doing the steps below. If anything breaks
live tomorrow, you play the video and keep talking. This is your insurance.

---

## The demo (5 minutes)

Open **http://localhost:8000** in your browser.

### Step 1 — Say the problem (45 seconds, no clicking)

> "An analyst can't use AI for anything that matters, because they can't tell if a
> number is real. Ask a chatbot about a filing and you get a confident figure with no
> way to check it — verifying it takes longer than just reading the filing yourself."

### Step 2 — Show a cited answer (60 seconds)

1. Click **Load sample report**
2. Type: `What was operating profit in FY26?`
3. When it answers, **click the citation chip** (`p4·b0`)
4. Say: *"It jumps to page 4 and highlights the exact row. Every number is traceable."*

### Step 3 — Show the calculator (45 seconds)

Type: `By how much did operating profit change year on year, in percent?`

Point at the chips as they appear: **lane: calc** → **calc: ((1052-985)/985)*100**

> "Number questions don't go to the language model. It writes one expression and a
> calculator evaluates it. The arithmetic is exact by construction."

### Step 4 — The autonomous brief (90 seconds) ← the important one

1. Click **Upload PDF** → choose `demo\filings\PDF-Solutions-Q1-2023.pdf`
   (it loads instantly — already cached)
2. Click the **📋 Brief** button
3. Watch the seven chips light up one by one

> "One instruction. It planned an analyst checklist and is now running each item
> through its full reasoning loop on its own — about twenty model and tool calls."

When it finishes, point at two things:
- A **"not disclosed"** row → *"the filing is silent on this. It says so instead of inventing a number."*
- The **⚠ unverified figure** warning → **this is the whole product**:

> "The model asserted a number its own citation doesn't support, and the system caught
> it. A tool that's confidently wrong is worse than no tool. This one tells you which
> number to double-check."

### Step 5 — Compare two filings (60 seconds)

1. Click **Upload PDF** → choose `demo\filings\PDF-Solutions-Q1-2022.pdf`
2. Go back to the Q1 2023 document
3. Use the **⇄ Compare…** dropdown → pick the Q1 2022 filing

> "Revenue 40.8 versus 33.5, up 21.8% — computed by the calculator, not generated.
> The filing itself says 'up 22%', so it agrees. Each figure is cited into its own
> document. And this row it refused to compare, because the two filings state it in
> different units — a wrong delta in a briefing note is worse than a missing one."

### Step 6 — Close (30 seconds)

> "It doesn't make decisions. It makes the evidence behind them fast and provable —
> so the analyst spends their time on judgment instead of transcription. It's deployed
> on Cloud Run with Vertex AI, every change is scored against committed quality floors,
> and the whole thing runs with no API keys in the container."

---

## If someone asks

**"Is this deployed?"** — Yes: `https://finsight-7vuichacoq-uc.a.run.app`
(agent + MCP retrieval sidecar + Qdrant on Cloud Run, Vertex AI by service-account
identity, GCS → Pub/Sub for async ingestion. I'm demoing locally because it's faster.)

**"How do you know it works?"** — Committed eval reports: citation hit 100%, verified
figures 100%, correctness 100% judged by a *different* model family. A regression gate
fails the build if any of those drop.

**"What can't it do?"** — It doesn't value, forecast, or recommend. Scanned documents
get page-level citations instead of block-level. It's the evidence layer, not the analyst.

**"What broke in production?"** — Cloud Run reserves `/healthz`; the `:latest` tag made
redeploys silent no-ops; an in-service vector store can't serve a separate ingest worker;
and the verifier flagged its own correct answer because production retrieval returned a
different evidence set. None reproducible locally — that's the argument for deploying early.

---

## Files here

- `filings/PDF-Solutions-Q1-2023.pdf` — the main demo document
- `filings/PDF-Solutions-Q1-2022.pdf` — the comparison document
- Demo document (slides): the artifact link in your chat
