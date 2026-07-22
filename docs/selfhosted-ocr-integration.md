# Integrating the self-hosted VDU pipeline (production-ocr-course) on GCP

The course repo (`E:\PROJECTS\production-ocr-course`) builds a self-hosted Visual
Document Understanding pipeline on GKE: **Rust (Axum) gateway → Redis → T4 layout
worker (GlmOcr SDK, `/dev/shm` handoff) → vLLM serving Qwen 3.5-4B on A100s**, KEDA
scale-to-zero, GCP API Gateway in front, MCP wrapper. Its API contract (from the code):

    POST /process           (binary PDF/image)  -> {"task_id": ...}
    GET  /status/<task_id>  -> {"status", "result": {"markdown", "layout"}}
    GET  /health

`layout` is GlmOcr's block-level JSON — **it carries geometry**, which matters to us
(see "Why bother", below).

## Why bother (and why NOT, for the demo)

finsight already solves OCR with hosted Gemini at fractions of a cent per page. The
self-hosted cluster earns its complexity only when at least one of these is true:

| Trigger | Why the cluster wins |
|---|---|
| **Data sovereignty** | documents never leave your VPC — no third-party model API sees them |
| **Sustained volume** | roughly >10k pages/day: A100 rental beats per-page API pricing, and 1.86 pages/s/replica of owned throughput is predictable |
| **Scanned-doc citation precision** | GlmOcr's layout stage returns per-block bboxes — restoring **block-level citations for scanned documents**, our current honest limitation (hosted OCR gives page-level only) |

For the portfolio demo, keep Gemini. The integration exists so scale is a *config
change*, which is itself the production-grade story.

## The three integration levels

### Level 1 — vLLM as a vision provider (DONE, config-only)
vLLM speaks the OpenAI API, so the cluster's Qwen endpoint is just another provider
in the role router:

    VLLM_BASE_URL=http://<ilb-or-api-gateway>/v1
    LLM_VISION=hosted_vllm/Qwen/Qwen3.5-4B,gemini/gemini-2.5-flash

Every existing vision call — scanned-page OCR, table/figure crop enrichment — now runs
on your GPUs, with Gemini as automatic fallback when the cluster is scaled to zero.
No finsight code changes; supported + tested in `llm/router.py`.

### Level 2 — the gateway as a whole-document parser (NEXT, needs a live cluster)
A third parser `ingestion/parsers/selfhosted.py` beside `textlayer`/`cloud_ocr`:

1. `POST /process` the PDF once (their pipeline does its own page decomposition,
   batching and `/dev/shm` handoff — far better GPU utilisation than our
   page-at-a-time calls)
2. poll `/status/<task_id>` (same pattern our UI already uses for parse progress)
3. map `result.layout` blocks → our `Block{type, bbox, page, content}` — this is the
   step that upgrades scanned docs to block-precise citations
4. routing: `parse_pdf(..., parser="auto")` prefers the gateway for scanned pages
   when `OCR_GATEWAY_URL` is set; per-page Gemini stays the fallback

Blocked on: a running cluster + one real `layout` JSON sample to write the field
mapping against (schema is the GlmOcr SDK's; don't guess it).

### Level 3 — full topology on GCP (both systems, one project)

    ┌─ Cloud Run ──────────────────────────┐        ┌─ GKE (private) ─────────────────┐
    │ finsight app (agent+mcp+qdrant)      │        │ API GW / ILB → Rust gateway     │
    │ finsight-ingest (Pub/Sub worker) ────┼─VPC────┼→ Redis → T4 layout → A100 vLLM  │
    └──────────────────────────────────────┘ conn.  │   (KEDA, scale-to-zero)         │
                GCS upload → Pub/Sub                └─────────────────────────────────┘

- Same project; Cloud Run reaches the cluster's **internal** load balancer via a
  Serverless VPC Access connector (or via API Gateway with JWT if exposed).
- `VLLM_BASE_URL` / `OCR_GATEWAY_URL` delivered via Secret Manager like every other
  endpoint; nothing about finsight's deploy scripts changes.
- Their `docs/gke_deployment.md` is the cluster runbook (AR repo, cluster,
  T4 + A100 pools with taints, Redis pool, KEDA). Budget note: A100 pools cost real
  money the moment KEDA scales above zero; set `--max-nodes` low and alerts first.

## Proven-steps order (when a cluster project exists)

1. Deploy their stack per `gke_deployment.md` — verify with their own client flow.
2. Point a laptop finsight at it: set `VLLM_BASE_URL`, ask a scanned-doc question,
   watch the vision call hit Qwen (Level 1 — zero code).
3. Capture one `/status` response, write the Level-2 block mapping + tests.
4. Wire VPC connector, move both URLs into Secret Manager, redeploy finsight.
5. Re-run the FinRAGBench-V chart/table categories — the before/after number for
   "did self-hosted VDU with real layout beat crop-enrichment".
