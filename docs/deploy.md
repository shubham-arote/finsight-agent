# Deploying finsight to GCP

Reference architecture (mirrors the course's week-3 ingestion + week-4 sidecar patterns):

```
                       ┌─ Cloud Run: finsight (public) ──────────────┐
 user ── HTTPS ──────▶ │ agent (UI/API/LangGraph, :8000)             │
                       │    │ MCP over localhost:3000/mcp/           │
                       │ mcp sidecar (retrieval)                     │
                       └───────────────┬─────────────────────────────┘
                                       │ shared collection
 gsutil cp report.pdf ─▶ GCS bucket ─▶ Pub/Sub ─push(OIDC)─▶ Cloud Run: finsight-ingest
                        (OBJECT_FINALIZE)                       (parse→chunk→index)
                                       │
                              Qdrant Cloud (free tier)
```

## Prerequisites
- A billing-enabled GCP project + `gcloud` CLI (or Cloud Shell).
- A [Qdrant Cloud](https://cloud.qdrant.io) free-tier cluster (URL + API key) — the
  shared vector store across all three containers.
- Free-tier keys as needed: Groq/Gemini (answers), Cohere (dense + rerank).

## Steps

```bash
cd deploy/gcp
PROJECT_ID=<your-project> ./setup.sh    # APIs, AR, bucket→topic, secrets, SA (idempotent)
# fill the CHANGE_ME secrets (console or `gcloud secrets versions add`)
PROJECT_ID=<your-project> ./deploy.sh   # Cloud Build → app (agent+mcp sidecar) + worker + subscription
```

Verify the live service exactly like the local stack:

```bash
uv run python scripts/verify_stack.py https://finsight-<hash>-uc.a.run.app
```

Async ingestion: `gsutil cp any_report.pdf gs://$BUCKET_NAME/` — the worker logs show
parse → chunk → index, and the document becomes searchable in the shared collection.

## Scale knobs
- **Vertex AI for large eval runs**: the service account already has `aiplatform.user`;
  set `LLM_JUDGE=vertex_ai/gemini-2.5-flash` (+ `GOOGLE_CLOUD_PROJECT`, injected in the
  service spec) — same role chains, production quotas, ADC auth (week-3 pattern).
- Cost guards are on by default: scale-to-zero, `maxScale: 2`, worker `max-instances=2`.

## Known limitations (Phase 7b)
- The app's document *registry* (page rendering, upload-via-UI persistence) lives on
  ephemeral container disk; Qdrant retains all indexed chunks, but after scale-to-zero
  the UI won't list previously uploaded docs. Fix planned: GCS-backed artifact/doc store
  + registry rebuild from Qdrant payloads.
- Conversation checkpointer is in-memory on Cloud Run (per-instance); Cloud SQL Postgres
  is the planned durable backend.
