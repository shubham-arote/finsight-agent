"""ingest_worker.py — async ingestion service (week-3 reference pattern).

    GCS upload ──(OBJECT_FINALIZE)──▶ Pub/Sub topic ──(push)──▶ this service
        ▶ download PDF ▶ ingest() (parse→chunk→contextualize, cached) ▶ shared Qdrant

Decouples parsing from the request path: uploads never block the app, and this service
scales independently. The Pub/Sub push envelope carries the GCS event in
`message.attributes` (bucketId/objectId — the notification format) with a JSON-in-`data`
fallback. Responses follow push semantics: 2xx acks (including skips), non-2xx retries.

Local emulation (no GCP): POST the same envelope shape to `/` — see tests.

Run:  uv run uvicorn finsight.ingest_worker:app --port 8080
"""

from __future__ import annotations

import base64
import json
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .ingestion import ArtifactStore, IngestError, ingest
from .llm import LLMRouter
from .retrieval import QdrantIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="finsight-ingest-worker")

STORE = ArtifactStore()
INDEX = QdrantIndex()          # shared collection (QDRANT_URL in deployment)
ROUTER = LLMRouter()


def download_gcs(bucket: str, name: str) -> bytes:
    """Fetch the uploaded object (module-level so tests/local emulation can stub it)."""
    from google.cloud import storage
    return storage.Client().bucket(bucket).blob(name).download_as_bytes()


def _parse_envelope(envelope: dict) -> tuple[str, str] | None:
    """Pub/Sub push envelope -> (bucket, object). Attributes first (GCS notification
    format, the week-3 path), then a JSON body in `data` as fallback."""
    msg = envelope.get("message") or {}
    attrs = msg.get("attributes") or {}
    bucket, name = attrs.get("bucketId"), attrs.get("objectId")
    if bucket and name:
        return bucket, name
    try:
        body = json.loads(base64.b64decode(msg.get("data", "")).decode("utf-8"))
        if body.get("bucket") and body.get("name"):
            return body["bucket"], body["name"]
    except Exception:
        pass
    return None


@app.post("/")
async def pubsub_push(request: Request):
    try:
        envelope = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    parsed = _parse_envelope(envelope or {})
    if not parsed:
        return JSONResponse({"error": "missing bucketId/objectId"}, status_code=400)
    bucket, name = parsed
    if not name.lower().endswith(".pdf"):
        logger.info("skipping non-PDF object: %s", name)
        return {"status": "skipped", "object": name}          # 200 = ack, don't redeliver

    logger.info("ingesting gs://%s/%s", bucket, name)
    try:
        data = download_gcs(bucket, name)
        res = ingest(data, router=ROUTER, store=STORE)
        INDEX.index_chunks(res.chunks)
        STORE.save_text(f"doc:{res.doc_id}:name", name.split("/")[-1])
        logger.info("ingested %s: pages=%d chunks=%d parser=%s cached=%d",
                    res.doc_id, res.page_count, len(res.chunks), res.parser,
                    res.cached_pages)
        return {"status": "ok", "doc_id": res.doc_id, "pages": res.page_count,
                "chunks": len(res.chunks), "parser": res.parser}
    except IngestError as e:                                   # e.g. scan w/o vision key
        logger.error("ingest failed for %s: %s", name, e)
        return JSONResponse({"error": str(e)}, status_code=500)   # 5xx -> Pub/Sub retries
    except Exception as e:
        logger.exception("unexpected failure for %s", name)
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "vision": ROUTER.available("vision")}
