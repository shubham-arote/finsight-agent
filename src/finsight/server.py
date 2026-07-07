r"""server.py — thin FastAPI surface. Routes only; lifecycle lives in services/documents.

Endpoints
  GET  /                        split-view UI
  GET  /api/status              cloud/offline + model labels for the header badge
  GET  /api/docs                registered documents
  POST /upload                  PDF -> ingest in background (page-level progress)
  POST /load-sample             the bundled synthetic annual report
  GET  /doc/{id}/status         ingest progress
  GET  /doc/{id}/page/{n}.png   page render (on demand)
  GET  /doc/{id}/page/{n}       page blocks (bboxes for citation highlighting)
  WS   /ws/ask/{id}             streams the agent's node events + the cited answer
  GET  /healthz                 liveness

Run:  uv run uvicorn finsight.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .agent import guards
from .ingestion.models import block_to_dict
from .services import documents

WEB_DIR = Path(__file__).parent / "web"


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    documents.load_persisted()          # restart recovery (page cache -> seconds)
    yield


app = FastAPI(title="finsight", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


async def _bridge(gen_factory):
    """Run a blocking generator in a thread; yield its events into the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def worker():
        try:
            for ev in gen_factory():
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    loop.run_in_executor(None, worker)
    while True:
        ev = await queue.get()
        if ev is DONE:
            return
        yield ev


# ── HTTP ────────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status():
    r = documents.ROUTER
    return {"cloud": r.available("answer"),
            "models": {"answer": r.label("answer"), "fast": r.label("fast"),
                       "vision": r.label("vision")}}


@app.get("/api/docs")
def list_docs():
    return {"docs": [documents.summary(d) for d in documents.DOCS]}


@app.post("/upload")
async def upload(file: UploadFile):
    if not (file.filename or "").lower().endswith(".pdf"):
        return JSONResponse({"error": "only PDF uploads are supported"}, status_code=400)
    data = await file.read()
    try:
        return documents.add_document(data, file.filename or "document.pdf")
    except Exception as e:
        return JSONResponse({"error": f"could not open PDF: {e}"}, status_code=400)


@app.post("/load-sample")
def load_sample():
    return documents.add_document(documents.sample_pdf_bytes(), "sample_report.pdf",
                                  background=False)


@app.get("/doc/{doc_id}/status")
def doc_status(doc_id: str):
    if doc_id not in documents.DOCS:
        return JSONResponse({"error": "unknown doc"}, status_code=404)
    return documents.summary(doc_id)


@app.get("/doc/{doc_id}/page/{n}.png")
def page_png(doc_id: str, n: int):
    d = documents.get(doc_id)
    if not d or n < 1 or n > d["page_count"]:
        return Response(status_code=404)
    pix = d["fitz"][n - 1].get_pixmap(dpi=130)
    return Response(content=pix.tobytes("png"), media_type="image/png")


@app.get("/doc/{doc_id}/page/{n}")
def page_blocks(doc_id: str, n: int):
    d = documents.get(doc_id)
    if not d or n < 1 or n > d["page_count"]:
        return JSONResponse({"error": "unknown page"}, status_code=404)
    w, h = d["sizes"][n - 1]
    blocks = (d["pages_blocks"][n - 1] if d.get("pages_blocks") else [])
    return {"page": n, "page_w": w, "page_h": h,
            "status": "ready" if blocks or d["status"] == "ready" else "pending",
            "blocks": [block_to_dict(b) for b in blocks]}


@app.get("/api/traces")
def api_traces(n: int = 20):
    """Recent agent traces (task, grades, retrieval, computation, claims, latency)."""
    from . import obs
    return {"traces": obs.recent(n), "langfuse": obs.enabled()}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "docs": len(documents.DOCS),
            "cloud": documents.ROUTER.available("answer")}


# ── WebSocket: ask ──────────────────────────────────────────────────────────
@app.websocket("/ws/ask/{doc_id}")
async def ws_ask(ws: WebSocket, doc_id: str):
    await ws.accept()
    thread_id = uuid.uuid4().hex                 # one conversation per connection
    try:
        while True:
            msg = await ws.receive_json()
            question = (msg.get("question") or "").strip()
            if not question:
                continue
            ok, reason = guards.check_question(question)
            if not ok:
                await ws.send_json({"type": "error", "error": reason})
                continue
            d = documents.get(doc_id)
            if not d or d["status"] == "error":
                await ws.send_json({"type": "error", "error": "document unavailable"})
                continue
            if d["status"] != "ready":
                await ws.send_json({"type": "error",
                                    "error": "still ingesting — try again in a moment"})
                continue
            engine = documents.get_engine(doc_id)

            def gen(q=question):
                yield from engine.run_streaming(q, thread_id=thread_id)

            async for ev in _bridge(gen):
                await ws.send_json(ev)
    except WebSocketDisconnect:
        return
