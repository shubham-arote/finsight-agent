"""documents.py — document lifecycle service.

Owns the in-memory registry (`DOCS`), the shared artifact store, and a per-document
Qdrant index (one collection per document; retrievers are doc_id-scoped). Upload -> ingest
in a background thread with page-level progress; uploads are content-addressed (same
bytes = same doc, ingestion is idempotent) and persisted, so a restart re-ingests from
the page cache in seconds and re-indexes.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import fitz

from ..config import settings
from ..ingestion import ArtifactStore, IngestError, doc_hash, ingest
from ..llm import LLMRouter
from ..retrieval import QdrantIndex

DOCS: dict[str, dict] = {}      # doc_id -> {name, status, page_count, parsed_pages, ...}

STORE = ArtifactStore()
# One Qdrant collection PER DOCUMENT (see doc_index) — sharing one collection
# corrupts the in-process sparse index as the vocabulary grows.
ROUTER = LLMRouter()


def get(doc_id: str) -> dict | None:
    return DOCS.get(doc_id)


def summary(doc_id: str) -> dict:
    d = DOCS[doc_id]
    return {"doc_id": doc_id, "name": d["name"], "status": d["status"],
            "page_count": d["page_count"], "parsed_pages": d["parsed_pages"],
            "chunks": d.get("chunks", 0), "parser": d.get("parser"),
            "error": d.get("error")}


def add_document(data: bytes, name: str, background: bool = True) -> dict:
    """Register + ingest a PDF. Content-addressed: re-uploading the same file returns
    the existing doc instead of re-ingesting."""
    doc_id = doc_hash(data)
    if doc_id in DOCS and DOCS[doc_id]["status"] in ("ready", "parsing"):
        return summary(doc_id)
    fdoc = fitz.open(stream=data, filetype="pdf")
    n = len(fdoc)
    DOCS[doc_id] = {"name": name, "status": "parsing", "page_count": n, "parsed_pages": 0,
                    "fitz": fdoc, "sizes": [(fdoc[i].rect.width, fdoc[i].rect.height)
                                            for i in range(n)],
                    "pages_blocks": None, "chunks": 0, "parser": None,
                    "error": None, "engine": None}
    STORE.save_blob(f"doc:{doc_id}:pdf", data)
    STORE.save_text(f"doc:{doc_id}:name", name)

    if background:
        threading.Thread(target=_ingest_worker, args=(doc_id, data), daemon=True).start()
    else:
        _ingest_worker(doc_id, data)
    return summary(doc_id)


def _ingest_worker(doc_id: str, data: bytes) -> None:
    d = DOCS[doc_id]

    def on_page(done: int, total: int) -> None:
        d["parsed_pages"] = done

    try:
        res = ingest(data, doc_id=doc_id, router=ROUTER, store=STORE, on_page=on_page)
        doc_index(doc_id).index_chunks(res.chunks)
        d.update(status="ready", parser=res.parser, chunks=len(res.chunks),
                 pages_blocks=res.pages_blocks, parsed_pages=res.page_count)
    except IngestError as e:
        d.update(status="error", error=str(e))
    except Exception as e:                              # never leave a doc stuck "parsing"
        d.update(status="error", error=f"{type(e).__name__}: {e}")


def doc_index(doc_id: str) -> QdrantIndex:
    """One collection per document.

    Retrieval in the app is always doc-scoped, so documents never need to share a
    collection — and sharing one is actively harmful in Qdrant's in-process mode, where
    the sparse index is sized to the vocabulary it was built with. Adding a second
    document grew it past that bound and corrupted the collection mid-demo
    ("index N is out of bounds for axis 0 with size N"). Per-document collections keep
    each index immutable once built. Corpus-wide work (the benchmark) still builds its
    own multi-document index explicitly.
    """
    d = DOCS[doc_id]
    if d.get("index") is None:
        d["index"] = QdrantIndex(collection=f"{settings.qdrant_collection}_{doc_id}")
    return d["index"]


def get_engine(doc_id: str):
    """The agent for one document, built once per doc (retriever is doc-scoped;
    in-process or the MCP sidecar depending on MCP_SERVER_URL)."""
    from ..agent import AgentEngine
    from ..retrieval import make_retriever
    d = DOCS[doc_id]
    if d.get("engine") is None:
        d["engine"] = AgentEngine(make_retriever(doc_index(doc_id), doc_id=doc_id),
                                  router=ROUTER)
    return d["engine"]


def sample_pdf_bytes() -> bytes:
    """The bundled 6-page synthetic annual report (samples/make_sample_pdf.py)."""
    samples = Path(__file__).resolve().parents[3] / "samples"
    if str(samples) not in sys.path:
        sys.path.insert(0, str(samples))
    from make_sample_pdf import build
    return build()


def load_persisted() -> int:
    """Restart recovery: re-ingest persisted uploads (page cache makes this seconds)."""
    n = 0
    for key in STORE.blob_keys("doc:"):
        doc_id = key.split(":")[1]
        if doc_id in DOCS:
            continue
        data = STORE.get_blob(key)
        if not data:
            continue
        name = STORE.get_text(f"doc:{doc_id}:name") or "document.pdf"
        try:
            add_document(data, name, background=True)
            n += 1
        except Exception:
            continue
    return n
