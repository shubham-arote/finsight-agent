"""MCP retrieval sidecar: tool handler, live Streamable-HTTP round trip, seam selection."""

import json
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from qdrant_client import QdrantClient

from finsight.config import settings
from finsight.ingestion import ArtifactStore, ingest
from finsight.retrieval import QdrantIndex, make_retriever
from finsight.retrieval.base import Retriever
from finsight.retrieval.mcp_client import MCPRetriever

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samples"))

PORT = 18321


@pytest.fixture(scope="module")
def mcp_main():
    """The MCP server module with its index swapped for a seeded in-memory one."""
    from conftest import keyless_settings
    from make_sample_pdf import build

    from finsight.llm.router import LLMRouter
    from finsight.mcp_server import main as mod

    res = ingest(build(), doc_id="sample", router=LLMRouter(keyless_settings()),
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="mcp", embedder=None)
    idx.index_chunks(res.chunks)
    old = mod.INDEX
    mod.INDEX = idx
    yield mod
    mod.INDEX = old


def test_tool_handler_returns_citation_ready_evidence(mcp_main):
    out = mcp_main.search_document(
        {"query": "What was operating profit in FY26?", "doc_id": "sample", "k": 4})
    hits = json.loads(out[0].text)
    assert hits and hits[0]["page"] in (3, 4)
    assert len(hits[0]["bbox"]) == 4 and hits[0]["parent_text"]


def test_tool_handler_reports_errors_as_payload(mcp_main, monkeypatch):
    monkeypatch.setattr(mcp_main, "INDEX", None)         # force a failure
    out = mcp_main.search_document({"query": "q"})
    assert "error" in json.loads(out[0].text)


@pytest.fixture(scope="module")
def mcp_url(mcp_main):
    """The real Streamable-HTTP server in a background thread."""
    config = uvicorn.Config(mcp_main.app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started, "MCP server failed to start"
    yield f"http://127.0.0.1:{PORT}/mcp"
    server.should_exit = True
    t.join(timeout=5)


def test_mcp_retriever_round_trip_over_http(mcp_url):
    retriever = MCPRetriever(mcp_url, doc_id="sample")
    assert isinstance(retriever, Retriever)              # same seam as in-process
    hits = retriever.retrieve("What was operating profit in FY26?", k=4)
    assert hits and hits[0]["page"] in (3, 4)
    assert hits[0]["parent_text"] and len(hits[0]["bbox"]) == 4


def test_mcp_retriever_scopes_doc_id(mcp_url):
    assert MCPRetriever(mcp_url, doc_id="other-doc").retrieve("revenue", k=3) == []


def test_make_retriever_selects_backend(monkeypatch):
    monkeypatch.setattr(settings, "mcp_server_url", "http://mcp:3000/mcp")
    assert isinstance(make_retriever(None, doc_id="d"), MCPRetriever)
    monkeypatch.setattr(settings, "mcp_server_url", "")
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="sel", embedder=None)
    assert not isinstance(make_retriever(idx, doc_id="d"), MCPRetriever)
