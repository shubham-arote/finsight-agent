"""Pub/Sub ingest worker: envelope parsing, ack/retry semantics, end-to-end emulation."""

import base64
import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from finsight.ingestion import ArtifactStore
from finsight.retrieval import QdrantIndex


@pytest.fixture()
def worker(monkeypatch, sample_pdf_bytes, keyless_router):
    from finsight import ingest_worker as mod
    monkeypatch.setattr(mod, "STORE", ArtifactStore(":memory:"))
    monkeypatch.setattr(mod, "INDEX", QdrantIndex(client=QdrantClient(":memory:"),
                                                  collection="ingest", embedder=None))
    monkeypatch.setattr(mod, "ROUTER", keyless_router)
    monkeypatch.setattr(mod, "download_gcs",
                        lambda bucket, name: sample_pdf_bytes)   # fake GCS
    return mod


def _envelope_attrs(bucket="uploads", name="report.pdf"):
    return {"message": {"attributes": {"bucketId": bucket, "objectId": name}}}


def test_push_ingests_and_indexes(worker):
    client = TestClient(worker.app)
    r = client.post("/", json=_envelope_attrs())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["pages"] == 4 and body["chunks"] > 0
    hits = worker.INDEX.search("operating profit 1,052", k=3)
    assert hits and hits[0]["page"] == 3                  # searchable in the shared index


def test_data_json_fallback_envelope(worker):
    data = base64.b64encode(json.dumps(
        {"bucket": "uploads", "name": "report.pdf"}).encode()).decode()
    r = TestClient(worker.app).post("/", json={"message": {"data": data}})
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_non_pdf_is_acked_not_retried(worker):
    r = TestClient(worker.app).post("/", json=_envelope_attrs(name="notes.txt"))
    assert r.status_code == 200 and r.json()["status"] == "skipped"


def test_bad_envelope_is_rejected(worker):
    assert TestClient(worker.app).post("/", json={"message": {}}).status_code == 400


def test_download_failure_returns_5xx_for_redelivery(worker, monkeypatch):
    def boom(bucket, name):
        raise ConnectionError("gcs unreachable")
    monkeypatch.setattr(worker, "download_gcs", boom)
    r = TestClient(worker.app).post("/", json=_envelope_attrs())
    assert r.status_code == 500                           # Pub/Sub will redeliver
