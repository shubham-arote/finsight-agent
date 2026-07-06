"""End-to-end (keyless): synthetic PDF -> ingest -> Qdrant (:memory:) -> search hits the
right page with a citation-ready payload."""

from qdrant_client import QdrantClient

from finsight.ingestion import ArtifactStore, ingest
from finsight.retrieval import QdrantIndex


def _index(res):
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="test", embedder=None)
    assert idx.index_chunks(res.chunks) == len(res.chunks)
    return idx


def test_ingest_then_search_finds_the_right_page(sample_pdf_bytes, keyless_router):
    res = ingest(sample_pdf_bytes, router=keyless_router,
                 store=ArtifactStore(":memory:"), contextual=False)
    assert res.parser == "textlayer" and res.page_count == 4
    assert res.stats["chunks"] > 0 and res.contextualized == 0    # keyless: plain chunks

    idx = _index(res)
    hits = idx.search("operating profit 1,052 million", k=3)
    assert hits and hits[0]["page"] == 3                          # the fact lives on page 3
    # citation-ready payload
    assert hits[0]["block_id"] is not None and len(hits[0]["bbox"]) == 4
    assert "1,052" in hits[0]["content"]

    hits2 = idx.search("dividend per share recommended", k=3)
    assert hits2 and hits2[0]["page"] == 4


def test_doc_id_filter_scopes_search(sample_pdf_bytes, keyless_router):
    res = ingest(sample_pdf_bytes, doc_id="docA", router=keyless_router,
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = _index(res)
    assert idx.search("revenue", k=3, doc_id="docA")
    assert idx.search("revenue", k=3, doc_id="docB") == []
