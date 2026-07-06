"""Retrieval stack: query typing, exact lookup, rerank fallback, hybrid fusion, protocol."""

import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from finsight.ingestion import ArtifactStore, ingest
from finsight.retrieval import (HybridRetriever, QdrantIndex, Retriever,
                                classify, lookup_terms, make_retriever)
from finsight.retrieval.rerank import _rerank_heuristic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samples"))


@pytest.fixture(scope="module")
def sample_retriever(keyless_router_module):
    from make_sample_pdf import build

    res = ingest(build(), doc_id="sample", router=keyless_router_module,
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="t2", embedder=None)
    idx.index_chunks(res.chunks)
    return HybridRetriever(idx, doc_id="sample")


@pytest.fixture(scope="module")
def keyless_router_module():
    from conftest import keyless_settings
    from finsight.llm.router import LLMRouter
    return LLMRouter(keyless_settings())


# ── query analysis ──────────────────────────────────────────────────────────
def test_classify_routes_query_types():
    assert classify("What was revenue in FY26?") == "lookup"
    assert classify("Why did margins decline?") == "diagnostic"
    assert classify("How do you calculate the ratio?") == "procedure"
    assert classify("Operating profit vs prior year") == "comparison"
    assert classify("Tell me something") == "general"


def test_lookup_terms_extracts_values_and_periods():
    assert lookup_terms("Was revenue 6,303 in FY26?") == ["6,303", "FY26"]
    assert lookup_terms("what about profitability?") == []


# ── rerank fallback ─────────────────────────────────────────────────────────
def test_heuristic_rerank_boosts_exact_numeric_and_table():
    cands = [
        {"chunk_id": "a", "type": "text", "heading": "", "content": "general prose", "score": 1.0},
        {"chunk_id": "b", "type": "table", "heading": "Income Statement",
         "content": "| Revenue | 6,303 | 5,952 |", "score": 0.5},
    ]
    ranked = _rerank_heuristic("What was revenue 6,303 in FY26?", cands)
    assert ranked[0]["chunk_id"] == "b"          # numeric+table beats higher base score


# ── the full keyless stack over the sample report ───────────────────────────
def test_retrieve_returns_citation_ready_evidence(sample_retriever):
    evs = sample_retriever.retrieve("What was operating profit in FY26?", k=5)
    assert evs
    assert isinstance(sample_retriever, Retriever)          # protocol satisfied
    top = evs[0]
    assert top["page"] in (3, 4)
    assert top["block_id"] is not None and len(top["bbox"]) == 4
    assert top["parent_text"]                                # small-to-big context attached


def test_exact_value_query_floats_deterministic_hit(sample_retriever):
    evs = sample_retriever.retrieve("Where does the figure 6,303 appear?", k=5)
    assert evs[0]["exact"] is True
    assert "6,303" in evs[0]["content"]
    assert evs[0]["page"] in (3, 4)


def test_make_retriever_scopes_doc(sample_retriever):
    scoped = make_retriever(sample_retriever.index, doc_id="other-doc")
    assert scoped.retrieve("revenue", k=3) == []


# ── dense fusion path (mock embedder — no keys, no network) ─────────────────
class MockEmbedder:
    """Maps texts mentioning 'dividend' near vec-A; the query embeds as vec-A too."""
    dim = 4

    def _vec(self, text):
        return [1.0, 0.0, 0.0, 0.0] if "dividend" in text.lower() else [0.0, 1.0, 0.0, 0.0]

    def embed_docs(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0, 0.0]


def test_dense_signal_contributes_to_fusion(keyless_router_module):
    from make_sample_pdf import build

    res = ingest(build(), doc_id="sample", router=keyless_router_module,
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="t3",
                      embedder=MockEmbedder())
    idx.index_chunks(res.chunks)
    # query shares no useful tokens with the dividend paragraph, but embeds next to it
    evs = HybridRetriever(idx, doc_id="sample").retrieve("shareholder payout policy", k=5)
    assert any("dividend" in e["content"].lower() for e in evs)


# ── tier-1 gate: retrieval baseline must not regress ────────────────────────
def test_baseline_hit_rate_above_floor(sample_retriever):
    from evals.retrieval_baseline import evaluate

    m = evaluate(sample_retriever, k=5)
    assert m["hit_rate"] >= 0.90, f"retrieval regressed: hit@5={m['hit_rate']:.0%}"
    assert m["mrr"] >= 0.70, f"retrieval regressed: MRR={m['mrr']:.3f}"
