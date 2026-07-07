"""FinRAGBench-V adapter + harness — offline, against a fixture mimicking the schema."""

import json

import pytest
from qdrant_client import QdrantClient

from evals.benchmarks.finragbench_v import load_cases, pdf_bytes
from evals.run_benchmark import build_corpus, evaluate
from finsight.agent import AgentEngine
from finsight.ingestion import ArtifactStore
from finsight.retrieval import HybridRetriever, QdrantIndex


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, sample_pdf_bytes):
    """Fixture data dir in the dataset's layout: queries_en.json + pdfs/."""
    root = tmp_path_factory.mktemp("finrag")
    (root / "queries").mkdir()
    (root / "pdfs").mkdir()
    (root / "pdfs" / "acme_report.pdf").write_bytes(sample_pdf_bytes)
    queries = [                              # real dataset id format: <stem>_<page>.png-<n>
        {"query-id": "acme_report_3.png-1", "query": "What was operating profit in FY26?",
         "answer": "1,052 million", "category": "Table-Information Extraction",
         "answer_type": "short", "from_pages": [3]},
        {"query-id": "acme_report_1.png-2", "query": "What was revenue for the year?",
         "answer": "6,303 million", "category": "Text Inference",
         "answer_type": "short", "from_pages": 1},                    # int form
        {"query-id": "acme_report_multipage_1-4.png-3", "query": "Multipage form parses",
         "answer": "x", "category": "Text-MultiPage", "answer_type": "short",
         "from_pages": [1, 4]},
        {"query-id": "missing_doc_2.png-1", "query": "Should be dropped",
         "answer": "x", "category": "Text Inference", "answer_type": "short",
         "from_pages": [2]},                                          # pdf not present
        {"query-id": "malformed id", "query": "Malformed id",
         "answer": "x", "category": "Text Inference", "answer_type": "short"},
    ]
    (root / "queries" / "queries_en.json").write_text(json.dumps(queries), encoding="utf-8")
    return root


def test_load_cases_parses_and_filters(data_dir):
    cases = load_cases(data_dir)
    assert [c.qid for c in cases] == ["acme_report_3.png-1", "acme_report_1.png-2",
                                      "acme_report_multipage_1-4.png-3"]
    assert cases[0].doc_name == "acme_report.pdf" and cases[0].gold_pages == [3]
    assert cases[1].gold_pages == [1]                       # int normalized to list
    assert cases[2].gold_pages == [1, 4]                    # multipage form


def test_sampling_is_seeded(data_dir):
    a = load_cases(data_dir, sample=1, seed=7)
    b = load_cases(data_dir, sample=1, seed=7)
    assert [c.qid for c in a] == [c.qid for c in b] and len(a) == 1


def test_category_filter(data_dir):
    cases = load_cases(data_dir, categories=["Text Inference"])
    assert [c.category for c in cases] == ["Text Inference"]


def test_harness_end_to_end_offline(data_dir, keyless_router, monkeypatch, tmp_path):
    from finsight.config import settings
    monkeypatch.setattr(settings, "traces_path", str(tmp_path / "t.jsonl"))
    cases = load_cases(data_dir)
    store = ArtifactStore(":memory:")
    index = QdrantIndex(client=QdrantClient(":memory:"), collection="bench", embedder=None)
    corpus = build_corpus(str(data_dir), cases, keyless_router, store, index,
                          contextual=False, enrich=False)
    assert corpus["docs"] == 1 and corpus["pages"] == 4 and corpus["chunks"] > 0
    assert pdf_bytes(data_dir, "acme_report.pdf")

    engine = AgentEngine(HybridRetriever(index), router=keyless_router)
    m = evaluate(cases, engine, keyless_router, k=5)
    assert m["n"] == 3
    assert m["hit_at_k"] >= 0.5                           # gold page retrieved
    assert m["correctness"] is None                       # keyless: no judge
    assert m["verified_rate"] == 1.0                      # extractive claims all verified
    # citation resolution produced page-level citations for at least one answer
    assert any(r["cite_p"] is not None for r in m["rows"])
