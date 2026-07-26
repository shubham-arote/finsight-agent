"""The autonomous analyst (brief) lane: intent, planning, the run loop, composition."""

import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from finsight.agent import AgentEngine, is_brief_request, run_brief
from finsight.agent.brief import compose_markdown
from finsight.ingestion import ArtifactStore, ingest
from finsight.retrieval import HybridRetriever, QdrantIndex

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samples"))


def test_brief_intent_detection():
    assert is_brief_request("analyze this filing")
    assert is_brief_request("give me a brief")
    assert is_brief_request("summarize the key metrics")
    assert is_brief_request("what are the highlights?")
    # a specific question is NOT a brief request
    assert not is_brief_request("What was total revenue in FY26?")
    assert not is_brief_request("By how much did operating profit grow year on year?")


@pytest.fixture(scope="module")
def engine(keyless_router_module):
    from make_sample_pdf import build
    res = ingest(build(), doc_id="sample", router=keyless_router_module,
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="brief", embedder=None)
    idx.index_chunks(res.chunks)
    return AgentEngine(HybridRetriever(idx, doc_id="sample"), router=keyless_router_module)


@pytest.fixture(scope="module")
def keyless_router_module():
    from conftest import keyless_settings

    from finsight.llm.router import LLMRouter
    return LLMRouter(keyless_settings())


def test_run_brief_streams_plan_then_sections_then_brief(engine):
    events = list(run_brief(engine, doc_label="ACME PLC Annual Report"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "brief_start" and kinds[-1] == "brief_done"
    start = events[0]
    assert start["sections"] and "Revenue" in start["sections"]
    # one autonomous agent run per checklist item
    steps = [e for e in events if e["type"] == "brief_step"]
    sections = [e for e in events if e["type"] == "brief_section"]
    assert len(steps) == len(start["sections"]) == len(sections)


def test_brief_answers_from_the_document_and_marks_gaps(engine):
    done = list(run_brief(engine, doc_label="ACME PLC"))[-1]
    assert done["total"] == 7
    # the sample report states revenue + operating profit -> answered from the doc
    by_head = {s["heading"]: s for s in done["sections"]}
    assert by_head["Revenue"]["status"] == "answered"
    assert any(c for cl in by_head["Revenue"]["claims"] for c in cl["citations"])
    # a metric the sample doesn't disclose is marked, never fabricated
    assert done["answered"] < done["total"]              # some honest "not disclosed"


def test_compose_markdown_is_cited_and_honest():
    sections = [
        {"heading": "Revenue", "status": "answered", "verified": True, "computed": False,
         "answer": "Revenue was 6,303 million pounds.",
         "claims": [{"text": "Revenue was 6,303.", "citations": [{"page": 4, "block_id": 0}]}]},
        {"heading": "Outlook", "status": "not_disclosed", "verified": True, "computed": False,
         "answer": "", "claims": []},
    ]
    md = compose_markdown(sections, "ACME PLC")
    assert "# Analyst brief — ACME PLC" in md
    assert "[p4·b0]" in md                               # citation marker survives export
    assert "Not disclosed" in md                         # gap shown honestly
    assert "1 of 2 items answered" in md
