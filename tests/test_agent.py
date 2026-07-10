"""The agent graph end-to-end (offline + scripted-LLM), node routing, citation contract."""

import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from finsight.agent import AgentEngine
from finsight.agent.citations import parse_structured, validate_citations
from finsight.agent.nodes import supervise
from finsight.agent.state import Deps
from finsight.ingestion import ArtifactStore, ingest
from finsight.retrieval import HybridRetriever, QdrantIndex

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samples"))


class ScriptedRouter:
    """Deterministic stand-in for LLMRouter: first prompt-marker match wins."""

    def __init__(self, script):
        self.script = script                 # list[(marker_in_prompt, response)]
        self.calls = []

    def available(self, role):
        return True

    def complete(self, role, prompt, system=None, **kw):
        self.calls.append(role)
        for marker, resp in self.script:
            if marker in prompt:
                return resp(prompt) if callable(resp) else resp
        return ""


@pytest.fixture(scope="module")
def retriever(keyless_router_mod):
    from make_sample_pdf import build
    res = ingest(build(), doc_id="sample", router=keyless_router_mod,
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="agent", embedder=None)
    idx.index_chunks(res.chunks)
    return HybridRetriever(idx, doc_id="sample")


@pytest.fixture(scope="module")
def keyless_router_mod():
    from conftest import keyless_settings
    from finsight.llm.router import LLMRouter
    return LLMRouter(keyless_settings())


# ── supervisor routing ──────────────────────────────────────────────────────
def test_supervisor_routes_lanes(keyless_router_mod):
    deps = Deps(router=keyless_router_mod, retriever=None)
    assert supervise({"question": "By how much did profit grow year on year?"}, deps)["task"] == "calc"
    assert supervise({"question": "What was revenue in FY26?"}, deps)["task"] == "qa"


# ── offline (keyless) e2e: extractive but still cited + verified ────────────
def test_offline_qa_answers_with_valid_citations(retriever, keyless_router_mod):
    engine = AgentEngine(retriever, router=keyless_router_mod)
    out = engine.run("What was operating profit in FY26?")
    assert "extractive" in out["answer"]
    assert out["claims"], "offline answers still carry structured citations"
    pages = {c["page"] for c in out["retrieved"]}
    for cl in out["claims"]:
        assert cl["verified"] is True
        assert all(ct["page"] in pages for ct in cl["citations"])
    assert out["unverified"] == []
    assert 1 <= len(out["sources"]) <= 3


def test_offline_abstains_on_out_of_scope(retriever, keyless_router_mod):
    engine = AgentEngine(retriever, router=keyless_router_mod)
    out = engine.run("Who is the chief executive officer?")
    assert out["answer"].startswith("I couldn't find")
    assert out["claims"] == [] and out["sources"] == []
    assert out["attempts"] == 3                        # rewrite budget exhausted


# ── calc lane with a scripted LLM: exact, verified computation ──────────────
def test_calc_lane_computes_exactly_and_verifies(retriever):
    router = ScriptedRouter([
        ("SINGLE arithmetic expression", "(1052-985)/985*100"),
        ("Respond with ONLY a JSON object",
         '{"answer": "Operating profit increased by 6.80% year on year.",'
         ' "claims": [{"text": "Operating profit increased by 6.80% year on year.",'
         '             "citations": [{"page": 4, "block_id": null}]}],'
         ' "insufficient": false}'),
    ])
    engine = AgentEngine(retriever, router=router)
    events = list(engine.run_streaming(
        "By how much did operating profit change year on year, in percent?"))
    tasks = [e for e in events if e.get("node") == "supervise"]
    assert tasks[0]["task"] == "calc"
    calc = next(e for e in events if e.get("node") == "calculate")
    assert calc["result"] == pytest.approx(6.8020, abs=1e-3)
    answer = next(e for e in events if e["type"] == "agent_answer")
    assert "6.80%" in answer["answer"]
    check = next(e for e in events if e.get("node") == "cite_check")
    assert check["unverified"] == []                   # 6.80 traces to the computation
    assert answer["claims"][0]["citations"] == [{"page": 4, "block_id": None}]


# ── cite_check catches a planted (hallucinated) figure ──────────────────────
def test_cite_check_flags_hallucinated_figure(retriever):
    router = ScriptedRouter([
        ("relevant or weak", "relevant"),
        ("Respond with ONLY a JSON object",
         '{"answer": "Revenue was 9,999 million pounds.",'
         ' "claims": [{"text": "Revenue was 9,999 million pounds.",'
         '             "citations": [{"page": 4, "block_id": null}]}],'
         ' "insufficient": false}'),
    ])
    engine = AgentEngine(retriever, router=router)
    out = engine.run("What was revenue in FY26?")
    assert "9999" in out["unverified"]
    assert "treat with caution" in out["answer"]       # transparent caveat, not silent
    assert out["claims"][0]["verified"] is False


# ── unparseable model output degrades to prose + number check ───────────────
def test_unparseable_json_falls_back_to_prose(retriever):
    router = ScriptedRouter([
        ("relevant or weak", "relevant"),
        ("Respond with ONLY a JSON object", "Revenue was 6,303 million pounds."),
    ])
    engine = AgentEngine(retriever, router=router)
    out = engine.run("What was revenue in FY26?")
    assert out["answer"].startswith("Revenue was 6,303")
    assert out["claims"] == []                         # no structured claims
    assert out["unverified"] == []                     # 6,303 is in the retrieved context


# ── citation contract units ─────────────────────────────────────────────────
def test_parse_structured_handles_fences_and_garbage():
    ok = parse_structured('```json\n{"answer": "A", "claims": []}\n```')
    assert ok and ok["answer"] == "A"
    assert parse_structured("not json at all") is None
    assert parse_structured('{"claims": []}') is None  # answer missing


def test_citations_snap_to_the_figure_bearing_block():
    """Model cites the section's tag id; the figure lives in a sibling block — the
    citation must snap to the block that contains it (found live on a real filing)."""
    from finsight.agent.citations import snap_citations
    retrieved = [
        # the year-token trap (found live): wrong block contains "2023" too — a year
        # must not anchor the citation when a real value block exists
        {"page": 3, "block_id": 27, "content": "we expect 2023 growth to reach mid-teens"},
        {"page": 3, "block_id": 7, "content": "Total revenues of $40.8 million, up 22%"},
    ]
    claims = [{"text": "Revenue in 2023 was $40.8 million.",
               "citations": [{"page": 3, "block_id": 27}]}]
    out = snap_citations(claims, retrieved)
    assert out[0]["citations"] == [{"page": 3, "block_id": 7}]
    # no retrieved block carries the figure -> degrade to page-level, never lie
    claims2 = [{"text": "EPS was $0.19.", "citations": [{"page": 3, "block_id": 27}]}]
    assert snap_citations(claims2, retrieved)[0]["citations"] == [{"page": 3, "block_id": None}]
    # figure already in the cited block -> unchanged
    claims3 = [{"text": "Revenue was $40.8 million.", "citations": [{"page": 3, "block_id": 7}]}]
    assert snap_citations(claims3, retrieved)[0]["citations"] == [{"page": 3, "block_id": 7}]


def test_bare_number_is_not_a_computation():
    from finsight.agent import extract_expression
    assert extract_expression("40.8") is None            # lookup, not math
    assert extract_expression("-40.8") is None
    assert extract_expression("(1052-985)/985*100") is not None


def test_invented_citations_are_dropped():
    retrieved = [{"page": 4, "block_id": 2}]
    claims = [{"text": "x is 1", "citations": [
        {"page": 4, "block_id": 2},      # real
        {"page": 99, "block_id": 1},     # invented page
        {"page": 4, "block_id": 77},     # invented block
    ]}]
    out = validate_citations(claims, retrieved)
    assert out[0]["citations"] == [{"page": 4, "block_id": 2}]
