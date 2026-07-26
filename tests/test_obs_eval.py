"""Observability (JSONL traces, optional Langfuse) + the full-agent eval gate."""


from types import SimpleNamespace

import litellm

from finsight import obs
from finsight.config import settings
from finsight.llm.router import LLMRouter


# ── Vertex AI routing (the scale path) ──────────────────────────────────────
def _vertex_settings(**over):
    from conftest import keyless_settings
    return keyless_settings(llm_judge="vertex_ai/gemini-2.5-flash", **over)


def test_vertex_models_authenticate_via_adc_project(monkeypatch):
    seen = {}

    def fake(model, messages, **kw):
        seen.update(kw, model=model)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setattr(litellm, "completion", fake)
    router = LLMRouter(_vertex_settings(google_cloud_project="my-proj"))
    assert router.available("judge")
    assert router.label("judge") == "vertex_ai/gemini-2.5-flash"
    assert router.complete("judge", "q") == "ok"
    assert seen["vertex_project"] == "my-proj"
    assert seen["vertex_location"] == "us-central1"
    assert "api_key" not in seen                       # ADC, not a key


def test_vertex_without_project_is_skipped():
    router = LLMRouter(_vertex_settings(google_cloud_project=""))
    assert not router.available("judge")


# ── JSONL traces ────────────────────────────────────────────────────────────
def test_trace_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "traces_path", str(tmp_path / "t.jsonl"))
    obs.log_trace({"question": "q1", "task": "qa"})
    obs.log_trace({"question": "q2", "task": "calc"})
    got = obs.recent(5)
    assert [t["question"] for t in got] == ["q2", "q1"]   # latest first


def test_engine_run_writes_a_trace(monkeypatch, tmp_path, sample_pdf_bytes, keyless_router):
    from qdrant_client import QdrantClient

    from finsight.agent import AgentEngine
    from finsight.ingestion import ArtifactStore, ingest
    from finsight.retrieval import HybridRetriever, QdrantIndex

    monkeypatch.setattr(settings, "traces_path", str(tmp_path / "traces.jsonl"))
    res = ingest(sample_pdf_bytes, doc_id="d", router=keyless_router,
                 store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="obs", embedder=None)
    idx.index_chunks(res.chunks)
    engine = AgentEngine(HybridRetriever(idx, doc_id="d"), router=keyless_router)
    list(engine.run_streaming("What was operating profit in FY26?"))

    trace = obs.recent(1)[0]
    assert trace["question"].startswith("What was operating")
    assert trace["task"] == "qa" and trace["grades"] == ["relevant"]
    assert trace["retrieved"] and "latency_s" in trace
    # the trace must record the version that actually ran — compared against the
    # registry, not a frozen literal, so improving a prompt doesn't fail the suite
    from finsight.llm import prompts
    assert trace["prompt_versions"]["generate_answer"] == prompts.get("generate_answer").id


def test_langfuse_off_means_no_callbacks_and_no_crash():
    assert obs.graph_callbacks() == [] or settings.langfuse_public_key


# ── full-agent eval gate (deterministic, offline) ───────────────────────────
def test_agent_eval_deterministic_floor(keyless_router, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "traces_path", str(tmp_path / "t.jsonl"))
    from evals.run_agent_eval import build_engine, evaluate

    engine = build_engine(router=keyless_router)
    m = evaluate(engine)
    # Offline floor is 3/4: lexical grading can't detect topic-adjacent OOS ("share
    # price" overlaps "per share"/"full price" text and routes to the calc lane).
    # The cloud path abstains via the structured `insufficient` flag — measured
    # separately when keys are configured.
    assert m["abstain_accuracy"] >= 0.75, "OOS questions must be refused"
    assert m["citation_hit"] >= 0.80, f"citation hit regressed: {m['citation_hit']:.0%}"
    assert m["verified_rate"] == 1.0, "no answer may carry unverified figures offline"
    assert m["claim_coverage"] == 1.0, "every answer must carry structured citations"
    assert m["correctness"] is None                    # keyless: judge unavailable
