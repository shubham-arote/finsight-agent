"""graph.py — assembly only: wire the node modules into the LangGraph loop + AgentEngine.

    contextualize → supervise → retrieve → grade ─┬─(qa)──────────────→ generate ─→ cite_check → END
                                                  ├─(calc/compare)──→ calculate ─→ generate ┘
                                     rewrite ↺────┴─(weak, attempts < MAX_ATTEMPTS)

The engine depends only on the `Retriever` protocol and the `LLMRouter` roles; nodes live
in `agent/nodes/` (one file each, unit-testable). Conversation memory via a checkpointer
(`CHECKPOINT=memory|sqlite|off`) keyed by thread_id.
"""

from __future__ import annotations

import time
import uuid
from functools import partial
from pathlib import Path
from typing import Iterator

from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..llm import LLMRouter
from ..retrieval.base import Retriever
from . import nodes
from .state import MAX_ATTEMPTS, Deps, RAGState


def _get_checkpointer():
    mode = settings.checkpoint
    if mode == "off":
        return None
    if mode == "sqlite":
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver
            path = settings.checkpoint_db
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        except Exception:
            pass                                       # fall through to memory
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


class AgentEngine:
    def __init__(self, retriever: Retriever, router: LLMRouter | None = None,
                 k: int = 6, checkpointer="default", multi_doc: bool = False):
        self.deps = Deps(router=router or LLMRouter(), retriever=retriever,
                         k=k, multi_doc=multi_doc)
        self.checkpointer = _get_checkpointer() if checkpointer == "default" else checkpointer
        self.graph = self._build()

    @property
    def mode(self) -> str:
        return ("cloud LLM" if self.deps.router.available("answer")
                else "offline (sparse + extractive)")

    def _build(self):
        g = StateGraph(RAGState)
        for name in ("contextualize", "supervise", "retrieve", "grade",
                     "rewrite", "calculate", "generate", "cite_check"):
            g.add_node(name, partial(getattr(nodes, name), deps=self.deps))
        g.add_edge(START, "contextualize")
        g.add_edge("contextualize", "supervise")
        g.add_edge("supervise", "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._route,
                                {"rewrite": "rewrite", "calculate": "calculate",
                                 "generate": "generate"})
        g.add_edge("rewrite", "retrieve")
        g.add_edge("calculate", "generate")
        g.add_edge("generate", "cite_check")
        g.add_edge("cite_check", END)
        return g.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _route(state: RAGState) -> str:
        if state.get("grade") == "relevant":
            return "calculate" if state.get("task") in ("calc", "compare") else "generate"
        if state.get("attempts", 0) >= MAX_ATTEMPTS:
            return "generate"                          # -> abstain path inside generate
        return "rewrite"

    # ── run ─────────────────────────────────────────────────────────────────
    def run_streaming(self, question: str, thread_id: str | None = None) -> Iterator[dict]:
        init: RAGState = {"user_question": question, "question": question,
                          "original_question": question, "attempts": 0}
        config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
        yield {"type": "agent_start", "question": question, "mode": self.mode}
        t0 = time.time()
        for update in self.graph.stream(init, config=config, stream_mode="updates"):
            for node, delta in update.items():
                ev = self._event(node, delta or {})
                if ev:
                    yield ev
        yield {"type": "agent_done", "latency_s": round(time.time() - t0, 2)}

    def run(self, question: str, thread_id: str | None = None) -> dict:
        config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
        return self.graph.invoke(
            {"user_question": question, "question": question,
             "original_question": question, "attempts": 0}, config=config)

    @staticmethod
    def _event(node: str, d: dict) -> dict | None:
        if node == "supervise":
            return {"type": "agent_node", "node": "supervise", "task": d.get("task")}
        if node == "retrieve":
            return {"type": "agent_node", "node": "retrieve", "attempt": d.get("attempts"),
                    "k": len(d.get("retrieved", []))}
        if node == "grade":
            return {"type": "agent_node", "node": "grade", "verdict": d.get("grade")}
        if node == "rewrite":
            return {"type": "agent_node", "node": "rewrite", "query": d.get("question")}
        if node == "calculate":
            comp = d.get("computation")
            return ({"type": "agent_node", "node": "calculate",
                     "expr": comp["expr"], "result": comp["result"]} if comp else None)
        if node == "generate":
            return {"type": "agent_answer", "answer": d.get("answer", ""),
                    "claims": d.get("claims", []), "sources": d.get("sources", [])}
        if node == "cite_check":
            return {"type": "agent_node", "node": "cite_check",
                    "unverified": d.get("unverified", []),
                    "answer": d.get("answer")}          # set only when a caveat was added
        return None


# ── CLI demo: python -m finsight.agent.graph  (sample report, offline-capable) ──
if __name__ == "__main__":
    import sys

    from qdrant_client import QdrantClient

    from ..ingestion import ArtifactStore, ingest
    from ..retrieval import HybridRetriever, QdrantIndex
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "samples"))
    from make_sample_pdf import build

    res = ingest(build(), doc_id="sample", store=ArtifactStore(":memory:"), contextual=False)
    idx = QdrantIndex(client=QdrantClient(":memory:"), collection="demo")
    idx.index_chunks(res.chunks)
    engine = AgentEngine(HybridRetriever(idx, doc_id="sample"))
    print("mode:", engine.mode)
    for q in ["What was operating profit in FY26?",
              "By how much did operating profit change year on year, in percent?",
              "Who is the chief executive officer?"]:
        print(f"\n=== Q: {q}")
        for ev in engine.run_streaming(q):
            if ev["type"] == "agent_node":
                print("  ", {k: v for k, v in ev.items() if k != "type"})
            elif ev["type"] == "agent_answer":
                print("  ANSWER:", ev["answer"][:220].replace("\n", " "))
                for cl in ev["claims"]:
                    print("    claim:", cl["text"][:80], "->", cl["citations"])
