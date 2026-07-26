"""obs.py — observability: JSONL traces always, Langfuse when keyed (week-5 lab-1 pattern).

Every answered question logs one structured JSONL trace (query, task, grades, rewrites,
retrieval pages+scores, computation, claims/unverified, prompt versions, latency) —
readable offline, served at GET /api/traces.

With LANGFUSE_PUBLIC_KEY/SECRET_KEY set, two extra sinks light up:
  * LiteLLM success-callback -> every LLM call with tokens/cost/latency per model
  * a LangChain CallbackHandler on the graph -> one span per node (the trace tree)
Both are optional and fail-silent: observability must never take the agent down.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .config import settings

_LOCK = threading.Lock()
_LANGFUSE_READY: bool | None = None


def enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


# ── JSONL traces (always on) ────────────────────────────────────────────────
def log_trace(trace: dict) -> None:
    try:
        path = Path(settings.traces_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass                                             # never fail the request over a trace


def recent(n: int = 20) -> list[dict]:
    path = Path(settings.traces_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ── Langfuse (optional) ─────────────────────────────────────────────────────
def _setup_langfuse() -> bool:
    """Idempotent: export creds for the SDK + register the LiteLLM cost callback."""
    global _LANGFUSE_READY
    if _LANGFUSE_READY is not None:
        return _LANGFUSE_READY
    if not enabled():
        _LANGFUSE_READY = False
        return False
    try:
        # the Langfuse SDK + LiteLLM's callback read these env vars directly
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
        import litellm
        # "langfuse_otel", not "langfuse": the plain callback targets the v2 SDK and
        # crash-loops against langfuse>=3 (module has no attribute 'version')
        if "langfuse_otel" not in (litellm.success_callback or []):
            litellm.success_callback = [*(litellm.success_callback or []), "langfuse_otel"]
        _LANGFUSE_READY = True
    except Exception:
        _LANGFUSE_READY = False
    return _LANGFUSE_READY


def graph_callbacks() -> list:
    """LangChain callbacks for the agent graph — a span per node in Langfuse."""
    if not _setup_langfuse():
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []
