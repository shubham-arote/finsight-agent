"""contextualize — resolve a follow-up into a standalone question using the conversation.

Memory only rewrites the *query*; answers still come from retrieved document context,
so grounding/faithfulness are unchanged. Passthrough on the first turn or keyless.
"""

from __future__ import annotations

from ...llm import LLMUnavailable, prompts
from ..state import Deps, RAGState


def contextualize(state: RAGState, deps: Deps) -> dict:
    history = state.get("history") or []
    if not history or not deps.router.available("fast"):
        return {}
    convo = "\n".join(f"User: {h['q']}\nAssistant: {h['a'][:200]}" for h in history[-4:])
    p = prompts.get("contextualize")
    try:
        standalone = deps.router.complete(
            "fast", p.render(conversation=convo, user_question=state["user_question"]),
            max_tokens=64).strip()
    except LLMUnavailable:
        return {}
    if not standalone:
        return {}
    return {"question": standalone, "original_question": standalone}
