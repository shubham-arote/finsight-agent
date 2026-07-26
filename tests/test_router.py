"""Router behaviour under provider failure — the Phase 0 gate.

These tests monkeypatch `litellm.completion`; no network, no keys.
"""

from types import SimpleNamespace

import litellm
import pytest

from finsight.config import Settings
from finsight.llm.router import LLMRouter, LLMUnavailable


def _settings(**over) -> Settings:
    base = {
        "groq_api_key": "k-groq", "gemini_api_key": "k-gemini",
        "openrouter_api_key": "", "cohere_api_key": "",
        "llm_fast": "groq/m1,gemini/m2",
        "llm_answer": "groq/m1,gemini/m2",
        "llm_vision": "gemini/m2",
        "llm_judge": "gemini/m2",
        "llm_cooldown_s": 60.0,
        "llm_retries": 0,
    }
    base.update(over)
    return Settings(_env_file=None, **base)


def _resp(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _rate_limit(model: str) -> litellm.RateLimitError:
    return litellm.RateLimitError(message="429", llm_provider="test", model=model)


def test_falls_back_to_next_model_on_rate_limit(monkeypatch):
    calls = []

    def fake(model, **kw):
        calls.append(model)
        if model == "groq/m1":
            raise _rate_limit(model)
        return _resp("answer from m2")

    monkeypatch.setattr(litellm, "completion", fake)
    router = LLMRouter(_settings())
    assert router.complete("fast", "q") == "answer from m2"
    assert calls == ["groq/m1", "gemini/m2"]


def test_cooldown_skips_recently_failed_model(monkeypatch):
    calls = []

    def fake(model, **kw):
        calls.append(model)
        if model == "groq/m1":
            raise _rate_limit(model)
        return _resp("ok")

    monkeypatch.setattr(litellm, "completion", fake)
    router = LLMRouter(_settings())
    router.complete("fast", "q1")          # m1 fails -> cooldown, m2 answers
    router.complete("fast", "q2")          # m1 must be skipped without a call
    assert calls == ["groq/m1", "gemini/m2", "gemini/m2"]


def test_models_without_keys_are_skipped(monkeypatch):
    calls = []

    def fake(model, **kw):
        calls.append(model)
        return _resp("ok")

    monkeypatch.setattr(litellm, "completion", fake)
    router = LLMRouter(_settings(groq_api_key=""))   # groq key absent
    assert router.complete("fast", "q") == "ok"
    assert calls == ["gemini/m2"]


def test_all_models_failing_raises_llm_unavailable(monkeypatch):
    def fake(model, **kw):
        raise _rate_limit(model)

    monkeypatch.setattr(litellm, "completion", fake)
    router = LLMRouter(_settings())
    with pytest.raises(LLMUnavailable, match="role=fast"):
        router.complete("fast", "q")


def test_no_keys_means_unavailable_not_crash():
    router = LLMRouter(_settings(groq_api_key="", gemini_api_key=""))
    assert not router.available("fast")
    with pytest.raises(LLMUnavailable):
        router.complete("fast", "q")


def test_available_reflects_keys():
    assert LLMRouter(_settings()).available("fast")
    assert not LLMRouter(_settings(gemini_api_key="")).available("judge")


def test_unknown_role_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        LLMRouter(_settings()).complete("nonsense", "q")


def test_empty_completion_falls_through_to_next_model(monkeypatch):
    """An empty completion is a failure, not an answer: returning "" made a one-word
    grader verdict read as 'weak' and abstain on answerable questions."""
    calls = []

    def fake(model, **kw):
        calls.append(model)
        return _resp("" if model == "groq/m1" else "real answer")

    monkeypatch.setattr(litellm, "completion", fake)
    assert LLMRouter(_settings()).complete("fast", "q") == "real answer"
    assert calls == ["groq/m1", "gemini/m2"]


def test_all_empty_raises_llm_unavailable(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda model, **kw: _resp("   "))
    with pytest.raises(LLMUnavailable, match="empty response"):
        LLMRouter(_settings()).complete("fast", "q")


def test_gemini_25_disables_thinking(monkeypatch):
    """Gemini 2.5 spends a small max_tokens on invisible reasoning and returns nothing;
    every call here wants short deterministic output."""
    seen = {}

    def fake(model, **kw):
        seen.update(kw)
        return _resp("ok")

    monkeypatch.setattr(litellm, "completion", fake)
    LLMRouter(_settings(gemini_api_key="k", llm_fast="gemini/gemini-2.5-flash")).complete("fast", "q")
    assert seen["reasoning_effort"] == "none"
    seen.clear()
    LLMRouter(_settings(llm_fast="groq/llama-3.3-70b-versatile")).complete("fast", "q")
    assert "reasoning_effort" not in seen        # only applied where it's needed


def test_hosted_vllm_uses_api_base_not_api_key(monkeypatch):
    """Self-hosted vLLM (production-ocr-course cluster): endpoint configured = usable;
    calls carry api_base; unset endpoint = skipped like any missing key."""
    seen = {}

    def fake(model, messages, **kw):
        seen.update(kw)
        return _resp("ok")

    monkeypatch.setattr(litellm, "completion", fake)
    s = _settings(llm_vision="hosted_vllm/Qwen/Qwen3.5-4B",
                  vllm_base_url="http://ocr-gateway/v1")
    router = LLMRouter(s)
    assert router.available("vision")
    assert router.complete("vision", "ocr this") == "ok"
    assert seen["api_base"] == "http://ocr-gateway/v1"

    assert not LLMRouter(_settings(llm_vision="hosted_vllm/Qwen/Qwen3.5-4B",
                                   vllm_base_url="")).available("vision")


def test_system_message_is_passed_through(monkeypatch):
    seen = {}

    def fake(model, messages, **kw):
        seen["messages"] = messages
        return _resp("ok")

    monkeypatch.setattr(litellm, "completion", fake)
    LLMRouter(_settings()).complete("answer", "q", system="be terse")
    assert seen["messages"][0] == {"role": "system", "content": "be terse"}
    assert seen["messages"][1]["content"] == "q"
