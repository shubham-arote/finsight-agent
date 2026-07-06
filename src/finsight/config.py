"""config.py — single Settings object. The ONLY place environment variables are read.

Every provider key is optional: features degrade gracefully when a key is absent
(the router skips models whose key is missing; callers check `router.available(role)`).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── provider keys (all optional) ──
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    cohere_api_key: str = ""

    # ── role → model fallback chains (comma-separated LiteLLM model ids) ──
    llm_fast: str = (
        "groq/llama-3.3-70b-versatile,"
        "gemini/gemini-2.5-flash,"
        "openrouter/meta-llama/llama-3.3-70b-instruct:free"
    )
    llm_answer: str = (
        "groq/llama-3.3-70b-versatile,"
        "cohere/command-a-03-2025,"
        "gemini/gemini-2.5-flash"
    )
    llm_vision: str = (
        "gemini/gemini-2.5-flash,"
        "groq/meta-llama/llama-4-scout-17b-16e-instruct"
    )
    # judge is Gemini-only on purpose: the eval judge must be a DIFFERENT model family
    # from the answering models (kills self-judging bias); Google models judge in cloud too.
    llm_judge: str = "gemini/gemini-2.5-flash"

    # seconds a model is skipped after a rate-limit / provider error
    llm_cooldown_s: float = 60.0
    # per-call retries within one model before falling to the next in the chain
    llm_retries: int = 1


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
