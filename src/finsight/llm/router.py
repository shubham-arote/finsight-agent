"""router.py — role-based LLM access with fallback chains.

All LLM calls in the app go through `LLMRouter.complete(role, ...)`. A *role* is what
the call is for, not which provider serves it:

    fast    grading / query-rewrite / contextualize (cheap, low-latency)
    answer  final answer generation
    vision  OCR of page/block images
    judge   evaluation only — kept on an independent model family (Gemini)

Each role maps to an ordered chain of LiteLLM model ids (from Settings, env-overridable).
On a rate-limit or provider error the failing model is put in a cooldown and the call
falls through to the next model in the chain. Models whose provider key is missing are
skipped entirely — key-optional degradation, never a hard crash. If no model in a chain
is usable, `LLMUnavailable` is raised; callers degrade (e.g. extractive answers).
"""

from __future__ import annotations

import time

import litellm

from ..config import Settings
from ..config import settings as _default_settings

# errors that mean "this model/provider is unhealthy right now — try the next one"
_FALLBACK_ERRORS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
    litellm.AuthenticationError,   # bad/revoked key: skip provider rather than crash
)

_PROVIDER_KEY_ATTR = {
    "groq": "groq_api_key",
    "gemini": "gemini_api_key",
    "openrouter": "openrouter_api_key",
    "cohere": "cohere_api_key",
    # Vertex AI authenticates via ADC, not an api key — "has key" = project configured.
    # This is the scale path: same chains, production quotas (e.g. for large eval runs).
    "vertex_ai": "google_cloud_project",
    # Self-hosted vLLM (OpenAI-compatible; e.g. the production-ocr-course Qwen cluster):
    # "has key" = endpoint configured. Auth is the endpoint's concern (gateway/ILB).
    "hosted_vllm": "vllm_base_url",
}

ROLES = ("fast", "answer", "vision", "judge")


class LLMUnavailable(RuntimeError):
    """No model in the role's chain is usable (missing keys, cooldowns, or all failed)."""


class LLMRouter:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or _default_settings
        self.chains: dict[str, list[str]] = {
            "fast": _parse(self.settings.llm_fast),
            "answer": _parse(self.settings.llm_answer),
            "vision": _parse(self.settings.llm_vision),
            "judge": _parse(self.settings.llm_judge),
        }
        self._cooldown_until: dict[str, float] = {}

    # ── public api ──────────────────────────────────────────────────────────
    def available(self, role: str) -> bool:
        """True if at least one model in the role's chain has its provider key set."""
        return any(self._has_key(m) for m in self._chain(role))

    def label(self, role: str) -> str:
        """First usable model in the role's chain (for status displays), or 'offline'."""
        for m in self._chain(role):
            if self._has_key(m):
                return m
        return "offline"

    def complete(self, role: str, prompt: str, *, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0, **kwargs) -> str:
        """Run the role's chain until one model answers; return the text content."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete_messages(role, messages, max_tokens=max_tokens,
                                      temperature=temperature, **kwargs)

    def complete_messages(self, role: str, messages: list[dict], *,
                          max_tokens: int = 1024, temperature: float = 0.0,
                          **kwargs) -> str:
        errors: list[str] = []
        now = time.monotonic()
        for model in self._chain(role):
            if not self._has_key(model):
                errors.append(f"{model}: no api key")
                continue
            if self._cooldown_until.get(model, 0.0) > now:
                errors.append(f"{model}: cooling down")
                continue
            try:
                resp = litellm.completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    num_retries=self.settings.llm_retries,
                    **self._auth_kwargs(model),
                    **kwargs,
                )
                return resp.choices[0].message.content or ""
            except _FALLBACK_ERRORS as e:
                self._cooldown_until[model] = time.monotonic() + self.settings.llm_cooldown_s
                errors.append(f"{model}: {type(e).__name__}")
        raise LLMUnavailable(f"role={role}: all models failed — " + "; ".join(errors))

    # ── internals ───────────────────────────────────────────────────────────
    def _chain(self, role: str) -> list[str]:
        if role not in self.chains:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
        return self.chains[role]

    def _provider(self, model: str) -> str:
        return model.split("/", 1)[0]

    def _has_key(self, model: str) -> bool:
        attr = _PROVIDER_KEY_ATTR.get(self._provider(model))
        return bool(attr and getattr(self.settings, attr, ""))

    def _key(self, model: str) -> str:
        return getattr(self.settings, _PROVIDER_KEY_ATTR[self._provider(model)])

    def _auth_kwargs(self, model: str) -> dict:
        """Provider auth: api_key for keyed providers; project+location (ADC) for Vertex;
        api_base for self-hosted vLLM."""
        provider = self._provider(model)
        if provider == "vertex_ai":
            return {"vertex_project": self.settings.google_cloud_project,
                    "vertex_location": self.settings.vertex_location}
        if provider == "hosted_vllm":
            return {"api_base": self.settings.vllm_base_url, "api_key": "EMPTY"}
        return {"api_key": self._key(model)}


def _parse(chain: str) -> list[str]:
    return [m.strip() for m in chain.split(",") if m.strip()]
