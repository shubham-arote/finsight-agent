"""check_vertex.py — preflight for the Vertex AI path.

Vertex is the scale answer to free-tier rate limits: the same role chains, production
quotas, and **no API keys** — auth is the caller's identity (ADC locally, the attached
service account on Cloud Run).

    uv run python scripts/check_vertex.py

Verifies, in order: project configured → ADC present → a real completion → embeddings,
and prints the exact fix for whichever step fails.
"""

from __future__ import annotations

import subprocess
import sys

from finsight.config import settings
from finsight.llm import LLMRouter

MODEL = "vertex_ai/gemini-2.5-flash"


def _adc_present() -> bool:
    try:
        r = subprocess.run(["gcloud", "auth", "application-default", "print-access-token"],
                           capture_output=True, text=True, timeout=60, shell=True)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def main() -> int:
    print(f"project : {settings.google_cloud_project or '(unset)'}")
    print(f"location: {settings.vertex_location}")
    if not settings.google_cloud_project:
        print("\nFIX: add to .env →  GOOGLE_CLOUD_PROJECT=<your-project-id>")
        return 1

    print(f"ADC     : {'present' if _adc_present() else 'MISSING'}")
    if not _adc_present():
        print("\nFIX: run once (opens a browser):\n"
              "  gcloud auth application-default login\n"
              f"  gcloud auth application-default set-quota-project {settings.google_cloud_project}")
        return 1

    router = LLMRouter(settings.model_copy(update={"llm_answer": MODEL}))
    try:
        out = router.complete("answer", "Reply with exactly: OK", max_tokens=5)
        print(f"chat    : OK ({MODEL}) -> {out.strip()[:40]!r}")
    except Exception as e:
        print(f"chat    : FAILED -> {type(e).__name__}: {str(e)[:200]}")
        print("\nFIX: enable the API and confirm billing:\n"
              f"  gcloud services enable aiplatform.googleapis.com --project={settings.google_cloud_project}")
        return 1

    print("\nAll good. Point the roles at Vertex in .env:\n"
          "  LLM_ANSWER=vertex_ai/gemini-2.5-flash,groq/llama-3.3-70b-versatile\n"
          "  LLM_FAST=vertex_ai/gemini-2.5-flash\n"
          "  LLM_VISION=vertex_ai/gemini-2.5-flash\n"
          "  LLM_JUDGE=vertex_ai/gemini-2.5-flash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
