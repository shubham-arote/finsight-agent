"""Shared fixtures: a deterministic synthetic annual-report PDF + a keyless router.

The suite is OFFLINE BY CONTRACT: a developer's local .env (real keys) must never leak
into tests — otherwise "offline" tests silently make network calls (rerank per retrieve,
cloud grading) and results drift with quota. Env overrides beat .env in pydantic-settings,
so blank them here BEFORE any finsight import constructs Settings.
"""

import os
import tempfile

for _k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY",
           "GOOGLE_CLOUD_PROJECT", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
           "MCP_SERVER_URL", "QDRANT_URL"):
    os.environ[_k] = ""

# Never read/write the developer's artifact cache: it accumulates real uploaded docs,
# so the server's startup-reload would re-index megabytes and hang the suite. Point
# every test at a throwaway store before Settings loads.
os.environ["ARTIFACTS_DB"] = os.path.join(tempfile.mkdtemp(prefix="finsight-test-"), "a.db")

import fitz
import pytest

from finsight.config import Settings
from finsight.llm.router import LLMRouter

PAGE_TEXTS = [
    "Revenue for the year was 6,303 million, up from 5,950 million in the prior year, "
    "driven by growth in the services segment across all regions.",
    "Gross profit reached 2,302 million with a gross margin improvement of 40 basis "
    "points, reflecting supply-chain efficiencies achieved during the year.",
    "Operating profit was 1,052 million compared with 985 million in the prior year, "
    "an increase of 6.8 percent on a reported basis.",
    "Net assets at the period end stood at 4,810 million, and the board recommends a "
    "total dividend per share of 12.4 pence for the year.",
]


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """4-page born-digital PDF: big heading + body paragraph per page, repeated footer
    (furniture) on every page. Deterministic — built from fixed text with fixed fonts."""
    doc = fitz.open()
    for i, body in enumerate(PAGE_TEXTS):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 90), f"Section {i + 1}: Financial Review", fontsize=20)
        page.insert_textbox(fitz.Rect(72, 130, 523, 420), body, fontsize=11)
        page.insert_text((72, 822), "Acme Holdings plc Annual Report 2026", fontsize=8)
    return doc.tobytes()


def keyless_settings(**over) -> Settings:
    base = dict(groq_api_key="", gemini_api_key="", openrouter_api_key="", cohere_api_key="")
    base.update(over)
    return Settings(_env_file=None, **base)


@pytest.fixture()
def keyless_router() -> LLMRouter:
    return LLMRouter(keyless_settings())
