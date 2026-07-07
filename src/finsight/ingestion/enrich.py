"""enrich.py — targeted VLM enrichment for what the text layer cannot see.

The text-layer parser is exact on prose and ruled tables but blind to two things common
in real financial documents:

  * FIGURES — charts/images whose numbers live in pixels, not text
  * BORDERLESS TABLES — whitespace-aligned statements `find_tables` (ruled-lines
    strategy) leaves as plain text, losing row/column structure

Instead of paying page-image OCR for every page (the week-3 pattern), this pass crops
exactly those blocks (we have their bboxes) and sends them to the `vision` role —
Gemini-first per config, Groq fallback. A 200-page 10-K needs tens of calls, not 200.
Results are cached by crop-content hash (paid once, ever), capped per document, and the
whole pass is a silent no-op without a vision key.
"""

from __future__ import annotations

import base64
import hashlib
import re

import fitz

from ..config import settings
from ..llm import LLMRouter, LLMUnavailable, prompts
from .models import Block, BlockType

_NUM_TOKEN = re.compile(r"\d[\d,.]*")


def looks_like_borderless_table(b: Block) -> bool:
    """Multi-line text where most lines carry >=2 numeric tokens — a whitespace-aligned
    financial table the ruled-lines detector could not see."""
    if b.type != BlockType.TEXT:
        return False
    lines = [ln for ln in (b.content or "").splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    numeric = sum(1 for ln in lines if len(_NUM_TOKEN.findall(ln)) >= 2)
    return numeric / len(lines) >= 0.5


def _needs_vision(b: Block) -> str | None:
    if b.type == BlockType.FIGURE and not (b.content or "").strip():
        return "figure"
    if looks_like_borderless_table(b):
        return "table"
    return None


def enrich_blocks(data: bytes, pages_blocks: list[list[Block]], router: LLMRouter,
                  store=None, max_blocks: int | None = None) -> int:
    """Fill figure/borderless-table blocks via the vision chain. Returns how many blocks
    were enriched. Key-optional: no vision -> 0, untouched blocks, never a crash."""
    if not router.available("vision"):
        return 0
    max_blocks = settings.enrich_max_blocks if max_blocks is None else max_blocks
    doc = fitz.open(stream=data, filetype="pdf")
    done = 0
    for page_idx, blocks in enumerate(pages_blocks):
        if done >= max_blocks:
            break
        page = doc[page_idx] if page_idx < len(doc) else None
        if page is None:
            continue
        for b in blocks:
            if done >= max_blocks:
                break
            kind = _needs_vision(b)
            if not kind:
                continue
            clip = fitz.Rect(b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1)
            if clip.width < 30 or clip.height < 20:
                continue
            png = page.get_pixmap(clip=clip, dpi=180).tobytes("png")
            key = "vlm1:" + hashlib.sha256(png).hexdigest()[:32]
            content = store.get_text(key) if store else None
            if content is None:
                prompt = prompts.get("ocr_table_region" if kind == "table"
                                     else "describe_figure")
                b64 = base64.b64encode(png).decode()
                try:
                    content = router.complete_messages(
                        "vision",
                        [{"role": "user", "content": [
                            {"type": "text", "text": prompt.render()},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ]}],
                        max_tokens=1200).strip()
                except LLMUnavailable:
                    return done                       # chain exhausted; cached work kept
                if store:
                    store.save_text(key, content)
            if content:
                b.content = content
                if kind == "table":
                    b.type = BlockType.TABLE          # structure recovered -> table chunk
                done += 1
    return done
