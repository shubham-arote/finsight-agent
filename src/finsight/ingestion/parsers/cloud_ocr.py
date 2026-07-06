"""cloud_ocr.py — scanned-PDF parsing via the `vision` role (cloud VLM, no local models).

Renders each page to PNG and asks the vision chain (Gemini → Groq fallback, via the
router) to transcribe it to structured Markdown. The Markdown is then split back into
Blocks (headings / tables / paragraphs) so downstream chunking and citations work the
same as the text-layer path — with page-level bboxes, since OCR gives no per-block boxes.

Rate-limit resilience comes from two layers: the router's fallback+cooldown chain, and
the pipeline's per-page artifact cache (a page is OCR'd exactly once; reruns resume).
"""

from __future__ import annotations

import base64

from ...config import settings
from ...llm import LLMRouter, prompts
from ..models import BBox, Block, BlockType


def extract_page(page, page_no: int, router: LLMRouter) -> list[Block]:
    """OCR one fitz page -> Blocks. Raises LLMUnavailable if the whole vision chain fails
    (the pipeline surfaces that as a doc-level error; already-parsed pages stay cached)."""
    pix = page.get_pixmap(dpi=settings.ocr_dpi)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    prompt = prompts.get("ocr_page")
    md = router.complete_messages(
        "vision",
        [{"role": "user", "content": [
            {"type": "text", "text": prompt.render()},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        max_tokens=4096,
    )
    return markdown_to_blocks(md, page_no, (page.rect.width, page.rect.height))


def markdown_to_blocks(md: str, page_no: int, page_size: tuple[float, float]) -> list[Block]:
    """Split OCR Markdown into TITLE / TABLE / TEXT blocks. All blocks carry the full-page
    bbox (OCR has no per-block geometry) — citations stay page-accurate."""
    pw, ph = page_size
    bbox = BBox(0, 0, pw, ph)
    blocks: list[Block] = []
    para: list[str] = []
    table: list[str] = []

    def flush_para():
        text = "\n".join(para).strip()
        if text:
            blocks.append(Block(BlockType.TEXT, bbox, content=text, page=page_no))
        para.clear()

    def flush_table():
        text = "\n".join(table).strip()
        if text:
            blocks.append(Block(BlockType.TABLE, bbox, content=text, page=page_no))
        table.clear()

    for line in (md or "").splitlines():
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.count("|") >= 2
        if is_table_row:
            if para:
                flush_para()
            table.append(stripped)
            continue
        if table:
            flush_table()
        if stripped.startswith("#"):
            flush_para()
            heading = stripped.lstrip("# ").strip()
            if heading:
                blocks.append(Block(BlockType.TITLE, bbox, content=heading, page=page_no))
        elif not stripped:
            flush_para()
        else:
            para.append(line)
    flush_para()
    flush_table()

    for i, b in enumerate(blocks):
        b.id, b.order = i, i
    return blocks
