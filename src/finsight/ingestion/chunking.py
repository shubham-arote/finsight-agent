"""chunking.py — structure-aware, parent-child, contextual chunking.

Small-to-big retrieval, ported from financial_analyst_agent and extended:

  * children = blocks (paragraph / table / heading) — precise, bbox-anchored, so every
    citation points to the exact spot on the page (FinRAGBench-V block-level citations)
  * parents  = sections (a heading + the blocks under it) — coherent context for the LLM
  * tables stay whole — never split mid-table; a table is one child chunk
  * contextual retrieval (Anthropic technique): an LLM-written 1-2 sentence prefix
    situates each chunk in the document before embedding, so "revenue ... 6,303" embeds
    as "FY26 income statement of X plc: revenue ... 6,303". Cached by content hash;
    skipped cleanly when no `fast` key is available. Tables get the same treatment,
    which doubles as the natural-language table summary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..llm import LLMRouter, LLMUnavailable, prompts
from .models import FURNITURE, Block, BlockType


@dataclass
class Section:
    """Parent: a heading plus the blocks under it (per page)."""
    id: int
    page: int
    heading: str
    text: str = ""
    block_ids: list[int] = field(default_factory=list)


@dataclass
class Chunk:
    """Child: one block, citation-anchored (page + block_id + bbox)."""
    id: str
    doc_id: str
    page: int
    block_id: int | None
    section_id: int | None
    type: str
    heading: str
    text: str                 # heading-tagged matching text
    content: str              # raw block content
    bbox: list[int]
    context: str = ""         # contextual-retrieval prefix (may stay empty)

    @property
    def embed_text(self) -> str:
        """What gets embedded / sparse-indexed: context prefix + matching text."""
        return f"{self.context}\n{self.text}" if self.context else self.text

    def payload(self) -> dict:
        """Vector-store payload — everything a citation or the agent's context needs."""
        return {"chunk_id": self.id, "doc_id": self.doc_id, "page": self.page,
                "block_id": self.block_id, "section_id": self.section_id,
                "type": self.type, "heading": self.heading, "text": self.text,
                "content": self.content, "bbox": self.bbox, "context": self.context}


def build_chunks(blocks: list[Block], doc_id: str, default_page: int = 1
                 ) -> tuple[list[Chunk], list[Section]]:
    """Blocks (any parser, any number of pages) -> child chunks + parent sections."""
    chunks: list[Chunk] = []
    sections: list[Section] = []
    heading = ""
    cur_page: int | None = None
    cur_sec: int | None = None

    def open_section(page: int, head: str) -> int:
        sections.append(Section(id=len(sections), page=page, heading=head or "(top)"))
        return len(sections) - 1

    def child(b: Block, text: str, content: str, page: int, sec_id: int | None) -> Chunk:
        return Chunk(id=f"{doc_id}:{page}:{b.id}", doc_id=doc_id, page=page,
                     block_id=b.id, section_id=sec_id, type=b.type.value,
                     heading=heading or "(top)", text=text, content=content,
                     bbox=[int(b.bbox.x0), int(b.bbox.y0), int(b.bbox.x1), int(b.bbox.y1)])

    # reading order so a heading precedes its body/table (correct section grouping)
    ordered = sorted(blocks, key=lambda b: (b.page or default_page,
                                            b.order if b.order is not None else 1e9))
    for b in ordered:
        page = b.page or default_page
        if page != cur_page:
            heading, cur_page, cur_sec = "", page, None
        if b.type in FURNITURE:
            continue
        content = (b.content or "").strip()
        if b.type == BlockType.TITLE:
            heading = content or heading
            cur_sec = open_section(page, heading)
            sections[cur_sec].text = content
            sections[cur_sec].block_ids.append(b.id)
            chunks.append(child(b, content, content, page, cur_sec))
            continue
        if not content:
            continue
        if cur_sec is None:
            cur_sec = open_section(page, heading)
        sec = sections[cur_sec]
        sec.text = (sec.text + "\n" + content).strip()
        sec.block_ids.append(b.id)
        text = f"{heading}\n{content}" if heading else content
        chunks.append(child(b, text, content, page, cur_sec))
    return chunks, sections


def add_context(chunks: list[Chunk], router: LLMRouter, cache, doc_label: str = "",
                min_chars: int = 60, max_excerpt: int = 1200) -> int:
    """Fill `chunk.context` via the `fast` role, cached by content hash. Returns how many
    chunks got a context. Degrades to 0 (plain chunks) without a key — never crashes."""
    if not router.available("fast"):
        return 0
    prompt = prompts.get("chunk_context")
    done = 0
    for c in chunks:
        if c.type != "table" and len(c.content) < min_chars:
            continue                       # tiny prose chunks don't need situating
        key = "ctx1:" + hashlib.sha256(
            f"{doc_label}|{c.heading}|{c.content}".encode()).hexdigest()[:32]
        ctx = cache.get_text(key) if cache else None
        if ctx is None:
            try:
                ctx = router.complete("fast", prompt.render(
                    doc_label=doc_label or "(unknown document)", heading=c.heading,
                    excerpt=c.content[:max_excerpt]), max_tokens=90).strip()
            except LLMUnavailable:
                return done               # chain exhausted mid-run; cached ones are kept
            if cache:
                cache.save_text(key, ctx)
        if ctx:
            c.context = ctx
            done += 1
    return done


def doc_label_from_blocks(blocks: list[Block]) -> str:
    """Best-effort document label: the first heading on the first parsed page."""
    for b in sorted(blocks, key=lambda b: (b.page, b.order if b.order is not None else 1e9)):
        if b.type == BlockType.TITLE and (b.content or "").strip():
            return b.content.strip()[:120]
    return ""
