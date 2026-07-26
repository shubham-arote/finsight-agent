"""Chunking: parent-child structure (golden file), table integrity, contextual prefixes."""

import json
from pathlib import Path

from finsight.ingestion.artifacts import ArtifactStore
from finsight.ingestion.chunking import add_context, build_chunks, doc_label_from_blocks
from finsight.ingestion.models import BBox, Block, BlockType

GOLDEN = Path(__file__).parent / "golden" / "chunks.json"


def sample_blocks() -> list[Block]:
    """Hand-built two-page block set: heading + prose + table, a heading-less page,
    and a footer that must be excluded. Fully deterministic."""
    def mk(bt, content, page, bid, order, y=None):
        y0 = y if y is not None else 100 + bid * 120
        b = Block(bt, BBox(72, y0, 523, y0 + 80), page=page, content=content)
        b.id, b.order = bid, order
        return b

    p1 = [
        mk(BlockType.TITLE, "Financial Highlights", 1, 0, 0),
        mk(BlockType.TEXT, "Revenue was 6,303 million (FY25: 5,950 million).", 1, 1, 1),
        mk(BlockType.TABLE, "| Metric | FY26 | FY25 |\n|---|---|---|\n"
                            "| Revenue | 6,303 | 5,950 |\n| Operating profit | 1,052 | 985 |",
           1, 2, 2),
    ]
    footer = mk(BlockType.FOOTER, "Acme plc", 1, 3, None, y=810)
    p2 = [mk(BlockType.TEXT, "Cash generated from operations improved during the year.",
             2, 0, 0)]
    return [*p1, footer, *p2]


def test_chunks_match_golden_file():
    chunks, sections = build_chunks(sample_blocks(), doc_id="doc1")
    got = {"chunks": [c.payload() for c in chunks],
           "sections": [vars(s) for s in sections]}
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert got == expected


def test_structure_invariants():
    chunks, sections = build_chunks(sample_blocks(), doc_id="doc1")
    # furniture excluded
    assert not any(c.content == "Acme plc" for c in chunks)
    # table kept whole, heading-tagged, bbox-anchored
    table = next(c for c in chunks if c.type == "table")
    assert "1,052" in table.content and "985" in table.content
    assert table.heading == "Financial Highlights" and table.text.startswith("Financial Highlights")
    assert table.bbox and table.block_id == 2 and table.page == 1
    # prose + table share the heading's parent section (small-to-big)
    sec = sections[table.section_id]
    assert "Revenue was 6,303" in sec.text and "Operating profit" in sec.text
    # heading-less page opens its own "(top)" section
    p2 = next(c for c in chunks if c.page == 2)
    assert sections[p2.section_id].heading == "(top)"
    # doc label = first heading
    assert doc_label_from_blocks(sample_blocks()) == "Financial Highlights"


class StubRouter:
    """Duck-typed router: returns a canned context and counts calls."""
    def __init__(self, up=True):
        self.up, self.calls = up, 0

    def available(self, role):
        return self.up

    def complete(self, role, prompt, **kw):
        self.calls += 1
        return "This excerpt is from Acme plc's FY26 annual report."


def test_add_context_prefixes_and_caches():
    cache = ArtifactStore(":memory:")
    router = StubRouter()
    chunks, _ = build_chunks(sample_blocks(), doc_id="doc1")
    n1 = add_context(chunks, router, cache, doc_label="Acme FY26")
    assert n1 > 0
    assert all("Acme plc's FY26" in c.embed_text for c in chunks if c.context)
    assert all(c.embed_text.endswith(c.text) for c in chunks)     # prefix, not replacement
    first_calls = router.calls
    # rerun: everything served from cache, zero LLM calls
    chunks2, _ = build_chunks(sample_blocks(), doc_id="doc1")
    n2 = add_context(chunks2, router, cache, doc_label="Acme FY26")
    assert n2 == n1 and router.calls == first_calls


def test_add_context_keyless_is_noop():
    chunks, _ = build_chunks(sample_blocks(), doc_id="doc1")
    assert add_context(chunks, StubRouter(up=False), ArtifactStore(":memory:")) == 0
    assert all(c.context == "" for c in chunks)
