"""Phase 1.5 hardening: per-page routing (mixed docs) + VLM enrichment of figures and
borderless tables (Gemini-first vision chain, cached, capped, keyless no-op)."""

import fitz

from finsight.ingestion import ArtifactStore
from finsight.ingestion.enrich import enrich_blocks, looks_like_borderless_table
from finsight.ingestion.models import BBox, Block, BlockType
from finsight.ingestion.pipeline import parse_pdf


class VisionStub:
    """Vision-capable router double; answers by prompt kind and counts calls."""

    def __init__(self):
        self.calls = []

    def available(self, role):
        return True

    def complete(self, role, prompt, **kw):
        return ""

    def complete_messages(self, role, messages, **kw):
        text = messages[0]["content"][0]["text"]
        self.calls.append(role)
        if "chart or figure" in text:                        # describe_figure prompt
            return "Bar chart of revenue by year: FY26 6,303m vs FY25 5,952m."
        if "table region" in text:                           # ocr_table_region prompt
            return "| Segment | FY26 | FY25 |\n|---|---|---|\n| Revenue | 6,303 | 5,952 |"
        return "## Appendix\n\nRecovered scanned text about warranty provisions."


def _mixed_pdf() -> bytes:
    """Two born-digital pages + one blank (scanned-like) page."""
    doc = fitz.open()
    for i in range(2):
        p = doc.new_page(width=595, height=842)
        p.insert_textbox(fitz.Rect(72, 90, 523, 300),
                         f"Text page {i + 1}. Revenue for the year was 6,303 million, "
                         "an increase on the prior year driven by services.", fontsize=11)
    doc.new_page(width=595, height=842)
    return doc.tobytes()


# ── per-page routing ────────────────────────────────────────────────────────
def test_mixed_doc_routes_per_page():
    router = VisionStub()
    parsed = parse_pdf(_mixed_pdf(), router=router, store=None)
    assert parsed.parser == "mixed" and parsed.skipped_pages == []
    assert router.calls == ["vision"]                        # OCR for the blank page only
    assert parsed.pages_blocks[0] and "6,303" in parsed.pages_blocks[0][0].content
    assert any("warranty" in (b.content or "") for b in parsed.pages_blocks[2])


def test_mixed_doc_keyless_keeps_text_pages_skips_scanned(keyless_router):
    parsed = parse_pdf(_mixed_pdf(), router=keyless_router, store=None)
    assert parsed.skipped_pages == [3]
    assert parsed.pages_blocks[2] == [] and parsed.pages_blocks[0]


def test_skipped_pages_are_not_cached_so_a_keyed_rerun_parses_them(keyless_router):
    store = ArtifactStore(":memory:")
    data = _mixed_pdf()          # same bytes both runs (fitz stamps a creation time,
    parse_pdf(data, router=keyless_router, store=store)   # so two builds hash apart)
    parsed = parse_pdf(data, router=VisionStub(), store=store)
    assert parsed.skipped_pages == [] and parsed.pages_blocks[2]
    assert parsed.cached_pages == 2                          # text pages came from cache


# ── borderless-table heuristic ──────────────────────────────────────────────
def _block(bt, content, x0=72, y0=100, x1=523, y1=320):
    b = Block(bt, BBox(x0, y0, x1, y1), page=1, content=content)
    b.id, b.order = 0, 0
    return b


def test_borderless_table_detection():
    table = _block(BlockType.TEXT,
                   "Revenue           6,303    5,952\n"
                   "Cost of sales    (4,001)  (3,820)\n"
                   "Gross profit      2,302    2,132\n"
                   "Operating profit  1,052      985")
    prose = _block(BlockType.TEXT,
                   "The Group delivered a resilient performance this year.\n"
                   "Trading conditions remained challenging throughout.\n"
                   "Our online platform continued to grow strongly.\n"
                   "The Board thanks all colleagues for their work.")
    assert looks_like_borderless_table(table)
    assert not looks_like_borderless_table(prose)


# ── enrichment ──────────────────────────────────────────────────────────────
def _one_page_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    return doc.tobytes()


def _enrichable_blocks():
    fig = _block(BlockType.FIGURE, "", 72, 100, 300, 260)
    tbl = _block(BlockType.TEXT,
                 "Revenue 6,303 5,952\nCost (4,001) (3,820)\n"
                 "Gross 2,302 2,132\nOperating 1,052 985", 72, 300, 523, 500)
    tbl.id = 1
    return [[fig, tbl]]


def test_enrich_fills_figures_and_recovers_tables():
    router, store = VisionStub(), ArtifactStore(":memory:")
    blocks = _enrichable_blocks()
    assert enrich_blocks(_one_page_pdf(), blocks, router, store) == 2
    fig, tbl = blocks[0]
    assert fig.content.startswith("Bar chart") and fig.type == BlockType.FIGURE
    assert tbl.type == BlockType.TABLE and "| Revenue | 6,303 |" in tbl.content
    calls_after_first = len(router.calls)
    # cache: identical crops -> zero new VLM calls
    assert enrich_blocks(_one_page_pdf(), _enrichable_blocks(), router, store) == 2
    assert len(router.calls) == calls_after_first


def test_enrich_is_keyless_noop(keyless_router):
    blocks = _enrichable_blocks()
    assert enrich_blocks(_one_page_pdf(), blocks, keyless_router, None) == 0
    assert blocks[0][0].content == "" and blocks[0][1].type == BlockType.TEXT


def test_enrich_respects_per_doc_cap():
    assert enrich_blocks(_one_page_pdf(), _enrichable_blocks(), VisionStub(),
                         ArtifactStore(":memory:"), max_blocks=1) == 1
