"""Ingestion: text-layer parsing, furniture, OCR markdown split, artifact resume."""

import fitz

from finsight.ingestion import ArtifactStore, BlockType, parse_pdf
from finsight.ingestion.parsers import cloud_ocr, textlayer
from finsight.ingestion import pipeline


def test_has_text_layer_true_for_borndigital(sample_pdf_bytes):
    doc = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    assert textlayer.has_text_layer(doc)


def test_extract_page_finds_heading_and_body(sample_pdf_bytes):
    doc = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    blocks = textlayer.extract_page(doc[0], 1)
    titles = [b for b in blocks if b.type == BlockType.TITLE]
    assert any("Financial Review" in (b.content or "") for b in titles)
    assert any("6,303" in (b.content or "") for b in blocks)
    body = [b for b in blocks if b.order is not None]
    assert body and all(b.bbox.area > 0 for b in body)


def test_repeated_footer_marked_as_furniture(sample_pdf_bytes, keyless_router):
    _, parser, pages_blocks, _, _ = parse_pdf(sample_pdf_bytes, router=keyless_router,
                                              store=None, parser="auto")
    assert parser == "textlayer"
    footers = [b for pb in pages_blocks for b in pb
               if "Annual Report 2026" in (b.content or "")]
    assert len(footers) == 4
    assert all(b.type in (BlockType.FOOTER, BlockType.HEADER) and b.order is None
               for b in footers)


def test_rerun_skips_already_parsed_pages(sample_pdf_bytes, keyless_router, monkeypatch):
    calls = {"n": 0}
    orig = textlayer.extract_page

    def counting(page, page_no, relation=None):
        calls["n"] += 1
        return orig(page, page_no, relation)

    monkeypatch.setattr(pipeline.textlayer, "extract_page", counting)
    store = ArtifactStore(":memory:")
    _, _, first, _, cached0 = parse_pdf(sample_pdf_bytes, router=keyless_router, store=store)
    assert calls["n"] == 4 and cached0 == 0
    _, _, second, _, cached1 = parse_pdf(sample_pdf_bytes, router=keyless_router, store=store)
    assert calls["n"] == 4 and cached1 == 4          # nothing re-parsed
    # cached round-trip preserves content + geometry
    assert [b.content for pb in second for b in pb] == [b.content for pb in first for b in pb]


def test_scanned_pdf_without_vision_key_fails_cleanly(keyless_router):
    doc = fitz.open()
    doc.new_page(width=595, height=842)              # image-only page: no text layer
    try:
        parse_pdf(doc.tobytes(), router=keyless_router, store=None)
        assert False, "expected IngestError"
    except pipeline.IngestError as e:
        assert "vision" in str(e)


def test_ocr_markdown_splits_into_typed_blocks():
    md = ("## Income Statement\n\nRevenue was 6,303 million for the year.\n\n"
          "| Metric | FY26 | FY25 |\n|---|---|---|\n| Revenue | 6,303 | 5,950 |\n\n"
          "Costs decreased slightly.")
    blocks = cloud_ocr.markdown_to_blocks(md, 3, (595.0, 842.0))
    types = [b.type for b in blocks]
    assert types == [BlockType.TITLE, BlockType.TEXT, BlockType.TABLE, BlockType.TEXT]
    assert all(b.page == 3 for b in blocks)
    table = blocks[2]
    assert "6,303" in table.content and "5,950" in table.content   # every cell preserved
    assert [b.id for b in blocks] == [0, 1, 2, 3]
