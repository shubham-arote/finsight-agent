"""pipeline.py — parse → chunk → (contextualize) for one document.

    ingest(pdf_bytes) -> IngestResult(chunks, sections, ...)

Routing: born-digital PDFs take the exact text-layer path (free, seconds); scanned PDFs
take the cloud-OCR path via the `vision` role. Every parsed page is cached in the
ArtifactStore by content hash, so reruns re-parse nothing and interrupted OCR resumes.

CLI (e2e proof):  python -m finsight.ingestion.pipeline report.pdf [--query "revenue"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from ..config import settings
from ..llm import LLMRouter
from .artifacts import ArtifactStore, doc_hash
from .chunking import Chunk, Section, add_context, build_chunks, doc_label_from_blocks
from .enrich import enrich_blocks
from .models import Block, block_from_dict, block_to_dict
from .parsers import cloud_ocr, textlayer


class IngestError(RuntimeError):
    pass


@dataclass
class IngestResult:
    doc_id: str
    parser: str
    page_count: int
    sizes: list[tuple[float, float]]
    pages_blocks: list[list[Block]]
    chunks: list[Chunk]
    sections: list[Section]
    doc_label: str = ""
    contextualized: int = 0
    cached_pages: int = 0
    stats: dict = field(default_factory=dict)


@dataclass
class ParsedDoc:
    doc_hash: str
    parser: str                      # textlayer | cloud_ocr | mixed
    pages_blocks: list[list[Block]]
    sizes: list[tuple[float, float]]
    cached_pages: int
    skipped_pages: list[int]         # scanned pages left empty (no vision key)


def _page_has_text(page) -> bool:
    return len(page.get_text("text").strip()) > 50


def parse_pdf(data: bytes, *, router: LLMRouter | None = None,
              store: ArtifactStore | None = None, parser: str = "auto",
              on_page=None) -> ParsedDoc:
    """Parse every page (cache-aware, routed PER PAGE). Born-digital pages take the exact
    text-layer path; scanned pages go to cloud OCR — so a filing with a scanned appendix
    keeps both halves. `on_page(done, total)` reports progress."""
    h = doc_hash(data)
    doc = fitz.open(stream=data, filetype="pdf")
    n = len(doc)
    if n == 0:
        raise IngestError("empty PDF")
    router = router or LLMRouter()
    vision_ok = router.available("vision")

    if parser == "auto":
        page_parsers = ["textlayer" if _page_has_text(doc[i]) else "cloud_ocr"
                        for i in range(n)]
    else:
        page_parsers = [parser] * n
    if all(p == "cloud_ocr" for p in page_parsers) and not vision_ok:
        raise IngestError("scanned PDF needs a vision-capable key "
                          "(GEMINI_API_KEY or GROQ_API_KEY)")

    relation = None
    pages_blocks: list[list[Block]] = []
    cached = 0
    skipped: list[int] = []
    for i, pparser in enumerate(page_parsers):
        saved = store.get_page(h, i + 1, pparser) if store else None
        if saved is not None:
            pages_blocks.append([block_from_dict(d) for d in saved])
            cached += 1
        elif pparser == "cloud_ocr" and not vision_ok:
            pages_blocks.append([])                    # mixed doc, no key: keep the rest
            skipped.append(i + 1)                      # NOT cached — a keyed rerun parses it
        else:
            if pparser == "textlayer":
                blocks = textlayer.extract_page(doc[i], i + 1, relation)
            else:
                blocks = cloud_ocr.extract_page(doc[i], i + 1, router)
            if store:
                store.save_page(h, i + 1, pparser, [block_to_dict(b) for b in blocks])
            pages_blocks.append(blocks)
        if on_page:
            on_page(i + 1, n)

    sizes = [(doc[i].rect.width, doc[i].rect.height) for i in range(n)]
    # furniture marking is document-wide and cheap — recomputed every run (never cached),
    # so cached raw pages and freshly parsed ones are treated identically
    textlayer.mark_repeated_furniture(pages_blocks, sizes)
    kinds = set(page_parsers)
    label = page_parsers[0] if len(kinds) == 1 else "mixed"
    return ParsedDoc(h, label, pages_blocks, sizes, cached, skipped)


def ingest(data: bytes, *, doc_id: str | None = None, router: LLMRouter | None = None,
           store: ArtifactStore | None = None, parser: str = "auto",
           contextual: bool | None = None, enrich: bool | None = None,
           on_page=None) -> IngestResult:
    """Full ingestion for one PDF:
    parse (per-page routed, cached) → VLM enrichment of figures/borderless tables
    (Gemini-first, cached, capped) → chunk → contextual prefixes (cached)."""
    router = router or LLMRouter()
    store = store if store is not None else ArtifactStore()
    parsed = parse_pdf(data, router=router, store=store, parser=parser, on_page=on_page)
    doc_id = doc_id or parsed.doc_hash

    enrich = settings.enrich_blocks if enrich is None else enrich
    n_enriched = (enrich_blocks(data, parsed.pages_blocks, router, store)
                  if enrich else 0)

    all_blocks = [b for pb in parsed.pages_blocks for b in pb]
    chunks, sections = build_chunks(all_blocks, doc_id)
    label = doc_label_from_blocks(all_blocks)

    contextual = settings.contextual_chunks if contextual is None else contextual
    n_ctx = add_context(chunks, router, store, doc_label=label) if contextual else 0

    return IngestResult(
        doc_id=doc_id, parser=parsed.parser, page_count=len(parsed.pages_blocks),
        sizes=parsed.sizes, pages_blocks=parsed.pages_blocks, chunks=chunks,
        sections=sections, doc_label=label,
        contextualized=n_ctx, cached_pages=parsed.cached_pages,
        stats={"blocks": len(all_blocks), "chunks": len(chunks),
               "sections": len(sections),
               "tables": sum(1 for c in chunks if c.type == "table"),
               "enriched": n_enriched, "skipped_pages": parsed.skipped_pages})


# ── CLI: e2e proof over a real PDF ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Ingest a PDF; optionally index + search it.")
    ap.add_argument("pdf")
    ap.add_argument("--parser", default="auto", choices=["auto", "textlayer", "cloud_ocr"])
    ap.add_argument("--no-contextual", action="store_true")
    ap.add_argument("--query", help="index into Qdrant (in-memory unless QDRANT_URL) and search")
    args = ap.parse_args()

    data = Path(args.pdf).read_bytes()
    t0 = time.time()
    res = ingest(data, parser=args.parser, contextual=not args.no_contextual)
    print(f"doc={res.doc_id} parser={res.parser} label={res.doc_label!r}")
    print(f"pages={res.page_count} (cached={res.cached_pages})  "
          f"blocks={res.stats['blocks']} chunks={res.stats['chunks']} "
          f"tables={res.stats['tables']} sections={res.stats['sections']} "
          f"contextualized={res.contextualized}  in {time.time()-t0:.1f}s")
    if args.query:
        from ..retrieval import QdrantIndex
        idx = QdrantIndex()
        idx.index_chunks(res.chunks)
        for hit in idx.search(args.query, k=5):
            print(f"  p{hit['page']:>3} score={hit['score']:.3f} "
                  f"[{hit['type']}] {hit['heading'][:40]!r} :: {hit['content'][:90]!r}")
