"""pipeline.py — parse → chunk → (contextualize) for one document.

    ingest(pdf_bytes) -> IngestResult(chunks, sections, ...)

Routing: born-digital PDFs take the exact text-layer path (free, seconds); scanned PDFs
take the cloud-OCR path via the `vision` role. Every parsed page is cached in the
ArtifactStore by content hash, so reruns re-parse nothing and interrupted OCR resumes.

CLI (e2e proof):  python -m finsight.ingestion.pipeline report.pdf [--query "revenue"]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import fitz  # PyMuPDF

from ..config import settings
from ..llm import LLMRouter
from .artifacts import ArtifactStore, doc_hash
from .chunking import Chunk, Section, add_context, build_chunks, doc_label_from_blocks
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


def parse_pdf(data: bytes, *, router: LLMRouter | None = None,
              store: ArtifactStore | None = None, parser: str = "auto",
              on_page=None
              ) -> tuple[str, str, list[list[Block]], list[tuple[float, float]], int]:
    """Parse every page (cache-aware). Returns (hash, parser, pages_blocks, sizes, n_cached).
    `on_page(done, total)` is called after each page — progress for background ingestion."""
    h = doc_hash(data)
    doc = fitz.open(stream=data, filetype="pdf")
    n = len(doc)
    if n == 0:
        raise IngestError("empty PDF")
    if parser == "auto":
        parser = "textlayer" if textlayer.has_text_layer(doc) else "cloud_ocr"
    if parser == "cloud_ocr":
        router = router or LLMRouter()
        if not router.available("vision"):
            raise IngestError("scanned PDF needs a vision-capable key "
                              "(GEMINI_API_KEY or GROQ_API_KEY)")

    relation = None
    pages_blocks: list[list[Block]] = []
    cached = 0
    for i in range(n):
        saved = store.get_page(h, i + 1, parser) if store else None
        if saved is not None:
            pages_blocks.append([block_from_dict(d) for d in saved])
            cached += 1
        else:
            if parser == "textlayer":
                blocks = textlayer.extract_page(doc[i], i + 1, relation)
            else:
                blocks = cloud_ocr.extract_page(doc[i], i + 1, router)
            if store:
                store.save_page(h, i + 1, parser, [block_to_dict(b) for b in blocks])
            pages_blocks.append(blocks)
        if on_page:
            on_page(i + 1, n)

    sizes = [(doc[i].rect.width, doc[i].rect.height) for i in range(n)]
    # furniture marking is document-wide and cheap — recomputed every run (never cached),
    # so cached raw pages and freshly parsed ones are treated identically
    textlayer.mark_repeated_furniture(pages_blocks, sizes)
    return h, parser, pages_blocks, sizes, cached


def ingest(data: bytes, *, doc_id: str | None = None, router: LLMRouter | None = None,
           store: ArtifactStore | None = None, parser: str = "auto",
           contextual: bool | None = None, on_page=None) -> IngestResult:
    """Full ingestion for one PDF: parse (cached) → chunk → contextual prefixes (cached)."""
    router = router or LLMRouter()
    store = store if store is not None else ArtifactStore()
    h, parser, pages_blocks, sizes, cached = parse_pdf(
        data, router=router, store=store, parser=parser, on_page=on_page)
    doc_id = doc_id or h

    all_blocks = [b for pb in pages_blocks for b in pb]
    chunks, sections = build_chunks(all_blocks, doc_id)
    label = doc_label_from_blocks(all_blocks)

    contextual = settings.contextual_chunks if contextual is None else contextual
    n_ctx = add_context(chunks, router, store, doc_label=label) if contextual else 0

    return IngestResult(
        doc_id=doc_id, parser=parser, page_count=len(pages_blocks), sizes=sizes,
        pages_blocks=pages_blocks, chunks=chunks, sections=sections, doc_label=label,
        contextualized=n_ctx, cached_pages=cached,
        stats={"blocks": len(all_blocks), "chunks": len(chunks),
               "sections": len(sections),
               "tables": sum(1 for c in chunks if c.type == "table")})


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

    data = open(args.pdf, "rb").read()
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
