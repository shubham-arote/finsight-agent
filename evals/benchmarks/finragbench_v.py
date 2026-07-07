"""finragbench_v.py — adapter for FinRAGBench-V (EMNLP 2025), English subset.

Dataset: https://huggingface.co/datasets/zhaosuifeng/FinRAGBench-V
Expected local layout (user-downloaded — the corpus is large):

    <data_dir>/queries/queries_en.json     [{query-id, query, answer, category,
                                             answer_type, from_pages}, ...]
    <data_dir>/pdfs/<name>.pdf             extracted from pdfs_for_QA/pdf_en.tar.gz

`query-id` embeds the source PDF name ("<name>.pdf" + page(s) + suffix); `from_pages`
are the gold evidence pages (1-based in the original PDF). We ingest the actual PDFs
through our own pipeline (text-layer/OCR routed), which is the honest way to benchmark
this system — the paper's baseline (RGenCite) retrieves page *images* instead.

Sampling is seeded and disclosed: the full English corpus (51k pages) exceeds free-tier
budgets; a sampled subset + its source documents is the documented protocol.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchCase:
    qid: str
    question: str
    reference: str
    category: str
    answer_type: str
    doc_name: str                 # source PDF file name (the doc_id we ingest under)
    gold_pages: list[int] = field(default_factory=list)


_QID = re.compile(r"^(?P<stem>.+)_(?:multipage_\d+-\d+|\d+)\.png-\d+$")


def _doc_from_qid(qid: str, stems: list[str] | None = None) -> str | None:
    """query-id = '<pdf stem>_<page>.png-<n>' (or '_multipage_<a>-<b>.png-<n>').

    When the pdfs directory is available, resolve by longest-stem prefix match against
    the actual files (robust to stems that themselves end in digits); otherwise fall
    back to the regex parse."""
    if stems:
        for stem in stems:                     # pre-sorted longest-first
            if qid.startswith(stem + "_"):
                return stem + ".pdf"
    m = _QID.match(qid)
    return f"{m.group('stem')}.pdf" if m else None


def load_cases(data_dir: str | Path, sample: int | None = None, seed: int = 42,
               categories: list[str] | None = None) -> list[BenchCase]:
    data_dir = Path(data_dir)
    qfile = data_dir / "queries" / "queries_en.json"
    entries = json.loads(qfile.read_text(encoding="utf-8"))
    pdf_dir = data_dir / "pdfs"
    stems = (sorted((p.stem for p in pdf_dir.glob("*.pdf")), key=len, reverse=True)
             if pdf_dir.exists() else None)
    cases: list[BenchCase] = []
    for e in entries:
        qid = e.get("query-id") or e.get("query_id") or ""
        doc = _doc_from_qid(qid, stems)
        if not doc or not e.get("query"):
            continue
        if categories and e.get("category") not in categories:
            continue
        pages = e.get("from_pages") or []
        if isinstance(pages, int):
            pages = [pages]
        elif isinstance(pages, str):                     # "16" or "1, 4"
            pages = [int(x) for x in re.findall(r"\d+", pages)]
        cases.append(BenchCase(
            qid=qid, question=e["query"], reference=str(e.get("answer") or ""),
            category=e.get("category", "unknown"),
            answer_type=e.get("answer_type", "short"),
            doc_name=doc, gold_pages=[int(p) for p in pages]))
    # keep only cases whose source PDF is actually present locally
    have = {p.name for p in (data_dir / "pdfs").glob("*.pdf")} if (data_dir / "pdfs").exists() else set()
    if have:
        cases = [c for c in cases if c.doc_name in have]
    if sample is not None and sample < len(cases):
        cases = random.Random(seed).sample(cases, sample)
    return cases


def pdf_bytes(data_dir: str | Path, doc_name: str) -> bytes:
    return (Path(data_dir) / "pdfs" / doc_name).read_bytes()
