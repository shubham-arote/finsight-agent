# FinRAGBench-V (English, sampled subset) — full-agent run

Date: 2026-07-08 · Agent: **offline (sparse + extractive)** · Judge: **offline** · Sample: **24 questions (seed 42)** over 18 source PDFs (1963 pages → 40037 chunks, 0 VLM-enriched, 37 pages skipped, 1 docs excluded as scanned-without-vision-key) · k=5 · doc-scoped (oracle document) retrieval

| Metric | Score |
|---|---|
| Retrieval hit@5 / MRR | **17%** / **0.074** |
| Retrieval recall@5 (gold pages) | **17%** |
| Citation precision / recall (page-level) | **2%** / **4%** |
| Verified rate (figures trace to citations) | **100%** |
| Answer rate | **83%** |
| Correctness (LLM-judge, n=0) | **—** |

## Per-category

| Category | n | hit@k | judge |
|---|---|---|---|
| Chart-Information Extraction | 5 | 20% | — |
| Chart-Numerical Calculation | 1 | 0% | — |
| Chart-Time Sensitive | 1 | 100% | — |
| Table-Compare and Sort | 6 | 0% | — |
| Table-Numerical Calculation | 7 | 29% | — |
| Text Inference | 2 | 0% | — |
| Text-MultiPage | 2 | 0% | — |

_Sampled-subset protocol (full EN corpus is 51k pages). Our system is text-first with block-level citations; the paper's RGenCite baseline is a page-image pipeline — numbers are indicative, not strictly comparable. Citation P/R here is page-level; FinRAGBench-V's box-level protocol is a planned addition using its citation_labels set._