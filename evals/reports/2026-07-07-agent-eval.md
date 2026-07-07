# Agent evaluation (full loop)

Date: 2026-07-07 · Agent mode: **offline (sparse + extractive)** · Judge: **offline** · Dataset: sample_report_qa.json (21 in-scope + 4 out-of-scope)

| Metric | Score |
|---|---|
| Abstain accuracy (OOS refused) | **75%** |
| Answer rate (in-scope answered) | **100%** |
| Citation hit (cited source on an expected page) | **90%** |
| Verified rate (zero unverified figures) | **100%** |
| Claim coverage (structured citations present) | **100%** |
| Correctness (LLM-judge, n=0) | **—** |

| Scope | Question | Task | Abstained | Cited p. | Hit | Verified | Judge |
|---|---|---|---|---|---|---|---|
| in | What was total revenue in FY26? | calc | ✗ | [2, 4, 5] | ✓ | ✓ | — |
| in | What was operating profit in FY26? | qa | ✗ | [2, 3, 4] | ✓ | ✓ | — |
| in | What were net assets at the period end? | qa | ✗ | [2, 5] | ✓ | ✓ | — |
| in | What total dividend per share was recommended for the year? | calc | ✗ | [2, 3, 6] | ✓ | ✓ | — |
| in | What was profit before tax in FY26? | qa | ✗ | [3, 4, 6] | ✓ | ✓ | — |
| in | What was the effective tax rate for the year? | qa | ✗ | [2, 3, 6] | ✓ | ✓ | — |
| in | How much was capital expenditure? | qa | ✗ | [3, 6] | ✓ | ✓ | — |
| in | What was the operating margin? | calc | ✗ | [3, 6] | ✓ | ✓ | — |
| in | What were total assets in FY26? | calc | ✗ | [2, 5] | ✓ | ✓ | — |
| in | What was cost of sales in FY26? | qa | ✗ | [2, 4] | ✓ | ✓ | — |
| in | What was profit after tax in FY26? | qa | ✗ | [3, 4, 6] | ✓ | ✓ | — |
| in | What were earnings per share? | calc | ✗ | [2, 3] | ✓ | ✓ | — |
| in | By how much did revenue grow year on year? | calc | ✗ | [2, 6] | ✗ | ✓ | — |
| in | What were current assets in FY26? | qa | ✗ | [2, 5] | ✓ | ✓ | — |
| in | What was gross profit in FY26? | qa | ✗ | [2, 3, 4] | ✓ | ✓ | — |
| in | What was revenue in the prior year, FY25? | calc | ✗ | [2, 6] | ✗ | ✓ | — |
| in | What were total liabilities in FY26? | calc | ✗ | [2, 4, 5] | ✓ | ✓ | — |
| in | What was profit before tax in the prior year, FY25? | calc | ✗ | [3, 4, 6] | ✓ | ✓ | — |
| in | By how much did net assets change year on year? | calc | ✗ | [2, 5] | ✓ | ✓ | — |
| in | Did operating profit rise or fall versus the prior year, and | calc | ✗ | [2, 3, 4] | ✓ | ✓ | — |
| in | Approximately what was the gross profit margin in FY26? | calc | ✗ | [3, 4, 6] | ✓ | ✓ | — |
| oos | Who is the chief executive officer? | qa | ✓ | [] | — | ✓ | — |
| oos | What was the company's share price at year end? | calc | ✗ | [2, 3, 6] | — | ✓ | — |
| oos | How many employees does the company have? | qa | ✓ | [] | — | ✓ | — |
| oos | What is the company's carbon emissions reduction target? | qa | ✓ | [] | — | ✓ | — |

_Synthetic 6-page document: a regression smoke set, not a generalization claim. The judge is an independent model family (Gemini/Vertex) from the answering models. External benchmark (FinRAGBench-V) is the Phase 5 target._