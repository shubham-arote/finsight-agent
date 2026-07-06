# Retrieval baseline

Date: 2026-07-06 · Stack: **sparse + lexical rerank + exact lookup (keyless)** · Dataset: sample_report_qa.json (21 in-scope questions) · k=5

| hit@5 | MRR |
|---|---|
| **100%** | **0.902** |

| Question | Expected p. | Got (top-k) | Rank |
|---|---|---|---|
| What was total revenue in FY26? | [3, 4] | [5, 4, 2, 2, 3] | 2 |
| What was operating profit in FY26? | [4] | [4, 3, 2, 5, 3] | 1 |
| What were net assets at the period end? | [5] | [5, 5, 2, 6, 6] | 1 |
| What total dividend per share was recommended for the year? | [2] | [2, 3, 6, 2, 5] | 1 |
| What was profit before tax in FY26? | [3, 4] | [4, 3, 6, 2, 5] | 1 |
| What was the effective tax rate for the year? | [6] | [6, 2, 3, 2, 6] | 1 |
| How much was capital expenditure? | [6] | [6, 3] | 1 |
| What was the operating margin? | [3] | [3, 6, 3, 2, 4] | 1 |
| What were total assets in FY26? | [5] | [5, 2, 5, 2, 4] | 1 |
| What was cost of sales in FY26? | [4] | [4, 2, 2, 3, 5] | 1 |
| What was profit after tax in FY26? | [4] | [4, 3, 6, 2, 5] | 1 |
| What were earnings per share? | [3] | [3, 2] | 1 |
| By how much did revenue grow year on year? | [3] | [2, 6, 2, 3, 4] | 4 |
| What were current assets in FY26? | [5] | [5, 2, 5, 4] | 1 |
| What was gross profit in FY26? | [4] | [4, 3, 2, 5, 6] | 1 |
| What was revenue in the prior year, FY25? | [4] | [2, 6, 2, 3, 4] | 5 |
| What were total liabilities in FY26? | [5] | [5, 2, 4, 2] | 1 |
| What was profit before tax in the prior year, FY25? | [4] | [4, 3, 6, 2, 2] | 1 |
| By how much did net assets change year on year? | [5] | [5, 2, 2, 6, 5] | 1 |
| Did operating profit rise or fall versus the prior year, and by roughly how much? | [4] | [4, 2, 3, 2, 3] | 1 |
| Approximately what was the gross profit margin in FY26? | [4] | [4, 3, 6, 2, 3] | 1 |

_Synthetic 6-page document — a smoke baseline for regressions, not a generalization claim. The external benchmark (FinRAGBench-V) lands in Phase 5._