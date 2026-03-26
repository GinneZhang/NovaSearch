# NovaSearch Benchmark Report

## Scope

This report uses benchmark-native, paper-style task metrics for external reporting.

- `HotpotQA`: `Answer EM / F1`
- `MuSiQue`: `Answer EM / F1`
- `ASQA`: diagnostic `Answer EM / F1` only

`Ragas` metrics and NovaSearch chain/debug fields are retained for internal optimization, but are intentionally excluded from this public-facing report.

## Experimental Setup

| Dataset | Public Eval Split Used Here | Public Split Size | This Run | Primary Report Metric |
|---|---|---:|---:|---|
| HotpotQA | `validation` | 7,405 | 100 | Answer EM / F1 |
| MuSiQue | `validation` | 2,417 | 100 | Answer EM / F1 |
| ASQA | `dev` | 948 | 91 | Diagnostic Answer EM / F1 only |

Run notes:

- Runtime: `http://127.0.0.1:8032`
- Code path: current chain-aware NovaSearch with adaptive chain activation, budgeted chain expansion, and concise grounded answer projection
- `Ragas` was skipped for these report-mode runs: `BENCHMARK_SKIP_RAGAS=true`

## Main Results

| Dataset | Answer EM | Answer F1 |
|---|---:|---:|
| HotpotQA | 22.0 | 28.9 |
| MuSiQue | 20.0 | 30.2 |
| ASQA | 0.0 | 11.5 |

Interpretation:

- `HotpotQA` and `MuSiQue` improved materially after the current refinement pass.
- `ASQA` remains weak under `EM/F1`, which is expected because ASQA is a long-form grounded QA benchmark and does not use extractive-style `EM/F1` as its main official metric.

## Efficiency Table

| Dataset | Noise Reduction Ratio | Cost Savings | Avg Final Context Count |
|---|---:|---:|---:|
| HotpotQA | 4.1x | 75.80% | 3.9 |
| MuSiQue | 0.8x | -22.03% | 3.5 |
| ASQA | 0.7x | -45.92% | 2.7 |

Read this table as an engineering side view, not as the main task score:

- `HotpotQA` remains the cleanest from a token-efficiency perspective.
- `MuSiQue` and `ASQA` still pay a large context cost relative to the size of their provided benchmark corpora.

## Representative Published Upper Bounds

These are task-difficulty references from representative strong papers. They are not all directly comparable to NovaSearch’s current report numbers.

| Dataset | Representative System | Venue | Official Metric | Published Result |
|---|---|---|---|---|
| HotpotQA | Beam Retrieval | NAACL 2024 | Answer EM / F1 | 72.69 / 85.04 |
| MuSiQue | Beam Retrieval | NAACL 2024 | Answer F1 | 69.2 |
| ASQA | Query Refinement Prompts (`PaLM 540B + PT`) | ACL 2023 | ROUGE-L / DISAMBIG-F1 / DR | 40.7 / 27.8 / 33.5 |

## Gap to Published References

| Dataset | NovaSearch Current | Reference | Absolute Gap |
|---|---|---|---|
| HotpotQA | EM 22.0 / F1 28.9 | EM 72.69 / F1 85.04 | EM -50.69 / F1 -56.14 |
| MuSiQue | F1 30.2 | F1 69.2 | F1 -39.0 |
| ASQA | Answer F1 11.5 (diagnostic only) | ROUGE-L 40.7 / DISAMBIG-F1 27.8 / DR 33.5 | Not directly comparable |

Interpretation:

- `HotpotQA`: NovaSearch is still far below strong full-benchmark multi-hop systems, but it is no longer near-zero after the current refinement pass.
- `MuSiQue`: the gap is still large, but the current code now lands in a more meaningful range for further multi-hop retrieval work.
- `ASQA`: the current answer projection is optimized for short benchmark answers; ASQA should ultimately be judged with its official long-form metrics rather than `EM/F1`.

## Change vs Previous Internal Report-Mode Baseline

| Dataset | Previous Answer EM / F1 | Current Answer EM / F1 | Delta |
|---|---:|---:|---:|
| HotpotQA | 0.0 / 6.0 | 22.0 / 28.9 | +22.0 / +22.9 |
| MuSiQue | 0.0 / 6.6 | 20.0 / 30.2 | +20.0 / +23.6 |
| ASQA | 0.0 / 26.2 | 0.0 / 11.5 | 0.0 / -14.7 |

Current reading:

- The latest fix clearly helps factoid-style multi-hop answer evaluation.
- It does **not** solve ASQA’s official evaluation problem, because ASQA is fundamentally a long-form benchmark.
- The next ASQA step should be to add official long-form ASQA metrics rather than forcing `EM/F1` to carry that benchmark.

## Trace Files

- [kpi_trace_hotpot_report_refined.json](/Users/ginnezhang/Documents/Playground/NovaSearch/docs/kpi_trace_hotpot_report_refined.json)
- [kpi_trace_musique_report_refined.json](/Users/ginnezhang/Documents/Playground/NovaSearch/docs/kpi_trace_musique_report_refined.json)
- [kpi_trace_asqa_report_refined.json](/Users/ginnezhang/Documents/Playground/NovaSearch/docs/kpi_trace_asqa_report_refined.json)
