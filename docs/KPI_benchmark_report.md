# AsterScope Benchmark Report

## Scope

This report uses benchmark-native, paper-style task metrics for external reporting.

- `HotpotQA`: `Answer EM / F1`, `Hit Rate @ 5`, `MRR @ 5`
- `MuSiQue`: `Answer EM / F1`, `Hit Rate @ 5`, `MRR @ 5`

`2WikiMultiHopQA` is kept as an optional experiment track, not a default report benchmark.  
Internal `Ragas` metrics and AsterScope chain/debug fields are still useful for optimization, but they are intentionally excluded from this public-facing report.

## Experimental Setup

| Dataset | Public Eval Split Used Here | Public Split Size | This Run | Primary Report Metrics |
|---|---|---:|---:|---|
| HotpotQA | `validation` | 7,405 | 100 | Answer EM / F1, Hit@5, MRR@5 |
| MuSiQue | `validation` | 2,417 | 100 | Answer EM / F1, Hit@5, MRR@5 |

Run notes:

- Runtime: warm benchmark runtime on `http://127.0.0.1:8060`
- Query concurrency: `8`
- Explicit warmup queries: `2`
- Code path: current chain-aware AsterScope with recursive multi-hop retrieval, bridge-aware packing, and concise benchmark answer projection
- Official summary path uses custom benchmark metrics; `Ragas` is not part of the external report

## Main Results

| Dataset | Hit Rate @ 5 | MRR @ 5 | Answer EM | Answer F1 |
|---|---:|---:|---:|---:|
| HotpotQA | 0.920 | 0.755 | 0.330 | 0.427 |
| MuSiQue | 0.970 | 0.879 | 0.260 | 0.393 |

Interpretation:

- `HotpotQA` remains the main benchmark because it best exposes evidence-chain assembly quality under realistic multi-hop pressure.
- `MuSiQue` is retained as the second main benchmark because it gives stronger answer-level signal than `2WikiMultiHopQA` on the current codebase.
- `MuSiQue` currently has better retrieval metrics than `HotpotQA`, but answer conversion still lags.

## Efficiency Table

| Dataset | Noise Reduction Ratio | Cost Savings | Avg Final Context Count |
|---|---:|---:|---:|
| HotpotQA | 4.5x | 77.81% | 3.9 |
| MuSiQue | 0.8x | -24.92% | 3.9 |

Read this table as an engineering side view, not as the main task score:

- `HotpotQA` remains much cleaner from a token-efficiency perspective.
- `MuSiQue` currently requires more context than the provided benchmark corpus would ideally justify, which indicates room for better pruning and answer-targeted packing.

## Optional Experiment Track

`2WikiMultiHopQA` was run as a comparison track but is **not** retained as a default report benchmark.

| Dataset | Hit Rate @ 5 | MRR @ 5 | Answer EM | Answer F1 |
|---|---:|---:|---:|---:|
| 2WikiMultiHopQA | 0.990 | 0.855 | 0.210 | 0.250 |

Why it is not retained as a main benchmark:

- retrieval is extremely strong, but answer conversion remains weaker than on `MuSiQue`
- it is currently less discriminative than `MuSiQue` for the chain-aware retrieval changes we are evaluating

## Representative Published Upper Bounds

These are task-difficulty references from representative strong papers. They are not directly comparable to AsterScope’s current report numbers in every detail, but they remain useful scale references.

| Dataset | Representative System | Venue | Official Metric | Published Result |
|---|---|---|---|---|
| HotpotQA | Beam Retrieval | NAACL 2024 | Answer EM / F1 | 72.69 / 85.04 |
| MuSiQue | Beam Retrieval | NAACL 2024 | Answer F1 | 69.2 |

## Gap to Published References

| Dataset | AsterScope Current | Reference | Absolute Gap |
|---|---|---|---|
| HotpotQA | EM 33.0 / F1 42.7 | EM 72.69 / F1 85.04 | EM -39.69 / F1 -42.34 |
| MuSiQue | F1 39.3 | F1 69.2 | F1 -29.9 |

Interpretation:

- `HotpotQA`: still meaningfully behind strong full-benchmark multi-hop systems, but no longer in the near-zero regime.
- `MuSiQue`: the gap remains large, but this is now the cleaner second benchmark for tracking multi-hop answer conversion improvements.

## Trace Files

- [kpi_trace.json](/Users/ginnezhang/Documents/Playground/NovaSearch/docs/kpi_trace.json)
- [kpi_trace_musique.json](/Users/ginnezhang/Documents/Playground/NovaSearch/docs/kpi_trace_musique.json)
- [kpi_trace_two_wiki.json](/Users/ginnezhang/Documents/Playground/NovaSearch/docs/kpi_trace_two_wiki.json)
