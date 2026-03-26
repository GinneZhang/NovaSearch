# Ragas Evaluation Migration

## What Changed

AsterScope benchmark scoring now uses Ragas as the primary evaluation framework.

The benchmark runner still:

- loads benchmark datasets
- ingests reference documents into AsterScope
- runs `/ask`
- preserves AsterScope runtime preflight and chain/debug metadata

But it no longer computes custom benchmark summary metrics such as hand-rolled faithfulness, custom context precision, EM/F1, or title-hit metrics in the final summary path.

## Ragas Metrics

The runner now reports this default core set when the required inputs are available:

- `context_precision`
- `context_recall`
- `answer_relevancy`
- `faithfulness`

These four are the default because they are the most common and interpretable set for
RAG-style paper tables:

- `context_precision`
- `context_recall`
- `answer_relevancy`
- `faithfulness`

Optional heavier metrics can still be enabled explicitly with:

```bash
BENCHMARK_RAGAS_METRICS=context_precision,context_recall,answer_relevancy,faithfulness,context_entities_recall,noise_sensitivity
```

### Context Modes

Two evaluation modes are supported:

1. `BENCHMARK_CONTEXT_MODE=generation`
   Uses AsterScope `generation_contexts`

2. `BENCHMARK_CONTEXT_MODE=retrieval`
   Uses AsterScope `retrieval_contexts`

## Benchmark Tracks

The current standardized tracks are:

- `hotpotqa`
- `squad`
- `squad_v2`

`squad` and `squad_v2` fit the Ragas migration better than the previous BeIR/NQ track because they provide official reference answers and reference contexts.

## What AsterScope Still Preserves

Ragas is now the official scorer, but AsterScope internal observability is still preserved per sample and in the benchmark summary. This includes fields such as:

- `chain_mode_selected`
- `chain_activation_reason`
- `candidate_chains`
- `selected_chains`
- `second_hop_candidates_added`
- `second_hop_candidates_kept`
- `bridge_budget_used`
- `weak_bridge_candidates_dropped`
- `final_context_bridge_fraction`
- `final_context_direct_support_fraction`
- `chain_score_components`
- `chain_vs_standalone_mix`

These fields do not affect Ragas scoring. They remain diagnostic-only.

## Notes On Metric Availability

- `context_entities_recall` and `noise_sensitivity` are optional because they are slower and
  less commonly used as primary headline metrics.
- They also require reference answers.
- For answerless rows, these metrics are skipped explicitly rather than being backfilled with custom substitutes.
- `context_precision` / `context_recall` can fall back to reference-context-based Ragas variants when a benchmark row has reference contexts but no canonical answer string.

## Recommended Next Step

Run fresh standardized reruns on the current adaptive/pruned chain-aware system:

```bash
BENCHMARK_NAME=hotpotqa BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation BENCHMARK_CONTEXT_MODE=generation python tests/benchmark_rag.py
```

```bash
BENCHMARK_NAME=hotpotqa BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation BENCHMARK_CONTEXT_MODE=retrieval python tests/benchmark_rag.py
```

```bash
BENCHMARK_NAME=squad BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation BENCHMARK_CONTEXT_MODE=generation python tests/benchmark_rag.py
```

```bash
BENCHMARK_NAME=squad_v2 BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation BENCHMARK_CONTEXT_MODE=retrieval python tests/benchmark_rag.py
```
