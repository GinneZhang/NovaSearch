# Evaluation Migration

## What Changed

AsterScope benchmark scoring no longer uses Ragas in the primary benchmark path.

The benchmark runner still:

- loads benchmark datasets
- ingests reference documents into AsterScope
- runs `/ask`
- preserves AsterScope runtime preflight and chain/debug metadata

The official benchmark summary now uses fast custom metrics instead of LLM-judge metrics so iteration speed stays practical during retrieval work.

## Official Metrics

The runner now reports:

- `Answer EM`
- `Answer F1`
- `Hit Rate @ 5`
- `MRR @ 5`
- `Noise Reduction Ratio`
- `Cost Savings`

These are deterministic, cheap to compute, and align better with fast ablation loops.

## Benchmark Tracks

The current standardized tracks are:

- `hotpotqa`
- `musique`
- `squad_v2`

## What AsterScope Still Preserves

Even though Ragas is no longer the official scorer, AsterScope internal observability is still preserved per sample and in the benchmark summary. This includes fields such as:

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

These fields do not affect official scoring. They remain diagnostic-only.

## Recommended Next Step

Run fresh standardized reruns on the current adaptive/pruned chain-aware system:

```bash
BENCHMARK_NAME=hotpotqa BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation python tests/benchmark_rag.py
```

```bash
BENCHMARK_NAME=musique BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation python tests/benchmark_rag.py
```

```bash
BENCHMARK_NAME=squad_v2 BENCHMARK_SAMPLE_SIZE=100 BENCHMARK_SPLIT=validation python tests/benchmark_rag.py
```
