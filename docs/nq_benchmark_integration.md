# Natural Questions Benchmark Integration Note

## Why NovaSearch uses BeIR/NQ instead of `nq_open`

NovaSearch's benchmark runner is built around a concrete retrieval pipeline:

1. ingest a benchmark-local corpus into NovaSearch
2. run live retrieval and grounded answering through `/ask`
3. evaluate retrieval quality, evidence quality, and answer behavior

`nq_open` is easy to load, but it only provides `question + answers`. It does **not** provide a benchmark-local corpus or passage-level relevance labels that fit NovaSearch's current ingestion-and-retrieval evaluation loop.

For this reason, the new Natural Questions track is integrated through **BeIR/NQ-style data**:

- `Hyukkyu/beir-nq` for `queries` and `corpus`
- `BeIR/nq-qrels` for relevance judgments

This is the cleanest fit for NovaSearch because it preserves the existing benchmark architecture:

- a real corpus is ingested into NovaSearch
- relevant passages are defined by qrels rather than synthetic heuristics
- retrieval hit metrics remain honest and benchmark-grounded

## What this benchmark can and cannot claim

### What it measures well

- retrieval hit rate and MRR against relevant passages
- faithfulness of generated answers to retrieved context
- final evidence-set quality through context-precision style auditing
- context size, noise indicators, and evidence composition

### What it does not measure directly

This BeIR/NQ package does **not** include canonical gold answers in the same way `HotpotQA` does. Because of that:

- answer `EM` / `F1` is **not reported** for the NQ track
- Hotpot-style supporting-facts supervision is **not forced** onto NQ

This is intentional. NovaSearch reports only metrics that are natural for the underlying annotation structure of each benchmark.

## Interpretation guidance

- If `NQ` looks much better than `HotpotQA`, NovaSearch is likely stronger on broad open-domain retrieval than on multi-hop evidence assembly.
- If both look weak, the problem is probably earlier in the retrieval stack, not only in multi-hop packing.
- If retrieval is strong but context precision is still weak, NovaSearch is finding the right region of the corpus but still assembling noisy final evidence sets.
