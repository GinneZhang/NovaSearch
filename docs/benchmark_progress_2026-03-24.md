# Benchmark Progress - 2026-03-24

## Scope

This note captures the latest benchmark state after the external GitHub pattern-mining round and evidence-quality refactor.

## HotpotQA 100-sample

Previous reference run:

- Hit Rate @ 5: `0.92`
- MRR @ 5: `0.79`
- Faithfulness: `0.800`
- Legacy Context Precision: `0.700`
- Generation Context Precision: `0.342`
- Supporting Context Precision: `0.650`
- Noise Reduction Ratio: `4.0x`
- Cost Savings: `75.03%`

Latest run:

- Hit Rate @ 5: `0.88`
- MRR @ 5: `0.78`
- Faithfulness: `0.937`
- Legacy Context Precision: `0.526`
- Generation Context Precision: `0.320`
- Supporting Context Precision: `0.526`
- Noise Reduction Ratio: `4.1x`
- Cost Savings: `75.70%`

Interpretation:

- Retrieval stayed broadly stable but did not improve.
- Faithfulness improved materially.
- Final evidence quality did not improve on HotpotQA; supporting and generation precision both regressed relative to the earlier reference run.

Trace:

- [kpi_trace_hotpot_post_research.json](/Users/ginnezhang/Documents/Playground/NovaSearch/docs/kpi_trace_hotpot_post_research.json)

## Natural Questions 100-sample

Previous reference run:

- Hit Rate @ 5: `0.88`
- MRR @ 5: `0.81`
- Faithfulness: `0.767`
- Legacy Context Precision: `0.556`
- Generation Context Precision: `0.352`
- Supporting Context Precision: `0.500`
- Noise Reduction Ratio: `0.7x`
- Cost Savings: `-51.42%`

Latest run:

- Hit Rate @ 5: `0.85`
- MRR @ 5: `0.81`
- Faithfulness: `0.935`
- Legacy Context Precision: `0.525`
- Generation Context Precision: `0.396`
- Supporting Context Precision: `0.550`
- Noise Reduction Ratio: `0.7x`
- Cost Savings: `-47.20%`

Interpretation:

- Retrieval stayed close to the earlier reference run.
- Faithfulness improved strongly.
- Supporting and generation context precision improved modestly.
- The post-refactor behavior looks more positive on NQ than on HotpotQA.

Trace:

- [kpi_trace_nq_post_research.json](/Users/ginnezhang/Documents/Playground/NovaSearch/docs/kpi_trace_nq_post_research.json)

## Current Read

- The latest retrieval/evidence changes increased answer conservatism and grounding discipline.
- That helped faithfulness on both benchmarks.
- However, the final evidence-set architecture is still not reliably solving Hotpot-style multi-hop evidence assembly.
- The next retrieval-quality iteration should focus on multi-hop evidence inheritance and supporting-to-generation transfer, not broad recall expansion.
