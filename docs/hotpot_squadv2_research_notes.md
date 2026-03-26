# HotpotQA and SQuAD 2.0 Targeted Research Notes

## Why This Pass

The latest trustworthy benchmarks show a split pattern:

- `SQuAD 2.0` is better than `HotpotQA` on the same runtime.
- `HotpotQA` still underperforms mainly because bridge evidence is not preserved strongly enough through final selection and packing.
- `SQuAD 2.0` remains useful as a fast check that chain-aware retrieval does not over-expand simple or mostly single-hop questions.

## External Methods Reviewed

### MDR

Primary source:

- [Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering](https://arxiv.org/abs/1911.10470)

Mechanism relevant to NovaSearch:

- retrieve later hops conditioned on earlier evidence
- keep reasoning paths, not just flat passages
- let the reader score whole paths, not isolated chunks

Adaptation for NovaSearch:

- preserve complete chains longer through final candidate selection
- do not let a chain-backed bridge chunk compete as if it were just another background chunk

### Beam Retrieval

Primary sources:

- [End-to-End Beam Retrieval for Multi-Hop Question Answering](https://openreview.net/forum?id=rFKfiuOdZBg)
- [Official repository](https://github.com/canghongjian/beam_retriever)

Mechanism relevant to NovaSearch:

- maintain several partial path hypotheses
- expand a path with bounded beam size instead of flattening candidates too early
- score and prune at path level

Adaptation for NovaSearch:

- preserve a small bundle from high-value complete chains
- prefer one direct/support chunk plus one bridge chunk from the same chain when the chain is already strong

### Adaptive-RAG

Primary source:

- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://aclanthology.org/2024.naacl-long.389/)

Mechanism relevant to NovaSearch:

- simple questions should not pay the cost of heavy multi-step retrieval
- strong direct evidence should trigger a lighter path

Adaptation for NovaSearch:

- more aggressive `bypass` for simple, low-bridge, high-direct-evidence queries
- use existing runtime signals instead of a dataset-specific classifier

### ChainRAG

Primary sources:

- [Official repository](https://github.com/nju-websoft/ChainRAG)

Mechanism relevant to NovaSearch:

- missing key entities in decomposition create "lost-in-retrieval"
- progressive retrieval and rewriting helps preserve the question-to-evidence chain

Adaptation for NovaSearch:

- downstream stages should respect already identified bridge entities and chain roles
- final context packing should preserve chain-backed support/bridge pairs instead of re-flattening them

## Applied Refinement in This Pass

This pass keeps the current chain-aware architecture and adds two narrow improvements:

1. More selective bypass for simple, strong-direct-evidence queries
2. Bundle preservation for complete high-value chains

The bundle logic is intentionally strict:

- it preserves at most a tiny number of rows per chain
- it still respects caps and pruning
- it does not expand retrieval width

## Expected Effect

- `SQuAD 2.0`: should benefit from stronger simple-query bypass and reduced unnecessary chain competition
- `HotpotQA`: should benefit from keeping one support chunk plus one bridge chunk together when a chain is already strong
