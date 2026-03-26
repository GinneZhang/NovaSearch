# Hotpot Multi-Hop Refinement Notes

## Why This Change

The current AsterScope chain-aware pipeline already built many candidate chains on HotpotQA, but the final evidence set still retained almost no bridge evidence. In the latest 100-sample Ragas run:

- `avg_candidate_chains = 24.75`
- `avg_selected_chains = 5.7`
- `avg_bridge_chunks_kept = 0.17`

This means the system was doing chain construction, but later stages were still effectively flattening the result into mostly direct evidence.

## External Research Signals

### MDR / Learning to Retrieve Reasoning Paths
Source: [ICLR 2020 OpenReview](https://openreview.net/forum?id=SJgVHkrYDH)

Key mechanism:
- retrieve evidence sequentially while conditioning on previously retrieved documents
- score reasoning paths rather than isolated paragraphs

Relevant takeaway for AsterScope:
- once a passage has already been identified as a bridge hop in a reasoning path, later stages should not require it to look like a strong direct-answer chunk before it survives.

### Beam Retrieval
Source: [NAACL 2024 ACL Anthology](https://aclanthology.org/2024.naacl-long.96/)

Key mechanism:
- maintain multiple partial hypotheses during retrieval
- reduce early retrieval errors by preserving plausible partial paths

Relevant takeaway for AsterScope:
- if a chain has already been selected, its bridge member should get path-level credit in final evidence selection and packing.
- otherwise chain-aware retrieval degenerates back into flat top-N reranking.

### Adaptive-RAG
Source: [NAACL 2024 OpenReview](https://openreview.net/forum?id=vZmtOnp6KF)

Key mechanism:
- apply heavier retrieval/reasoning only when query complexity warrants it

Relevant takeaway for AsterScope:
- we should keep the existing adaptive chain mode logic, but when `full` mode is selected, the system should actually preserve chain-completing evidence rather than immediately collapsing back to direct-only evidence.

## Concrete Fix Applied

Two targeted changes were added:

1. Chain-role propagation in retrieval finalization
- `primary_chain_member_role=bridge` and `chain_bridge_signal` now contribute directly to bridge classification in final evidence selection.
- bridge candidates no longer need explicit follow-up-query lexical overlap to count as bridge evidence.

2. Chain-role propagation in generation packing
- generation packing now treats chain-annotated bridge members as bridge candidates even when their bridge utility is mostly path-derived rather than lexical.
- chain-seeded rows are allowed to survive into the packed generation context based on chain role, not only on flat answer/bridge overlap scores.

## Why This Is Generalizable

This refinement does **not** use Hotpot-specific templates, supporting-facts labels, or relation keyword rules.

It only assumes a general multi-hop principle:

- if upstream retrieval already identified a passage as a bridge member of a plausible reasoning chain,
- downstream evidence selection should preserve that structured signal instead of discarding it because the passage is not directly answer-bearing on its own.
