# NovaSearch Retrieval & Evidence Pipeline Audit

## Scope
This audit targets general multi-hop grounded QA quality in NovaSearch, with emphasis on **final evidence-set quality** and **context precision**.

The changes and recommendations below are intentionally dataset-agnostic and do not depend on benchmark-specific labels, templates, or supporting-facts artifacts.

## Phase 1 — Where Noise Enters

### 1) Merge-to-final path was under-structured
- Candidate pools were merged from dense/sparse/follow-up flows, but final selection used a mostly flattened score path.
- `finalize_candidates()` prioritized relevance score + rerank score, but not explicit evidence role utility.
- This allowed semantically related background chunks to survive into final top-k.

### 2) Symbolic retrieval was opportunistic
- Dynamic Cypher paths were appended whenever available.
- There was no explicit gate checking whether symbolic retrieval was likely to add support value for the current query state.

### 3) Evidence roles were implicit, not first-class
- Direct support chunks, bridge chunks, graph-enriched chunks, and symbolic chunks were treated similarly.
- This caused objective mismatch: high semantic similarity often outranked bridge-useful evidence.

### 4) Observability was fragmented
- There was no unified stage-level signal for:
  - role mix
  - source-type mix
  - dedupe impact
  - bridge/support coverage
- Debug output was mostly benchmark-scoped rather than retrieval-stack scoped.

## Relevant External Design Patterns (Conceptual)

- **ApeRAG / flexible-graphrag style orchestration**: keep hybrid retrieval broad, but enforce structured downstream selection.
- **Neo4j GraphRAG pattern**: graph/symbolic retrieval should be selective and query-state dependent.
- **Microsoft GraphRAG pattern**: organize evidence into utility-aware groups, not one undifferentiated ranking objective.

These were applied as architecture patterns, not code copying.

## Phase 2 — Prioritized Improvements

### P1: Role-aware evidence selection (implemented)
- Introduce explicit role modeling in final selection:
  - `direct`
  - `bridge`
  - `graph`
  - `symbolic`
  - `background`
- Use a role-aware composite score with:
  - direct support signal
  - bridge utility signal
  - graph confidence
  - source priors
  - background penalties
  - duplicate suppression

### P2: Selective symbolic retrieval (implemented)
- Add a gate that uses query structure and focus coverage before invoking dynamic Cypher.
- Only trigger symbolic retrieval when it is likely to contribute support value.

### P3: Stage instrumentation (implemented)
- Track stage-wise and composition-wise quality signals:
  - `first_stage_candidates`
  - `reranked_candidates`
  - `graph_expansion_enriched`
  - `dynamic_cypher_added`
  - `duplicates_removed`
  - `source_type_mix_pre`
  - `source_type_mix`
  - `evidence_role_mix`
  - `bridge_coverage_score`
  - `support_coverage_score`
  - `final_context_count`

### P4: API-level visibility (implemented)
- Expose retrieval debug metadata in `answer_metadata` when `ENABLE_RETRIEVAL_DEBUG=true` (default).
- Keep benchmark behavior compatible while allowing non-benchmark inspection.

## Phase 3 — Implemented Code Changes

### A) `retrieval/hybrid_search.py`
- Added role-aware selection helpers:
  - `_infer_source_type`
  - `_is_near_duplicate_text`
  - `_should_use_symbolic_search`
  - `_select_role_aware_candidates`
- Refactored `finalize_candidates()` to:
  - compute rerank stage as before
  - apply graph enrichment
  - invoke symbolic retrieval only through selectivity gate
  - run role-aware final selection
  - persist stage instrumentation into `self.last_search_debug`

### B) `agent/copilot_agent.py`
- Added general debug emission switch:
  - `ENABLE_RETRIEVAL_DEBUG=true|false`
- Merged retriever debug metrics into agent debug payload.
- Returned debug metadata in `answer_metadata` when debug is enabled (not benchmark-only).

### C) `tests/benchmark_rag.py`
- Extended evaluation aggregation to include:
  - duplicate-removal average
  - bridge/support coverage averages
  - source-type totals
  - evidence-role totals
- Included new per-case instrumentation fields in trace output.

### D) Tests
- Added `tests/test_evidence_selection.py` for:
  - role-aware preservation of direct + bridge evidence
  - symbolic gate behavior
  - finalize debug instrumentation presence

## Phase 4 — Test Notes

- Added targeted unit tests for evidence-selection behavior.
- Added a runtime sanity script check for role-aware selection and finalize instrumentation.
- Existing integration tests remain unchanged; they can now consume richer debug metadata.

## Phase 5 — Expected Impact

### Context precision / support precision
- Improved by reducing undifferentiated selection and enforcing role-aware utility.
- Explicit background caps and duplicate suppression reduce noisy survivors.

### Final-context noise
- Reduced via:
  - background penalties
  - per-title and role caps
  - near-duplicate filtering
  - symbolic gating

### Recall retention
- Preserved by keeping broad early retrieval and rerank candidate generation.
- Precision improvements are applied in final selection, not by narrowing first-stage retrieval.

## Tradeoffs

- Slight risk of over-pruning niche evidence if role thresholds are too strict.
- Symbolic gating may skip useful graph paths in some edge cases; this is tunable through focus-coverage thresholds.
- More debug metadata increases payload size when enabled.

## Why This Is Generalizable (Explicit)

The implemented logic is based on retrieval mechanics (evidence roles, overlap signals, diversity, dedupe, stage gating) rather than dataset-specific assumptions.

It does **not** use:
- benchmark keyword whitelists
- benchmark question templates
- benchmark annotations/supporting-facts schema
- dataset-specific branches

These changes are applicable to enterprise corpora, policy docs, technical manuals, and mixed multi-document QA where multi-hop grounding quality matters.
