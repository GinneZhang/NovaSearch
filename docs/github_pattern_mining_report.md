# GitHub Pattern Mining for Evidence-Quality Improvements

## Scope

This report documents a benchmark-agnostic pattern-mining pass across strong open-source retrieval systems, followed by targeted adaptations into NovaSearch.

Guardrails for this pass:

- No HotpotQA-specific logic
- No Natural Questions-specific logic
- No benchmark template matching or label-aware shortcuts
- No metric redefinition to manufacture gains
- Focus on cleaner final evidence sets for general grounded QA over enterprise, wiki, manual, and policy corpora

## Repositories Studied

| Repository | Why it was studied | Relevant subsystem(s) | Fit for NovaSearch |
| --- | --- | --- | --- |
| [RAGFlow](https://github.com/infiniflow/ragflow) | Large, production-oriented hybrid RAG engine with agent and graph layers | Hybrid search orchestration, graph retrieval selectivity, citation-oriented grounding | Strong conceptual fit without replacing NovaSearch's existing retrievers |
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | High-quality graph-guided retrieval and context organization patterns | Mixed-context budgeting, graph-context selection, structured source presentation | Strong fit for evidence budgeting and graph gating |
| [Haystack](https://github.com/deepset-ai/haystack) | Mature retrieval pipeline toolkit with modular joiners and routing | Document joiners, distribution-aware fusion, auto-merging, routing | Strong fit for cleaner candidate merging and anti-redundancy |
| [ColBERT](https://github.com/stanford-futuredata/ColBERT) | Canonical late-interaction retrieval system for support-sensitive matching | Late interaction, iterative evidence retrieval via Baleen | Useful as a ranking objective reference, but full adoption would be a larger redesign |
| [LightRAG](https://github.com/HKUDS/LightRAG) | Fast mixed KG/vector RAG with explicit chunk processing and context budgeting | Unified chunk processing, KG/vector mixing, thresholded context building | Strong fit for final packing and confidence-controlled graph evidence |

These repositories are all code-rich and actively used in practice. At the time of inspection, their GitHub pages showed substantial adoption, for example RAGFlow at about 76k stars and LightRAG at about 30k stars, with both showing multi-thousand-commit histories on their repository pages.

## What Each Repository Does Better

### 1. RAGFlow

Relevant code studied:

- `rag/nlp/search.py`
- `rag/graphrag/search.py`
- `rag/prompts/citation_plus.md`

Relevant mechanisms:

- Hybrid dense-plus-text retrieval is fused before answer generation rather than trusting one channel.
- Graph retrieval is not always treated as equal to lexical/dense retrieval; it is introduced as a distinct reasoning aid.
- Citation-oriented answering keeps retrieval provenance close to the response.

Why it matters for NovaSearch:

- NovaSearch was already retrieving useful material, but final evidence quality suffered because graph and symbolic evidence could enter the pack too opportunistically.
- RAGFlow reinforces the idea that graph output should be a selective supplement, not an always-on peer to stronger textual evidence.

Adaptation chosen:

- Strengthened graph/symbolic confidence control in `retrieval/hybrid_search.py`
- Exposed evidence-role and source-type markers in `agent/copilot_agent.py` so the model sees what kind of evidence it is using

Tradeoff:

- Tighter graph gating can reduce some recall on genuinely graph-heavy queries if corroboration thresholds are too strict

### 2. Microsoft GraphRAG

Relevant code studied:

- `graphrag/query/structured_search/local_search/mixed_context.py`
- `graphrag/query/context_builder/dynamic_community_selection.py`
- `graphrag/query/context_builder/source_context.py`

Relevant mechanisms:

- Context is assembled with explicit budget shares across evidence classes instead of taking a flat top-N list.
- Graph/community context is selected dynamically instead of being injected indiscriminately.
- Final context is structured by source tables and provenance rather than a loose text blob.

Why it matters for NovaSearch:

- NovaSearch's biggest weakness was not early-stage collapse; it was late-stage evidence mixing.
- GraphRAG's budgeting model is directly relevant to role-aware evidence selection and selective graph contribution.

Adaptation chosen:

- Added role-aware and source-aware caps in `retrieval/hybrid_search.py`
- Added corroboration-based graph retention so graph-like items need supporting family evidence
- Added structured context markers in `_format_context()` for source type, evidence role, and corroboration state

Tradeoff:

- More structured budgets make selection behavior more explicit, but require careful tuning to avoid underfilling the context on sparse corpora

### 3. Haystack

Relevant code studied:

- `haystack/components/joiners/document_joiner.py`
- `haystack/components/retrievers/sentence_window_retriever.py`
- `haystack/components/retrievers/auto_merging_retriever.py`
- `haystack/components/routers/conditional_router.py`

Relevant mechanisms:

- Distribution-aware document joining and deduplication prevent one retrieval channel from dominating on raw score scale alone.
- Auto-merging and sentence-window retrieval compact fragmented evidence from the same source.
- Routing decisions are explicit modules rather than hidden conditionals.

Why it matters for NovaSearch:

- NovaSearch merged multiple channels and then still suffered from redundancy and noisy same-family chunks.
- Haystack's join and compaction patterns are a strong fit for source-aware calibration and adjacent-chunk merging.

Adaptation chosen:

- Added per-source score calibration and source-rank fractions before final evidence selection
- Added near-duplicate suppression and same-family compaction in generation packing
- Added richer trace fields so source-type composition can be inspected per run

Tradeoff:

- Family compaction must avoid over-merging semantically distinct evidence from the same title

### 4. ColBERT

Relevant code studied:

- `README.md`
- `baleen/engine.py`
- `colbert/modeling/segmented_maxsim.cpp`

Relevant mechanisms:

- Late interaction emphasizes support usefulness at token level instead of relying only on whole-chunk topical similarity.
- Baleen shows iterative evidence gathering for multi-step questions rather than assuming a single retrieval step is enough.

Why it matters for NovaSearch:

- NovaSearch's ranking weakness was not total failure to retrieve; it was overvaluing semantically related but weakly useful chunks.
- Full ColBERT adoption would be a larger indexing and serving redesign, but its objective is still instructive.

Adaptation chosen:

- Kept NovaSearch's existing rerank stack, but pushed the final packer closer to support usefulness via answer/bridge utility, corroboration, and redundancy penalties

Tradeoff:

- This is an approximation, not true late interaction; it improves packing and ranking signals without replacing the retrieval backend

### 5. LightRAG

Relevant code studied:

- `lightrag/utils.py` around unified chunk processing
- `lightrag/operate.py` around context construction
- `lightrag/base.py`

Relevant mechanisms:

- Unified chunk processing applies reranking, thresholds, deduplication, and token budgeting in one place.
- KG/vector context is built under explicit budgets, not through open-ended accumulation.
- Mixed retrieval modes account for token cost explicitly.

Why it matters for NovaSearch:

- NovaSearch needed a cleaner handoff from merged candidate pool to final generation context.
- LightRAG's thresholded chunk processing maps well to final evidence-set pruning and graph/symbolic downweighting.

Adaptation chosen:

- Tightened final evidence assembly with utility thresholds, graph/symbolic corroboration requirements, and post-selection compaction
- Extended benchmark traces with `pre_pack_count`, `post_pack_count`, graph/symbolic kept-dropped counts, and generation compaction counts

Tradeoff:

- More aggressive packing can hurt recall if thresholds are pushed too high on thin corpora

## Comparative Diagnosis

| NovaSearch weakness | External repo(s) | Mechanism used there | NovaSearch adaptation | Expected upside | Tradeoff / risk |
| --- | --- | --- | --- | --- | --- |
| Final evidence set too noisy | GraphRAG, Haystack, LightRAG | Explicit context budgeting, calibrated merging, thresholded chunk processing | Role-aware selection with source caps, corroboration gating, and utility thresholds | Higher context precision, less background bleed | Over-tight budgets can underfill context |
| Supporting to generation inheritance too weak | GraphRAG, RAGFlow | Structured context and provenance-aware grounding | Supporting-selected chunks get stronger inheritance priority and prompt-visible metadata | Better survival of already-validated evidence | Can over-anchor on bad seeds if selector is wrong |
| Graph/symbolic evidence enters with too little confidence control | GraphRAG, RAGFlow, LightRAG | Dynamic graph selection and graph used as selective supplement | Uncorroborated graph/symbolic candidates are penalized or blocked | Cleaner final context, fewer speculative KG paths | May miss some graph-only answers |
| Final packing is not role-aware enough | GraphRAG, ColBERT, LightRAG | Separate evidence classes and support-oriented context building | Direct, bridge, graph, symbolic, and background evidence now carry explicit roles | Better answer usefulness per token | Role inference remains heuristic, not learned |
| Reranking overvalues topical similarity | ColBERT, Haystack | Support-sensitive scoring and channel-aware fusion | Added source calibration plus bridge/direct utility-aware final selection | Better preservation of answer-bearing chunks | Not a full late-interaction reranker |
| Near-duplicate chunks survive too often | Haystack, LightRAG | Deduplication plus auto-merging / sentence windows | Near-duplicate suppression and adjacent same-family chunk compaction | Lower redundancy, better budget efficiency | Over-merging can hide contrastive details |
| Multi-channel merge is too naive | Haystack, RAGFlow | Explicit join policies and selective routing | Distribution-style source calibration and graph-symbolic caps | Better channel balance without widening recall | Some tuning sensitivity by corpus |
| Weak observability around evidence quality | GraphRAG, LightRAG, RAGFlow | Structured context reports and retrieval tracing | Added pre/post pack counts, graph/symbolic kept-dropped, role/source mixes, compaction stats | Easier diagnosis of where noise enters | Slightly larger metadata payloads |

## Prioritized Plan

Ranked by likely impact on final evidence quality, generalizability, feasibility, and recall risk:

1. **Source-calibrated, role-aware final evidence selection**
   - Highest leverage because NovaSearch's measured weakness is late-stage evidence cleanliness
   - Implemented in `retrieval/hybrid_search.py`

2. **Corroboration-gated graph and symbolic evidence**
   - High value because graph/symbolic material is useful but noisy when ungated
   - Implemented in `retrieval/hybrid_search.py` and respected by generation packing

3. **Generation context family compaction and anti-redundancy control**
   - High value because repeated same-family chunks waste budget and lower support density
   - Implemented in `agent/copilot_agent.py`

4. **Prompt-visible evidence-role and source metadata**
   - Medium-to-high value because the model should know whether a chunk is direct evidence, bridge evidence, or graph-derived
   - Implemented in `_format_context()`

5. **Richer observability for evidence quality**
   - Medium value operationally, but crucial for verifying whether future changes help or only move noise around
   - Implemented in retrieval debug and benchmark summary output

Deferred rather than implemented:

- Full ColBERT-style late interaction replacement
- Larger routing/runtime redesign
- Major graph schema or index rebuild

Those could help, but they would be materially heavier changes than needed for NovaSearch's current bottleneck.

## Implemented Adaptations

### Retrieval-side selection

Implemented in [retrieval/hybrid_search.py](/Users/ginnezhang/Documents/Playground/NovaSearch/retrieval/hybrid_search.py).

- Added source-aware score calibration inspired by Haystack-style join normalization
- Added family-level corroboration tracking before keeping graph/symbolic evidence
- Added source-type and role caps so graph/symbolic/background channels do not flood the final set
- Added debug counters for graph/symbolic candidates kept and dropped

### Generation-side packing

Implemented in [agent/copilot_agent.py](/Users/ginnezhang/Documents/Playground/NovaSearch/agent/copilot_agent.py).

- Added stronger utility gating for generation chunks
- Compacts adjacent chunks from the same document family
- Penalizes uncorroborated graph/symbolic filler
- Preserves supporting-selected and planner-confirmed evidence with higher priority

### Instrumentation

Implemented across [retrieval/hybrid_search.py](/Users/ginnezhang/Documents/Playground/NovaSearch/retrieval/hybrid_search.py), [agent/copilot_agent.py](/Users/ginnezhang/Documents/Playground/NovaSearch/agent/copilot_agent.py), and [tests/benchmark_rag.py](/Users/ginnezhang/Documents/Playground/NovaSearch/tests/benchmark_rag.py).

Added:

- `pre_pack_count`
- `post_pack_count`
- `graph_symbolic_candidates`
- `graph_symbolic_kept`
- `graph_symbolic_dropped`
- `corroborated_graph_kept`
- `source_type_mix`
- `evidence_role_mix`
- `generation_compacted_count`
- `generation_uncorroborated_graph_count`

## Why These Changes Are Generalizable

These changes do not depend on benchmark templates, supporting-fact annotations, or dataset-specific keywords. They operate on generally useful retrieval-quality principles:

- evidence role separation
- source-aware calibration
- corroboration before trusting graph-only evidence
- duplicate suppression
- compacting same-family evidence
- measuring evidence quality at the packer boundary

Those principles remain valid across:

- enterprise policy corpora
- internal SOPs
- technical manuals
- wiki-style knowledge bases
- mixed graph-plus-document retrieval systems

## Validation Added

Relevant tests:

- [tests/test_evidence_selection.py](/Users/ginnezhang/Documents/Playground/NovaSearch/tests/test_evidence_selection.py)
- [tests/test_generation_packing.py](/Users/ginnezhang/Documents/Playground/NovaSearch/tests/test_generation_packing.py)
- [tests/test_benchmark_reporting.py](/Users/ginnezhang/Documents/Playground/NovaSearch/tests/test_benchmark_reporting.py)

These tests cover:

- role-aware evidence preservation
- blocking uncorroborated symbolic evidence
- keeping corroborated graph evidence
- generation context compaction
- benchmark summary propagation for the new evidence-quality metrics
