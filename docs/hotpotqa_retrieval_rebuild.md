# NovaSearch HotpotQA Retrieval Rebuild

## Brutal Assessment

`Hit Rate@5 = 0.04` is not a reranker problem. It is a retrieval architecture failure.

The current stack is optimized for:

- safe answer generation once evidence is found
- multimodal `/ask_vision`
- chunk dehydration and cost control

It is not optimized for:

- Wikipedia-style dense passage retrieval
- multi-hop retrieval sequencing
- title/entity-preserving graph traversal

That mismatch shows up clearly in the code:

- text retrieval is still single-shot dense search over one embedding space in [retrieval/dense/vector_search.py](/Users/ginnezhang/Documents/Playground/NovaSearch/retrieval/dense/vector_search.py)
- multimodal CLIP search is already isolated in [retrieval/dense/vision_search.py](/Users/ginnezhang/Documents/Playground/NovaSearch/retrieval/dense/vision_search.py)
- query decomposition exists but is too generic and not evidence-gated in [agent/planner.py](/Users/ginnezhang/Documents/Playground/NovaSearch/agent/planner.py)
- the agent fans out sub-queries, but not as ordered hops with stop/go evaluation in [agent/copilot_agent.py](/Users/ginnezhang/Documents/Playground/NovaSearch/agent/copilot_agent.py)
- graph search exists, but mostly as optional enrichment after vector search in [retrieval/hybrid_search.py](/Users/ginnezhang/Documents/Playground/NovaSearch/retrieval/hybrid_search.py)

There is also a benchmark distortion:

- [tests/benchmark_rag.py](/Users/ginnezhang/Documents/Playground/NovaSearch/tests/benchmark_rag.py) ingests each sample as one synthetic document title (`HotpotQA Sample {idx}`), which discards the original Wikipedia page titles used by HotpotQA supporting facts.

## Target Architecture

Build four retrieval lanes and route queries intentionally:

1. `text_dense`: text-only embeddings for passage retrieval
2. `text_sparse`: BM25 / FTS for exact entity and title recall
3. `graph_path`: Neo4j traversal for relationship hops
4. `vision_clip`: CLIP-only lane for image and cross-modal search

The rule is simple:

- `/ask` never touches CLIP for primary ranking
- `/ask_vision` always uses CLIP
- text queries with attached images can do late fusion between `text_dense` and `vision_clip`
- multi-hop questions run a planner -> hop retriever -> critic loop before final fusion

## 1. Dual-Embedding Strategy

### Recommendation

Do not keep CLIP as the universal text index for enterprise QA.

Use:

- `bge-m3` if you want one self-hosted model that supports dense + lexical-style benefits and works well on passage retrieval
- `text-embedding-3-small` if you want strong API quality and easier operations
- keep `clip-ViT-B-32` only for image and cross-modal retrieval

### Storage Design

Split Postgres tables instead of forcing one common vector schema:

```sql
CREATE TABLE IF NOT EXISTS text_chunks (
    id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    title TEXT,
    chunk_text TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS text_chunks_embedding_hnsw_idx
ON text_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS vision_embeddings (
    id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT,
    embedding_model TEXT NOT NULL,
    embedding vector(512)
);
```

### Router

```python
from dataclasses import dataclass
from typing import Literal
import re


@dataclass
class RetrievalRoute:
    mode: Literal["text", "vision", "multimodal"]
    require_multi_hop: bool
    require_graph: bool


class QueryRouter:
    MULTIHOP_CUES = (
        "wife of", "husband of", "founder of", "parent of", "born in",
        "capital of", "member of", "director of", "author of", "where did",
        "who was", "what city", "what country"
    )

    def route(self, query: str, has_image: bool = False) -> RetrievalRoute:
        lowered = query.lower()
        require_multi_hop = any(cue in lowered for cue in self.MULTIHOP_CUES)
        require_graph = require_multi_hop or bool(re.search(r"\b(of|by|from|after|before)\b", lowered))

        if has_image and query.strip():
            return RetrievalRoute(mode="multimodal", require_multi_hop=require_multi_hop, require_graph=require_graph)
        if has_image:
            return RetrievalRoute(mode="vision", require_multi_hop=False, require_graph=False)
        return RetrievalRoute(mode="text", require_multi_hop=require_multi_hop, require_graph=require_graph)
```

### Text Retriever Replacement

This should replace the default text lane used by [retrieval/dense/vector_search.py](/Users/ginnezhang/Documents/Playground/NovaSearch/retrieval/dense/vector_search.py).

```python
from typing import Any
from sentence_transformers import SentenceTransformer
from psycopg2.extras import DictCursor


class PGVectorTextRetriever:
    def __init__(self, pg_conn, embedding_model_name: str = "BAAI/bge-m3"):
        self.pg_conn = pg_conn
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        query_vec = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()
        vec = "[" + ",".join(str(x) for x in query_vec) + "]"

        sql = """
            SELECT
                id,
                doc_id,
                chunk_index,
                title,
                chunk_text,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM text_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        with self.pg_conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(sql, (vec, vec, top_k))
            rows = cur.fetchall()

        return [
            {
                "id": row["id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "title": row["title"],
                "chunk_text": row["chunk_text"],
                "metadata": row["metadata"],
                "score": float(row["score"]),
                "source": "text_dense",
            }
            for row in rows
        ]
```

### Ingestion Rule

At ingest time:

- write text chunks to `text_chunks` using text embeddings
- write image and OCR-derived vision chunks to `vision_embeddings` using CLIP
- keep shared `doc_id` and `chunk_index` so late fusion can merge them

Do not store the same text chunk in CLIP and pretend it is a competitive passage retriever. It is not.

## 2. Multi-Hop Query Decomposition With Planner-Critic

### Problem In Current Flow

The current agent decomposes into parallel sub-queries and merges hits. That is not multi-hop reasoning. It is query fan-out.

HotpotQA needs:

1. retrieve hop 1 evidence
2. extract bridge entity or relation target
3. form hop 2 query from evidence, not just from the raw question
4. stop only when the critic says the bridge fact is grounded

### Planner Output Schema

Use a structured planner, not free-form lines:

```python
from pydantic import BaseModel, Field


class RetrievalHop(BaseModel):
    hop_id: int
    goal: str = Field(description="What this hop must establish")
    query: str = Field(description="Search query for this hop")
    expected_answer_type: str = Field(description="person, place, date, org, etc.")
    depends_on: list[int] = Field(default_factory=list)


class RetrievalPlan(BaseModel):
    question: str
    hops: list[RetrievalHop]
```

### Planner Prompt

```python
PLANNER_SYSTEM = """
You are a retrieval planner for a multi-hop QA engine.
Break the user question into ordered retrieval hops.

Rules:
- Return 1 hop for simple lookup questions.
- Return 2-3 hops for bridge-comparison or relational questions.
- Each hop must be answerable via retrieval.
- Do not answer the question.
- Output JSON matching the provided schema.
"""
```

### Critic Contract

The critic should decide:

- did hop 1 retrieve evidence answering its own goal?
- is there a bridge entity or attribute to drive the next hop?
- if not, should the system reformulate, broaden, or switch to graph traversal?

```python
from pydantic import BaseModel


class CriticDecision(BaseModel):
    sufficient: bool
    extracted_bridge_entity: str | None = None
    missing_fact: str | None = None
    retry_query: str | None = None
    switch_to_graph: bool = False
```

### Production Loop

```python
class MultiHopRetriever:
    def __init__(self, planner, critic, text_retriever, sparse_retriever, graph_retriever, reranker):
        self.planner = planner
        self.critic = critic
        self.text_retriever = text_retriever
        self.sparse_retriever = sparse_retriever
        self.graph_retriever = graph_retriever
        self.reranker = reranker

    def retrieve(self, question: str, top_k: int = 5) -> dict:
        plan = self.planner.plan(question)
        accumulated = []
        bridge_state = {}

        for hop in plan.hops:
            hop_query = self._materialize_query(hop.query, bridge_state)

            dense_hits = self.text_retriever.search(hop_query, top_k=20)
            sparse_hits = self.sparse_retriever.search(hop_query, top_k=20)
            graph_hits = self.graph_retriever.search(hop_query, bridge_state=bridge_state, top_k=10)

            candidates = self.reranker.rerank(hop_query, dense_hits + sparse_hits + graph_hits, top_k=8)
            decision = self.critic.evaluate(
                question=question,
                hop_goal=hop.goal,
                hop_query=hop_query,
                candidates=candidates,
                prior_state=bridge_state,
            )

            accumulated.extend(candidates)

            if decision.sufficient:
                if decision.extracted_bridge_entity:
                    bridge_state[f"hop_{hop.hop_id}_entity"] = decision.extracted_bridge_entity
                continue

            if decision.switch_to_graph:
                graph_only = self.graph_retriever.search(
                    question,
                    bridge_state=bridge_state,
                    top_k=10,
                    force_path=True,
                )
                accumulated.extend(graph_only)
                continue

            if decision.retry_query:
                retry_dense = self.text_retriever.search(decision.retry_query, top_k=20)
                retry_sparse = self.sparse_retriever.search(decision.retry_query, top_k=20)
                retry_candidates = self.reranker.rerank(
                    decision.retry_query,
                    retry_dense + retry_sparse,
                    top_k=8,
                )
                accumulated.extend(retry_candidates)

        final_hits = self._dedupe_and_sort(accumulated)[:top_k]
        return {
            "plan": plan.model_dump(),
            "bridge_state": bridge_state,
            "hits": final_hits,
        }

    def _materialize_query(self, query_template: str, bridge_state: dict) -> str:
        result = query_template
        for key, value in bridge_state.items():
            result = result.replace("{" + key + "}", value)
        return result

    def _dedupe_and_sort(self, hits: list[dict]) -> list[dict]:
        seen = {}
        for hit in hits:
            key = hit.get("id") or f"{hit.get('doc_id')}::{hit.get('chunk_index')}::{hit.get('source')}"
            if key not in seen or hit.get("score", 0.0) > seen[key].get("score", 0.0):
                seen[key] = hit
        return sorted(seen.values(), key=lambda x: x.get("score", 0.0), reverse=True)
```

### Critic Heuristics

For HotpotQA, the critic should explicitly score:

- answerability: does any candidate likely answer the current hop goal?
- bridge extraction: can we name the intermediate entity needed for the next hop?
- support diversity: do we have at least 2 supporting chunks from different sources when needed?

Example:

- question: `Who is the wife of the founder of Patagonia?`
- hop 1 goal: identify founder of Patagonia
- critic passes only if top evidence contains `Yvon Chouinard`
- hop 2 query becomes `wife of Yvon Chouinard`

If hop 1 evidence only says Patagonia is an outdoor clothing company, the critic must fail it and request retry, not allow the chain to continue.

## 3. Graph-RAG Integration

### What To Change

Right now graph retrieval is mostly enrichment or dynamic Cypher after the fact. For HotpotQA-class questions, graph traversal has to become a first-class candidate generator.

Use the graph for:

- entity disambiguation
- bridge entity extraction
- constrained path traversal
- evidence expansion back into source chunks

### Graph Model Requirements

Minimum nodes:

- `Document`
- `Chunk`
- `Entity`

Recommended additions:

- `Claim`
- `Page`
- `Alias`

Minimum relationships:

- `(:Document)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:MENTIONS]->(:Entity)`
- `(:Entity)-[:ALIAS_OF]->(:Entity)`
- `(:Entity)-[:RELATED_TO {relation_type: ...}]->(:Entity)`
- `(:Claim)-[:SUPPORTED_BY]->(:Chunk)`
- `(:Claim)-[:SUBJECT]->(:Entity)`
- `(:Claim)-[:OBJECT]->(:Entity)`

That lets you do symbolic traversal while still grounding back to text.

### Dynamic Cypher Strategy

Do not ask the LLM to hallucinate arbitrary Cypher from scratch for every question. Generate Cypher from a constrained path template library first, and only use LLM repair if template filling fails.

### Example Cypher Complementing Vector Search

Step 1: vector lane finds likely bridge entity candidates:

```python
bridge_candidates = text_retriever.search("founder of Patagonia", top_k=5)
```

Step 2: graph lane resolves and traverses:

```python
def graph_follow_founder_spouse(driver, company_name: str) -> list[dict]:
    cypher = """
    MATCH (company:Entity {name: $company_name})
    MATCH (founder:Entity)-[:RELATED_TO {relation_type: 'FOUNDER_OF'}]->(company)
    OPTIONAL MATCH (founder)-[:RELATED_TO {relation_type: 'SPOUSE_OF'}]->(spouse:Entity)
    OPTIONAL MATCH (claim:Claim)-[:SUBJECT]->(founder)
    OPTIONAL MATCH (claim)-[:OBJECT]->(spouse)
    OPTIONAL MATCH (claim)-[:SUPPORTED_BY]->(chunk:Chunk)<-[:HAS_CHUNK]-(doc:Document)
    RETURN founder.name AS founder,
           spouse.name AS spouse,
           collect(DISTINCT {
               doc_id: doc.id,
               title: doc.title,
               chunk_text: chunk.chunk_text
           })[0..5] AS evidence
    LIMIT 5
    """

    with driver.session() as session:
        rows = session.run(cypher, company_name=company_name)
        return [row.data() for row in rows]
```

### Fallback From Graph To Text

If the graph has the relation path but weak supporting text:

- use the traversed entities to launch targeted dense retrieval
- rerank by claim support rather than raw similarity

```python
def expand_graph_path_with_text(text_retriever, founder: str, spouse: str | None) -> list[dict]:
    followups = [
        f"{founder} founder biography",
        f"{founder} wife" if spouse is None else f"{founder} {spouse}",
    ]
    hits = []
    for query in followups:
        hits.extend(text_retriever.search(query, top_k=5))
    return hits
```

### Safe Dynamic Cypher Generation

The current [retrieval/graph/cypher_generator.py](/Users/ginnezhang/Documents/Playground/NovaSearch/retrieval/graph/cypher_generator.py) is a good repair layer, but not a great primary planner. Use this staged policy:

1. classify question into path type
2. fill a safe Cypher template
3. validate against schema
4. execute
5. use LLM self-healing only if schema-valid query still fails

That reduces token cost and improves determinism.

## Benchmark Fixes You Should Make Immediately

Before trusting the next score:

1. Preserve original HotpotQA page titles during ingestion instead of collapsing each sample into `HotpotQA Sample {idx}`.
2. Evaluate retrieval against `title + chunk_text + graph_context.doc_title`, not `chunk_text` alone.
3. Separate benchmark modes:
   - `single-hop text`
   - `multi-hop text`
   - `graph-assisted`
   - `vision`
4. Report recall before and after query decomposition so you can isolate whether the planner is helping.

## Rollout Order

If you want the fastest path from 4% to something respectable, do this in order:

1. remove CLIP from `/ask` text retrieval ranking
2. create a dedicated `text_chunks` index with `bge-m3` or `text-embedding-3-small`
3. preserve article titles and entity aliases in ingestion
4. replace parallel sub-query fan-out with ordered planner-critic hops
5. make graph traversal a candidate generator, not just enrichment
6. rerank the merged pool with a strong cross-encoder

## Expected Outcome

What I would expect after the rebuild:

- the first large jump comes from replacing CLIP for text and preserving titles/entities
- the second jump comes from planner-critic hop sequencing
- the third jump comes from graph-assisted bridge traversal on relation-heavy questions

If you keep CLIP as the main text retriever and keep treating multi-hop as parallel query fan-out, the hit rate will stay bad no matter how much you polish the reranker.
