"""
Table-Specific Retrieval Path for NovaSearch.

Provides a dedicated search path for structured table data,
prioritizing chunks with table metadata and applying table-aware
reranking logic. Includes deep table embedding via Schema Summary
headers and structured LLM extraction routing.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import openai
except ImportError:
    openai = None


class TableRetriever:
    """
    Dedicated retrieval path for table chunks.
    Filters on metadata.type == 'table' and applies table-aware
    scoring that treats Markdown structure as structured data.
    
    Features:
        - Schema Summary embedding headers for deep table vectors
        - Structured LLM extraction for table-related queries
        - Table-query heuristic routing
    """
    
    TABLE_QUERY_INDICATORS = [
        "compare", "table", "column", "row", "values", "total",
        "average", "sum", "count", "list", "breakdown", "statistics",
        "entries", "records", "data", "metrics", "figures"
    ]
    
    def __init__(self):
        self.model = None
        self.conn = None
        self.llm_client = None
        
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"TableRetriever: Failed to load model: {e}")
        
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                dbname=os.getenv("POSTGRES_DB", "novasearch"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )
            logger.info("TableRetriever: Connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"TableRetriever: No PostgreSQL connection: {e}")
        
        # Initialize OpenAI client for structured extraction
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and openai:
            try:
                self.llm_client = openai.OpenAI(api_key=api_key)
            except Exception:
                pass
    
    @staticmethod
    def is_table_query(query: str) -> bool:
        """
        Heuristic to determine if a query targets structured/tabular data.
        """
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in TableRetriever.TABLE_QUERY_INDICATORS)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search specifically for table-type chunks using vector similarity.
        Only returns chunks that were marked as table content during ingestion.
        """
        if not self.model or not self.conn:
            return []
        
        try:
            query_embedding = self.model.encode(query).tolist()
            
            cur = self.conn.cursor()
            # Search only table-type chunks using PGVector cosine distance
            cur.execute("""
                SELECT id, doc_id, chunk_index, chunk_text, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE metadata->>'type' = 'table'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "id": row[0],
                    "doc_id": row[1],
                    "chunk_index": row[2],
                    "chunk_text": row[3],
                    "metadata": row[4],
                    "score": float(row[5]) if row[5] else 0.0,
                    "source": "table_retriever"
                })
            
            cur.close()
            
            if results:
                logger.info(f"TableRetriever: Found {len(results)} table chunks for query.")
            return results
            
        except Exception as e:
            logger.error(f"TableRetriever search failed: {e}")
            return []
    
    def rerank_for_tables(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Table-aware reranking: boosts chunks with table metadata
        and applies structural scoring for Markdown table content.
        """
        for hit in hits:
            base_score = hit.get("score", 0.0)
            text = hit.get("chunk_text", "")
            metadata = hit.get("metadata", {})
            
            # Boost factor for confirmed table chunks
            if isinstance(metadata, dict) and metadata.get("type") == "table":
                hit["score"] = base_score * 1.3  # 30% boost for table chunks
            
            # Additional boost if the chunk contains Markdown table syntax
            if "| " in text and "---" in text:
                hit["score"] = hit.get("score", base_score) * 1.1  # 10% extra for Markdown tables
        
        # Re-sort by updated score
        hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return hits
    
    @staticmethod
    def generate_schema_summary(markdown_table: str, title: str = "") -> str:
        """
        Generate a Schema Summary header for embedding alongside the table.
        
        Analyzes column headers and infers types from data to produce a
        structured summary: 'Table: [title]. Schema: col1(type), col2(type).'
        This is embedded alongside the raw Markdown for richer retrieval.
        """
        lines = [l.strip() for l in markdown_table.strip().split("\n") if l.strip()]
        
        if not lines:
            return markdown_table
        
        # Extract column headers
        header_line = lines[0]
        columns = [col.strip() for col in header_line.split("|") if col.strip()]
        
        # Skip separator line and get data lines
        data_lines = [l for l in lines[1:] if not all(c in "-| " for c in l)]
        
        # Infer column types from first data row
        col_types = []
        if data_lines:
            first_row = [c.strip() for c in data_lines[0].split("|") if c.strip()]
            for i, col in enumerate(columns):
                cell = first_row[i] if i < len(first_row) else ""
                # Type heuristic
                cell_clean = cell.replace(",", "").replace("$", "").replace("%", "").strip()
                try:
                    float(cell_clean)
                    col_type = "numeric"
                except ValueError:
                    if any(c.isdigit() for c in cell) and "/" in cell or "-" in cell:
                        col_type = "date"
                    else:
                        col_type = "text"
                col_types.append(f"{col}({col_type})")
        else:
            col_types = [f"{col}(unknown)" for col in columns]
        
        # Build Schema Summary
        parts = []
        if title:
            parts.append(f"Table: {title}.")
        parts.append(f"Schema: {', '.join(col_types)}.")
        parts.append(f"Rows: {len(data_lines)}.")
        
        return " ".join(parts)
    
    @staticmethod
    def generate_table_embedding_text(markdown_table: str, title: str = "") -> str:
        """
        Generate a specialized embedding representation for a Markdown table.
        
        Creates a 'Table Summary + Column Description' header that preserves
        structural intent for the embedding model, appended with the Schema Summary.
        """
        lines = [l.strip() for l in markdown_table.strip().split("\n") if l.strip()]
        
        if not lines:
            return markdown_table
        
        # Extract column headers
        header_line = lines[0]
        columns = [col.strip() for col in header_line.split("|") if col.strip()]
        
        # Skip separator line
        data_lines = [l for l in lines[1:] if not all(c in "-| " for c in l)]
        
        # Build semantic representation
        parts = []
        
        # Add Schema Summary header
        schema_summary = TableRetriever.generate_schema_summary(markdown_table, title)
        parts.append(schema_summary)
        
        if columns:
            parts.append(f"Columns: {', '.join(columns)}.")
        
        for i, row_line in enumerate(data_lines[:10]):  # Cap at 10 rows
            cells = [c.strip() for c in row_line.split("|") if c.strip()]
            if cells:
                parts.append(f"Row {i+1}: {', '.join(cells)}.")
        
        return " ".join(parts) if parts else markdown_table
    
    @staticmethod
    def extract_structured_values(markdown_table: str) -> List[Dict[str, str]]:
        """
        Extract structured key-value pairs from a Markdown table
        for exact value matching in queries.
        """
        lines = [l.strip() for l in markdown_table.strip().split("\n") if l.strip()]
        
        if len(lines) < 2:
            return []
        
        # Parse header
        headers = [col.strip() for col in lines[0].split("|") if col.strip()]
        
        # Parse data rows
        rows = []
        for line in lines[1:]:
            if all(c in "-| " for c in line):
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
        
        return rows
    
    def extract_structured_answer(self, query: str, table_chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Structured LLM extraction: for table-related queries, uses the LLM
        to extract specific row/column values from table chunks instead
        of relying on simple vector similarity.
        
        Returns a structured JSON response with extracted values.
        """
        if not self.llm_client or not table_chunks:
            return None
        
        # Concatenate table content
        table_context = "\n\n".join([
            chunk.get("chunk_text", "") for chunk in table_chunks[:5]
        ])
        
        prompt = f"""You are a precise data extraction agent. The user is asking about structured table data.

Extract the specific values requested from the following table context.

<Table Context>
{table_context}
</Table Context>

User Query: {query}

Instructions:
1. Extract ONLY the specific values the user asked about.
2. Return a structured JSON response with:
   - "answer": A natural language answer to the query
   - "extracted_values": A list of key-value pairs from the table
   - "source_table": Which table the data came from (if identifiable)
3. If the data is not in the tables, respond with {{"answer": "Data not found in provided tables", "extracted_values": [], "source_table": null}}

Respond with valid JSON only."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Table structured extraction failed: {e}")
            return None
