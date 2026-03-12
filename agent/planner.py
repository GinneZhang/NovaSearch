"""
Task Decomposer using LangChain to break down complex queries.
"""
import os
import logging
from typing import List, Union

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
except ImportError:
    pass

logger = logging.getLogger(__name__)

class TaskDecomposer:
    """
    Uses LangChain to conceptually break down complex user queries into 
    atomic sub-queries for broader retrieval coverage.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        try:
            if not self.openai_api_key:
                logger.warning("OPENAI_API_KEY missing. TaskDecomposer will bypass execution.")
                self.llm = None
            else:
                self.llm = ChatOpenAI(temperature=0.0, model=model_name, api_key=self.openai_api_key) # type: ignore
        except Exception as e:
            logger.error(f"Failed to initialize LangChain ChatOpenAI: {e}")
            self.llm = None

        if self.llm:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", 
                "You are an expert query analyzer for an enterprise search system.\n"
                "Your task is to take a complex user query and decompose it into 2-3 atomic, standalone sub-queries.\n"
                "If the query is already simple and atomic, just return the original query.\n"
                "CRITICAL: Do NOT answer the question or inject external parametric knowledge. You must ONLY break down the literal concepts provided in the user's prompt into sub-queries.\n"
                "CRITICAL EXCEPTION: If the user's query is critically ambiguous (e.g., missing a subject, too broad, or nonsensical), you MUST NOT decompose it. Instead, you MUST output EXACTLY this JSON format and nothing else: `{\"type\": \"clarification\", \"content\": \"<your clarifying question here>\"}`\n\n"
                "Example 1:\n"
                "User: Compare blackout periods for SVPs and regular employees\n"
                "Output:\n"
                "What is the blackout period for SVPs?\n"
                "What is the blackout period for regular employees?"),
                ("user", "{query}")
            ])
            self.chain = self.prompt | self.llm | StrOutputParser()
        else:
            self.chain = None

    def decompose(self, query: str) -> Union[List[str], dict]:
        """Decomposes a query into atomic sub-queries, or returns a clarification dict if ambiguous."""
        if not self.chain:
            return [query]
            
        try:
            logger.info(f"Decomposing query using LangChain: '{query}'")
            output = self.chain.invoke({"query": query})
            
            # Check for clarification JSON
            if output.strip().startswith("{") and "clarification" in output:
                import json
                try:
                    parsed = json.loads(output)
                    if parsed.get("type") == "clarification":
                        logger.info(f"Query is ambiguous, requesting clarification: {parsed.get('content')}")
                        return parsed
                except json.JSONDecodeError:
                    pass
            
            sub_queries = [line.strip() for line in output.split("\n") if line.strip()] # type: ignore
            
            if not sub_queries:
                return [query]
                
            logger.info(f"Decomposed into {len(sub_queries)} queries: {sub_queries}")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]
