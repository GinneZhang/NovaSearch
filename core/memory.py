"""
Redis Session Memory Manager for AsterScope.
"""
import os
import json
import logging
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import redis
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

logger = logging.getLogger(__name__)

class RedisMemoryManager:
    """Manages multi-turn conversation history using Redis."""
    
    def __init__(self):
        """Initializes the Redis connection."""
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        
        try:
            self.client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                password=redis_password,
                decode_responses=True # Automatically decode bytes to strings
            )
            self.client.ping()
            logger.info("Connected to Redis for Semantic Memory.")
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", str(e))
            self.client = None
            
        # Default TTL for session keys: 24 hours (in seconds)
        self.ttl = 60 * 60 * 24
        
        # Initialize embedding model for semantic context search
        self.model_name = "all-MiniLM-L6-v2"
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info("Loaded SentenceTransformer for Redis Memory.")
        except Exception as e:
            logger.error(f"Failed to load embedding model for memory: {e}")
            self.model = None

    def add_message(self, session_id: str, role: str, content: str, user_id: str = "default_user"):
        """
        Appends a message to the session's conversation history.
        Also performs lightweight Cross-Session Thread Linking by tagging the first user query.
        """
        if not self.client:
            return
            
        key = f"session:{session_id}:history"
        
        try:
            # If this is the absolute first user message in this session, map it to the user's cross-session index
            if role == "user" and self.client.llen(key) == 0:
                if self.model:
                    # Generate semantic topic vector
                    topic_vector = self.model.encode(content).tolist()
                    
                    # Store session-topic mapping (for metadata/display)
                    mapping_key = f"user:{user_id}:sessions"
                    self.client.hset(mapping_key, session_id, content[:100]) # Store snippet
                    
                    # Store the vector for similarity search
                    vector_key = f"user:{user_id}:session_vectors"
                    self.client.hset(vector_key, session_id, json.dumps(topic_vector))
        
            message = json.dumps({"role": role, "content": content})
            # RPUSH appends to the end of the list
            self.client.rpush(key, message)
            # Reset the TTL so active sessions don't expire mid-conversation
            self.client.expire(key, self.ttl)
        except Exception as e:
            logger.error("Failed to append message to Redis for session %s: %s", session_id, str(e))

    def get_related_sessions(self, user_id: str, current_query: str, threshold: float = 0.6) -> List[str]:
        """
        Fetches past session IDs that are semantically related to the current query.
        """
        if not self.client or not self.model:
            return []
            
        vector_key = f"user:{user_id}:session_vectors"
        try:
            # 1. Embed current query
            query_vec = self.model.encode(current_query)
            
            # 2. Fetch all session vectors for the user
            # Node: For massive scale, we'd use Redis Stack (RediSearch/Vector similarity)
            # For this enterprise prototype, we do in-memory scan for the user's sessions.
            all_vecs_raw = self.client.hgetall(vector_key)
            if not all_vecs_raw:
                return []
                
            related = []
            for sid, v_json in all_vecs_raw.items():
                v = np.array(json.loads(v_json))
                # Cosine Similarity
                sim = np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v))
                if sim >= threshold:
                    related.append(sid)
            
            logger.info(f"Found {len(related)} semantically related sessions for user {user_id}")
            return related
        except Exception as e:
            logger.error("Failed to fetch related sessions for user %s: %s", user_id, str(e))
            return []

    def get_history(self, session_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
        """
        Retrieves the last N turns of the conversation history.
        """
        if not self.client:
            return []
            
        key = f"session:{session_id}:history"
        fetch_count = max_turns * 2
        
        try:
            # We want the LAST fetch_count items
            raw_history = self.client.lrange(key, -fetch_count, -1)
            history = [json.loads(msg) for msg in raw_history]
            return history
        except Exception as e:
            logger.error("Failed to retrieve history from Redis for session %s: %s", session_id, str(e))
            return []
