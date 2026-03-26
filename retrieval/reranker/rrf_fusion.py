"""
Reciprocal Rank Fusion Reranker Module for AsterScope.
"""
from typing import List, Dict, Any, Tuple

def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]], 
    sparse_results: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Fuses Dense and Sparse results using Reciprocal Rank Fusion (RRF).
    formula: RRF_score = 1 / (k + rank)
    """
    rrf_scores: Dict[Tuple[str, int], Dict[str, Any]] = {}
    
    # Process Dense
    for rank, hit in enumerate(dense_results):
        key = (hit["doc_id"], hit["chunk_index"])
        if key not in rrf_scores:
            rrf_scores[key] = {**hit, "rrf_score": 0.0, "sources": []}
        rrf_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
        rrf_scores[key]["sources"].append("dense")
        
    # Process Sparse
    for rank, hit in enumerate(sparse_results):
         key = (hit["doc_id"], hit["chunk_index"])
         if key not in rrf_scores:
             rrf_scores[key] = {**hit, "rrf_score": 0.0, "sources": []}
         rrf_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
         if "sparse" not in rrf_scores[key]["sources"]:
             rrf_scores[key]["sources"].append("sparse")
             
    # Sort by final RRF score descending
    fused_results = list(rrf_scores.values())
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_results
