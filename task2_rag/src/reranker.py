"""
Cross-Encoder Re-ranking Stage for RAG pipeline.
Performs full cross-attention between query and retrieved candidate passages.
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class Reranker:
    """
    Re-ranks candidate passages retrieved from bi-encoder or BM25 stages
    using a full Cross-Encoder transformer model.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                print(f"Loading Cross-Encoder model: {model_name}...")
                self.model = CrossEncoder(model_name)
                print("Cross-Encoder loaded successfully.")
            except Exception as e:
                print(f"Could not load Cross-Encoder ({e}). Operating in heuristic cross-scoring mode.")
        else:
            print("sentence_transformers CrossEncoder not available. Using heuristic scoring.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidate passages against the query.
        
        Args:
            query: The user query string.
            candidates: List of candidate chunk dictionaries.
            top_k: Number of re-ranked chunks to return.
            
        Returns:
            Sorted list of top_k re-ranked chunk dictionaries.
        """
        if not candidates:
            return []
            
        if self.model is not None:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.model.predict(pairs)
        else:
            # Fallback heuristic: query coverage + dense retrieval score
            scores = []
            q_terms = set(query.lower().split())
            for c in candidates:
                c_text = c["text"].lower()
                overlap = sum(1 for t in q_terms if t in c_text) / max(len(q_terms), 1)
                prior_score = c.get("retrieval_score", 0.5)
                scores.append(overlap * 0.5 + prior_score * 0.5)
            scores = np.array(scores)
            
        # Sort candidates by rerank score
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        reranked = []
        for rank, idx in enumerate(sorted_indices):
            chunk_copy = dict(candidates[idx])
            chunk_copy["rerank_score"] = float(scores[idx])
            chunk_copy["rerank_rank"] = rank + 1
            chunk_copy["retrieval_mode"] = "cross_encoder_reranked"
            reranked.append(chunk_copy)
            
        return reranked
