"""
Evaluation framework for RAG: calculates Recall@5 and MRR@5 across chunk sizes and retrieval modes,
captures concrete retrieval failures, and details actionable engineering fixes.
"""

import os
import json
from typing import List, Dict, Any, Tuple

def evaluate_retrieval(
    questions: List[Dict[str, Any]],
    retriever_func,
    top_k: int = 5,
    tag: str = "eval_run"
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality against ground truth questions and target passages.
    
    Args:
        questions: 15 verified ground truth QA records.
        retriever_func: Function (query: str, top_k: int) -> List[chunk dicts].
        top_k: Number of retrieved chunks to consider (default: 5).
        tag: Identifier for the experiment run.
        
    Returns:
        Dictionary of metrics, hit status per query, and failure cases.
    """
    total_q = len(questions)
    hits = 0
    reciprocal_ranks = []
    query_details = []
    failures = []
    
    for q in questions:
        qid = q["id"]
        query_text = q["question"]
        target_page = q.get("target_page")
        key_phrases = [kp.lower() for kp in q.get("key_phrases", [])]
        
        # Retrieve chunks
        retrieved_chunks = retriever_func(query_text, top_k=top_k)
        
        hit_found = False
        hit_rank = 0
        hit_chunk_id = None
        
        for rank, chunk in enumerate(retrieved_chunks):
            c_text = chunk.get("text", "").lower()
            c_page = chunk.get("page_number")
            
            # Check page match and phrase containment
            page_match = (target_page is not None and c_page == target_page)
            phrase_match = any(kp in c_text for kp in key_phrases)
            
            if phrase_match or page_match:
                hit_found = True
                hit_rank = rank + 1
                hit_chunk_id = chunk.get("chunk_id")
                break
                
        if hit_found:
            hits += 1
            reciprocal_ranks.append(1.0 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append({
                "id": qid,
                "question": query_text,
                "target_page": target_page,
                "expected_key_phrases": key_phrases,
                "ground_truth_answer": q.get("ground_truth_answer"),
                "top_retrieved": [
                    {
                        "rank": i + 1,
                        "chunk_id": c.get("chunk_id"),
                        "page_number": c.get("page_number"),
                        "score": c.get("retrieval_score", c.get("rrf_score", c.get("rerank_score", 0.0))),
                        "excerpt": c.get("text", "")[:120] + "..."
                    }
                    for i, c in enumerate(retrieved_chunks[:3])
                ]
            })
            
        query_details.append({
            "id": qid,
            "hit": hit_found,
            "hit_rank": hit_rank,
            "hit_chunk_id": hit_chunk_id
        })
        
    recall_at_k = hits / max(total_q, 1)
    mrr_at_k = sum(reciprocal_ranks) / max(total_q, 1)
    
    results = {
        "tag": tag,
        "total_queries": total_q,
        "top_k": top_k,
        "hits": hits,
        "recall_at_k": float(recall_at_k),
        "mrr_at_k": float(mrr_at_k),
        "failures": failures,
        "query_details": query_details
    }
    
    return results
