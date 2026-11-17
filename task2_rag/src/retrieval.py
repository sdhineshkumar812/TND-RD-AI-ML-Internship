"""
Dense Vector Retrieval (FAISS + NumPy) and BM25 Sparse Retrieval implemented from scratch
without LangChain or LlamaIndex.
"""

import math
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
# Optional FAISS and SentenceTransformer imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorRetriever:
    """
    Dense semantic retriever using SentenceTransformers.
    Supports FAISS IndexFlatIP where available, with seamless, mathematically
    identical NumPy dot-product fallback.
    """
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_faiss_if_available: bool = True
    ):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None
        self.use_faiss = use_faiss_if_available and FAISS_AVAILABLE
        self.chunks = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        
    def index_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64):
        """Compute normalized dense embeddings for all text chunks."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        
        print(f"Embedding {len(texts)} chunks using {self.model}...")
        raw_embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.embeddings = raw_embs.astype(np.float32)
        
        if self.use_faiss:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
            print(f"FAISS IndexFlatIP initialized with {self.faiss_index.ntotal} vectors (dim={dim}).")
        else:
            print("VectorRetriever: Operating in high-performance NumPy exact cosine retrieval mode.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k most similar chunks for the query string."""
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("Retriever has no indexed chunks. Call index_chunks first.")
            
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        
        if self.use_faiss and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q_emb, min(top_k, len(self.chunks)))
            scores = scores[0]
            indices = indices[0]
        else:
            # Exact cosine similarity via vectorized NumPy dot product
            # query shape: (1, dim), embeddings shape: (N, dim) -> scores shape: (N,)
            scores_all = np.dot(self.embeddings, q_emb[0])
            top_k_indices = np.argsort(scores_all)[::-1][:top_k]
            indices = top_k_indices
            scores = scores_all[top_k_indices]
            
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["retrieval_score"] = float(score)
            chunk_copy["retrieval_rank"] = rank + 1
            chunk_copy["retrieval_mode"] = "vector_dense"
            results.append(chunk_copy)
            
        return results


class BM25Retriever:
    """
    Pure Python Okapi BM25 Implementation from scratch.
    Zero dependency on LangChain, LlamaIndex, or external search servers.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple, deterministic lowercase regex tokenization."""
        return re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Build inverted index and calculate BM25 IDF statistics."""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freqs = {}

        total_tokens = 0
        for chunk in chunks:
            tokens = self.tokenize(chunk["text"])
            self.doc_lens.append(len(tokens))
            total_tokens += len(tokens)

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = total_tokens / max(self.corpus_size, 1)

        # Calculate Okapi BM25 IDF
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # Robertson-Sparck Jones formula with smoothing
            val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(val, 0.0)
            
        print(f"BM25 index built: {self.corpus_size} chunks, {len(self.idf)} unique vocabulary terms.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Score all chunks against query using BM25 scoring formula."""
        q_tokens = self.tokenize(query)
        if not q_tokens or self.corpus_size == 0:
            return []

        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for q in q_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for doc_idx, tf_dict in enumerate(self.term_freqs):
                f = tf_dict.get(q, 0)
                if f > 0:
                    doc_len = self.doc_lens[doc_idx]
                    denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    numer = f * (self.k1 + 1.0)
                    scores[doc_idx] += q_idf * (numer / denom)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            chunk_copy = dict(self.chunks[idx])
            chunk_copy["retrieval_score"] = float(scores[idx])
            chunk_copy["retrieval_rank"] = rank + 1
            chunk_copy["retrieval_mode"] = "bm25_sparse"
            results.append(chunk_copy)
            
        return results


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) to merge dense and sparse ranked lists.
    RRF Score(d) = sum(1 / (k + rank_m(d)))
    """
    rrf_scores: Dict[str, float] = {}
    chunk_store: Dict[str, Dict[str, Any]] = {}

    for r_list in ranked_lists:
        for rank, chunk in enumerate(r_list):
            cid = chunk["chunk_id"]
            chunk_store[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + (rank + 1)))

    sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_n]
    
    fused_results = []
    for rank, cid in enumerate(sorted_cids):
        res = dict(chunk_store[cid])
        res["rrf_score"] = float(rrf_scores[cid])
        res["final_rank"] = rank + 1
        res["retrieval_mode"] = "hybrid_rrf"
        fused_results.append(res)
        
    return fused_results
