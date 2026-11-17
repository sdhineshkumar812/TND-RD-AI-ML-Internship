# Task 2: Modular RAG From Scratch (Zero LangChain / LlamaIndex)

## Overview
This module implements a complete, production-grade **Retrieval-Augmented Generation (RAG)** pipeline completely from scratch using **pure PyTorch, NumPy, PyMuPDF, and SentenceTransformers**. 

In strict adherence to project specifications:
- **Zero reliance on LangChain or LlamaIndex**.
- Full transparency: all chunking, inverted indexing, BM25 scoring, dense vector indexing, and reciprocal rank fusion are written from first principles.
- Tested against a public document collection containing **77 pages** ($\ge 30$ pages): the seminal *Llama 2: Open Foundation and Fine-Tuned Chat Models* research paper (arXiv:2307.09288).

---

## Directory Structure
```
task2_rag/
├── data/
│   ├── llama2_research_paper.pdf      # 77-page benchmark document (arXiv:2307.09288)
│   └── ground_truth_qa.json           # 15 manually authored and verified QA pairs
├── src/
│   ├── __init__.py
│   ├── download_pdf.py                # Download & verify document page count (>= 30 pages)
│   ├── pdf_extractor.py               # PyMuPDF structured page extraction
│   ├── chunking.py                    # Token-bounded sliding window chunker (256 vs 512 tokens)
│   ├── retrieval.py                   # FAISS + NumPy Dense Retriever & Pure BM25 Sparse Retriever
│   ├── reranker.py                    # Cross-Encoder (Full cross-attention re-ranker)
│   ├── generator.py                   # Grounded generation with [Page X, Chunk Y] citations
│   └── evaluate_rag.py                # Recall@5 and MRR@5 evaluation harness
├── notebooks/
│   └── task2_rag_interactive.ipynb    # Interactive exploration and visual inspection
├── results/
│   ├── benchmark_metrics.json         # Raw Recall@5 and MRR@5 across chunk sizes & modes
│   ├── retrieval_failures.json        # Documented failure cases with root-cause analysis
│   └── sample_answers_with_citations.json # Grounded answers with exact citations
├── run_task2.py                       # Unified CLI orchestrator
└── README.md
```

---

## Retrieval Architecture & Components

```
                           +------------------------------+
                           |  77-Page PDF (arXiv:2307.09) |
                           +--------------+---------------+
                                          | (PyMuPDF Extraction)
                                          v
                           +------------------------------+
                           |   Structured Page Text       |
                           +--------------+---------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
          +-----------------------+               +-----------------------+
          | Chunk Size 256        |               | Chunk Size 512        |
          | (Overlap: 32 tokens)  |               | (Overlap: 64 tokens)  |
          +-----------+-----------+               +-----------+-----------+
                      |                                       |
          +-----------+-----------+               +-----------+-----------+
          |                       |               |                       |
          v                       v               v                       v
    +-----------+           +-----------+   +-----------+           +-----------+
    | Dense Vec |           | BM25      |   | Dense Vec |           | BM25      |
    | (MiniLM)  |           | (Okapi)   |   | (MiniLM)  |           | (Okapi)   |
    +-----+-----+           +-----+-----+   +-----+-----+           +-----+-----+
          |                       |               |                       |
          +----------+------------+               +----------+------------+
                     | (RRF Fusion)                          | (RRF Fusion)
                     v                                       v
             +---------------+                       +---------------+
             | Hybrid Recall |                       | Hybrid Top-10 |
             +---------------+                       +-------+-------+
                                                             | (Cross-Encoder)
                                                             v
                                                     +---------------+
                                                     | Re-ranked Top |
                                                     +-------+-------+
                                                             |
                                                             v
                                                     +---------------+
                                                     | Grounded Gen  |
                                                     | with Citation |
                                                     +---------------+
```

### 1. Token-Based Chunking (Dual Chunk Sizes)
Text is chunked based on token IDs using `sentence-transformers/all-MiniLM-L6-v2` tokenizer:
- **Configuration A (Fine-Grained)**: `chunk_size = 256` tokens, `overlap = 32` tokens.
- **Configuration B (Coarse-Grained)**: `chunk_size = 512` tokens, `overlap = 64` tokens.

### 2. Dense Vector Retrieval (FAISS + NumPy)
- Encodes chunks into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
- Normalizes vectors to unit $L_2$ norm.
- Implements `VectorRetriever` with FAISS `IndexFlatIP` support and high-performance vectorized NumPy cosine search fallback (`np.dot(q, doc_embs.T)`).

### 3. Sparse Retrieval: Okapi BM25 from Scratch
Zero external framework dependencies:
- Inverted index construction with token frequencies and document lengths.
- Robertson-Sparck Jones formula for Inverse Document Frequency (IDF):
  $$\text{IDF}(q_i) = \ln \left( \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1.0 \right)$$
- Non-linear term saturation with $k_1 = 1.5, b = 0.75$:
  $$\text{Score}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

### 4. Hybrid Fusion & Cross-Encoder Re-Ranking
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings without requiring score normalization:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- **Cross-Encoder Re-ranker**: Jointly encodes `(query, passage)` pairs via `cross-encoder/ms-marco-MiniLM-L-6-v2` with full cross-attention.

---

## 15 Verified Benchmark Questions
The evaluation dataset (`task2_rag/data/ground_truth_qa.json`) contains 15 manually authored, technically rigorous questions covering diverse sections of the Llama 2 paper:
1. Pretraining token count (2 Trillion)
2. Context window length (4096 tokens)
3. Attention architecture for 34B/70B (Grouped-Query Attention)
4. AdamW optimizer hyperparameters and warmup steps
5. Two distinct reward models (Helpfulness & Safety)
6. Reinforcement learning algorithms (Rejection Sampling & PPO)
7. Ghost Attention (GAtt) purpose in multi-turn dialogues
8. Cumulative GPU compute consumed (3.3M GPU hours on A100-80GB)
9. Privacy-preserving data cleaning (PII scrub)
10. Released model parameter scales (7B, 13B, 70B)
11. Peak learning rate variation by scale (3e-4 vs 1.5e-4)
12. Adversarial probing and red-teaming methodology
13. Reward model loss function formulation (Binary ranking loss with margin)
14. MMLU 5-shot benchmark score (68.9%)
15. Tokenizer choice and vocabulary size (SentencePiece BPE, 32k)

---

## Reproduction Commands

### 1. Download Document Collection & Run Full Pipeline
```bash
python -m task2_rag.run_task2 --mode all
```

### 2. Run Only Retrieval Benchmark
```bash
python -m task2_rag.run_task2 --mode benchmark
```

All metrics, comparison tables, and failure case analyses will be saved to `task2_rag/results/`.
