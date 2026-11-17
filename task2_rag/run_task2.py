"""
Unified CLI orchestrator for Task 2: RAG From Scratch.

Usage:
    python -m task2_rag.run_task2 --download
    python -m task2_rag.run_task2 --benchmark
    python -m task2_rag.run_task2 --all
"""

import os
import sys
import json
import argparse
import numpy as np

from task2_rag.src.pdf_extractor import extract_pdf_pages
from task2_rag.src.chunking import TokenChunker
from task2_rag.src.retrieval import VectorRetriever, BM25Retriever, reciprocal_rank_fusion
from task2_rag.src.reranker import Reranker
from task2_rag.src.generator import RAGGenerator
from task2_rag.src.evaluate_rag import evaluate_retrieval
from task2_rag.src.download_pdf import download_benchmark_pdf, DEFAULT_SAVE_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: RAG From Scratch Orchestrator")
    parser.add_argument(
        "--pdf_path",
        type=str,
        default=DEFAULT_SAVE_PATH,
        help="Path to PDF source document"
    )
    parser.add_argument(
        "--qa_path",
        type=str,
        default="./task2_rag/data/ground_truth_qa.json",
        help="Path to 15 verified questions JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./task2_rag/results",
        help="Directory to save benchmarks and reports"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "download", "benchmark", "demo_qa"],
        help="Execution mode"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 65)
    print(" TnDrr AI/ML Internship Assignment - Task 2: RAG From Scratch")
    print(" Document Collection: Llama 2 Research Paper (arXiv:2307.09288, 77 pages)")
    print(" Evaluation Benchmark: 15 Verified Technical Ground-Truth Questions")
    print(" Vector Retrieval   : FAISS & NumPy Exact Cosine Search")
    print(" Keyword Retrieval  : Custom Okapi BM25 (Zero LangChain/LlamaIndex)")
    print(" Re-ranking         : Cross-Encoder (Full Cross-Attention)")
    print("=" * 65)
    
    # 1. Download / Verify PDF
    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path) or args.mode == "download":
        print("\n>>> Ensuring Public Document Collection (>= 30 pages) is present...")
        pdf_path = download_benchmark_pdf(save_path=pdf_path, min_pages=30)
        if args.mode == "download":
            return
            
    # 2. Extract Text with PyMuPDF
    print("\n>>> Step 1: Extracting Structured Pages with PyMuPDF...")
    pages = extract_pdf_pages(pdf_path, min_pages=30)
    print(f"Verified {len(pages)} pages extracted from {pdf_path}.")
    
    # 3. Load 15 Verified Questions
    with open(args.qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} manually authored and verified benchmark questions.")
    
    # 4. Token-Based Chunking for Two Chunk Sizes
    print("\n>>> Step 2: Token-Based Chunking with Dual Chunk Sizes...")
    chunker = TokenChunker()
    
    # Setting A: Chunk Size 256 (overlap 32)
    chunks_256 = chunker.chunk_document(pages, chunk_size=256, overlap=32)
    
    # Setting B: Chunk Size 512 (overlap 64)
    chunks_512 = chunker.chunk_document(pages, chunk_size=512, overlap=64)
    
    # 5. Build Retrievers & Index Chunks
    print("\n>>> Step 3: Indexing Chunks for Retrieval...")
    
    # Retrievers for 256 tokens
    dense_256 = VectorRetriever()
    dense_256.index_chunks(chunks_256)
    
    bm25_256 = BM25Retriever()
    bm25_256.index_chunks(chunks_256)
    
    # Retrievers for 512 tokens
    dense_512 = VectorRetriever()
    dense_512.index_chunks(chunks_512)
    
    bm25_512 = BM25Retriever()
    bm25_512.index_chunks(chunks_512)
    
    reranker = Reranker()
    
    # 6. Benchmark Evaluation Across Chunk Sizes and Modes
    print("\n>>> Step 4: Running Comprehensive Benchmark across Chunk Sizes & Retrieval Modes...")
    
    experiments = {}
    
    # Experiment A1: Chunk 256 - Dense Vector Retrieval
    experiments["chunk256_dense"] = evaluate_retrieval(
        qa_data,
        retriever_func=lambda q, top_k: dense_256.search(q, top_k=top_k),
        top_k=5,
        tag="Chunk 256 (Dense Vector)"
    )
    
    # Experiment A2: Chunk 256 - BM25 Retrieval
    experiments["chunk256_bm25"] = evaluate_retrieval(
        qa_data,
        retriever_func=lambda q, top_k: bm25_256.search(q, top_k=top_k),
        top_k=5,
        tag="Chunk 256 (BM25 Sparse)"
    )
    
    # Experiment A3: Chunk 256 - Hybrid RRF
    def hybrid_search_256(q, top_k=5):
        d_res = dense_256.search(q, top_k=10)
        b_res = bm25_256.search(q, top_k=10)
        return reciprocal_rank_fusion([d_res, b_res], top_n=top_k)
        
    experiments["chunk256_hybrid"] = evaluate_retrieval(
        qa_data,
        retriever_func=hybrid_search_256,
        top_k=5,
        tag="Chunk 256 (Hybrid RRF)"
    )
    
    # Experiment B1: Chunk 512 - Dense Vector Retrieval
    experiments["chunk512_dense"] = evaluate_retrieval(
        qa_data,
        retriever_func=lambda q, top_k: dense_512.search(q, top_k=top_k),
        top_k=5,
        tag="Chunk 512 (Dense Vector)"
    )
    
    # Experiment B2: Chunk 512 - BM25 Retrieval
    experiments["chunk512_bm25"] = evaluate_retrieval(
        qa_data,
        retriever_func=lambda q, top_k: bm25_512.search(q, top_k=top_k),
        top_k=5,
        tag="Chunk 512 (BM25 Sparse)"
    )
    
    # Experiment B3: Chunk 512 - Hybrid RRF + Re-ranking
    def hybrid_rerank_512(q, top_k=5):
        d_res = dense_512.search(q, top_k=10)
        b_res = bm25_512.search(q, top_k=10)
        fused = reciprocal_rank_fusion([d_res, b_res], top_n=10)
        return reranker.rerank(q, fused, top_k=top_k)
        
    experiments["chunk512_hybrid_rerank"] = evaluate_retrieval(
        qa_data,
        retriever_func=hybrid_rerank_512,
        top_k=5,
        tag="Chunk 512 (Hybrid + Cross-Encoder)"
    )
    
    # Save Benchmark Metrics
    benchmark_file = os.path.join(args.output_dir, "benchmark_metrics.json")
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)
    print(f"\nAll benchmark results saved to: {benchmark_file}")
    
    # Print Comparison Table
    print("\n" + "=" * 70)
    print(f"{'Retrieval Configuration':<35} | {'Recall@5':<12} | {'MRR@5':<10} | {'Hits':<6}")
    print("-" * 70)
    for exp_id, res in experiments.items():
        tag = res["tag"]
        r5 = f"{res['recall_at_k']*100:.1f}%"
        mrr = f"{res['mrr_at_k']:.4f}"
        hits = f"{res['hits']}/{res['total_queries']}"
        print(f"{tag:<35} | {r5:<12} | {mrr:<10} | {hits:<6}")
    print("=" * 70)
    
    # 7. Document Two Actual Retrieval Failures and Proposed Fixes
    print("\n>>> Step 5: Documenting Concrete Retrieval Failures & Fixes...")
    all_failures = []
    # Collect failures from chunk256_dense and chunk256_bm25
    for f in experiments["chunk256_dense"]["failures"]:
        f["failed_regime"] = "Chunk 256 Dense"
        all_failures.append(f)
    for f in experiments["chunk256_bm25"]["failures"]:
        f["failed_regime"] = "Chunk 256 BM25"
        all_failures.append(f)
        
    failures_file = os.path.join(args.output_dir, "retrieval_failures.json")
    with open(failures_file, "w", encoding="utf-8") as f:
        json.dump(all_failures[:5], f, indent=2)
        
    # Generate Answers with Source Citations Demo
    print("\n>>> Step 6: Generating Sample Answers with Grounded Citations [Page X, Chunk Y]...")
    generator = RAGGenerator()
    sample_queries = [
        "What is the total number of pretraining tokens used to train the Llama 2 models?",
        "Which architectural attention mechanism was introduced for the larger 34B and 70B models to improve inference scalability?",
        "What two separate reward models are trained during RLHF to reflect human preferences?"
    ]
    sample_answers = []
    for q in sample_queries:
        top_chunks = hybrid_rerank_512(q, top_k=2)
        ans = generator.generate_answer(q, top_chunks)
        sample_answers.append(ans)
        print(f"\nQ: {q}")
        print(f"A: {ans['answer']}")
        print("Citations:")
        for c in ans["citations"]:
            print(f"  - [{c['chunk_id']} | Page {c['page_number']}]: {c['excerpt']}")
            
    answers_file = os.path.join(args.output_dir, "sample_answers_with_citations.json")
    with open(answers_file, "w", encoding="utf-8") as f:
        json.dump(sample_answers, f, indent=2)
        
    print(f"\nSample answers with citations saved to: {answers_file}")

if __name__ == "__main__":
    main()
