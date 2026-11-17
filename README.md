# TnDrr AI/ML Internship Assignment: Full Modular Implementation

Act as a Senior AI/ML Engineer and Research Mentor. This repository contains the complete, production-grade, reproducible implementation of all three tasks specified in the **TnDrr AI/ML Internship Assignment**.

---

## Architectural Principles & Strict Compliance
- **Zero LangChain / LlamaIndex**: All retrieval algorithms (token chunking, Okapi BM25 inverted index, Reciprocal Rank Fusion, vector similarity, and citation synthesis) are built completely from first principles.
- **No Fabricated Results**: Every metric, VRAM calculation, and failure case analysis is mathematically derived from exact formulas or generated through actual code execution.
- **Hardware-Aware Multi-Tier Execution**:
  - Local environment audit: Windows 64-bit, Intel Core 5 320 (6 threads), 15.54 GB RAM, CPU-only.
  - Interactive, GPU-accelerated Google Colab notebooks provided for full GPU-dependent fine-tuning and quantization profiling.
- **Reproducibility**: Explicit random seeds (`seed = 42`), strictly deterministic splits, held-out evaluation sets ($\ge 50$ samples), and standardized evaluation prompts.

---

## Repository Structure

```
TnDrr-AI-ML/
├── task1_lora/                        # Task 1: LoRA Fine-Tuning & Evaluation
│   ├── configs/
│   │   └── lora_config.json           # LoRA hyperparameter configuration
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_loader.py             # AG News loading, prompt formatting, label mapping
│   │   ├── model_utils.py             # Qwen2.5-0.5B-Instruct loading & PEFT LoRA wrapping
│   │   ├── evaluate.py                # Deterministic greedy evaluation & metrics
│   │   ├── memory_calc.py             # 7B model GPU VRAM calculation engine
│   │   └── train_lora.py              # SFT training loop with prompt target masking
│   ├── notebooks/
│   │   └── task1_lora_colab.ipynb     # Interactive Google Colab notebook (T4 GPU ready)
│   ├── results/
│   │   └── 7b_memory_calculation.md   # Mathematical GPU VRAM derivation report
│   ├── run_task1.py                   # Unified CLI runner for Task 1
│   └── README.md
│
├── task2_rag/                         # Task 2: RAG From Scratch (Zero LangChain/LlamaIndex)
│   ├── data/
│   │   ├── llama2_research_paper.pdf  # 77-page public document (arXiv:2307.09288)
│   │   └── ground_truth_qa.json       # 15 manually authored and verified QA pairs
│   ├── src/
│   │   ├── __init__.py
│   │   ├── download_pdf.py            # Automatic PDF download and verification (>= 30 pages)
│   │   ├── pdf_extractor.py           # PyMuPDF structured page extraction
│   │   ├── chunking.py                # Token-based chunker (256 vs 512 tokens with overlap)
│   │   ├── retrieval.py               # FAISS + NumPy Dense Retriever & Pure BM25 Sparse
│   │   ├── reranker.py                # Cross-Encoder (Full cross-attention re-ranker)
│   │   ├── generator.py               # Grounded answer synthesis with [Page X, Chunk Y]
│   │   └── evaluate_rag.py            # Recall@5 and MRR@5 evaluation harness
│   ├── run_task2.py                   # Unified CLI runner for Task 2
│   └── README.md
│
├── task3_optimization/                # Task 3: Local LLM Optimization & Capacity Planning
│   ├── configs/
│   │   └── benchmark_config.json      # 10 standardized prompts across 10 distinct domains
│   ├── src/
│   │   ├── __init__.py
│   │   ├── benchmark.py               # Latency, throughput (tok/s), RAM, and VRAM profiler
│   │   ├── kv_cache_and_quantization_guide.md # Graduate-level reference guide
│   │   └── serving_capacity_plan.md   # Transparent 20-user serving plan on 24 GB GPU
│   ├── notebooks/
│   │   └── task3_optimization_colab.ipynb # Free T4 GPU Colab benchmark (FP16 vs 4-bit)
│   ├── run_task3.py                   # Unified CLI runner for Task 3
│   └── README.md
│
├── tests/                             # Unit and Integration Test Suite
│   └── test_suite.py                  # End-to-end tests for all modular components
├── requirements.txt                   # Complete project dependencies
├── FINAL_VERIFIED_REPORT.md           # 4-page comprehensive technical report
└── README.md                          # Project overview and reproduction guide
```

---

## Quickstart & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/TnDrr-AI-ML.git
cd TnDrr-AI-ML

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
# source venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Comprehensive Unit Test Suite
```bash
python -m unittest tests/test_suite.py
```

---

## Task Summary & Reproduction

### Task 1: LoRA Fine-Tuning (`task1_lora/`)
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Dataset**: AG News (4 classes: World, Sports, Business, Sci/Tech)
- **LoRA Hyperparameters**: $r=16, \alpha=32$, target projections: all linear layers (`q, k, v, o, gate, up, down`).
- **Reproducible Split**: Seed = 42, 2,000 train samples, 100 held-out test samples ($\ge 50$).
- **7B Model GPU Memory Calculation**:
  - Full Fine-Tuning (FP16): **110.31 GB** (Requires multi-GPU cluster, e.g. 2x A100-80GB).
  - LoRA (FP16 Base, $r=16$): **18.84 GB** (Fits on a single 24 GB GPU like RTX 3090/4090).
  - QLoRA (NF4 4-bit Base, $r=16$): **9.38 GB** (Fits comfortably on a free 16 GB Colab T4 GPU).
- **Reproduction**:
  ```bash
  python -m task1_lora.run_task1 --mode calc_memory
  ```
  For GPU fine-tuning, run `task1_lora/notebooks/task1_lora_colab.ipynb` on Google Colab T4.

---

### Task 2: RAG From Scratch (`task2_rag/`)
- **Document Collection**: 77-page research paper (*Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv:2307.09288).
- **Extraction**: PyMuPDF (`fitz`) extracting structured text page-by-page.
- **Dual Chunk Sizes**:
  - Setting A: 256 tokens (overlap 32 tokens).
  - Setting B: 512 tokens (overlap 64 tokens).
- **Retrievers**:
  - Dense Vector: `sentence-transformers/all-MiniLM-L6-v2` with FAISS & NumPy exact cosine search.
  - Sparse Keyword: Pure Python Okapi BM25 from scratch ($k_1=1.5, b=0.75$, inverted index).
  - Hybrid: Reciprocal Rank Fusion (RRF).
  - Re-ranking: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Evaluation**: 15 manually authored ground-truth questions with verified target passages and pages.
- **Reproduction**:
  ```bash
  python -m task2_rag.run_task2 --mode all
  ```

---

### Task 3: Local LLM Optimization (`task3_optimization/`)
- **Quantization Comparison**: FP16 vs 4-bit (GGUF `Q4_K_M` / bitsandbytes NF4).
- **Workload**: 10 standardized prompts across 10 distinct task domains.
- **KV Cache & Quantization Guide**: In-depth explanations of memory-bandwidth bottlenecks, $O(N)$ vs $O(N^2)$ scaling, GQA vs MHA, and quantization mathematics.
- **Serving Capacity Plan (20 Concurrent Users on a 24 GB GPU)**:
  - 4-bit 7B Model Weights: 4.10 GB
  - CUDA Overhead: 1.50 GB
  - MHA KV Cache for 20 users @ 2k context: 21.40 GB $\implies$ **27.00 GB total (OOM)**.
  - GQA KV Cache for 20 users @ 2k context: 5.36 GB $\implies$ **10.96 GB total (FEASIBLE, 13.04 GB headroom!)**.
- **Reproduction**:
  ```bash
  python -m task3_optimization.run_task3 --mode capacity_plan
  python -m task3_optimization.run_task3 --mode guide
  ```
  For GPU benchmark profiling, run `task3_optimization/notebooks/task3_optimization_colab.ipynb` on Google Colab T4.

---

## Final Verification Checklist
- [x] Python used exclusively across all modules.
- [x] Clean, modular, well-documented source code in separate folders.
- [x] Zero usage of LangChain or LlamaIndex.
- [x] No fabricated experimental results; mathematical derivations and code execution only.
- [x] 7B model GPU VRAM calculation provided for Full FT, LoRA, and QLoRA.
- [x] AG News 4-class classification pipeline with reproducible splits and prompt loss masking.
- [x] Public PDF document collection $\ge 30$ pages used (77-page Llama 2 paper).
- [x] PyMuPDF structured page extraction verified.
- [x] Dual token chunk sizes implemented (256 and 512 tokens).
- [x] FAISS & NumPy dense vector search implemented.
- [x] Pure Python Okapi BM25 implemented from scratch.
- [x] Cross-Encoder re-ranking stage implemented.
- [x] 15 verified ground-truth questions authored with exact citations.
- [x] Concrete retrieval failures documented with root causes and fixes.
- [x] 10 standardized LLM optimization prompts across 10 domains.
- [x] Serving capacity plan for 20 concurrent users on a 24 GB GPU fully calculated.
- [x] Comprehensive unit and integration test suite passing.
- [x] Google Colab notebooks provided for GPU execution.
