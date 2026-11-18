# TND R&D AI/ML Internship Project

A modular, reproducible Python implementation of the three core engineering tasks for the **TND R&D AI/ML Internship Assignment**:

1. **Task 1 — Parameter-Efficient Fine-Tuning (LoRA)**: Fine-tuning `Qwen/Qwen2.5-0.5B-Instruct` on four-class AG News classification with prompt loss masking, paired with a first-principles 7B GPU VRAM analysis.
2. **Task 2 — Modular RAG From Scratch**: End-to-end Retrieval-Augmented Generation over the 77-page Llama 2 research paper (`arXiv:2307.09288`) using PyMuPDF, dual token chunking, Sentence Transformers, FAISS/NumPy, pure-Python Okapi BM25, Reciprocal Rank Fusion (RRF), and Cross-Encoder re-ranking — with **zero LangChain or LlamaIndex dependencies**.
3. **Task 3 — Local LLM Optimization & Serving Capacity**: Empirical FP16 vs. 4-bit NF4 inference benchmarking of `Qwen/Qwen2.5-0.5B-Instruct` on an NVIDIA Tesla T4 GPU, plus an analytical 20-user serving capacity plan on a 24 GiB GPU.

For the complete verified technical report, see **[`report/FINAL_VERIFIED_REPORT.md`](report/FINAL_VERIFIED_REPORT.md)** (also synchronized at [`FINAL_VERIFIED_REPORT.md`](FINAL_VERIFIED_REPORT.md)). For the full repository verification audit, see [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md).

---

## Project Objectives & Evidence Standards

- **First-Principles Implementation**: No LangChain or LlamaIndex; all chunking, inverted indexing, Okapi BM25 scoring, RRF, and citation assembly are written directly in Python.
- **Strict Separation of Evidence Tiers**:
  - **Measured Results**: Metrics collected from actual execution (Google Colab Tesla T4 GPU runs for Task 1 and Task 3; local host runs for Task 2 PDF extraction, chunking, BM25 evaluation, and 10/10 automated unit tests).
  - **Code-Verified Components**: Modular pipelines verified via Python syntax compilation (20/20 `.py` files) and automated tests ([`tests/test_suite.py`](tests/test_suite.py)).
  - **Theoretical / Conditional Estimates**: 7B training VRAM and 20-user serving capacity models derived analytically in consistent binary **GiB** ($1\text{ GiB} = 1024^3\text{ bytes}$), with activation and CUDA overheads explicitly labeled as estimates.
- **Reproducibility**: Deterministic splits (`seed = 42`), standardized evaluation prompts, saved JSON configs, and Google Colab notebooks for GPU-dependent workloads.

---

## Task Summaries & Verified Results

### Task 1: LoRA Fine-Tuning on AG News ([`task1_lora/`](task1_lora/README.md))
- **Model & Dataset**: `Qwen/Qwen2.5-0.5B-Instruct` on AG News (`World`, `Sports`, `Business`, `Sci/Tech`), using **2,000** training samples and **100** held-out test samples (`seed = 42`, strictly disjoint `train`/`test` splits).
- **LoRA Setup**: `CAUSAL_LM`, rank $r = 16$, $\alpha = 32$, dropout $0.05$, targeting all 7 linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with prompt token loss masking (`-100`).
- **Measured Results (Tesla T4 GPU)**:
  - **Trainable Parameters**: **8,798,208** (**1.7497%** of total model parameters)
  - **Training**: **3 epochs**, final training loss **0.27636**
  - **Base Model Accuracy (zero-shot)**: **65.0%** (100 test samples)
  - **LoRA Fine-Tuned Accuracy**: **86.0%** (**+21.0 percentage points** improvement)
  - **Verified Per-Class LoRA F1** (from [`task1_lora/results/7b_memory_calculation.md`](task1_lora/results/7b_memory_calculation.md)): World **0.87** (P 1.00, R 0.77, n=22), Sports **0.96** (P 0.93, R 1.00, n=27), Business **0.81** (P 0.81, R 0.81, n=32), Sci/Tech **0.78** (P 0.73, R 0.84, n=19); Macro F1 **0.86**.

---

### Task 2: Modular RAG From Scratch ([`task2_rag/`](task2_rag/README.md))
- **Corpus & Extraction (Measured)**: *Llama 2: Open Foundation and Fine-Tuned Chat Models* (`arXiv:2307.09288`), **77 pages** (**263,678 characters**) extracted page-by-page via PyMuPDF.
- **Dual Token Chunking (Measured)**: `all-MiniLM-L6-v2` sliding-window tokenization producing **326 chunks** (256-token window, 32-token overlap) and **177 chunks** (512-token window, 64-token overlap).
- **Retrieval & Re-Ranking Stack (Code-Verified)**:
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (**384 dimensions**, $L_2$-normalized).
  - **Vector Retrieval**: FAISS `IndexFlatIP` with exact NumPy cosine similarity fallback.
  - **Sparse Retrieval**: Pure-Python Okapi BM25 ($k_1 = 1.5, b = 0.75$).
  - **Hybrid & Re-Ranking**: Reciprocal Rank Fusion (`k = 60`) and Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **Evaluation (15 Verified Questions, Target-Page Matching)**:
  - **BM25 256-token (Measured Locally)**: Recall@5 = **100.0%** (15/15), MRR@5 = **0.8000**; Recall@1 = **66.7%** (10/15; Top-1 misses: `q01`, `q06`, `q09`, `q11`, `q13`).
  - **BM25 512-token (Measured Locally)**: Recall@5 = **100.0%** (15/15), MRR@5 = **0.8722** — **highest MRR@5 among all reported configurations**.
  - **Dense, Hybrid & Cross-Encoder**: **Hybrid 256-token** ranked best specifically among the Dense, Hybrid, and Cross-Encoder pipelines (not overall best). Raw JSON outputs for these three modes are currently missing from disk (see *Missing Artifacts* below).

---

### Task 3: Local LLM Optimization ([`task3_optimization/`](task3_optimization/README.md))
- **Measured Quantization Benchmark (Tesla T4 GPU, 0.5B Model)**:
  Evaluated **`Qwen/Qwen2.5-0.5B-Instruct`** (a **0.5B** model, *not* a 7B model) on **10 standardized prompts** ([`benchmark_config.json`](task3_optimization/configs/benchmark_config.json)), generating up to **32 tokens** per prompt:

| Metric | FP16 (Unquantized) | 4-bit NF4 (`bitsandbytes`) | Delta (4-bit vs. FP16) |
| :--- | :---: | :---: | :---: |
| **Total Time (10 prompts)** | **9.758 s** | **18.175 s** | +86.26% slower ($1.86\times$) |
| **Average Latency / Prompt** | **0.976 s** | **1.817 s** | +86.17% slower |
| **Throughput** | **32.794 tok/s** | **17.607 tok/s** | -46.31% throughput |
| **Peak Allocated VRAM** | **1.105 GB** | **0.615 GB** | **-44.34% VRAM (-0.490 GB)** |

- **Memory vs. Speed Trade-Off**: 4-bit NF4 reduces peak allocated VRAM by **44.3%**, but runs slower than FP16 at 0.5B scale and batch size 1 because on-the-fly weight dequantization and kernel launch overhead dominate when FP16 weights (~1 GB) already fit easily within T4 HBM bandwidth.

---

### 7B Model VRAM Analysis *(Theoretical & Conditional Estimates)*
All analytical memory estimates use consistent binary **GiB** ($1\text{ GiB} = 1024^3\text{ bytes}$). Activation (~4.00 GiB) and CUDA runtime (~1.50–2.00 GiB) figures are unmeasured estimates, and hardware feasibility notes are conditional estimates (not measured 7B runs).

- **Training VRAM (Llama 2 7B-Class, $L = 32$, $d_{\text{model}} = 4096$, $d_{\text{ff}} = 11008$, $B = 1$, $S = 2048$, gradient checkpointing)**:
  - Verified LoRA adapter parameter count for $r = 16$ across all 7 projections (`q, k, v, o, gate, up, down`): $32 \times [4 \times 16 \times (4096 + 4096) + 3 \times 16 \times (4096 + 11008)] = \mathbf{39,976,960 \approx 39.98\text{M}}$ trainable parameters (FP16 adapter weights/grads = 0.0745 GiB each; FP32 AdamW = 0.4468 GiB; 8-bit AdamW = 0.2234 GiB).

| Training Paradigm (Nominal 7.0B) | Weights | Gradients | Optimizer (AdamW) | Activations *(est.)* | CUDA *(est.)* | Total *(est.)* | Conditional Hardware Estimate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Fine-Tuning (FP16)** | 13.04 GiB | 13.04 GiB | 78.23 GiB | ~4.00 GiB | ~2.00 GiB | **~110.31 GiB** *(106.41 @ 6.74B)* | Multi-GPU (e.g., $2\times 80$ GiB A100) |
| **LoRA (FP16 Base, $r=16$, 39.98M)** | 13.11 GiB | 0.074 GiB | 0.447 GiB | ~4.00 GiB | ~1.50 GiB | **~19.13 GiB** *(18.65 @ 6.74B)* | Conditionally feasible on 24 GiB GPU |
| **QLoRA (NF4 Base, $r=16$, 39.98M)** | 3.66 GiB | 0.074 GiB | 0.447 GiB *(0.223 8-bit)* | ~4.00 GiB | ~1.50 GiB | **~9.68 GiB** *(~9.46 w/ 8-bit)* | Conditionally feasible on 16 GiB T4 |

- **20-User Serving Capacity Plan on a 24 GiB GPU ($B = 20, S = 2048$, FP16 KV Cache, 4-bit 7B Model)**:
  - **MHA ($n_{\text{kv}} = 32$, e.g., Llama 2 7B)**: 3.59 GiB weights + ~1.50 GiB overhead + 20.00 GiB KV cache = **~25.09 GiB** (conditional OOM on 24 GiB).
  - **GQA ($n_{\text{kv}} = 8$, e.g., Qwen2.5-7B / Mistral 7B)**: 3.59 GiB weights + ~1.50 GiB overhead + 5.00 GiB KV cache = **~10.09 GiB** (conditionally feasible with ~13.91 GiB headroom using PagedAttention and continuous batching).

---

## Technology Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Core Language & Numerics** | Python 3, NumPy, Pandas, SciPy, scikit-learn, Matplotlib |
| **LLM & Fine-Tuning** | PyTorch, Hugging Face Transformers, PEFT (LoRA), Datasets, Accelerate, `bitsandbytes` (Colab) |
| **RAG & Document Processing** | PyMuPDF (`fitz`), Sentence Transformers (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`), FAISS (with NumPy fallback), Custom Okapi BM25 |
| **Benchmarking & Testing** | `psutil`, `requests`, Ollama HTTP API client, `unittest`, Google Colab (NVIDIA Tesla T4) |

---

## Project Folder Structure

```text
TnDrr-AI-ML/
├── report/
│   └── FINAL_VERIFIED_REPORT.md            # Complete submission-ready final technical report
├── FINAL_VERIFIED_REPORT.md                # Synchronized root copy of the final report
├── PROJECT_AUDIT.md                        # Repository technical audit report
├── README.md                               # Project overview and reproduction guide
├── requirements.txt                        # Core Python dependencies
├── task1_lora/                             # Task 1: LoRA Fine-Tuning & 7B VRAM Analysis
│   ├── configs/lora_config.json
│   ├── notebooks/task1_lora_colab.ipynb    # Colab T4 GPU notebook
│   ├── results/7b_memory_calculation.md    # VRAM report + verified LoRA classification report
│   ├── src/{__init__,data_loader,model_utils,evaluate,memory_calc,train_lora}.py
│   ├── run_task1.py
│   └── README.md
├── task2_rag/                              # Task 2: Modular RAG From Scratch
│   ├── data/
│   │   ├── ground_truth_qa.json            # 15 manually authored ground-truth QA pairs
│   │   └── llama2_research_paper.pdf       # 77-page Llama 2 paper (gitignored download)
│   ├── src/{__init__,download_pdf,pdf_extractor,chunking,retrieval,reranker,generator,evaluate_rag}.py
│   ├── run_task2.py
│   └── README.md
├── task3_optimization/                     # Task 3: LLM Quantization & Capacity Planning
│   ├── configs/benchmark_config.json       # 10 standardized evaluation prompts
│   ├── notebooks/task3_optimization_colab.ipynb # Colab T4 GPU benchmark notebook
│   ├── src/
│   │   ├── __init__.py
│   │   ├── benchmark.py                    # Ollama/HTTP latency & telemetry profiler
│   │   ├── kv_cache_and_quantization_guide.md
│   │   └── serving_capacity_plan.md
│   ├── run_task3.py
│   └── README.md
└── tests/
    └── test_suite.py                       # 10/10 passing automated unit tests
```

---

## Setup & Reproduction Instructions

### 1. Local Environment Setup
```bash
git clone <repository-url>
cd TnDrr-AI-ML

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

pip install -r requirements.txt
```
*Environment Note*: On Windows hosts with active Windows Application Control (WDAC) or without a CUDA GPU, local execution covers pure-Python modules, PyMuPDF extraction, BM25 evaluation, and the unit test suite. Use the provided Google Colab notebooks for GPU-accelerated training and 4-bit quantization.

### 2. Local Verification & CLI Commands
```bash
# Run automated unit test suite (10/10 passing)
python -m unittest tests/test_suite.py

# Task 1: Generate 7B VRAM derivation report
python -m task1_lora.run_task1 --mode calc_memory

# Task 2: Download 77-page Llama 2 PDF and run RAG pipeline
python -m task2_rag.src.download_pdf
python -m task2_rag.run_task2 --mode all

# Task 3: View serving capacity plan and KV cache / quantization guide
python -m task3_optimization.run_task3 --mode capacity_plan
python -m task3_optimization.run_task3 --mode guide
```

### 3. Google Colab GPU Execution
1. **Task 1 (LoRA Fine-Tuning)**: Upload [`task1_lora/notebooks/task1_lora_colab.ipynb`](task1_lora/notebooks/task1_lora_colab.ipynb) to Google Colab, set runtime to **T4 GPU**, run all cells, and extract exported artifacts into `task1_lora/results/`.
2. **Task 3 (FP16 vs. 4-bit NF4 Benchmark)**: Upload [`task3_optimization/notebooks/task3_optimization_colab.ipynb`](task3_optimization/notebooks/task3_optimization_colab.ipynb) to Google Colab (**T4 GPU**), run all cells, and save `benchmark_results.json` to `task3_optimization/results/`.

---

## Missing Raw Artifacts (Transparency Notice)

To avoid fabricating results or claiming unarchived outputs exist on disk, the following raw Colab/benchmark artifacts are **not currently stored in the repository** and should be imported into their respective `results/` folders for complete raw-artifact reproducibility:

- **Task 1 (`task1_lora/results/`)**: Base-model per-class classification report, base and LoRA confusion matrix arrays/PNG plots (`confusion_matrix_*.png`), and raw JSON metric/failure exports (`metrics_*.json`, `failures_*.json`).
- **Task 2 (`task2_rag/results/`)**: Saved Dense, Hybrid, and Cross-Encoder metric tables (`benchmark_metrics.json`), saved retrieval failure logs (`retrieval_failures.json`), and grounded citation answer outputs (`sample_answers_with_citations.json`).
- **Task 3 (`task3_optimization/results/`)**: Raw per-prompt generated text outputs (`benchmark_results.json`), host RAM logs per prompt, and local Ollama/llama.cpp daemon benchmark logs.
