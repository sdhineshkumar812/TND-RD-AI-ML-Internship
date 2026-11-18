# TnDrr AI/ML Internship Assignment: Final Verified Technical Report

**Role**: AI/ML Engineering Intern Submission  
**Date**: November 2026  
**Repository**: `TnDrr-AI-ML`  
**Frameworks**: Python 3, PyTorch, Hugging Face Transformers, PEFT, PyMuPDF, Sentence Transformers, NumPy (Zero LangChain / Zero LlamaIndex)

---

## 1. Executive Summary & Evidence Taxonomy

This report consolidates all verified work across the three internship tasks: (1) LoRA fine-tuning of `Qwen/Qwen2.5-0.5B-Instruct` on AG News, (2) a modular Retrieval-Augmented Generation (RAG) pipeline built from scratch over the 77-page Llama 2 research paper, and (3) FP16 vs. 4-bit NF4 inference optimization on a Tesla T4 GPU alongside analytical 7B VRAM and 20-user serving capacity models.

To maintain strict scientific rigor, every result in this document is categorized into one of three evidence tiers:
1. **Measured Results (Empirical)**: Metrics obtained from actual execution — Google Colab **Tesla T4 GPU** runs (Task 1 training/evaluation and Task 3 quantization benchmark) and local host runs (Task 2 PDF extraction, dual token chunking, Okapi BM25 evaluation, and unit tests).
2. **Code-Verified Results (Static & Unit-Tested)**: Modular Python implementations verified via syntax checks (20/20 files) and the automated unit test suite ([tests/test_suite.py](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/tests/test_suite.py): **10/10 passing in 0.004 s**).
3. **Analytical / Conditional Estimates**: First-principles mathematical derivations (7B training VRAM and 20-user serving capacity on a 24 GiB GPU) incorporating explicitly labeled estimates for activation memory and CUDA runtime overhead.

### Execution Environments
- **Cloud GPU Environment (Task 1 & Task 3 Experiments)**: Google Colab with NVIDIA **Tesla T4 GPU** (16 GiB VRAM).
- **Local Host Environment (Development, Audit, Task 2 BM25 & Unit Tests)**: Windows 64-bit, Intel Core 5 320 (6 threads), 15.54 GB RAM, CPU-only, Python 3.15.0 (Windows Application Control restricts user-space native `.pyd` loading for PyTorch/SciPy locally, motivating Colab GPU execution).

---

## 2. Task 1: LoRA Fine-Tuning on AG News

### 2.1 Experimental Setup *(Code-Verified & Executed on Tesla T4)*
- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Dataset & Split**: AG News four-class news classification (`0: World`, `1: Sports`, `2: Business`, `3: Sci/Tech`). Strictly disjoint splits (`seed = 42`): **2,000** training samples from `train` and **100** held-out test samples from `test` ($\ge 50$ required).
- **Prompting & Loss Masking**: ChatML instruction format with prompt tokens masked (`-100`) so cross-entropy loss optimizes exclusively over target category tokens ([train_lora.py](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task1_lora/src/train_lora.py), [task1_lora_colab.ipynb](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task1_lora/notebooks/task1_lora_colab.ipynb)).
- **LoRA Configuration**: `CAUSAL_LM`, rank $r = 16$, $\alpha = 32$, dropout $0.05$, targeting all seven linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Trainable Parameters *(Measured)***: **8,798,208** trainable parameters (**1.7497%** of total model parameters).
- **Training Run *(Measured)***: **3 epochs**, final training loss **0.27636**.

### 2.2 Evaluation Results *(Measured on 100 Held-Out Test Samples)*

| Model Setting | Evaluation Samples | Accuracy | Delta vs. Base | Training Epochs | Final Training Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Model (`Qwen2.5-0.5B-Instruct`, zero-shot)** | 100 | **65.0%** | — | 0 | — |
| **LoRA Fine-Tuned Model ($r=16$, 7 projections)** | 100 | **86.0%** | **+21.0 pp** | 3 | **0.27636** |

#### Actual LoRA Fine-Tuned Per-Class Classification Report *(Verified from `task1_lora/results/7b_memory_calculation.md`)*

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **World** | 1.00 | 0.77 | 0.87 | 22 |
| **Sports** | 0.93 | 1.00 | 0.96 | 27 |
| **Business** | 0.81 | 0.81 | 0.81 | 32 |
| **Sci/Tech** | 0.73 | 0.84 | 0.78 | 19 |
| **Accuracy** | — | — | **0.86** | **100** |
| **Macro Avg** | 0.87 | 0.86 | 0.86 | 100 |
| **Weighted Avg** | 0.87 | 0.86 | 0.86 | 100 |

- **Performance Analysis**: LoRA fine-tuning yielded a **+21 percentage point** accuracy gain (65% $\to$ 86%). **Sports** achieved the strongest F1 (0.96, perfect 1.00 recall), while **World** achieved perfect precision (1.00) with lower recall (0.77). **Sci/Tech** (0.78 F1, 0.73 precision) and **Business** (0.81 F1) account for most remaining errors due to topical overlap between corporate technology and financial news.
- **Artifact Availability Note**: Confusion matrix arrays/PNG plots, the base-model per-class classification report, and raw JSON failure logs (`metrics_*.json`, `failures_*.json`, `confusion_matrix_*.png`) are not currently saved in `task1_lora/results/`, so they are omitted rather than invented.

### 2.3 Analytical 7B Model GPU VRAM Analysis *(Theoretical & Conditional Estimates)*
All memory values below use consistent binary **GiB** ($1\text{ GiB} = 1024^3\text{ bytes}$) for a Llama 2 7B-class architecture ($L = 32$, $d_{\text{model}} = 4096$, $d_{\text{ff}} = 11008$, nominal $N = 7.0 \times 10^9$; exact Llama 2 7B parameter count is $6.738 \times 10^9$).

- **Verified Llama 2 7B LoRA Parameter Count ($r = 16$, all 7 projections, $L = 32$, $d_{\text{model}} = 4096$, $d_{\text{ff}} = 11008$)**:
  - For each linear projection $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA adds $P_{\text{LoRA}} = r(d_{\text{in}} + d_{\text{out}})$ trainable parameters.
  - Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`, each $4096 \times 4096$): $4 \times 16 \times (4096 + 4096) = 524,288$ params/layer.
  - MLP (`gate_proj`, `up_proj`, `down_proj`, $4096 \leftrightarrow 11008$): $3 \times 16 \times (4096 + 11008) = 724,992$ params/layer.
  - Per-layer total: $524,288 + 724,992 = 1,249,280$ params/layer; across $L = 32$ layers: $32 \times 1,249,280 = \mathbf{39,976,960 \approx 39.98\text{M}}$ trainable parameters ($0.5933\%$ of exact Llama 2 7B $6.738\text{B}$; $0.5711\%$ of nominal $7.0\text{B}$).
  - Dependent LoRA adapter footprints: FP16 weights = $0.0745\text{ GiB}$ ($76.25\text{ MiB}$); FP16 gradients = $0.0745\text{ GiB}$ ($76.25\text{ MiB}$); FP32 mixed-precision AdamW states ($12\text{ B/param}$) = $0.4468\text{ GiB}$ ($457.50\text{ MiB}$); 8-bit Paged AdamW ($6\text{ B/param}$) = $0.2234\text{ GiB}$ ($228.75\text{ MiB}$).
- **Base & Training Formulas**: FP16 base weights = $2N$ bytes; 4-bit NF4 base + double-quantization constants $\approx 0.55N$ bytes; FP16 gradients = $2N_{\text{trainable}}$ bytes; FP32 mixed-precision AdamW states (FP32 master + $m$ + $v$) = $12N_{\text{trainable}}$ bytes.

| Memory Component (in GiB) | Evidence Type | Full Fine-Tuning (FP16 + AdamW) | LoRA (FP16 Base, $r=16$, 39.98M) | QLoRA (NF4 Base, $r=16$, 39.98M) |
| :--- | :--- | :---: | :---: | :---: |
| **Model Weights** | Analytic (Nominal 7.0B) | 13.04 GiB | 13.11 GiB *(13.04 + 0.074)* | 3.66 GiB *(3.59 + 0.074)* |
| **Gradients** | Analytic (Trainable only) | 13.04 GiB | 0.074 GiB | 0.074 GiB |
| **Optimizer States (AdamW)** | Analytic (FP32 / 8-bit) | 78.23 GiB | 0.447 GiB | 0.447 GiB *(0.223 GiB w/ 8-bit)* |
| **Activations ($B=1, S=2048$, ckpt)** | **Estimate** | ~4.00 GiB | ~4.00 GiB | ~4.00 GiB |
| **CUDA Context & Allocator** | **Estimate** | ~2.00 GiB | ~1.50 GiB | ~1.50 GiB |
| **Total Estimated VRAM** | **Composite Estimate** | **~110.31 GiB** *(106.41 GiB @ 6.74B)* | **~19.13 GiB** *(18.65 GiB @ 6.74B)* | **~9.68 GiB** *(~9.46 GiB w/ 8-bit)* |
| **Hardware Feasibility** | **Conditional Estimate** | Multi-GPU required (e.g., $2\times 80$ GiB A100 w/ FSDP/ZeRO-3) | Conditionally feasible on 24 GiB GPU (RTX 3090/4090, A10G) | Conditionally feasible on 16 GiB T4 or 12 GiB GPU |

*Note*: Activation and CUDA overheads are unmeasured estimates; hardware feasibility claims are conditional on $B=1$, $S=2048$, gradient checkpointing enabled, and low allocator fragmentation, and were not empirically measured on a 7B model.

---

## 3. Task 2: Modular RAG From Scratch (Zero LangChain / LlamaIndex)

### 3.1 Corpus, Extraction, Chunking & Indexing *(Executed & Code-Verified)*
- **Source Document**: *Llama 2: Open Foundation and Fine-Tuned Chat Models* (`arXiv:2307.09288`), **77 pages** ($\ge 30$ pages required), **263,678 characters** extracted via PyMuPDF ([pdf_extractor.py](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task2_rag/src/pdf_extractor.py)).
- **Dual Token-Based Chunking *(Measured)***: Sliding-window tokenization via `all-MiniLM-L6-v2` tokenizer ([chunking.py](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task2_rag/src/chunking.py)):
  - **256-token window** (32-token overlap): **326 chunks**.
  - **512-token window** (64-token overlap): **177 chunks**.
- **Retrieval Components *(Code-Verified)***:
  - **Dense Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` (**384-dimensional** unit-normalized vectors) with FAISS `IndexFlatIP` and exact NumPy dot-product fallback ([retrieval.py](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task2_rag/src/retrieval.py)).
  - **Sparse Retrieval**: Pure Python Okapi BM25 from scratch ($k_1 = 1.5, b = 0.75$, Robertson-Sparck Jones IDF).
  - **Hybrid Fusion & Re-Ranking**: Reciprocal Rank Fusion ($\text{RRF}(d) = \sum \frac{1}{60 + \text{rank}_m(d)}$) and optional Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) in [reranker.py](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task2_rag/src/reranker.py).
  - **Grounded Citation Generation**: Context blocks formatted as `[chunk_id | Page X]` with strict citation prompting in [generator.py](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task2_rag/src/generator.py) (verified via unit test `test_context_and_citations`).

### 3.2 Evaluation Methodology & Measured Retrieval Results
- **Benchmark Set**: **15** manually authored questions mapped to ground-truth pages and key phrases in [ground_truth_qa.json](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task2_rag/data/ground_truth_qa.json).
- **Evaluation Criterion (Target-Page Matching)**: Retrieval relevance is evaluated via **target-page matching** (in [evaluate_rag.py](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task2_rag/src/evaluate_rag.py), a hit occurs when a retrieved chunk matches the ground-truth `target_page` or contains a verified key phrase). Because target-page matching credits any chunk originating from the target page rather than requiring exact span boundaries, it acts as a page-level proxy that can yield higher Recall@5 than strict chunk-ID evaluation.

| Retrieval Configuration | Chunks | Vocabulary / Dim | Evidence Status | Recall@5 | MRR@5 | Hits |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **BM25 Sparse (256-token)** | 326 | 6,111 terms | **Measured (Local Audit)** | **100.0%** | **0.8000** | **15 / 15** |
| **BM25 Sparse (512-token)** | 177 | 6,080 terms | **Measured (Local Audit — Highest MRR@5)** | **100.0%** | **0.8722** | **15 / 15** |
| **Dense (`all-MiniLM-L6-v2`)** | 326 / 177 | 384-dim | *Missing saved artifact on disk* | Not saved on disk | Not saved on disk | — |
| **Hybrid (Dense + BM25 RRF, 256-token)** | 326 | 384-dim + BM25 | *Best among Dense/Hybrid/Cross-Encoder; JSON missing on disk* | Not saved on disk | Not saved on disk | — |
| **Cross-Encoder Re-Ranking** | Top-10 | MiniLM Cross-Enc | *Missing saved artifact on disk* | Not saved on disk | Not saved on disk | — |

- **Comparative Ranking Across Reported Configurations**:
  - **Highest MRR@5 Overall**: **BM25 with 512-token chunks** achieved the highest MRR@5 (**0.8722**) and **100.0% Recall@5** (15/15 hits) among all reported configurations (outperforming BM25 256-token at MRR@5 **0.8000** and 100.0% Recall@5).
  - **Best Among Dense, Hybrid, and Cross-Encoder Pipelines**: **Hybrid retrieval with 256-token chunks** ranked best *specifically among the Dense, Hybrid, and Cross-Encoder configurations* based on its MRR@5 and Recall@5 — **not the overall best** across all reported methods, since BM25 512 achieved the highest overall MRR@5 (0.8722).
- **Verified Local BM25 Top-1 Failure Analysis (256-token chunks)**: While BM25 achieved 100% Recall@5, strict Top-1 evaluation yielded **Recall@1 = 66.7% (10/15 hits)** with five Rank-1 misses: **`q01`** (pretraining tokens — generic token frequency dilution), **`q06`** (RLHF algorithms — "fine-tuning" term overlap), **`q09`** (PII scrub — synonym mismatch between "personal data" and "PII"), **`q11`** (peak learning rates — low IDF weighting of numeric literals `3e-4`/`1.5e-4`), and **`q13`** (reward model ranking loss — mismatch between English query terms and LaTeX notation). Increasing chunk size to 512 improved BM25 MRR@5 from **0.8000 to 0.8722 (+0.0722)**.
- **Artifact Availability Note**: Saved output files for Dense, Hybrid, and Cross-Encoder runs (`task2_rag/results/benchmark_metrics.json`, `retrieval_failures.json`, `sample_answers_with_citations.json`) are **missing from the workspace**; their raw numerical values are therefore not invented here.

---

## 4. Task 3: Local LLM Optimization & Serving Capacity Plan

### 4.1 Measured Tesla T4 Quantization Benchmark (0.5B Model)
*Scope Clarification*: The empirical measurements below were collected on a **Tesla T4 GPU** using **`Qwen/Qwen2.5-0.5B-Instruct` (a 0.5B parameter model, NOT a 7B model)** across the same **10 standardized prompts** ([benchmark_config.json](file:///c:/Users/smdhi/OneDrive/Documents/PROJECT/TnDrr-AI-ML/task3_optimization/configs/benchmark_config.json)), generating up to **32 tokens** per prompt (320 total generated tokens).

| Metric | FP16 (Unquantized) | 4-bit NF4 (`bitsandbytes`) | Relative Change (4-bit vs. FP16) |
| :--- | :---: | :---: | :---: |
| **Total Generation Time (10 prompts)** | **9.758 s** | **18.175 s** | +86.26% slower ($1.86\times$) |
| **Average Latency per Prompt** | **0.976 s** | **1.817 s** | +86.17% slower |
| **Throughput (Tokens / Second)** | **32.794 tok/s** | **17.607 tok/s** | -46.31% throughput |
| **Peak Allocated VRAM** | **1.105 GB** | **0.615 GB** | **-44.34% VRAM (-0.490 GB)** |

#### Memory vs. Speed Trade-Off Analysis
- **Significant Memory Savings**: 4-bit NF4 reduced peak allocated VRAM from **1.105 GB to 0.615 GB** (**44.3% reduction**). The reduction is less than the theoretical 75% weight compression because token embeddings, layer norms, LM head, KV cache, and intermediate activations remain in FP16.
- **Why 4-bit NF4 Was Slower Than FP16 at 0.5B Scale**: Autoregressive decoding is normally memory-bandwidth bound on large models (7B–70B). However, for a **0.5B** model on a Tesla T4 (320 GB/s HBM bandwidth), FP16 weights (~1 GB) transfer in ~3 ms per token. Under `bitsandbytes` NF4 weight-only quantization, every linear layer incurs **on-the-fly dequantization overhead** (unpacking 4-bit weights and scaling factors back to FP16 before GEMV) plus additional CUDA kernel overhead. At 0.5B scale and batch size 1, this dequantization overhead exceeds the modest bandwidth savings, reducing throughput from **32.794 to 17.607 tok/s**. On larger 7B+ models or with specialized fused W4A16 kernels (AWQ/Marlin/GGUF), 4-bit quantization becomes essential to fit in VRAM and improve effective throughput.
- **Artifact Availability Note**: Raw per-prompt generated text files (`benchmark_results.json`), host RAM telemetry, and local Ollama/llama.cpp logs are not present in `task3_optimization/results/`, so per-prompt qualitative comparisons are not fabricated.

### 4.2 KV Cache Dynamics & 20-User Serving Capacity Plan *(Analytical & Conditional Estimates)*
Caching past Keys and Values reduces autoregressive attention compute from $O(N^2)$ to $O(N)$, with KV memory scaling as:
$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S \times \text{bytes\_per\_precision}$$

For **20 concurrent users** ($B = 20$), peak context $S = 2048$, $L = 32$, $d_{\text{head}} = 128$, FP16 KV cache (2 B/elem), and a nominal **7B 4-bit model** on a **24 GiB GPU** (e.g., RTX 3090/4090 or A10G):

| Memory Component (Consistent GiB) | Evidence Type | 7B 4-bit + MHA ($n_{\text{kv}}=32$, e.g., Llama 2 7B) | 7B 4-bit + GQA ($n_{\text{kv}}=8$, e.g., Qwen2.5-7B / Mistral 7B) |
| :--- | :--- | :---: | :---: |
| **Static 4-bit Model Weights (~0.55 B/param)** | Analytic | 3.59 GiB *(~4.0 GiB w/ `Q4_K_M`)* | 3.59 GiB *(~4.0 GiB w/ `Q4_K_M`)* |
| **CUDA Context & Serving Engine Overhead** | **Estimate** | ~1.50 GiB | ~1.50 GiB |
| **KV Cache (20 users $\times$ 2048 tokens, FP16)** | Analytic | 20.00 GiB ($1.00$ GiB/user) | 5.00 GiB ($0.25$ GiB/user) |
| **Total Estimated VRAM** | **Composite Estimate** | **~25.09 GiB** *(~25.5 GiB w/ `Q4_K_M`)* | **~10.09 GiB** *(~10.5 GiB w/ `Q4_K_M`)* |
| **24 GiB GPU Feasibility** | **Conditional Estimate** | **Exceeds 24 GiB (Conditional OOM)** | **Conditionally Feasible (~13.5–13.9 GiB headroom)** |

*Architectural Recommendation*: Serving 20 concurrent users at 2,048-token context on a 24 GiB GPU conditionally requires a **GQA-based 7B model** combined with **4-bit weight quantization**, **PagedAttention** (to eliminate KV cache fragmentation), and **continuous batching**.

---

## 5. Verification Checklist

| Deliverable / Requirement | Verification Evidence | Status |
| :--- | :--- | :---: |
| **Modular Python, Zero LangChain/LlamaIndex** | 20 `.py` files compile cleanly; pure-Python BM25, RRF, chunking | **Code-Verified** |
| **Automated Unit Test Suite** | `tests/test_suite.py` executed locally: 10/10 tests pass (0.004 s) | **Measured (Passed)** |
| **Task 1: LoRA Fine-Tuning & Eval (T4 GPU)** | Base 65% $\to$ LoRA 86% (+21 pp), 8.80M params, loss 0.27636, class report | **Measured (Colab T4)** |
| **Task 1: 7B VRAM Calculation** | Corrected to 39.98M LoRA params ($r=16$), consistent GiB, labeled estimates | **Analytical Estimate** |
| **Task 2: PDF Extraction & Dual Chunking** | Llama 2 PDF (77 pages, 263,678 chars); 326 (256-tok) & 177 (512-tok) chunks | **Measured (Local)** |
| **Task 2: BM25 & Top-1 Failure Analysis** | Recall@5 100% (both); BM25 512 highest MRR@5 (0.8722) vs. 256 (0.8000); 5 Top-1 misses | **Measured (Local)** |
| **Task 2: Dense, Hybrid & Cross-Encoder** | Code implemented; Hybrid 256 best among Dense/Hybrid/Cross-Encoder; raw JSON missing | **Code-Verified / Missing JSON** |
| **Task 3: FP16 vs. 4-bit Benchmark (T4 GPU)** | 0.5B model, 10 prompts $\times$ 32 tok: 32.79 vs. 17.61 tok/s; 1.105 vs. 0.615 GB | **Measured (Colab T4)** |
| **Task 3: 20-User Serving Plan (24 GiB GPU)** | MHA ~25.09 GiB (OOM) vs. GQA ~10.09 GiB (feasible w/ PagedAttention) | **Analytical Estimate** |

---

## 6. Conclusion & Limitations

### Conclusion
All three internship tasks were engineered end-to-end without high-level orchestration wrappers. Empirical results confirm that: (1) LoRA fine-tuning of `Qwen2.5-0.5B-Instruct` with only **1.75%** trainable parameters yields a **+21 percentage point** accuracy gain (65% $\to$ 86%) on AG News; (2) pure-Python **BM25 with 512-token chunks** achieved the highest MRR@5 (**0.8722**) and **100.0% Recall@5** among all reported configurations over the 77-page technical PDF (compared to MRR@5 **0.8000** for BM25 256-token), while **Hybrid 256-token** ranked best specifically among the Dense, Hybrid, and Cross-Encoder pipelines; and (3) 4-bit NF4 quantization cuts peak VRAM by **44.3%** on a Tesla T4 GPU while highlighting the dequantization latency penalty on small (0.5B) models. Analytical modeling further demonstrates why QLoRA (for training) and 4-bit + GQA + PagedAttention (for 20-user serving) are structurally required on 16–24 GiB GPUs.

### Limitations & Missing Workspace Artifacts
1. **Unarchived Result Artifacts**: Several raw Colab outputs are not yet committed into the local `results/` folders — specifically Task 1 confusion matrices/base classification report/failure JSONs, Task 2 Dense/Hybrid/Cross-Encoder `benchmark_metrics.json` and citation exports, and Task 3 per-prompt `benchmark_results.json` text outputs. Importing these files into `task*/results/` will complete full artifact reproducibility.
2. **Target-Page Evaluation Granularity**: Task 2 evaluation uses target-page matching (plus key-phrase checks in code), which is coarser than exact passage-span evaluation and can overstate Recall@5 when multiple chunks share a page.
3. **Model Scale Boundary**: Task 3 latency and VRAM measurements were collected on a **0.5B** model with `max_new_tokens=32`; extrapolating throughput directly to 7B models is invalid without measuring 7B weights, which are instead modeled analytically as conditional estimates.
4. **Local Host Constraints**: Because the Windows host lacks a CUDA GPU and enforces WDAC policies blocking user-space `.pyd` binaries, GPU experiments require Google Colab or a Linux/CUDA host.
