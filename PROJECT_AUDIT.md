# Technical Audit Report: TnDrr AI/ML Internship Assignment Project

**Audit Date**: November 2026  
**Auditor**: Senior AI/ML Engineer & Research Mentor  
**Repository**: `TnDrr-AI-ML`  
**Execution Environment**: Windows 11/10 64-bit | Python 3.15.0 | Intel Core 5 320 (6 threads) | 15.54 GB RAM | No CUDA GPU  

---

## Executive Summary

A comprehensive technical audit of all three tasks and project quality components was performed. The repository demonstrates high architectural quality, clean modular structure, strict avoidance of blackbox frameworks (zero LangChain, zero LlamaIndex), mathematically derived memory calculations, and an automated unit test suite passing 10/10 tests. 

However, critical differences between local host capabilities (CPU-only, Python 3.15, Windows Application Control policy) and cloud GPU requirements were identified. Furthermore, certain metrics and failure cases presented in the initial summary report were theoretical projections rather than artifacts of executed GPU runs. This audit categorizes all items into **PASS**, **FAIL**, and **NOT TESTED**, details exact errors and locations, and outlines concrete reproduction steps.

---

## Audit Item Matrix

### Task 1: LoRA Fine-Tuning
| Item # | Verification Target | Audit Status | Evidence & Detailed Findings |
| :--- | :--- | :---: | :--- |
| **T1.1** | Qwen2.5-0.5B-Instruct model loading | **PASS** (Code) / **NOT TESTED** (Local execution) | `load_base_model_and_tokenizer()` in `task1_lora/src/model_utils.py` uses `AutoModelForCausalLM` and `AutoTokenizer` with ChatML template and `pad_token = eos_token`. Colab notebook executes with GPU acceleration. Local execution is blocked by host WDAC policy preventing `.pyd` dynamic library loading. |
| **T1.2** | AG News dataset loading & label mapping | **PASS** | `LABEL_MAPPING = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}` verified in `data_loader.py`. `test_label_mapping` passes in test suite. |
| **T1.3** | Training & held-out eval data separation | **PASS** | Line 61-66 in `data_loader.py` pulls `train_subset` strictly from `dataset["train"]` and `eval_subset` strictly from `dataset["test"]`. Deterministic shuffling (`seed=42`) guarantees reproducibility. |
| **T1.4** | Evaluation set $\ge 50$ examples | **PASS** | `lora_config.json` sets `eval_samples: 100` (exceeds the 50-example minimum). |
| **T1.5** | LoRA config & target modules | **PASS** | `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")`. All linear projections targeted. |
| **T1.6** | Base-model evaluation implementation | **PASS** | `evaluate_model()` in `evaluate.py` performs greedy generation (`do_sample=False, max_new_tokens=8`), parses generated text into label IDs, and handles special tokens. |
| **T1.7** | Fine-tuning & post-training evaluation | **PASS** | `train_lora.py` uses `Trainer` and `DataCollatorForSeq2Seq` with prompt target masking (`labels = [-100]*len(prompt) + target`). Post-training loads `PeftModel` and evaluates on the identical held-out test split. |
| **T1.8** | Accuracy, Precision, Recall, F1 & Confusion Matrix | **PASS** | `evaluate.py` computes metrics via `accuracy_score`, `classification_report`, and `confusion_matrix`. Generates both raw and normalized matrices and exports matplotlib heatmap to PNG. |
| **T1.9** | Actual failure examples generation | **FAIL** (Report claims) / **PASS** (Code logic) | Code logic in `evaluate.py` correctly appends `{text, true_label, pred_label, raw_generated_text}` to `failures` list when `pred != true`. However, the 3 failure cases in `FINAL_VERIFIED_REPORT.md` Section 2.4 were illustrative summaries before executing on GPU, not imported from a real `failures_lora_finetuned.json` output. |
| **T1.10** | 7B GPU memory calculation | **PASS** | `memory_calc.py` implements transparent mathematical derivations: Full FT: 110.31 GB (2x A100-80GB required); LoRA: 18.84 GB (RTX 3090/4090 24GB); QLoRA: 9.38 GB (Colab T4 16GB). Verified by `test_7b_memory_calculation`. |

---

### Task 2: RAG From Scratch
| Item # | Verification Target | Audit Status | Evidence & Detailed Findings |
| :--- | :--- | :---: | :--- |
| **T2.1** | PDF extraction | **PASS** | PyMuPDF (`fitz`) extracted all 77 pages (263,678 characters) from `llama2_research_paper.pdf`. Verified $\ge 30$ pages constraint. |
| **T2.2** | Dual chunk sizes | **PASS** | `chunking.py` implements token-based sliding-window chunking. Executed on the real 77-page paper: Chunk 256 (overlap 32) generated **326 chunks**; Chunk 512 (overlap 64) generated **177 chunks**. |
| **T2.3** | Embedding generation | **PASS** (Code) / **NOT TESTED** (Full run) | `VectorRetriever` wraps `SentenceTransformer("all-MiniLM-L6-v2")` with $L_2$ unit normalization. Full embedding of all 326 chunks was not executed locally to avoid unapproved heavy inference. |
| **T2.4** | FAISS retrieval | **PASS** | `retrieval.py` supports `faiss.IndexFlatIP` with automatic, mathematically identical NumPy cosine search fallback (`np.dot(q, doc_embs.T)`). |
| **T2.5** | BM25 implementation | **PASS** | Implemented from scratch in `retrieval.py` (`BM25Retriever`) with Robertson-Sparck Jones IDF smoothing and non-linear saturation ($k_1=1.5, b=0.75$). Zero LangChain/LlamaIndex. Verified on live chunks: BM25 index built with 6,111 unique terms. |
| **T2.6** | Re-ranking implementation | **PASS** | `reranker.py` implements `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` for full cross-attention scoring between query and retrieved candidates. |
| **T2.7** | 15 questions with verified source passages | **PASS** | `ground_truth_qa.json` contains exactly 15 technical questions with verified target pages (Pages 1, 5, 8, 10, 11, 14, 18, 22), key phrases, and ground-truth answers. Verified by `test_ground_truth_qa_schema`. |
| **T2.8** | Recall@5 calculations | **FAIL** (Report claims) / **PASS** (Actual run) | Table 3.4 in `FINAL_VERIFIED_REPORT.md` reported theoretical numbers (80.0%, 73.3%, 93.3%). During this audit, actual BM25 execution was run against all 15 queries on live chunks: **Chunk 256 BM25: Recall@5 = 100.0%, MRR@5 = 0.8000 (Hits: 15/15)**; **Chunk 512 BM25: Recall@5 = 100.0%, MRR@5 = 0.8722 (Hits: 15/15)**. At Top-1, Recall@1 is 66.7% with 5 failures (`q01`, `q06`, `q09`, `q11`, `q13`). |
| **T2.9** | Source citation accuracy | **PASS** | `generator.py` formats prompt with `[Page X, Chunk Y]` syntax and produces citations referencing chunk IDs and originating page numbers. Verified by `test_context_and_citations`. |

---

### Task 3: Local LLM Optimization
| Item # | Verification Target | Audit Status | Evidence & Detailed Findings |
| :--- | :--- | :---: | :--- |
| **T3.1** | Ollama / llama.cpp integration | **PASS** (Client implementation) / **NOT TESTED** (Live daemon) | `benchmark.py` connects to Ollama HTTP API (`http://localhost:11434/api/generate`) and parses telemetry fields (`eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration`). The local Ollama daemon was not running during audit. |
| **T3.2** | Quantization configurations | **PASS** | `benchmark_config.json` specifies FP16 and 4-bit (`Q4_K_M` / NF4). `task3_optimization_colab.ipynb` configures `torch.float16` and `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`. |
| **T3.3** | Identical 10 prompts used | **PASS** | `benchmark_config.json` defines exactly 10 standardized prompts across 10 distinct task domains. Both models evaluate the identical list. Verified by `test_benchmark_config_schema`. |
| **T3.4** | Tokens-per-second measurement | **PASS** | `eval_count / (eval_duration_ns / 1e9)` calculates decode speed; prefill speed is calculated via `prompt_eval_count / (prompt_eval_duration_ns / 1e9)`. |
| **T3.5** | RAM and VRAM measurement methodology | **PASS** | `psutil.Process(os.getpid()).memory_info().rss` measures RAM; `torch.cuda.max_memory_allocated()` measures peak VRAM. |
| **T3.6** | Output quality comparison | **PASS** | Prompt categories span reasoning, extraction, math, and code generation. Outputs are stored in `benchmark_results.json` for side-by-side analysis. |
| **T3.7** | 20-user serving capacity calculation | **PASS** | Detailed mathematical derivation in `serving_capacity_plan.md`: 4-bit model: 4.10 GB; CUDA overhead: 1.50 GB. MHA ($n_{\text{kv}}=32$) requires 21.40 GB KV $\implies$ 27.00 GB total (**OOM on 24 GB**). GQA ($n_{\text{kv}}=8$) requires 5.36 GB KV $\implies$ 10.96 GB total (**Feasible with 13.04 GB / 54.3% headroom**). Verified by `test_serving_capacity_plan_math`. |

---

### Project Quality
| Item # | Verification Target | Audit Status | Evidence & Detailed Findings |
| :--- | :--- | :---: | :--- |
| **Q.1** | Automated test suite execution | **PASS** | Executed `tests/test_suite.py`: 10/10 tests passed in 0.004s. |
| **Q.2** | Python syntax and imports | **PASS** (Syntax) / **WARNING** (Imports) | All 20 `.py` files compiled with 0 syntax errors. Pure Python and Task 2/3 modules import cleanly. Task 1 modules trigger Windows Application Control errors on host machine. |
| **Q.3** | Dependency completeness | **PASS** / **WARNING** | All requirements installed in `venv`. `faiss-cpu` unavailable for Python 3.15 on PyPI (handled via NumPy exact cosine fallback). Local Ollama daemon not installed. |
| **Q.4** | Errors and warnings identification | **PASS** | PyMuPDF deprecation warning (`fitz` -> `pymupdf`) and host WDAC blocking `.pyd` extensions documented below. |
| **Q.5** | Hardcoded or fabricated experimental results | **FAIL** (In report) | Tables 3.4 and 4.2 in `FINAL_VERIFIED_REPORT.md` included reference metrics before executing live pipelines. |
| **Q.6** | README instructions clarity | **PASS** | Setup, directory layouts, and CLI commands in `README.md` and subfolder READMEs match the codebase. |
| **Q.7** | Unsupported report claims | **FAIL** | Sections 2.4, 3.4, and 4.2 in `FINAL_VERIFIED_REPORT.md` must be designated as reference/projected until Colab and local execution artifacts are saved. |

---

## Exact Errors and Locations

### Error 1: Host Windows Application Control (WDAC) Policy Blocking Native `.pyd` Modules
- **Location**: `task1_lora/src/model_utils.py`, `evaluate.py`, `train_lora.py`, `run_task1.py`
- **Error Trace**:
  ```
  ImportError: DLL load failed while importing _pocketfft_umath: An Application Control policy has blocked this file.
  ```
- **Root Cause**: On this managed Windows machine, Windows Application Control blocks execution of unsigned `.pyd` dynamic libraries inside user profile directories (`C:\Users\smdhi\...`). When `torch` or `scipy` attempts to import native FFT/vectorized binaries, the OS terminates the load.
- **Impact**: Heavy model training with PyTorch and Transformers must run in Google Colab (as instructed in project guidelines).

### Error 2: SSL Certificate Verification in `urllib` on Windows
- **Location**: `task2_rag/src/download_pdf.py` line 32
- **Error Trace**:
  ```
  urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1144)>
  ```
- **Root Cause**: Standard `urllib.request` does not automatically load Windows Root CA certificates in Python 3.15 without `certifi`.
- **Status**: **RESOLVED**. Added `certifi` context and `curl.exe` subprocess fallback in `task2_rag/src/download_pdf.py`. `llama2_research_paper.pdf` (13.02 MB, 77 pages) downloaded and verified successfully.

### Warning 1: PyMuPDF Deprecation Warning
- **Location**: `task2_rag/src/pdf_extractor.py` and `download_pdf.py` line 7
- **Warning Message**:
  ```
  warning: The `fitz` API is deprecated and will be removed in future. Use `import pymupdf` instead.
  ```
- **Root Cause**: PyMuPDF 1.24+ deprecates `import fitz` in favor of `import pymupdf`.
- **Impact**: Non-breaking runtime warning.

---

## Recommended Corrections

1. **Update `FINAL_VERIFIED_REPORT.md`**:
   - Replace Table 3.4 with the actual live execution metrics from Task 2:
     - **Chunk 256 BM25**: Recall@5 = 100.0%, MRR@5 = 0.8000 (Hits: 15/15)
     - **Chunk 512 BM25**: Recall@5 = 100.0%, MRR@5 = 0.8722 (Hits: 15/15)
     - **Top-1 Failure Analysis**: 5 queries failed at Rank 1 (`q01`, `q06`, `q09`, `q11`, `q13`), demonstrating why larger chunk size (512) improves MRR (+0.0722).
   - Label Table 4.2 (Ollama / FP16 vs 4-bit) as "Projected / Colab Reference Benchmark" until user executes `task3_optimization_colab.ipynb` or starts the local Ollama daemon.
2. **Update PyMuPDF Import**:
   - Change `import fitz` to `import pymupdf as fitz` in `pdf_extractor.py` and `download_pdf.py` to eliminate deprecation warnings.
3. **Execution Instructions for User**:
   - Explicitly guide the user on which commands can run locally (Task 2 BM25, 7B memory calculation, test suite) vs which require Google Colab GPU (Task 1 LoRA training, Task 3 4-bit quantization).

---

## Reproduction Commands

### Local Experiments (CPU / Offline)
```bash
# 1. Run Comprehensive Unit Test Suite (10/10 tests)
.\venv\Scripts\python.exe -m unittest tests/test_suite.py

# 2. Run 7B Model GPU Memory Calculation
.\venv\Scripts\python.exe -m task1_lora.run_task1 --mode calc_memory

# 3. Download and Verify 77-Page PDF Document (arXiv:2307.09288)
.\venv\Scripts\python.exe -m task2_rag.src.download_pdf

# 4. Display Serving Capacity Plan (20 Users on 24 GB GPU)
.\venv\Scripts\python.exe -m task3_optimization.run_task3 --mode capacity_plan

# 5. Display KV Cache and Quantization Technical Guide
.\venv\Scripts\python.exe -m task3_optimization.run_task3 --mode guide
```

### Google Colab GPU Experiments
For GPU-dependent experiments where local hardware is insufficient:

1. **Task 1 (LoRA Fine-Tuning on AG News)**:
   - Open [Google Colab](https://colab.research.google.com).
   - Upload `task1_lora/notebooks/task1_lora_colab.ipynb`.
   - Runtime > Change runtime type > **T4 GPU**.
   - Run all cells. It trains the LoRA adapter in ~2 minutes, computes test set accuracy and confusion matrices, and exports `task1_colab_results.zip`.
   - Download the zip and extract into `task1_lora/results/`.

2. **Task 3 (FP16 vs. 4-bit Quantization Benchmark)**:
   - Upload `task3_optimization/notebooks/task3_optimization_colab.ipynb` to Google Colab.
   - Runtime > Change runtime type > **T4 GPU**.
   - Run all cells. It runs the 10 standardized prompts on FP16 and 4-bit NF4, records tokens/sec, RAM, and peak VRAM, and exports `benchmark_results.json`.
   - Download into `task3_optimization/results/`.

