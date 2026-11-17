# Task 3: Local LLM Optimization & Serving Benchmarking

## Overview
This module benchmarks and optimizes local Large Language Model serving across two distinct quantization regimes:
- **Setting 1 (Baseline)**: **FP16 (Unquantized / 16-bit Floating Point)**
- **Setting 2 (Optimized)**: **4-bit Quantization (`Q4_K_M` / NormalFloat4 NF4)**

It provides comprehensive profiling across:
1. **Decode Throughput**: Tokens generated per second.
2. **Latency & TTFT**: Time to First Token and total end-to-end latency.
3. **Hardware Utilization**: System RAM (MB) and GPU VRAM (MB) consumption.
4. **Architectural Foundations**: Rigorous mathematical breakdowns of **KV Caching** and **Quantization**.
5. **Production Sizing**: A fully transparent capacity plan for serving **20 concurrent users** on a **single 24 GB GPU**.

---

## Directory Structure
```
task3_optimization/
├── configs/
│   └── benchmark_config.json          # 10 standardized prompts and generation settings
├── src/
│   ├── __init__.py
│   ├── benchmark.py                   # Latency, throughput, RAM, and VRAM benchmarking engine
│   ├── kv_cache_and_quantization_guide.md # Graduate-level reference on KV cache & quantization
│   └── serving_capacity_plan.md       # 20-user concurrent serving plan on 24 GB GPU
├── notebooks/
│   └── task3_optimization_colab.ipynb # Free T4 GPU Colab benchmark (FP16 vs 4-bit NF4)
├── results/
│   ├── benchmark_results.json         # Raw benchmark data, token counts, and generated texts
│   └── serving_capacity_summary.md    # Executive summary table
├── run_task3.py                       # CLI orchestrator
└── README.md
```

---

## The 10 Standardized Evaluation Prompts
The benchmark evaluates 10 standardized prompts designed to probe distinct LLM capabilities:
1. **Summarization (`p01`)**: Executive summary of multimodal AI research.
2. **Code Generation (`p02`)**: Type-hinted binary search implementation.
3. **Mathematical Reasoning (`p03`)**: Multi-leg velocity and distance computation.
4. **Information Extraction (`p04`)**: Entity and M&A deal value extraction.
5. **Multi-step Logic (`p05`)**: Transitive ordering puzzle.
6. **Instruction Adherence (`p06`)**: Exactly 3 bullet points explaining Unix permissions (`chmod 755`).
7. **Technical Explanation (`p07`)**: Key, Query, Value intuition analogy.
8. **Creative Writing (`p08`)**: Edge model compilation haiku.
9. **Few-Shot Classification (`p09`)**: Sentiment classification in few-shot format.
10. **Factual Retrieval (`p10`)**: Capital and geography query.

---

## Serving Capacity Plan: 20 Concurrent Users on a 24 GB GPU

### The Capacity Planning Math
- **Target Hardware**: Single 24 GB GPU (NVIDIA RTX 3090, RTX 4090, or A10G).
- **Target Model**: 7B Parameter LLM (e.g. Llama-2-7B, Qwen2.5-7B, Mistral-7B).
- **Static 4-bit Model Footprint**: $7.0 \times 10^9 \times 0.55\text{ bytes} \approx \mathbf{4.10\text{ GB}}$.
- **CUDA Overhead & Buffers**: $\approx \mathbf{1.50\text{ GB}}$.
- **Usable Headroom for KV Cache**: $24.00 - 4.10 - 1.50 = \mathbf{18.40\text{ GB}}$.

### Multi-Head Attention (MHA) vs Grouped-Query Attention (GQA) for 20 Users @ 2048 Context:

$$\text{KV Memory} = 2 \times B \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S \times \text{bytes\_per\_precision}$$

- **MHA ($n_{\text{kv}} = 32$)**:
  - KV Memory per user = $1.07\text{ GB}$.
  - For 20 users = $21.40\text{ GB}$.
  - Total VRAM: $4.10 + 1.50 + 21.40 = \mathbf{27.00\text{ GB}} > 24\text{ GB} \implies$ **OUT OF MEMORY (OOM)**.
- **GQA ($n_{\text{kv}} = 8$, Recommended)**:
  - KV Memory per user = $0.268\text{ GB (268 MB)}$.
  - For 20 users = $5.36\text{ GB}$.
  - Total VRAM: $4.10 + 1.50 + 5.36 = \mathbf{10.96\text{ GB}} \implies$ **FEASIBLE with 13.04 GB (54.3%) Headroom!**

---

## Reproduction Commands

### 1. Inspect Serving Capacity Plan
```bash
python -m task3_optimization.run_task3 --mode capacity_plan
```

### 2. Inspect KV Cache & Quantization Technical Guide
```bash
python -m task3_optimization.run_task3 --mode guide
```

### 3. Run Benchmark Against Local Ollama Instance
Start your Ollama daemon with `ollama run qwen2.5:0.5b`, then execute:
```bash
python -m task3_optimization.run_task3 --mode benchmark_ollama --model_tag qwen2.5:0.5b
```

### 4. Run GPU Benchmark in Google Colab
1. Upload `task3_optimization/notebooks/task3_optimization_colab.ipynb` to Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run all cells to benchmark FP16 vs 4-bit NF4 with GPU acceleration.
4. Download the generated `benchmark_results.json` and save to `task3_optimization/results/`.
