# Transparent Serving Capacity Plan: 20 Concurrent Users on a 24 GB GPU

## 1. Executive Summary & Objective
This document outlines a mathematically rigorous, verified serving-capacity plan for deploying a **7-Billion parameter Large Language Model** to serve **20 concurrent users** on a **single 24 GB GPU** (e.g., NVIDIA RTX 3090, RTX 4090, or A10G 24GB).

---

## 2. Workload & Hardware Parameters

### Hardware Profile
- **GPU**: 1x NVIDIA RTX 3090 / 4090 / A10G
- **Total Physical VRAM**: $24.0\text{ GB}$ ($24,576\text{ MB}$)
- **Target Concurrent Streams ($B$)**: $20$ active simultaneous users
- **Serving Framework**: vLLM / llama.cpp / TGI with PagedAttention and Continuous Batching

### Workload Profile
- **Target Model Architecture**: 7B Parameter LLM (Layers $L=32$, Hidden Dim $h=4096$, Head Dim $d_{\text{head}}=128$, Attention Heads $n_q=32$)
- **Average Prompt Length ($S_{\text{prompt}}$)**: $512$ tokens
- **Average Completion Length ($S_{\text{gen}}$)**: $256$ tokens
- **Peak SLA Context Window ($S_{\text{peak}}$)**: $2048$ tokens per user

---

## 3. GPU VRAM Budgeting

### Component 1: Static Model Weights Footprint
- **FP16 (Unquantized)**:
  $$\text{Memory}_{\text{FP16}} = 7.0 \times 10^9 \text{ params} \times 2 \text{ bytes} \approx 14.0\text{ GB}$$
- **4-bit Quantized (GGUF `Q4_K_M` or AWQ INT4)**:
  $$\text{Memory}_{\text{4-bit}} = 7.0 \times 10^9 \text{ params} \times 0.55 \text{ bytes} \approx \mathbf{4.10\text{ GB}}$$

### Component 2: CUDA Context & Workspace Overhead
- PyTorch CUDA context, memory allocator reserves, and execution scratchpad buffers: $\approx \mathbf{1.50\text{ GB}}$.

### Component 3: Dynamic KV Cache Memory Breakdown

$$\text{KV Memory} = 2 \times B \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S \times \text{bytes\_per\_precision}$$

#### Scenario A: Legacy Multi-Head Attention (MHA, $n_{\text{kv}} = 32$)
- Memory per user at $S=2048$:
  $$2 \times 1 \times 32 \times 32 \times 128 \times 2048 \times 2\text{ bytes} = 1.07\text{ GB per user}$$
- For 20 concurrent users:
  $$20 \times 1.07\text{ GB} = \mathbf{21.4\text{ GB}}$$
- **Total VRAM required**:
  $$4.10\text{ GB (Weights)} + 1.50\text{ GB (Overhead)} + 21.40\text{ GB (KV)} = \mathbf{27.00\text{ GB}} > 24\text{ GB}$$
- **Outcome**: **FAILED (Out of Memory - OOM)**. MHA cannot serve 20 concurrent users at 2k context on a 24GB GPU.

#### Scenario B: Grouped-Query Attention (GQA, $n_{\text{kv}} = 8$, e.g. Llama 2 / Qwen2.5 / Mistral)
- Memory per user at $S=2048$:
  $$2 \times 1 \times 32 \times 8 \times 128 \times 2048 \times 2\text{ bytes} = 0.268\text{ GB (268 MB) per user}$$
- For 20 concurrent users:
  $$20 \times 0.268\text{ GB} = \mathbf{5.36\text{ GB}}$$
- **Total VRAM required**:
  $$4.10\text{ GB (Weights)} + 1.50\text{ GB (Overhead)} + 5.36\text{ GB (KV)} = \mathbf{10.96\text{ GB}}$$
- **Headroom Remaining**:
  $$24.00\text{ GB} - 10.96\text{ GB} = \mathbf{13.04\text{ GB (54.3% Headroom)}}$$
- **Outcome**: **SUCCESS**. Comfortably fits with substantial safety buffer!

---

## 4. Comprehensive VRAM Budget Allocation Table

| Memory Component | FP16 Base + MHA | 4-bit Base + MHA | 4-bit Base + GQA (Recommended) | 4-bit Base + GQA + FP8 KV |
| :--- | :---: | :---: | :---: | :---: |
| **Model Weights** | $14.00\text{ GB}$ | $4.10\text{ GB}$ | $4.10\text{ GB}$ | $4.10\text{ GB}$ |
| **CUDA / Engine Overhead** | $1.50\text{ GB}$ | $1.50\text{ GB}$ | $1.50\text{ GB}$ | $1.50\text{ GB}$ |
| **KV Cache (20 users @ 2k)** | $21.40\text{ GB}$ | $21.40\text{ GB}$ | $5.36\text{ GB}$ | $2.68\text{ GB}$ |
| **Total VRAM Consumed** | **$36.90\text{ GB}$** | **$27.00\text{ GB}$** | **$10.96\text{ GB}$** | **$8.28\text{ GB}$** |
| **Status on 24 GB GPU** | **OOM** (Needs A100) | **OOM** (Exceeds 24GB) | **FEASIBLE** ($13.0\text{ GB}$ headroom) | **FEASIBLE** ($15.7\text{ GB}$ headroom) |

---

## 5. Serving Architecture & Throughput Optimization

To guarantee high throughput and sub-second latency across all 20 streams:

1. **Continuous Batching (Iteration-Level Scheduling)**:
   - Avoids waiting for the slowest generation in a batch to complete.
   - Newly arriving prompts join the active forward pass immediately, and finished sequences release their KV blocks instantly.
2. **PagedAttention**:
   - Eliminates external and internal fragmentation by paging KV cache in 16-token physical blocks.
   - Dynamic block allocation ensures zero memory wasted on ungenerated tokens.
3. **Latency & Throughput Expectations**:
   - **Time To First Token (TTFT)**: $\approx 150\text{--}250\text{ ms}$ (prefill for 512 tokens).
   - **Per-Token Generation Latency**: $\approx 25\text{--}35\text{ ms}$ per token across the batch.
   - **Aggregate System Throughput**: $\approx 180\text{--}240\text{ tokens/second}$ across all 20 streams.
   - **Headroom Exploitation**: The $13.0\text{ GB}$ remaining headroom allows scaling to **up to 45 concurrent users** or handling expanded context lengths up to **8192 tokens**.
