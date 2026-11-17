# Comprehensive Engineering Guide: KV Cache Architecture and Model Quantization

## Part 1: The KV Cache in Autoregressive Large Language Models

### 1. The Core Bottleneck: Compute-Bound vs. Memory-Bandwidth Bound
During the **prefill stage** (processing the input prompt), all prompt tokens are processed simultaneously in parallel via standard matrix multiplications (GEMM). This phase achieves high arithmetic intensity (FLOPs per byte transferred) and is **compute-bound**.

During the **decoding / generation stage**, tokens are generated strictly sequentially: each new token $x_{t}$ depends on all previous tokens $x_{1}, \dots, x_{t-1}$. At each forward pass, the model generates only a single token vector ($b=1$). Matrix-vector multiplications (GEMV) have very low arithmetic intensity: the GPU must transfer billions of model weight bytes from high-bandwidth memory (HBM/VRAM) into on-chip cache (SRAM) just to compute a single output token. Consequently, token generation is strictly **memory-bandwidth bound**.

---

### 2. Why the KV Cache is Essential
In multi-head self-attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Without caching:
- To generate token $t$, the model must re-project all tokens $x_1, \dots, x_{t-1}$ to compute their Key and Value vectors.
- Total operations scale quadratically: $O(N^2)$ in context length $N$.

With the **Key-Value (KV) Cache**:
- The model caches the Key ($K$) and Value ($V$) tensors of all past tokens in GPU VRAM.
- For each new token $x_t$, the model computes only its single Query $q_t$, single key $k_t$, and single value $v_t$.
- $k_t$ and $v_t$ are appended to the KV cache.
- The single query vector $q_t$ attends over the full cached keys and values.
- Compute scales linearly: $O(N)$ per sequence, drastically speeding up generation.

---

### 3. KV Cache Memory Footprint Formula
The exact memory consumed by the KV cache across a batch of concurrent requests is given by:

$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S \times \text{bytes\_per\_precision}$$

Where:
- $2$: Accounts for both Key and Value tensors.
- $B$: Batch size (number of concurrent sequences).
- $L$: Total number of transformer layers.
- $n_{\text{kv\_heads}}$: Number of key-value attention heads per layer.
- $d_{\text{head}}$: Hidden dimension per head ($d_{\text{model}} / n_{\text{q\_heads}}$).
- $S$: Sequence length (prompt tokens + generated tokens).
- $\text{bytes\_per\_precision}$: Precision bytes ($2.0$ for FP16/BF16, $1.0$ for FP8, $0.5$ for INT4).

#### Example: 7B Parameter Model (FP16, $L=32, d_{\text{head}}=128, S=2048, B=1$)
- **Standard Multi-Head Attention (MHA)** ($n_{\text{kv}} = 32$):
  $$\text{KV Cache} = 2 \times 1 \times 32 \times 32 \times 128 \times 2048 \times 2 \text{ bytes} = 1,073,741,824\text{ bytes} = \mathbf{1.00\text{ GB per user}}$$
- **Grouped-Query Attention (GQA)** ($n_{\text{kv}} = 8$, e.g. Llama 2 / Qwen):
  $$\text{KV Cache} = 2 \times 1 \times 32 \times 8 \times 128 \times 2048 \times 2 \text{ bytes} = 268,435,456\text{ bytes} = \mathbf{0.25\text{ GB (256 MB) per user}}$$

---

### 4. Attention Architectures: MHA vs. MQA vs. GQA

| Feature | Multi-Head Attention (MHA) | Multi-Query Attention (MQA) | Grouped-Query Attention (GQA) |
| :--- | :---: | :---: | :---: |
| **Query Heads ($n_q$)** | $32$ | $32$ | $32$ |
| **KV Heads ($n_{kv}$)** | $32$ | $1$ | $8$ |
| **KV Memory Ratio** | $1.00\times$ (Baseline) | $0.031\times$ (32x reduction) | $0.25\times$ (4x reduction) |
| **Quality Degradation** | None (Reference) | Notable on complex tasks | Statistically indistinguishable from MHA |
| **Adopted By** | Llama 1, GPT-3 | Falcon-7B | Llama 2 (34B/70B), Llama 3, Qwen2.5 |

---

### 5. PagedAttention and Memory Fragmentation
Traditional LLM inference engines allocate contiguous GPU memory buffers sized for the *maximum possible sequence length* ($S_{\text{max}}$). This causes two massive memory inefficiencies:
1. **Internal Fragmentation**: If a user generates 200 tokens but the buffer reserved 2048 tokens, 90% of reserved VRAM is wasted.
2. **External Fragmentation**: Varying request lengths create non-contiguous memory gaps that prevent new requests from being scheduled.

**PagedAttention** (introduced in vLLM) solves this by adapting the OS virtual memory paging concept:
- Physical GPU memory is divided into fixed-size contiguous blocks (e.g. 16 or 32 tokens per block).
- The KV cache for a sequence is stored in non-contiguous physical blocks mapped via a logical page table.
- Blocks are allocated dynamically on-demand as tokens are generated.
- Memory waste is reduced to $<4\%$, enabling **2x to 4x higher concurrency** on the same GPU.

---

## Part 2: Quantization Fundamentals

### 1. Definition & Mathematical Formulation
Quantization compresses high-precision continuous floating-point weights ($W \in \mathbb{R}$) into discrete low-bit integer or sub-byte representations ($q \in \mathbb{Z}$).

#### Affine (Asymmetric) Quantization
Maps real values $[\alpha, \beta]$ to an integer range $[q_{\text{min}}, q_{\text{max}}]$ (e.g., $[0, 255]$ for UINT8 or $[0, 15]$ for UINT4):
$$q = \text{clamp}\left( \left\lfloor \frac{w}{S} \right\rceil + Z, q_{\text{min}}, q_{\text{max}} \right)$$

$$\hat{w} = S \cdot (q - Z)$$

Where:
- $S = \frac{\beta - \alpha}{q_{\text{max}} - q_{\text{min}}}$ is the **scale factor** (step size).
- $Z = \text{round}\left( \frac{-\alpha}{S} \right) + q_{\text{min}}$ is the integer **zero-point offset**.

#### Symmetric Quantization
Centers the quantization range around zero, enforcing $Z = 0$:
$$S = \frac{\max(|w|)}{q_{\text{max}}}, \quad q = \text{clamp}\left( \left\lfloor \frac{w}{S} \right\rceil, -q_{\text{max}}, q_{\text{max}} \right)$$
$$\hat{w} = S \cdot q$$
Symmetric quantization is computationally faster because zero-point arithmetic cancels out during matrix multiplication.

---

### 2. Quantization Granularities

1. **Per-Tensor**: A single scale factor $S$ for the entire weight matrix. Highly vulnerable to outlier activations, resulting in significant perplexity degradation.
2. **Per-Channel (Per-Column/Row)**: Each row/column of the weight matrix has its own independent scale factor. Standard for INT8 weight quantization.
3. **Block-wise / Group-wise (e.g. Group Size 32 or 128)**: Divides each row into small blocks of $G=32$ or $128$ weights, each with an independent scale and zero point. Outliers only affect their specific group, dramatically preserving accuracy at 4-bit (used in AWQ, GPTQ, and GGUF).

---

### 3. Weight-Only vs. Weight-Activation Quantization

- **Weight-Only (W4A16 / W8A16)**:
  - Model weights are stored in 4-bit or 8-bit in VRAM.
  - During forward pass, weights are unpacked/dequantized to FP16 in fast on-chip SRAM before GEMM computation.
  - **Advantage**: Reduces memory footprint by up to 75% and memory bandwidth bottleneck without requiring special integer Tensor Cores.
  - **Tools**: GGUF/llama.cpp (`Q4_K_M`), AWQ, bitsandbytes.
- **Weight-Activation (W8A8 / W4A4)**:
  - Both weights and intermediate activations are quantized.
  - Enables integer INT8/INT4 Tensor Core execution.
  - **Advantage**: Increases compute throughput (FLOPs/sec) for prefill / high-batch scenarios.
  - **Challenge**: Activation quantization requires calibration to handle activation outliers without catastrophic degradation.

---

### 4. NormalFloat4 (NF4) in QLoRA
Neural network weights typically follow a normal distribution $\mathcal{N}(0, \sigma^2)$ rather than a uniform distribution. Standard uniform quantization assigns equal interval lengths across the whole range, wasting bins on low-probability tails.

**NormalFloat4 (NF4)** is an information-theoretically optimal quantile quantization:
- Divides the standard normal distribution into 16 discrete intervals with **equal probability mass** ($1/16 = 6.25\%$ each).
- Preserves significantly higher accuracy than standard INT4 at identical bitwidth.
