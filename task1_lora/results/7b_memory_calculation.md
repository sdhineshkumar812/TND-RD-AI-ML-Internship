# 7B Parameter Model GPU VRAM Requirements Analysis

This breakdown calculates the exact GPU memory required to train a standard 7-Billion parameter LLM 
(e.g., Llama-2-7B, Qwen2.5-7B, Mistral-7B) across three distinct fine-tuning paradigms:
1. **Full Fine-Tuning (FP16/BF16 Mixed Precision)**
2. **LoRA (Low-Rank Adaptation with FP16 Base Model)**
3. **QLoRA (Quantized LoRA with 4-bit NormalFloat NF4 Base Model)**

### Architectural Assumptions & Hyperparameters
- **Base Architecture**: Standard 7B transformer ($L = 32$ layers, hidden dimension $d_{\text{model}} = 4096$, intermediate MLP dimension $d_{\text{ff}} = 11008$; exact Llama 2 7B parameter count is $6.738 \times 10^9$, nominal $N = 7.0 \times 10^9$).
- **LoRA Target Modules**: All 7 linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) across all 32 layers.
- **LoRA Hyperparameters**: Rank $r = 16$, Alpha $\alpha = 32$.
  - For each linear projection $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, LoRA introduces low-rank matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$, adding $P_{\text{LoRA}} = r(d_{\text{in}} + d_{\text{out}})$ parameters.
  - **Attention Projections** (`q, k, v, o`, each $4096 \times 4096$): $4 \times 16 \times (4096 + 4096) = 524,288\text{ params/layer}$.
  - **MLP Projections** (`gate, up, down`, $4096 \leftrightarrow 11008$): $3 \times 16 \times (4096 + 11008) = 724,992\text{ params/layer}$.
  - **Per-Layer Total**: $524,288 + 724,992 = 1,249,280\text{ params/layer}$.
  - **Total Trainable Parameters across 32 layers**:
    $$32 \times 1,249,280 = \mathbf{39,976,960 \approx 39.98\text{M}}$$
    This corresponds to **0.5933%** of exact Llama 2 7B ($6.738\text{B}$) and **0.5711%** of nominal 7.0B across all 7 linear projections.
- **Batch Size ($B$)**: 1, **Sequence Length ($S$)**: 2048 tokens.
- **Activation Checkpointing (Gradient Checkpointing)**: Enabled.
- **Optimizer**: Standard AdamW (FP32 master weights + momentum + variance = 12 bytes/trainable param) or 8-bit Paged AdamW (6 bytes/trainable param).

---

## Summary Comparison Table

All values use consistent binary **GiB** ($1\text{ GiB} = 1024^3\text{ bytes}$), with decimal **GB** ($10^9\text{ bytes}$) in parentheses for clarity.

| Memory Component | Full Fine-Tuning (FP16) | LoRA (FP16 Base + $r=16$) | QLoRA (NF4 4-bit Base + $r=16$) |
| :--- | :---: | :---: | :---: |
| **Base Model Weights** | 13.04 GiB (14.00 GB) | 13.04 GiB (14.00 GB) | 3.59 GiB (3.85 GB) |
| **LoRA Adapter Weights** | — | 0.074 GiB (0.080 GB) | 0.074 GiB (0.080 GB) |
| **Total Model Weights** | 13.04 GiB (14.00 GB) | 13.11 GiB (14.08 GB) | 3.66 GiB (3.93 GB) |
| **Gradients (Trainable)** | 13.04 GiB (14.00 GB) | 0.074 GiB (0.080 GB) | 0.074 GiB (0.080 GB) |
| **Optimizer States (AdamW)** | 78.23 GiB (84.00 GB) | 0.447 GiB (0.480 GB) | 0.447 GiB *(0.223 GiB w/ 8-bit)* |
| **Activations (w/ checkpointing)** | ~4.00 GiB (~4.29 GB) | ~4.00 GiB (~4.29 GB) | ~4.00 GiB (~4.29 GB) |
| **CUDA & Framework Overhead** | ~2.00 GiB (~2.15 GB) | ~1.50 GiB (~1.61 GB) | ~1.50 GiB (~1.61 GB) |
| **Total Estimated VRAM** | **~110.31 GiB (~118.44 GB)** | **~19.13 GiB (~20.54 GB)** | **~9.68 GiB (~10.40 GB)** |
| **Hardware Feasibility** | Multi-GPU required (e.g. $2\times$ A100-80GB or $4\times$ A100-40GB w/ FSDP) | 1x RTX 3090 / 4090 (24GB) or 1x A10G (24GB) | 1x Free Google Colab T4 (16GB) or RTX 3060/4070 (12GB) |

*Note: Activations and CUDA context overhead are estimates based on standard PyTorch / CUDA runtime overhead. Hardware feasibility claims are conditional upon $B=1$, $S=2048$, and activation checkpointing.*

---

## Detailed Mathematical Derivations

### 1. Model Weights Memory
- **Full Fine-Tuning & LoRA Base (FP16)**: Each parameter occupies 16 bits (2 bytes).
  $$\text{Base Weight Memory} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.00 \times 10^9\text{ bytes} \approx 13.04\text{ GiB}$$
- **LoRA Adapter Weights (FP16)**: 39,976,960 trainable parameters in 16-bit precision.
  $$\text{Adapter Weight Memory} = 39,976,960 \times 2 \text{ bytes} = 79,953,920\text{ bytes} \approx 0.0745\text{ GiB (76.25 MiB)}$$
- **QLoRA Base (NF4 4-bit)**: Each parameter occupies 4 bits (0.5 bytes), plus block quantization constants and double-quantization overhead ($\approx 0.05$ bytes/param):
  $$\text{QLoRA Base Memory} \approx 7.0 \times 10^9 \times 0.55 \text{ bytes} = 3.85 \times 10^9\text{ bytes} \approx 3.59\text{ GiB}$$
  Adding FP16 adapter weights yields $3.59 + 0.0745 = \mathbf{3.66\text{ GiB}}$.

### 2. Gradient Memory
- **Full Fine-Tuning**: Backpropagation computes and stores gradients for all $7.0\text{B}$ parameters in FP16 (2 bytes/param).
  $$\text{Gradient Memory} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.00 \times 10^9\text{ bytes} \approx 13.04\text{ GiB}$$
- **LoRA & QLoRA**: Base model weights are strictly frozen. Gradients are only computed and tracked for the $39,976,960$ adapter parameters.
  $$\text{Gradient Memory} = 39,976,960 \times 2 \text{ bytes} = 79,953,920\text{ bytes} \approx \mathbf{0.0745\text{ GiB (76.25 MiB)}}$$

### 3. Optimizer State Memory (AdamW)
- Standard 32-bit mixed-precision AdamW maintains three FP32 tensors per trainable parameter:
  1. FP32 master copy of weights (4 bytes/param)
  2. First momentum term $m_t$ (4 bytes/param)
  3. Second momentum term $v_t$ (4 bytes/param)
  $$\text{AdamW Memory Footprint} = 4 + 4 + 4 = 12\text{ bytes per trainable parameter}$$
- **Full Fine-Tuning**: $7.0 \times 10^9 \times 12 \text{ bytes} = 84.00 \times 10^9\text{ bytes} \approx \mathbf{78.23\text{ GiB}}$ solely for optimizer state!
- **LoRA & QLoRA**: $39,976,960 \times 12 \text{ bytes} = 479,723,520\text{ bytes} \approx \mathbf{0.4468\text{ GiB (457.50 MiB)}}$.
- **8-bit Paged AdamW (QLoRA option)**: Using `bitsandbytes` 8-bit optimizer reduces state to 6 bytes/param:
  $$39,976,960 \times 6 \text{ bytes} = 239,861,760\text{ bytes} \approx \mathbf{0.2234\text{ GiB (228.75 MiB)}}$$

### 4. Activation Memory
- Without gradient checkpointing, intermediate activations must be retained across all 32 transformer layers for the backward pass ($\approx 25\text{--}35\text{ GiB}$).
- With **Activation (Gradient) Checkpointing**, activations are discarded during the forward pass and recomputed on demand during backward pass. Memory scales with layer activations and attention matrices for one layer:
  $$\text{Memory}_{\text{act}} \approx 3.5\text{--}4.0\text{ GiB for } B=1, S=2048, d_{\text{model}}=4096$$

### 5. Conclusion & Practical Takeaway
- **Full fine-tuning (~110.3 GiB)** is completely infeasible on consumer or single-node GPUs, mandating multi-GPU clusters ($2\times$ A100-80GB or $4\times$ A100-40GB) with FSDP or DeepSpeed ZeRO-3.
- **LoRA (~19.1 GiB)** allows full instruction tuning on a single consumer GPU with 24 GiB VRAM (RTX 3090 / 4090 or A10G).
- **QLoRA (~9.7 GiB)** allows high-quality 7B model training directly on a **free Google Colab T4 (16GB)** or budget 12 GiB GPU (RTX 3060 / 4070).
