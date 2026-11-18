# 7B Parameter Model GPU VRAM Requirements Analysis

This breakdown calculates the exact GPU memory required to train a standard 7-Billion parameter LLM 
(e.g., Llama-2-7B, Qwen2.5-7B, Mistral-7B) across three distinct fine-tuning paradigms:
1. **Full Fine-Tuning (FP16/BF16 Mixed Precision)**
2. **LoRA (Low-Rank Adaptation with FP16 Base Model)**
3. **QLoRA (Quantized LoRA with 4-bit NormalFloat NF4 Base Model)**

### Assumptions & Hyperparameters
- **Model Parameter Count ($N$)**: 7.0 Billion ($7.0 \times 10^9$)
- **LoRA Target Modules**: All attention and MLP projections (`q, k, v, o, gate, up, down`)
- **LoRA Rank ($r$)**: 16, **Alpha**: 32 (Trainable parameters: $\approx 19.6\text{M}$, or $0.28\%$ of total)
- **Batch Size**: 1, **Sequence Length**: 2048 tokens
- **Activation Checkpointing (Gradient Checkpointing)**: Enabled
- **Optimizer**: Standard AdamW (stores FP32 copy + momentum + variance = 12 bytes/trainable param)

---

## Summary Comparison Table

| Memory Component | Full Fine-Tuning (FP16) | LoRA (FP16 Base) | QLoRA (NF4 4-bit Base) |
| :--- | :---: | :---: | :---: |
| **Model Weights (GB)** | 13.04 | 13.08 | 3.62 |
| **Gradients (GB)** | 13.04 | 0.037 | 0.037 |
| **Optimizer States (AdamW) (GB)** | 78.23 | 0.219 | 0.219 |
| **Activations (w/ checkpointing) (GB)** | 4.0 | 4.0 | 4.0 |
| **CUDA & Memory Overhead (GB)** | 2.0 | 1.5 | 1.5 |
| **Total Estimated VRAM (GB)** | 110.31 | 18.83 | 9.38 |
| **Feasible Hardware** | 2x A100 (80GB) or 4x A100 (40GB) with FSDP/ZeRO-3 | 1x RTX 3090 / 4090 (24GB) or 1x A10G (24GB) | 1x Free Google Colab T4 (16GB) or RTX 3060/4070 (12GB) |

---

## Detailed Mathematical Derivation

### 1. Model Weights Memory
- **Full Fine-Tuning & LoRA (FP16)**: Each parameter occupies 16 bits (2 bytes).
  $$\text{Weight Memory} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.00\text{ GB}$$
- **QLoRA (NF4 4-bit)**: Each parameter occupies 4 bits (0.5 bytes), plus small quantization metadata (double quantization adds $\approx 0.05$ bytes/param).
  $$\text{Weight Memory} = 7.0 \times 10^9 \times 0.55 \text{ bytes} = 3.59\text{ GB}$$

### 2. Gradient Memory
- **Full Fine-Tuning**: Backpropagation requires storing a gradient for every single trainable parameter in FP16 (2 bytes/param).
  $$\text{Gradient Memory} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.00\text{ GB}$$
- **LoRA & QLoRA**: Base model weights are strictly frozen. Gradients are only computed and stored for the $\approx 19.6\text{M}$ adapter parameters.
  $$\text{Gradient Memory} = 19.6 \times 10^6 \times 2 \text{ bytes} \approx 0.037\text{ GB (37 MB)}$$

### 3. Optimizer State Memory (AdamW)
- Standard AdamW maintains:
  1. Master weights in FP32 (4 bytes/param)
  2. First moment vector $m_t$ in FP32 (4 bytes/param)
  3. Second moment vector $v_t$ in FP32 (4 bytes/param)
  $$\text{Total AdamW Memory} = 4 + 4 + 4 = 12\text{ bytes per trainable parameter}$$
- **Full Fine-Tuning**: $7.0 \times 10^9 \times 12 \text{ bytes} = \mathbf{84.00\text{ GB}}$ alone just for optimizer state!
- **LoRA & QLoRA**: $19.6 \times 10^6 \times 12 \text{ bytes} = \mathbf{0.22\text{ GB (220 MB)}}$.

### 4. Activation Memory
- Without gradient checkpointing, intermediate activations must be retained across all 32 transformer layers for backward pass: $\approx 25\text{--}35\text{ GB}$.
- With **Activation (Gradient) Checkpointing**, intermediate activations are discarded during forward pass and recomputed on demand during backward pass. Memory scales only with 1 layer: $\approx 3.5\text{--}4.0\text{ GB}$ for $s=2048, b=1$.

### 5. Conclusion & Practical Takeaway
- **Full fine-tuning (118 GB)** is prohibitively expensive for consumer or single-GPU setups, mandating multi-GPU clusters (e.g. 2x A100-80GB or 4x A100-40GB) with FSDP or DeepSpeed ZeRO-3.
- **LoRA (19.8 GB)** allows fine-tuning on a single high-end consumer GPU (RTX 3090 / 4090 24GB).
- **QLoRA (9.3 GB)** democratizes 7B model training, allowing high-quality instruction fine-tuning directly on a **free Google Colab T4 (16GB)** or a budget RTX 3060 (12GB) GPU.
              precision    recall  f1-score   support

       World       1.00      0.77      0.87        22
      Sports       0.93      1.00      0.96        27
    Business       0.81      0.81      0.81        32
    Sci/Tech       0.73      0.84      0.78        19

    accuracy                           0.86       100
   macro avg       0.87      0.86      0.86       100
weighted avg       0.87      0.86      0.86       100
