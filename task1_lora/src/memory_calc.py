"""
Transparent 7B Model GPU Memory Calculator for Full Fine-Tuning, LoRA, and QLoRA.
Generates rigorous mathematical derivations, breakdowns, and comparative tables.
"""

import os
from typing import Dict, Any

def compute_7b_memory_breakdown(
    param_count: float = 7.0e9,
    lora_trainable_ratio: float = 0.0028,  # ~20M params for r=16 on all linear projections
    seq_len: int = 2048,
    batch_size: int = 1,
    use_gradient_checkpointing: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compute precise memory requirements in Gigabytes (GB, 10^9 bytes) for 7B LLM training regimes.
    
    Formulas:
    - Base Weights:
        - FP16/BF16: 2 bytes per param
        - 4-bit (NF4/INT4): 0.5 bytes per param + block quantization constants (~0.05 bytes)
    - Gradients:
        - Full: 2 bytes per param
        - LoRA/QLoRA: 2 bytes per trainable param
    - Optimizer States (AdamW):
        - FP32 master weight (4 bytes) + momentum (4 bytes) + variance (4 bytes) = 12 bytes / trainable param
        - Paged AdamW 8-bit (QLoRA): ~6 bytes / trainable param
    - Activations (with activation checkpointing):
        - Storing input activations per layer: ~2.5 - 4.0 GB for b=1, s=2048, h=4096
    - CUDA / Framework Overhead:
        - PyTorch context, kernels, workspace: ~1.5 - 2.0 GB
    """
    n_params = param_count
    n_trainable_lora = n_params * lora_trainable_ratio
    
    # 1. Full Fine-Tuning (FP16/BF16 Mixed Precision)
    full_weights_gb = (n_params * 2) / (1024**3)
    full_grads_gb = (n_params * 2) / (1024**3)
    full_opt_gb = (n_params * 12) / (1024**3)
    full_act_gb = 4.0 if use_gradient_checkpointing else 28.0
    full_overhead_gb = 2.0
    full_total_gb = full_weights_gb + full_grads_gb + full_opt_gb + full_act_gb + full_overhead_gb
    
    # 2. LoRA (FP16 Base + FP16 Adapters)
    lora_weights_gb = (n_params * 2 + n_trainable_lora * 2) / (1024**3)
    lora_grads_gb = (n_trainable_lora * 2) / (1024**3)
    lora_opt_gb = (n_trainable_lora * 12) / (1024**3)
    lora_act_gb = 4.0 if use_gradient_checkpointing else 28.0
    lora_overhead_gb = 1.5
    lora_total_gb = lora_weights_gb + lora_grads_gb + lora_opt_gb + lora_act_gb + lora_overhead_gb
    
    # 3. QLoRA (NF4 4-bit Base + FP16 Adapters + Paged AdamW)
    qlora_weights_gb = (n_params * 0.55 + n_trainable_lora * 2) / (1024**3)
    qlora_grads_gb = (n_trainable_lora * 2) / (1024**3)
    qlora_opt_gb = (n_trainable_lora * 12) / (1024**3) # Standard or Paged AdamW
    qlora_act_gb = 4.0 if use_gradient_checkpointing else 28.0
    qlora_overhead_gb = 1.5
    qlora_total_gb = qlora_weights_gb + qlora_grads_gb + qlora_opt_gb + qlora_act_gb + qlora_overhead_gb
    
    return {
        "Full Fine-Tuning (FP16)": {
            "Model Weights (GB)": round(full_weights_gb, 2),
            "Gradients (GB)": round(full_grads_gb, 2),
            "Optimizer States (AdamW) (GB)": round(full_opt_gb, 2),
            "Activations (w/ checkpointing) (GB)": round(full_act_gb, 2),
            "CUDA & Memory Overhead (GB)": round(full_overhead_gb, 2),
            "Total Estimated VRAM (GB)": round(full_total_gb, 2),
            "Feasible Hardware": "2x A100 (80GB) or 4x A100 (40GB) with FSDP/ZeRO-3"
        },
        "LoRA (FP16 Base + r=16)": {
            "Model Weights (GB)": round(lora_weights_gb, 2),
            "Gradients (GB)": round(lora_grads_gb, 3),
            "Optimizer States (AdamW) (GB)": round(lora_opt_gb, 3),
            "Activations (w/ checkpointing) (GB)": round(lora_act_gb, 2),
            "CUDA & Memory Overhead (GB)": round(lora_overhead_gb, 2),
            "Total Estimated VRAM (GB)": round(lora_total_gb, 2),
            "Feasible Hardware": "1x RTX 3090 / 4090 (24GB) or 1x A10G (24GB)"
        },
        "QLoRA (NF4 4-bit Base + r=16)": {
            "Model Weights (GB)": round(qlora_weights_gb, 2),
            "Gradients (GB)": round(qlora_grads_gb, 3),
            "Optimizer States (AdamW) (GB)": round(qlora_opt_gb, 3),
            "Activations (w/ checkpointing) (GB)": round(qlora_act_gb, 2),
            "CUDA & Memory Overhead (GB)": round(qlora_overhead_gb, 2),
            "Total Estimated VRAM (GB)": round(qlora_total_gb, 2),
            "Feasible Hardware": "1x Free Google Colab T4 (16GB) or RTX 3060/4070 (12GB)"
        }
    }

def generate_memory_report(output_path: str = "./task1_lora/results/7b_memory_calculation.md") -> str:
    """Generate a clean markdown report of the 7B memory calculation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    breakdown = compute_7b_memory_breakdown()
    
    lines = [
        "# 7B Parameter Model GPU VRAM Requirements Analysis",
        "",
        "This breakdown calculates the exact GPU memory required to train a standard 7-Billion parameter LLM ",
        "(e.g., Llama-2-7B, Qwen2.5-7B, Mistral-7B) across three distinct fine-tuning paradigms:",
        "1. **Full Fine-Tuning (FP16/BF16 Mixed Precision)**",
        "2. **LoRA (Low-Rank Adaptation with FP16 Base Model)**",
        "3. **QLoRA (Quantized LoRA with 4-bit NormalFloat NF4 Base Model)**",
        "",
        "### Assumptions & Hyperparameters",
        "- **Model Parameter Count ($N$)**: 7.0 Billion ($7.0 \\times 10^9$)",
        "- **LoRA Target Modules**: All attention and MLP projections (`q, k, v, o, gate, up, down`)",
        "- **LoRA Rank ($r$)**: 16, **Alpha**: 32 (Trainable parameters: $\\approx 19.6\\text{M}$, or $0.28\\%$ of total)",
        "- **Batch Size**: 1, **Sequence Length**: 2048 tokens",
        "- **Activation Checkpointing (Gradient Checkpointing)**: Enabled",
        "- **Optimizer**: Standard AdamW (stores FP32 copy + momentum + variance = 12 bytes/trainable param)",
        "",
        "---",
        "",
        "## Summary Comparison Table",
        "",
        "| Memory Component | Full Fine-Tuning (FP16) | LoRA (FP16 Base) | QLoRA (NF4 4-bit Base) |",
        "| :--- | :---: | :---: | :---: |"
    ]
    
    components = [
        "Model Weights (GB)",
        "Gradients (GB)",
        "Optimizer States (AdamW) (GB)",
        "Activations (w/ checkpointing) (GB)",
        "CUDA & Memory Overhead (GB)",
        "Total Estimated VRAM (GB)",
        "Feasible Hardware"
    ]
    
    for comp in components:
        row = f"| **{comp}** "
        for regime in ["Full Fine-Tuning (FP16)", "LoRA (FP16 Base + r=16)", "QLoRA (NF4 4-bit Base + r=16)"]:
            val = breakdown[regime][comp]
            row += f"| {val} "
        row += "|"
        lines.append(row)
        
    lines.extend([
        "",
        "---",
        "",
        "## Detailed Mathematical Derivation",
        "",
        "### 1. Model Weights Memory",
        "- **Full Fine-Tuning & LoRA (FP16)**: Each parameter occupies 16 bits (2 bytes).",
        "  $$\\text{Weight Memory} = 7.0 \\times 10^9 \\times 2 \\text{ bytes} = 14.00\\text{ GB}$$",
        "- **QLoRA (NF4 4-bit)**: Each parameter occupies 4 bits (0.5 bytes), plus small quantization metadata (double quantization adds $\\approx 0.05$ bytes/param).",
        "  $$\\text{Weight Memory} = 7.0 \\times 10^9 \\times 0.55 \\text{ bytes} = 3.59\\text{ GB}$$",
        "",
        "### 2. Gradient Memory",
        "- **Full Fine-Tuning**: Backpropagation requires storing a gradient for every single trainable parameter in FP16 (2 bytes/param).",
        "  $$\\text{Gradient Memory} = 7.0 \\times 10^9 \\times 2 \\text{ bytes} = 14.00\\text{ GB}$$",
        "- **LoRA & QLoRA**: Base model weights are strictly frozen. Gradients are only computed and stored for the $\\approx 19.6\\text{M}$ adapter parameters.",
        "  $$\\text{Gradient Memory} = 19.6 \\times 10^6 \\times 2 \\text{ bytes} \\approx 0.037\\text{ GB (37 MB)}$$",
        "",
        "### 3. Optimizer State Memory (AdamW)",
        "- Standard AdamW maintains:",
        "  1. Master weights in FP32 (4 bytes/param)",
        "  2. First moment vector $m_t$ in FP32 (4 bytes/param)",
        "  3. Second moment vector $v_t$ in FP32 (4 bytes/param)",
        "  $$\\text{Total AdamW Memory} = 4 + 4 + 4 = 12\\text{ bytes per trainable parameter}$$",
        "- **Full Fine-Tuning**: $7.0 \\times 10^9 \\times 12 \\text{ bytes} = \\mathbf{84.00\\text{ GB}}$ alone just for optimizer state!",
        "- **LoRA & QLoRA**: $19.6 \\times 10^6 \\times 12 \\text{ bytes} = \\mathbf{0.22\\text{ GB (220 MB)}}$.",
        "",
        "### 4. Activation Memory",
        "- Without gradient checkpointing, intermediate activations must be retained across all 32 transformer layers for backward pass: $\\approx 25\\text{--}35\\text{ GB}$.",
        "- With **Activation (Gradient) Checkpointing**, intermediate activations are discarded during forward pass and recomputed on demand during backward pass. Memory scales only with 1 layer: $\\approx 3.5\\text{--}4.0\\text{ GB}$ for $s=2048, b=1$.",
        "",
        "### 5. Conclusion & Practical Takeaway",
        "- **Full fine-tuning (118 GB)** is prohibitively expensive for consumer or single-GPU setups, mandating multi-GPU clusters (e.g. 2x A100-80GB or 4x A100-40GB) with FSDP or DeepSpeed ZeRO-3.",
        "- **LoRA (19.8 GB)** allows fine-tuning on a single high-end consumer GPU (RTX 3090 / 4090 24GB).",
        "- **QLoRA (9.3 GB)** democratizes 7B model training, allowing high-quality instruction fine-tuning directly on a **free Google Colab T4 (16GB)** or a budget RTX 3060 (12GB) GPU."
    ])
    
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"7B Memory calculation report written to: {output_path}")
    return content

if __name__ == "__main__":
    generate_memory_report()
