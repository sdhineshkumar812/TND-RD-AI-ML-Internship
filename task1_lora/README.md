# Task 1: LoRA Fine-Tuning for Four-Class News Classification

## Overview
This module implements Parameter-Efficient Fine-Tuning (PEFT) using **LoRA (Low-Rank Adaptation)** on **`Qwen/Qwen2.5-0.5B-Instruct`** for four-class news classification on the **AG News** benchmark dataset.

The four news categories in AG News are:
- `0`: **World**
- `1`: **Sports**
- `2`: **Business**
- `3`: **Sci/Tech**

---

## Directory Structure
```
task1_lora/
├── configs/
│   └── lora_config.json          # Experiment hyperparameters and model settings
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # AG News loading, prompt formatting, label mapping
│   ├── model_utils.py            # Model loading, tokenizer, and PEFT LoRA wrapping
│   ├── evaluate.py               # Deterministic greedy evaluation, metrics, and plots
│   ├── memory_calc.py            # Mathematical 7B model VRAM derivation
│   └── train_lora.py             # SFT training loop with target prompt masking
├── notebooks/
│   └── task1_lora_colab.ipynb    # Google Colab GPU-ready interactive notebook
├── results/                      # Generated evaluation metrics, plots, and failure cases
│   ├── 7b_memory_calculation.md  # Detailed GPU memory report
│   ├── metrics_base_model.json
│   ├── metrics_lora_finetuned.json
│   ├── confusion_matrix_base_model.png
│   ├── confusion_matrix_lora_finetuned.png
│   └── failures_lora_finetuned.json
├── run_task1.py                  # CLI pipeline runner
└── README.md
```

---

## Technical Specifications

### Model & LoRA Parameters
- **Base Architecture**: `Qwen/Qwen2.5-0.5B-Instruct` (490M parameters)
- **LoRA Rank ($r$)**: 16
- **LoRA Alpha ($\alpha$)**: 32
- **LoRA Dropout**: 0.05
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Task Type**: `CAUSAL_LM`
- **Trainable Parameters**: ~4.1M parameters (~0.84% of total model weights)

### Reproducibility
- **Random Seed**: `42`
- **Training Subset**: 2,000 samples shuffled deterministically from the AG News train split
- **Held-Out Evaluation Split**: 100 samples shuffled deterministically from the test split (verified $\ge 50$ samples)
- **Prompt Format**: Qwen ChatML syntax with target label loss masking (only response tokens are trained with Cross-Entropy Loss)

---

## Reproduction Commands

### 1. GPU Memory Calculation (7B Model)
Run the transparent memory calculation script to generate the mathematical derivation for Full Fine-Tuning, LoRA, and QLoRA on a 7B parameter model:
```bash
python -m task1_lora.run_task1 --mode calc_memory
```
The report is saved to `task1_lora/results/7b_memory_calculation.md`.

### 2. Base Model Zero-Shot Evaluation
Evaluate `Qwen/Qwen2.5-0.5B-Instruct` zero-shot on the held-out test set:
```bash
python -m task1_lora.run_task1 --mode eval_base --eval_samples 100
```

### 3. LoRA Fine-Tuning
Execute LoRA fine-tuning:
```bash
python -m task1_lora.run_task1 --mode train --train_samples 2000
```

### 4. Fine-Tuned Model Evaluation & Comparison
Evaluate the adapted model on the exact same held-out test split:
```bash
python -m task1_lora.run_task1 --mode eval_lora --eval_samples 100
```
To print the comparison table:
```bash
python -m task1_lora.run_task1 --mode compare
```

### 5. Google Colab GPU Execution
When local hardware is CPU-only, execute the full pipeline in Google Colab (free T4 GPU):
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `task1_lora/notebooks/task1_lora_colab.ipynb`.
3. In Colab, select **Runtime > Change runtime type > T4 GPU**.
4. Run all cells. Training completes in ~2 minutes on T4.
5. Cell 12 downloads `task1_colab_results.zip`. Extract this archive into `task1_lora/results/` on your local machine.

---

## 7B Model GPU Memory Calculation Summary

| Memory Component | Full Fine-Tuning (FP16) | LoRA (FP16 Base) | QLoRA (NF4 4-bit Base) |
| :--- | :---: | :---: | :---: |
| **Model Weights (GB)** | 13.04 GB | 13.08 GB | 3.62 GB |
| **Gradients (GB)** | 13.04 GB | 0.04 GB | 0.04 GB |
| **Optimizer States (AdamW) (GB)** | 78.23 GB | 0.22 GB | 0.22 GB |
| **Activations (w/ checkpointing) (GB)** | 4.00 GB | 4.00 GB | 4.00 GB |
| **CUDA Overhead & Buffers (GB)** | 2.00 GB | 1.50 GB | 1.50 GB |
| **Total Estimated VRAM** | **110.31 GB** | **18.84 GB** | **9.38 GB** |
| **Hardware Requirement** | Multi-GPU (2x A100 80GB) | Single RTX 3090/4090 24GB | Free Colab T4 (16GB) |
