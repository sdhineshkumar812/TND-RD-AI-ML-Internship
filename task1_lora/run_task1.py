"""
Unified CLI orchestrator for Task 1: LoRA Fine-Tuning & Evaluation.

Usage:
    python -m task1_lora.run_task1 --mode calc_memory
    python -m task1_lora.run_task1 --mode eval_base
    python -m task1_lora.run_task1 --mode train
    python -m task1_lora.run_task1 --mode eval_lora
    python -m task1_lora.run_task1 --mode all
"""

import os
import sys
import json
import argparse
import torch

from task1_lora.src.data_loader import load_ag_news_data
from task1_lora.src.model_utils import (
    load_base_model_and_tokenizer,
    load_finetuned_model,
    get_device
)
from task1_lora.src.evaluate import evaluate_model
from task1_lora.src.memory_calc import generate_memory_report
from task1_lora.src.train_lora import train_lora

def parse_args():
    parser = argparse.ArgumentParser(description="Task 1: LoRA Fine-Tuning Orchestrator")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "calc_memory", "eval_base", "train", "eval_lora", "compare"],
        help="Pipeline phase to execute"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./task1_lora/configs/lora_config.json",
        help="Path to experiment configuration JSON"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Override training sample count in config"
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=None,
        help="Override evaluation sample count in config"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    train_samples = args.train_samples or config.get("train_samples", 2000)
    eval_samples = args.eval_samples or config.get("eval_samples", 100)
    seed = config.get("seed", 42)
    output_dir = "./task1_lora/results"
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    
    device = get_device()
    print("=" * 60)
    print(" TnDrr AI/ML Internship Assignment - Task 1: LoRA Fine-Tuning")
    print(f" Target Model : {config.get('model_name_or_path')}")
    print(f" Dataset      : AG News (4 classes)")
    print(f" Device       : {device}")
    print(f" Train Samples: {train_samples}")
    print(f" Eval Samples : {eval_samples} (Reproducible seed={seed})")
    print("=" * 60)
    
    # Phase: 7B GPU Memory Calculation
    if args.mode in ["all", "calc_memory"]:
        print("\n>>> Phase 1: 7B Parameter Model GPU Memory Calculation")
        mem_report = generate_memory_report(
            os.path.join(output_dir, "7b_memory_calculation.md")
        )
        if args.mode == "calc_memory":
            return
            
    # Load Data Splits
    print("\n>>> Loading AG News Data Splits...")
    train_dataset, eval_dataset, eval_records = load_ag_news_data(
        train_samples=train_samples,
        eval_samples=eval_samples,
        seed=seed
    )
    print(f"Train subset size: {len(train_dataset)}")
    print(f"Held-out test size: {len(eval_records)}")
    
    # Phase: Evaluate Base Model
    if args.mode in ["all", "eval_base"]:
        print("\n>>> Phase 2: Zero-Shot Base Model Evaluation")
        base_model, tokenizer = load_base_model_and_tokenizer(config.get("model_name_or_path"))
        evaluate_model(
            model=base_model,
            tokenizer=tokenizer,
            eval_records=eval_records,
            device=device,
            tag="base_model",
            output_dir=output_dir
        )
        # Clear model from memory if training follows
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if args.mode == "eval_base":
            return
            
    # Phase: Train LoRA
    if args.mode in ["all", "train"]:
        print("\n>>> Phase 3: LoRA Training")
        train_lora(train_dataset, config, output_dir=adapter_dir)
        if args.mode == "train":
            return
            
    # Phase: Evaluate LoRA Model
    if args.mode in ["all", "eval_lora"]:
        print("\n>>> Phase 4: Fine-Tuned LoRA Model Evaluation")
        base_model, tokenizer = load_base_model_and_tokenizer(config.get("model_name_or_path"))
        lora_model = load_finetuned_model(base_model, adapter_dir)
        lora_model = lora_model.to(device)
        evaluate_model(
            model=lora_model,
            tokenizer=tokenizer,
            eval_records=eval_records,
            device=device,
            tag="lora_finetuned",
            output_dir=output_dir
        )
        if args.mode == "eval_lora":
            return

    # Phase: Comparison
    if args.mode in ["all", "compare"]:
        base_metrics_path = os.path.join(output_dir, "metrics_base_model.json")
        lora_metrics_path = os.path.join(output_dir, "metrics_lora_finetuned.json")
        
        if os.path.exists(base_metrics_path) and os.path.exists(lora_metrics_path):
            with open(base_metrics_path, "r", encoding="utf-8") as f:
                base_m = json.load(f)
            with open(lora_metrics_path, "r", encoding="utf-8") as f:
                lora_m = json.load(f)
                
            print("\n" + "=" * 60)
            print(" FINAL VERIFIED COMPARISON: BASE MODEL vs. LoRA FINE-TUNED")
            print("=" * 60)
            print(f"Overall Accuracy: Base: {base_m['accuracy']*100:.2f}%  -->  LoRA: {lora_m['accuracy']*100:.2f}%")
            print(f"Macro Avg F1    : Base: {base_m['macro_avg_f1']:.4f}  -->  LoRA: {lora_m['macro_avg_f1']:.4f}")
            print("\nPer-Class F1 Score Comparison:")
            for cname in ["World", "Sports", "Business", "Sci/Tech"]:
                b_f1 = base_m["per_class"][cname]["f1_score"]
                l_f1 = lora_m["per_class"][cname]["f1_score"]
                diff = (l_f1 - b_f1) * 100
                print(f"  {cname:<10}: Base={b_f1:.4f} | LoRA={l_f1:.4f} ({diff:+.2f}%)")
            print("=" * 60)

if __name__ == "__main__":
    main()
