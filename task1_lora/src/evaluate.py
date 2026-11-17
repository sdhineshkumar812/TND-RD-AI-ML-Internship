"""
Evaluation module for AG News classification: computes accuracy, per-class metrics,
confusion matrices, and captures concrete failure cases.
"""

import os
import json
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from task1_lora.src.data_loader import LABEL_MAPPING, parse_predicted_label

def evaluate_model(
    model,
    tokenizer,
    eval_records: List[Dict[str, Any]],
    device: torch.device,
    tag: str = "base_model",
    output_dir: str = "./task1_lora/results"
) -> Dict[str, Any]:
    """
    Evaluate a causal LM on AG News classification records.
    
    Args:
        model: Hugging Face or PEFT model.
        tokenizer: Hugging Face tokenizer.
        eval_records: List of dicts containing 'prompt', 'label_id', 'text', etc.
        device: Device to run evaluation on.
        tag: Identifier for the evaluation run (e.g., 'base_model' or 'lora_finetuned').
        output_dir: Directory where results and plots are saved.
        
    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    y_true = []
    y_pred = []
    raw_outputs = []
    failures = []
    
    print(f"\n--- Running Evaluation: [{tag}] on {len(eval_records)} examples ---")
    
    with torch.no_grad():
        for i, item in enumerate(eval_records):
            prompt = item["prompt"]
            true_label = item["label_id"]
            true_name = LABEL_MAPPING[true_label]
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]
            
            # Greedy generation for deterministic classification
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Extract newly generated tokens
            gen_tokens = outputs[0][input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            
            pred_label = parse_predicted_label(gen_text)
            pred_name = LABEL_MAPPING.get(pred_label, "Unknown")
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            record = {
                "idx": item["idx"],
                "text": item["text"],
                "true_label": true_label,
                "true_name": true_name,
                "pred_label": pred_label,
                "pred_name": pred_name,
                "raw_generated_text": gen_text
            }
            raw_outputs.append(record)
            
            if pred_label != true_label:
                failures.append(record)
                
            if (i + 1) % 25 == 0 or (i + 1) == len(eval_records):
                print(f"Evaluated {i + 1}/{len(eval_records)} samples...")
                
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    target_names = [LABEL_MAPPING[i] for i in range(4)]
    # Filter valid predicted labels (0..3) for scikit-learn metrics
    valid_classes = [0, 1, 2, 3]
    
    cls_report = classification_report(
        y_true,
        y_pred,
        labels=valid_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=valid_classes)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    # Save Confusion Matrix Plot
    cm_plot_path = os.path.join(output_dir, f"confusion_matrix_{tag}.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{z}', ha='center', va='center',
                color='white' if z > cm.max() / 2 else 'black')
                
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(target_names, rotation=30, ha='left')
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix: {tag}", pad=20)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    
    # Save Metrics JSON
    results = {
        "tag": tag,
        "total_samples": len(eval_records),
        "accuracy": float(accuracy),
        "macro_avg_f1": float(cls_report["macro avg"]["f1-score"]),
        "weighted_avg_f1": float(cls_report["weighted avg"]["f1-score"]),
        "per_class": {
            name: {
                "precision": float(cls_report[name]["precision"]),
                "recall": float(cls_report[name]["recall"]),
                "f1_score": float(cls_report[name]["f1-score"]),
                "support": int(cls_report[name]["support"])
            }
            for name in target_names
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "total_failures": len(failures),
        "failures": failures[:10],  # Save first 10 failures
        "cm_plot_path": cm_plot_path
    }
    
    results_json_path = os.path.join(output_dir, f"metrics_{tag}.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Also save failure cases to dedicated JSON
    failure_file_path = os.path.join(output_dir, f"failures_{tag}.json")
    with open(failure_file_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
        
    print(f"\n==========================================")
    print(f"Results for [{tag}]:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {cls_report['macro avg']['f1-score']:.4f}")
    print(f"Failures recorded: {len(failures)}/{len(eval_records)}")
    print(f"Metrics saved to: {results_json_path}")
    print(f"Plot saved to: {cm_plot_path}")
    print(f"==========================================\n")
    
    return results
