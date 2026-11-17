"""
LoRA Fine-tuning pipeline for Qwen2.5-0.5B-Instruct on AG News four-class classification.
Supports CPU and GPU environments with proper prompt masking and loss tracking.
"""

import os
import json
from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

from task1_lora.src.data_loader import (
    LABEL_MAPPING,
    format_instruction_prompt
)
from task1_lora.src.model_utils import (
    load_base_model_and_tokenizer,
    create_lora_model,
    get_device
)

class AGNewsTokenizedDataset(TorchDataset):
    """
    Dataset that formats AG News into prompt + response pairs,
    masking the prompt with -100 so loss is computed solely on the target label token.
    """
    def __init__(self, raw_dataset, tokenizer, max_length: int = 256):
        self.features = []
        for item in raw_dataset:
            text = item["text"]
            label_id = item["label"]
            target_str = LABEL_MAPPING[label_id] + "<|im_end|>"
            prompt_str = format_instruction_prompt(text)
            
            # Tokenize prompt and target separately
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            target_ids = tokenizer.encode(target_str, add_special_tokens=False)
            
            input_ids = prompt_ids + target_ids
            # Mask prompt tokens with -100
            labels = [-100] * len(prompt_ids) + target_ids
            
            # Truncate if exceeds max_length
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
                
            self.features.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx]

class LossLoggingCallback(TrainerCallback):
    """Callback to record step-wise training loss."""
    def __init__(self):
        self.losses = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append({
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": logs["loss"]
            })

def train_lora(
    train_dataset,
    config: Dict[str, Any],
    output_dir: str = "./task1_lora/results/lora_adapter"
) -> Dict[str, Any]:
    """
    Execute LoRA fine-tuning on the given dataset.
    
    Args:
        train_dataset: HuggingFace dataset subset for training.
        config: Full configuration dictionary.
        output_dir: Output path for saving the adapter.
        
    Returns:
        Summary of training history and parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"\n--- Initializing LoRA Fine-Tuning Pipeline on {device} ---")
    
    model_name = config.get("model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
    base_model, tokenizer = load_base_model_and_tokenizer(model_name)
    
    lora_cfg = config.get("lora", {})
    peft_model, param_summary = create_lora_model(
        base_model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", None)
    )
    
    print("Tokenizing training dataset...")
    tokenized_train = AGNewsTokenizedDataset(
        train_dataset,
        tokenizer,
        max_length=config.get("max_seq_length", 256)
    )
    
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8 if device.type == "cuda" else None
    )
    
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("num_train_epochs", 2)
    batch_size = train_cfg.get("per_device_train_batch_size", 4)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
    lr = train_cfg.get("learning_rate", 2e-4)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy="no",
        fp16=(device.type == "cuda"),
        report_to="none",
        dataloader_drop_last=False
    )
    
    loss_callback = LossLoggingCallback()
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=collator,
        callbacks=[loss_callback]
    )
    
    print("Starting training loop...")
    train_result = trainer.train()
    
    print(f"Training completed. Saving LoRA adapter weights to: {output_dir}")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training log history
    history_file = os.path.join(output_dir, "training_history.json")
    history_data = {
        "global_step": train_result.global_step,
        "training_loss": train_result.training_loss,
        "metrics": train_result.metrics,
        "loss_logs": loss_callback.losses,
        "param_summary": param_summary
    }
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
        
    print(f"Training history saved to: {history_file}")
    return history_data
