"""
Model initialization, tokenizer configuration, and PEFT LoRA setup for Qwen2.5-0.5B-Instruct.
"""

from typing import Tuple, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def get_device() -> torch.device:
    """Determine the optimal device available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_base_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    torch_dtype: Any = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base causal LM and its tokenizer.
    
    Args:
        model_name: Hugging Face model identifier.
        torch_dtype: Explicit torch dtype (defaults to bfloat16 on CUDA, float32 on CPU).
        
    Returns:
        (model, tokenizer)
    """
    device = get_device()
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"Loading base model: {model_name} on {device} (dtype={torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True
    )
    if device.type == "cpu":
        model = model.to(device)
        
    return model, tokenizer

def create_lora_model(
    base_model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM"
) -> Tuple[PeftModel, Dict[str, Any]]:
    """
    Wrap base model with LoRA configuration.
    
    Returns:
        (peft_model, param_summary)
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type
    )
    
    peft_model = get_peft_model(base_model, lora_config)
    
    trainable_params, all_param = peft_model.get_nb_trainable_parameters()
    percent = 100 * trainable_params / all_param
    param_summary = {
        "trainable_params": trainable_params,
        "all_params": all_param,
        "trainable_percent": percent,
        "lora_r": r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules
    }
    
    print(
        f"Trainable params: {trainable_params:,} / {all_param:,} "
        f"({percent:.4f}% trainable)"
    )
    return peft_model, param_summary

def load_finetuned_model(
    base_model: PreTrainedModel,
    adapter_path: str
) -> PeftModel:
    """Load trained LoRA weights onto the base model."""
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model
