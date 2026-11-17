"""
Data loading and preprocessing utilities for AG News classification with Qwen2.5-0.5B-Instruct.
"""

from typing import Dict, List, Tuple, Any

LABEL_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

LABEL_TO_ID = {v.lower(): k for k, v in LABEL_MAPPING.items()}
# Support common aliases
LABEL_TO_ID["sci/tech"] = 3
LABEL_TO_ID["scitech"] = 3
LABEL_TO_ID["science/tech"] = 3
LABEL_TO_ID["science"] = 3
LABEL_TO_ID["tech"] = 3
LABEL_TO_ID["technology"] = 3

SYSTEM_PROMPT = (
    "You are an expert news classifier. Classify the given news article into exactly one "
    "of the following four categories: World, Sports, Business, Sci/Tech. "
    "Respond with only the category name."
)

def format_instruction_prompt(text: str) -> str:
    """Format an article into standard Qwen instruction prompt format."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nNews Article: {text.strip()}\nCategory:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def format_training_example(text: str, label_id: int) -> str:
    """Format full prompt + response for Causal LM fine-tuning."""
    prompt = format_instruction_prompt(text)
    target = LABEL_MAPPING[label_id]
    return f"{prompt}{target}<|im_end|>"

def load_ag_news_data(
    train_samples: int = 2000,
    eval_samples: int = 100,
    seed: int = 42
) -> Tuple[Dataset, Dataset, List[Dict[str, Any]]]:
    """
    Load AG News dataset and create reproducible train and held-out test splits.
    
    Args:
        train_samples: Number of training samples to take from the train split.
        eval_samples: Number of evaluation samples to take from the test split.
        seed: Random seed for reproducibility.
        
    Returns:
        (train_dataset, eval_dataset, raw_eval_records)
    """
    from datasets import load_dataset
    dataset = load_dataset("ag_news")
    
    # Shuffle splits deterministically
    shuffled_train = dataset["train"].shuffle(seed=seed)
    shuffled_test = dataset["test"].shuffle(seed=seed)
    
    # Select subsets
    train_subset = shuffled_train.select(range(min(train_samples, len(shuffled_train))))
    eval_subset = shuffled_test.select(range(min(eval_samples, len(shuffled_test))))
    
    raw_eval_records = []
    for i, item in enumerate(eval_subset):
        raw_eval_records.append({
            "idx": i,
            "text": item["text"],
            "label_id": item["label"],
            "label_name": LABEL_MAPPING[item["label"]],
            "prompt": format_instruction_prompt(item["text"])
        })
        
    return train_subset, eval_subset, raw_eval_records

def parse_predicted_label(prediction_text: str) -> int:
    """
    Parse model output string into a class label ID (0-3).
    Returns -1 if unparseable.
    """
    cleaned = prediction_text.strip().lower()
    
    # Check direct match or substring
    for key, label_id in LABEL_TO_ID.items():
        if key in cleaned:
            return label_id
            
    # Check startswith
    for key, label_id in LABEL_TO_ID.items():
        if cleaned.startswith(key):
            return label_id
            
    return -1
