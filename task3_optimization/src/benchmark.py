"""
Benchmarking harness for Local LLM Optimization.
Measures tokens per second, latency, TTFT, RAM, and VRAM across quantization settings.
Supports Ollama, llama.cpp HTTP server, or Hugging Face runtime.
"""

import os
import sys
import time
import json
from typing import List, Dict, Any, Optional
import psutil

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def get_current_ram_mb() -> float:
    """Return process RAM usage in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_ollama_model(
    model_tag: str,
    prompts: List[Dict[str, Any]],
    endpoint: str = "http://localhost:11434/api/generate",
    quant_label: str = "4-bit"
) -> Dict[str, Any]:
    """
    Benchmark Ollama model over 10 prompts.
    
    Args:
        model_tag: Ollama model tag (e.g., 'qwen2.5:0.5b').
        prompts: 10 evaluation prompts.
        endpoint: Ollama HTTP API endpoint.
        quant_label: Label for quantization regime (e.g. 'FP16' or '4-bit').
        
    Returns:
        Structured benchmark metrics including tok/s, RAM, VRAM, and actual outputs.
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library is required for Ollama benchmarking.")
        
    print(f"\n--- Benchmarking Ollama Model: '{model_tag}' ({quant_label}) ---")
    results = []
    total_tokens = 0
    total_eval_time_sec = 0.0
    
    for item in prompts:
        pid = item["id"]
        category = item["category"]
        prompt_text = item["prompt"]
        
        payload = {
            "model": model_tag,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 128
            }
        }
        
        ram_before = get_current_ram_mb()
        start_time = time.perf_counter()
        
        try:
            resp = requests.post(endpoint, json=payload, timeout=60)
            resp.raise_for_exception()
            data = resp.json()
        except Exception as e:
            print(f"Error querying Ollama endpoint: {e}")
            raise
            
        elapsed_sec = time.perf_counter() - start_time
        ram_after = get_current_ram_mb()
        
        # Ollama telemetry fields (in nanoseconds)
        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 1)
        
        # Decode speed in tokens per second
        eval_duration_sec = eval_duration_ns / 1e9
        tokens_per_second = eval_count / max(eval_duration_sec, 1e-6)
        
        # Prompt eval / prefill speed
        prefill_tok_per_sec = prompt_eval_count / max(prompt_eval_duration_ns / 1e9, 1e-6)
        
        total_tokens += eval_count
        total_eval_time_sec += eval_duration_sec
        
        response_text = data.get("response", "").strip()
        
        record = {
            "prompt_id": pid,
            "category": category,
            "prompt": prompt_text,
            "response": response_text,
            "generated_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "decode_tokens_per_sec": round(tokens_per_second, 2),
            "prefill_tokens_per_sec": round(prefill_tok_per_sec, 2),
            "total_latency_sec": round(elapsed_sec, 3),
            "ram_used_mb": round(ram_after, 2),
            "vram_used_mb": "Logged from host or GPU monitor"
        }
        results.append(record)
        print(f"[{pid} | {category:<22}]: {eval_count} tokens in {elapsed_sec:.2f}s ({tokens_per_second:.1f} tok/s)")
        
    avg_speed = total_tokens / max(total_eval_time_sec, 1e-6)
    
    summary = {
        "model_tag": model_tag,
        "quantization": quant_label,
        "total_prompts": len(prompts),
        "total_tokens_generated": total_tokens,
        "average_decode_tokens_per_sec": round(avg_speed, 2),
        "results": results
    }
    return summary
