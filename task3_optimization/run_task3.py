"""
Unified CLI orchestrator for Task 3: Local LLM Optimization & Benchmarking.

Usage:
    python -m task3_optimization.run_task3 --mode capacity_plan
    python -m task3_optimization.run_task3 --mode guide
    python -m task3_optimization.run_task3 --mode benchmark_ollama --model_tag qwen2.5:0.5b
    python -m task3_optimization.run_task3 --mode all
"""

import os
import sys
import json
import argparse

from task3_optimization.src.benchmark import benchmark_ollama_model

def parse_args():
    parser = argparse.ArgumentParser(description="Task 3: Local LLM Optimization Orchestrator")
    parser.add_argument(
        "--mode",
        type=str,
        default="capacity_plan",
        choices=["all", "capacity_plan", "guide", "benchmark_ollama"],
        help="Pipeline phase to execute"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./task3_optimization/configs/benchmark_config.json",
        help="Path to benchmark config JSON"
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="qwen2.5:0.5b",
        help="Ollama model tag to benchmark"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama or llama.cpp HTTP API endpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./task3_optimization/results",
        help="Directory to save benchmark metrics"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 65)
    print(" TnDrr AI/ML Internship Assignment - Task 3: LLM Optimization")
    print(" Target Analysis: Quantization (FP16 vs 4-bit) & Serving Capacity")
    print(" Benchmark Set  : 10 Standardized Prompts across 10 Task Domains")
    print(" Serving Target : 20 Concurrent Users on a Single 24 GB GPU")
    print("=" * 65)
    
    # 1. Serving Capacity Plan
    if args.mode in ["all", "capacity_plan"]:
        plan_path = "./task3_optimization/src/serving_capacity_plan.md"
        print(f"\n>>> Displaying Serving Capacity Plan ({plan_path}):\n")
        if os.path.exists(plan_path):
            with open(plan_path, "r", encoding="utf-8") as f:
                print(f.read())
        if args.mode == "capacity_plan":
            return
            
    # 2. KV Cache & Quantization Guide
    if args.mode in ["all", "guide"]:
        guide_path = "./task3_optimization/src/kv_cache_and_quantization_guide.md"
        print(f"\n>>> Displaying KV Cache & Quantization Technical Guide ({guide_path}):\n")
        if os.path.exists(guide_path):
            with open(guide_path, "r", encoding="utf-8") as f:
                print(f.read())
        if args.mode == "guide":
            return
            
    # 3. Ollama Benchmarking
    if args.mode in ["all", "benchmark_ollama"]:
        print(f"\n>>> Running Benchmark on Ollama endpoint: {args.endpoint}...")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        prompts = cfg["prompts"]
        
        try:
            results = benchmark_ollama_model(
                model_tag=args.model_tag,
                prompts=prompts,
                endpoint=args.endpoint
            )
            out_file = os.path.join(args.output_dir, f"ollama_benchmark_{args.model_tag.replace(':', '_')}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nBenchmark completed successfully! Saved to: {out_file}")
        except Exception as e:
            print(f"\nNote: Ollama server not reachable at {args.endpoint} ({e}).")
            print("To benchmark with GPU acceleration, open `task3_optimization/notebooks/task3_optimization_colab.ipynb` in Google Colab.")

if __name__ == "__main__":
    main()
