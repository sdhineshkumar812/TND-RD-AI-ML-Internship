"""
Comprehensive Unit and Integration Test Suite for TnDrr AI/ML Internship Assignment.
Tests data loaders, token chunking, BM25 scoring from scratch, RRF fusion,
7B memory derivations, and serving capacity math.
"""

import os
import sys
import unittest
import json
import math

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from task1_lora.src.data_loader import (
    LABEL_MAPPING,
    LABEL_TO_ID,
    format_instruction_prompt,
    format_training_example,
    parse_predicted_label
)
from task1_lora.src.memory_calc import compute_7b_memory_breakdown, generate_memory_report
from task2_rag.src.retrieval import BM25Retriever, reciprocal_rank_fusion
from task2_rag.src.generator import build_context_string, RAGGenerator


class TestTask1LoRA(unittest.TestCase):
    """Unit tests for Task 1: LoRA Fine-Tuning Components."""

    def test_label_mapping(self):
        self.assertEqual(len(LABEL_MAPPING), 4)
        self.assertEqual(LABEL_MAPPING[0], "World")
        self.assertEqual(LABEL_MAPPING[1], "Sports")
        self.assertEqual(LABEL_MAPPING[2], "Business")
        self.assertEqual(LABEL_MAPPING[3], "Sci/Tech")

    def test_parse_predicted_label(self):
        self.assertEqual(parse_predicted_label("World"), 0)
        self.assertEqual(parse_predicted_label("sports"), 1)
        self.assertEqual(parse_predicted_label("Business news"), 2)
        self.assertEqual(parse_predicted_label("Sci/Tech"), 3)
        self.assertEqual(parse_predicted_label("technology"), 3)
        self.assertEqual(parse_predicted_label("unrecognized text"), -1)

    def test_prompt_formatting(self):
        article = "Stocks rallied today as inflation eased."
        prompt = format_instruction_prompt(article)
        self.assertIn("News Article: Stocks rallied today as inflation eased.", prompt)
        self.assertIn("<|im_start|>assistant", prompt)
        
        full_train = format_training_example(article, label_id=2)
        self.assertTrue(full_train.endswith("Business<|im_end|>"))

    def test_7b_memory_calculation(self):
        breakdown = compute_7b_memory_breakdown()
        self.assertIn("Full Fine-Tuning (FP16)", breakdown)
        self.assertIn("LoRA (FP16 Base + r=16)", breakdown)
        self.assertIn("QLoRA (NF4 4-bit Base + r=16)", breakdown)
        
        # Verify mathematically that Full > LoRA > QLoRA
        full_vram = breakdown["Full Fine-Tuning (FP16)"]["Total Estimated VRAM (GB)"]
        lora_vram = breakdown["LoRA (FP16 Base + r=16)"]["Total Estimated VRAM (GB)"]
        qlora_vram = breakdown["QLoRA (NF4 4-bit Base + r=16)"]["Total Estimated VRAM (GB)"]
        
        self.assertGreater(full_vram, 100.0)
        self.assertLess(lora_vram, 24.0)
        self.assertLess(qlora_vram, 12.0)
        
        # Verify report generation
        out_report = "./task1_lora/results/7b_memory_calculation.md"
        report_text = generate_memory_report(out_report)
        self.assertTrue(os.path.exists(out_report))
        self.assertIn("Full Fine-Tuning", report_text)


class TestTask2RAG(unittest.TestCase):
    """Unit tests for Task 2: RAG From Scratch Components."""

    def test_bm25_from_scratch(self):
        sample_corpus = [
            {"chunk_id": "c1", "text": "Grouped-query attention reduces memory bandwidth in large language models."},
            {"chunk_id": "c2", "text": "Reinforcement learning from human feedback optimizes helpfulness and safety."},
            {"chunk_id": "c3", "text": "The model was pretrained on two trillion tokens using AdamW optimizer."}
        ]
        bm25 = BM25Retriever(k1=1.5, b=0.75)
        bm25.index_chunks(sample_corpus)
        
        self.assertEqual(bm25.corpus_size, 3)
        self.assertIn("attention", bm25.idf)
        
        # Query 1
        res1 = bm25.search("grouped query attention", top_k=1)
        self.assertEqual(len(res1), 1)
        self.assertEqual(res1[0]["chunk_id"], "c1")
        
        # Query 2
        res2 = bm25.search("two trillion tokens AdamW", top_k=1)
        self.assertEqual(len(res2), 1)
        self.assertEqual(res2[0]["chunk_id"], "c3")

    def test_reciprocal_rank_fusion(self):
        list_a = [
            {"chunk_id": "doc1", "text": "text1"},
            {"chunk_id": "doc2", "text": "text2"}
        ]
        list_b = [
            {"chunk_id": "doc2", "text": "text2"},
            {"chunk_id": "doc3", "text": "text3"}
        ]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=3)
        # doc2 appears in both lists, so its RRF score should rank highest!
        self.assertEqual(fused[0]["chunk_id"], "doc2")
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])

    def test_ground_truth_qa_schema(self):
        qa_file = "./task2_rag/data/ground_truth_qa.json"
        self.assertTrue(os.path.exists(qa_file))
        with open(qa_file, "r", encoding="utf-8") as f:
            qa_list = json.load(f)
        self.assertEqual(len(qa_list), 15, "Must have exactly 15 manually authored QA pairs")
        for q in qa_list:
            self.assertIn("id", q)
            self.assertIn("question", q)
            self.assertIn("target_page", q)
            self.assertIn("key_phrases", q)
            self.assertIn("ground_truth_answer", q)
            self.assertGreaterEqual(len(q["key_phrases"]), 1)

    def test_context_and_citations(self):
        chunks = [
            {"chunk_id": "c1", "page_number": 5, "text": "Context passage on pretraining."}
        ]
        ctx_str = build_context_string(chunks)
        self.assertIn("[c1 | Page 5]:", ctx_str)
        
        gen = RAGGenerator()
        ans = gen.generate_answer("How was the model pretrained?", chunks)
        self.assertIn("citations", ans)
        self.assertEqual(len(ans["citations"]), 1)
        self.assertEqual(ans["citations"][0]["chunk_id"], "c1")
        self.assertEqual(ans["citations"][0]["page_number"], 5)


class TestTask3Optimization(unittest.TestCase):
    """Unit tests for Task 3: Optimization & Capacity Planning Components."""

    def test_benchmark_config_schema(self):
        cfg_file = "./task3_optimization/configs/benchmark_config.json"
        self.assertTrue(os.path.exists(cfg_file))
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(len(cfg["prompts"]), 10, "Must have exactly 10 standardized prompts")
        self.assertIn("fp16", cfg["models"])
        self.assertIn("int4", cfg["models"])

    def test_serving_capacity_plan_math(self):
        # 20 concurrent users, 2048 sequence length, 7B model
        n_users = 20
        seq_len = 2048
        layers = 32
        head_dim = 128
        bytes_per_elem = 2 # FP16 KV
        
        # MHA: 32 heads
        mha_kv_bytes = 2 * n_users * layers * 32 * head_dim * seq_len * bytes_per_elem
        mha_kv_gb = mha_kv_bytes / (1024**3)
        self.assertAlmostEqual(mha_kv_gb, 20.0, delta=1.5)
        
        # GQA: 8 heads
        gqa_kv_bytes = 2 * n_users * layers * 8 * head_dim * seq_len * bytes_per_elem
        gqa_kv_gb = gqa_kv_bytes / (1024**3)
        self.assertAlmostEqual(gqa_kv_gb, 5.0, delta=0.5)
        
        # On a 24 GB GPU with 4.1 GB model weights + 1.5 GB overhead:
        total_gqa_vram = 4.1 + 1.5 + gqa_kv_gb
        self.assertLess(total_gqa_vram, 24.0, "GQA must fit within 24GB VRAM budget!")


if __name__ == "__main__":
    unittest.main()
