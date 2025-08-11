#!/usr/bin/env python3
"""Test configurations for different models"""

# Standard test prompt for all models
TEST_PROMPT = "What color is the sky?\nA. Red\nB. Blue\nC. Green\nD. Yellow"

# Expected answers (B is correct)
EXPECTED_ANSWERS = ["B", "Blue", "blue", "B.", "B)", "(B)", "B:", "B -", "B-", "B Blue"]

# MiniCPM4-8B parameter combinations
MINICPM4_8B_CONFIGS = [
    {
        "name": "MiniCPM4-8B-baseline",
        "apply_sparse": False,
        "apply_quant": False,
        "apply_eagle": False,
        "apply_eagle_quant": False,
        "minicpm4_yarn": False,
        "description": "Baseline configuration without any optimizations"
    },
    {
        "name": "MiniCPM4-8B-sparse",
        "apply_sparse": True,
        "apply_quant": False,
        "apply_eagle": False,
        "apply_eagle_quant": False,
        "minicpm4_yarn": False,
        "description": "Sparse attention only"
    },
    {
        "name": "MiniCPM4-8B-quant",
        "apply_sparse": False,
        "apply_quant": True,
        "apply_eagle": False,
        "apply_eagle_quant": False,
        "minicpm4_yarn": False,
        "description": "Quantization only"
    },
    {
        "name": "MiniCPM4-8B-eagle2",
        "spec_type": "eagle2",
        "apply_sparse": False,
        "apply_quant": False,
        "apply_eagle": True,
        "apply_eagle_quant": False,
        "minicpm4_yarn": False,
        "description": "Eagle2 speculative decoding only"
    },
    {
        "name": "MiniCPM4-8B-eagle3",
        "spec_type": "eagle3",
        "apply_sparse": False,
        "apply_quant": False,
        "apply_eagle": True,
        "apply_eagle_quant": False,
        "minicpm4_yarn": False,
        "description": "Eagle3 speculative decoding only"
    },
    {
        "name": "MiniCPM4-8B-sparse-quant",
        "apply_sparse": True,
        "apply_quant": True,
        "apply_eagle": False,
        "apply_eagle_quant": False,
        "minicpm4_yarn": False,
        "description": "Sparse attention + quantization"
    },
    {
        "name": "MiniCPM4-8B-sparse-eagle2",
        "spec_type": "eagle2",
        "apply_sparse": True,
        "apply_quant": False,
        "apply_eagle": True,
        "apply_eagle_quant": False,
        "minicpm4_yarn": False,
        "description": "Sparse attention + Eagle2 speculative decoding"
    },
    {
        "name": "MiniCPM4-8B-sparse-eagle3",
        "spec_type": "eagle3",
        "apply_sparse": True,
        "apply_quant": False,
        "apply_eagle": True,
        "apply_eagle_quant": False,
        "minicpm4_yarn": False,
        "description": "Sparse attention + Eagle3 speculative decoding"
    },
    {
        "name": "MiniCPM4-8B-quant-eagle2",
        "spec_type": "eagle2",
        "apply_sparse": False,
        "apply_quant": True,
        "apply_eagle": True,
        "apply_eagle_quant": True,
        "minicpm4_yarn": False,
        "description": "Quantization + Eagle2 with quantization"
    },
    {
        "name": "MiniCPM4-8B-full-optimized",
        "apply_sparse": True,
        "apply_quant": True,
        "apply_eagle": True,
        "apply_eagle_quant": True,
        "minicpm4_yarn": True,
        "description": "All optimizations enabled"
    }
]

# Simple model configurations
SIMPLE_MODEL_CONFIGS = [
    {
        "name": "MiniCPM-1B",
        "model_path": "openbmb/minicpm-1b-sft-bf16",
        "description": "MiniCPM 1B parameter model"
    },
    {
        "name": "MiniCPM4-0.5B",
        "model_path": "openbmb/minicpm4-0.5b",
        "description": "MiniCPM4 0.5B parameter model"
    },
    {
        "name": "Qwen1.5-0.5B",
        "model_path": "qwen/qwen1.5-0.5b-chat",
        "description": "Qwen1.5 0.5B parameter model"
    },
    {
        "name": "Qwen2-0.5B",
        "model_path": "qwen/qwen2-0.5b-instruct",
        "description": "Qwen2 0.5B parameter model"
    },
    {
        "name": "Qwen3-0.6B",
        "model_path": "qwen/qwen3-0.6b",
        "description": "Qwen3 0.6B parameter model"
    },
    {
        "name": "Llama3.2-1B",
        "model_path": "unsloth/llama-3.2-1b-instruct",
        "description": "Llama3.2 1B parameter model"
    },
    {
        "name": "Llama2-7B",
        "model_path": "unsloth/llama-2-7b-chat",
        "description": "Llama2 7B parameter model"
    },
]

# Test timeout settings
TEST_TIMEOUT = 1800  # 5 minutes timeout for each test 