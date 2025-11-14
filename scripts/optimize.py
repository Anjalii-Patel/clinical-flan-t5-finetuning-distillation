"""
CLI entry point for model pruning and dynamic INT8 quantization.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.pruning import (
    apply_magnitude_pruning,
    calculate_model_sparsity,
    make_pruning_permanent,
)
from src.optimization.quantization import (
    apply_dynamic_quantization,
    get_model_size_mb,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("optimize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune and quantize FLAN-T5 model.")
    parser.add_argument("--config", type=str, default=None, help="Path to optimization YAML config.")
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base", help="Model path or HF hub ID.")
    parser.add_argument("--prune", action="store_true", default=False, help="Enable magnitude pruning.")
    parser.add_argument("--pruning_ratio", type=float, default=0.20, help="Fraction of weights to prune.")
    parser.add_argument("--pruning_method", type=str, default="global", choices=["global", "local"], help="Pruning method.")
    parser.add_argument("--quantize", action="store_true", default=False, help="Enable dynamic INT8 quantization.")
    parser.add_argument("--output_dir", type=str, default="./models/flan_t5_optimized", help="Path to save optimized model.")
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    config = {}
    if args.config and Path(args.config).exists():
        logger.info("Loading optimization config from %s", args.config)
        config = load_yaml_config(args.config)

    model_name = config.get("model", {}).get("name_or_path", args.model_name_or_path)
    do_prune = config.get("optimization", {}).get("prune", args.prune)
    pruning_ratio = float(config.get("optimization", {}).get("pruning_ratio", args.pruning_ratio))
    pruning_method = config.get("optimization", {}).get("pruning_method", args.pruning_method)
    do_quantize = config.get("optimization", {}).get("quantize", args.quantize)
    output_dir = config.get("output", {}).get("save_dir", args.output_dir)

    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    orig_stats = get_model_size_mb(model)
    logger.info("Original Model Size: %.2f MB (%d parameters)", orig_stats["model_size_mb"], orig_stats["total_parameters"])

    if do_prune:
        logger.info("Applying %s pruning with ratio: %.2f", pruning_method, pruning_ratio)
        model, sparsity_stats = apply_magnitude_pruning(
            model=model,
            pruning_ratio=pruning_ratio,
            method=pruning_method,
        )
        make_pruning_permanent(model)
        logger.info("Model Sparsity: %.2f%% zeroed parameters", sparsity_stats["global_sparsity"] * 100)

    if do_quantize:
        logger.info("Applying Dynamic INT8 Quantization...")
        model = apply_dynamic_quantization(model)
        quant_stats = get_model_size_mb(model)
        logger.info("Quantized Model Size: %.2f MB", quant_stats["model_size_mb"])

    # Verification smoke test
    logger.info("Running smoke inference on optimized model...")
    test_prompt = "Question: What are chief complaints? Context: Patient reports mild headache and fatigue."
    inputs = tokenizer(test_prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=20)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info("Smoke test answer: '%s'", answer)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving optimized model state and tokenizer to %s", out_path)
    tokenizer.save_pretrained(out_path)
    # For quantized models, state_dict saving is recommended
    import torch
    torch.save(model.state_dict(), out_path / "pytorch_model.bin")
    logger.info("Optimization complete.")


if __name__ == "__main__":
    main()

