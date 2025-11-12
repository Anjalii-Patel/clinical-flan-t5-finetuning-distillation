"""
CLI entry point for running evaluation on clinical test split and generating metric reports.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Union

import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import ClinicalModelEvaluator


def sanitize_tokenizer_config(directory: Union[str, Path]) -> None:
    """Fix bug in transformers where extra_special_tokens is saved as a list, causing AttributeError on reload."""
    cfg_path = Path(directory) / "tokenizer_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "extra_special_tokens" in data and isinstance(data["extra_special_tokens"], list):
                del data["extra_special_tokens"]
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FLAN-T5 model on Clinical QA test set.")
    parser.add_argument("--config", type=str, default=None, help="Path to evaluation YAML config.")
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base", help="Model path or HF hub identifier.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Tokenizer path (defaults to model).")
    parser.add_argument("--test_path", type=str, default="data/processed/test.csv", help="Path to test CSV.")
    parser.add_argument("--output_results_json", type=str, default="results/evaluation_results.json", help="Path to save results JSON.")
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--num_beams", type=int, default=2, help="Number of beams for generation.")
    parser.add_argument("--max_input_length", type=int, default=512, help="Max input sequence length.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max generated tokens.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit for quick smoke evaluation.")
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    config = {}
    if args.config and Path(args.config).exists():
        logger.info("Loading evaluation configuration from %s", args.config)
        config = load_yaml_config(args.config)

    model_name = config.get("model", {}).get("name_or_path", args.model_name_or_path)
    tok_name = config.get("model", {}).get("tokenizer_name", args.tokenizer_name_or_path or model_name)
    test_path = config.get("data", {}).get("test_path", args.test_path)
    out_json = config.get("evaluation", {}).get("output_results_json", args.output_results_json)
    batch_size = int(config.get("evaluation", {}).get("batch_size", args.batch_size))
    num_beams = int(config.get("model", {}).get("num_beams", args.num_beams))
    max_input_length = int(config.get("model", {}).get("max_input_length", args.max_input_length))
    max_new_tokens = int(config.get("model", {}).get("max_new_tokens", args.max_new_tokens))
    max_samples = config.get("data", {}).get("max_samples", args.max_samples)

    logger.info("Evaluating model: %s on %s", model_name, test_path)
    if Path(tok_name).is_dir():
        sanitize_tokenizer_config(tok_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
    except AttributeError:
        logger.warning("Tokenizer attribute mismatch in %s; falling back to 'google/flan-t5-base'", tok_name)
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    evaluator = ClinicalModelEvaluator(model=model, tokenizer=tokenizer)
    eval_output = evaluator.evaluate_csv(
        csv_path=test_path,
        output_results_json=out_json,
        max_samples=max_samples,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        batch_size=batch_size,
    )

    metrics = eval_output["metrics"]
    print("\n" + "=" * 55)
    print("           CLINICAL EVALUATION REPORT")
    print("=" * 55)
    print(f"Model:                {model_name}")
    print(f"Test samples:         {metrics['num_samples']}")
    print(f"ROUGE-1:              {metrics['rouge1']:.4f}")
    print(f"ROUGE-2:              {metrics['rouge2']:.4f}")
    print(f"ROUGE-L:              {metrics['rougeL']:.4f}")
    print(f"BLEU:                 {metrics['bleu']:.4f}")
    print(f"Exact Match:          {metrics['exact_match']:.4f}")
    print(f"Avg Pred Word Count:  {metrics['avg_pred_length']}")
    print(f"Avg Latency:          {metrics['avg_latency_ms']:.2f} ms / sample")
    print(f"Total Evaluation Time:{metrics['total_eval_time_sec']:.2f} s")
    print("=" * 55)
    print(f"Detailed predictions saved to: {out_json}\n")


if __name__ == "__main__":
    main()

