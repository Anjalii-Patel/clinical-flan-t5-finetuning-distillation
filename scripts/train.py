"""
CLI entry point for Supervised Fine-Tuning of FLAN-T5 models on clinical QA data.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_clinical_splits
from src.training.trainer import build_training_args, train_seq2seq_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FLAN-T5 on Clinical QA dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base", help="Model checkpoint identifier.")
    parser.add_argument("--train_path", type=str, default="data/processed/train.csv", help="Path to train CSV.")
    parser.add_argument("--val_path", type=str, default="data/processed/val.csv", help="Path to validation CSV.")
    parser.add_argument("--output_dir", type=str, default="./models/flan_t5_finetuned", help="Directory to save model.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device training batch size.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 for full epoch training).")
    parser.add_argument("--max_input_length", type=int, default=512, help="Max input sequence length.")
    parser.add_argument("--max_target_length", type=int, default=128, help="Max target sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint directory to resume from.")
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    config = {}
    if args.config and Path(args.config).exists():
        logger.info("Loading configuration from %s", args.config)
        config = load_yaml_config(args.config)

    model_name = config.get("model", {}).get("name_or_path", args.model_name_or_path)
    train_path = config.get("data", {}).get("train_path", args.train_path)
    val_path = config.get("data", {}).get("val_path", args.val_path)
    output_dir = config.get("training", {}).get("output_dir", args.output_dir)
    learning_rate = float(config.get("training", {}).get("learning_rate", args.learning_rate))
    batch_size = int(config.get("training", {}).get("per_device_train_batch_size", args.batch_size))
    num_epochs = int(config.get("training", {}).get("num_train_epochs", args.num_epochs))
    max_steps = int(config.get("training", {}).get("max_steps", args.max_steps))
    max_input_length = int(config.get("model", {}).get("max_input_length", args.max_input_length))
    max_target_length = int(config.get("model", {}).get("max_target_length", args.max_target_length))
    seed = int(config.get("training", {}).get("seed", args.seed))

    logger.info("Initializing FLAN-T5 fine-tuning:")
    logger.info("  Model: %s", model_name)
    logger.info("  Train data: %s", train_path)
    logger.info("  Validation data: %s", val_path)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Batch size: %d, LR: %s, Epochs: %d, Max Steps: %d", batch_size, learning_rate, num_epochs, max_steps)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset_dict = load_clinical_splits(
        train_path=train_path,
        val_path=val_path,
        test_path=None,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    training_args = build_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        seed=seed,
    )

    train_seq2seq_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict.get("validation"),
        training_args=training_args,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()

