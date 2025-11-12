"""
CLI entry point for Knowledge Distillation from a Teacher model to a Student model.
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

from src.data.dataset import load_clinical_splits
from src.distillation.distillation_trainer import Seq2SeqDistillationTrainer
from src.training.collator import ClinicalSeq2SeqCollator
from src.training.trainer import build_training_args, sanitize_tokenizer_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("distill")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge Distillation for FLAN-T5.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--teacher_model", type=str, default="google/flan-t5-large", help="Teacher model identifier.")
    parser.add_argument("--student_model", type=str, default="google/flan-t5-base", help="Student model identifier.")
    parser.add_argument("--train_path", type=str, default="data/processed/train.csv", help="Path to train CSV.")
    parser.add_argument("--val_path", type=str, default="data/processed/val.csv", help="Path to validation CSV.")
    parser.add_argument("--output_dir", type=str, default="./models/flan_t5_distilled", help="Directory to save distilled student.")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation softmax temperature.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of soft loss (1-alpha for CE loss).")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device training batch size.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 for full epochs).")
    parser.add_argument("--max_input_length", type=int, default=512, help="Max input sequence length.")
    parser.add_argument("--max_target_length", type=int, default=128, help="Max target sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint to resume from.")
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    config = {}
    if args.config and Path(args.config).exists():
        logger.info("Loading distillation configuration from %s", args.config)
        config = load_yaml_config(args.config)

    teacher_name = config.get("teacher", {}).get("name_or_path", args.teacher_model)
    student_name = config.get("student", {}).get("name_or_path", args.student_model)
    train_path = config.get("data", {}).get("train_path", args.train_path)
    val_path = config.get("data", {}).get("val_path", args.val_path)
    output_dir = config.get("training", {}).get("output_dir", args.output_dir)
    temperature = float(config.get("distillation", {}).get("temperature", args.temperature))
    alpha = float(config.get("distillation", {}).get("alpha", args.alpha))
    learning_rate = float(config.get("training", {}).get("learning_rate", args.learning_rate))
    batch_size = int(config.get("training", {}).get("per_device_train_batch_size", args.batch_size))
    num_epochs = int(config.get("training", {}).get("num_train_epochs", args.num_epochs))
    max_steps = int(config.get("training", {}).get("max_steps", args.max_steps))
    max_input_length = int(config.get("student", {}).get("max_input_length", args.max_input_length))
    max_target_length = int(config.get("student", {}).get("max_target_length", args.max_target_length))
    seed = int(config.get("training", {}).get("seed", args.seed))

    logger.info("Initializing Knowledge Distillation:")
    logger.info("  Teacher: %s", teacher_name)
    logger.info("  Student: %s", student_name)
    logger.info("  Temperature: %.2f, Alpha: %.2f", temperature, alpha)
    logger.info("  Output dir: %s", output_dir)

    student_tokenizer = AutoTokenizer.from_pretrained(student_name)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(student_name)
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_name)

    dataset_dict = load_clinical_splits(
        train_path=train_path,
        val_path=val_path,
        test_path=None,
        tokenizer=student_tokenizer,
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

    collator = ClinicalSeq2SeqCollator(tokenizer=student_tokenizer, model=student_model)

    trainer_kwargs = {
        "model": student_model,
        "teacher_model": teacher_model,
        "temperature": temperature,
        "alpha": alpha,
        "args": training_args,
        "train_dataset": dataset_dict["train"],
        "eval_dataset": dataset_dict.get("validation"),
        "data_collator": collator,
    }

    try:
        trainer = Seq2SeqDistillationTrainer(processing_class=student_tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Seq2SeqDistillationTrainer(tokenizer=student_tokenizer, **trainer_kwargs)

    logger.info("Starting student distillation training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving distilled student model to %s", output_dir)
    if hasattr(student_model, "config"):
        student_model.config.tie_word_embeddings = False
    trainer.save_model(output_dir)
    student_tokenizer.save_pretrained(output_dir)
    sanitize_tokenizer_config(output_dir)
    logger.info("Distillation completed successfully.")


if __name__ == "__main__":
    main()

