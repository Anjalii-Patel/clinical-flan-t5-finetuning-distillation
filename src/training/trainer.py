"""
High-level trainer interface for FLAN-T5 fine-tuning.
"""

import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .collator import ClinicalSeq2SeqCollator

logger = logging.getLogger(__name__)


def build_training_args(
    output_dir: Union[str, Path] = "./models/flan_t5_finetuned",
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    num_train_epochs: int = 3,
    max_steps: int = -1,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    logging_steps: int = 20,
    eval_steps: int = 100,
    save_steps: int = 100,
    save_total_limit: int = 3,
    fp16: Optional[bool] = None,
    bf16: Optional[bool] = None,
    seed: int = 42,
    predict_with_generate: bool = True,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    report_to: str = "none",
) -> Seq2SeqTrainingArguments:
    """Construct Seq2SeqTrainingArguments with portable defaults."""
    output_dir = str(output_dir)
    has_cuda = torch.cuda.is_available()

    if fp16 is None:
        fp16 = has_cuda and not torch.cuda.is_bf16_supported()
    if bf16 is None:
        bf16 = has_cuda and torch.cuda.is_bf16_supported()

    # If CPU only, ensure fp16/bf16 are disabled
    if not has_cuda:
        fp16 = False
        bf16 = False

    eval_strategy_val = "steps" if eval_steps > 0 else "epoch"
    save_strategy_val = "steps" if save_steps > 0 else "epoch"

    candidates: Dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "logging_dir": f"{output_dir}/logs",
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "eval_strategy": eval_strategy_val,
        "evaluation_strategy": eval_strategy_val,
        "eval_steps": eval_steps if eval_strategy_val == "steps" else None,
        "save_strategy": save_strategy_val,
        "save_steps": save_steps if save_strategy_val == "steps" else None,
        "save_total_limit": save_total_limit,
        "predict_with_generate": predict_with_generate,
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "fp16": fp16,
        "bf16": bf16,
        "seed": seed,
        "report_to": report_to,
    }

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    accepted = set(sig.parameters.keys())

    # Dynamically filter parameters accepted by the installed transformers version
    final_args = {k: v for k, v in candidates.items() if k in accepted}

    # If modern eval_strategy is accepted, do not pass legacy evaluation_strategy
    if "eval_strategy" in accepted and "evaluation_strategy" in final_args:
        del final_args["evaluation_strategy"]

    return Seq2SeqTrainingArguments(**final_args)


def sanitize_tokenizer_config(directory: Union[str, Path]) -> None:
    """Fix bug in transformers where extra_special_tokens is saved as a list, causing AttributeError on reload."""
    import json
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


def train_seq2seq_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    training_args: Optional[Seq2SeqTrainingArguments] = None,
    compute_metrics_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
    save_final_model_dir: Optional[Union[str, Path]] = None,
    resume_from_checkpoint: Optional[Union[str, bool]] = None,
) -> Seq2SeqTrainer:
    """Run supervised fine-tuning using Seq2SeqTrainer."""
    if training_args is None:
        training_args = build_training_args()

    collator = ClinicalSeq2SeqCollator(tokenizer=tokenizer, model=model)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "compute_metrics": compute_metrics_fn,
    }

    # Provide tokenizer / processing_class depending on transformers version
    try:
        trainer = Seq2SeqTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Seq2SeqTrainer(tokenizer=tokenizer, **trainer_kwargs)

    logger.info("Starting FLAN-T5 training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info("Training complete. Global steps: %d", train_result.global_step)

    save_dir = save_final_model_dir or training_args.output_dir
    logger.info("Saving final model to: %s", save_dir)
    # Ensure FLAN-T5 keeps tie_word_embeddings=False in saved config
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    sanitize_tokenizer_config(save_dir)

    return trainer

