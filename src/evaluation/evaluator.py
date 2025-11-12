"""
Evaluator harness for running batch generation, metrics aggregation, and comparative model auditing.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .metrics import compute_clinical_metrics

logger = logging.getLogger(__name__)


class ClinicalModelEvaluator:
    """
    Evaluator that executes text generation, tracks latency, and aggregates
    clinical QA performance metrics.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        max_samples: Optional[int] = None,
        max_input_length: int = 512,
        max_new_tokens: int = 128,
        num_beams: int = 2,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a pandas DataFrame with ['input', 'target'] or ['Question', 'Context', 'Answer'].
        """
        if "input" not in df.columns or "target" not in df.columns:
            if {"Question", "Context", "Answer"}.issubset(df.columns):
                df = df.copy()
                df["input"] = [
                    f"Question: {q} Context: {c}" for q, c in zip(df["Question"], df["Context"])
                ]
                df["target"] = df["Answer"].astype(str)
            else:
                raise ValueError("DataFrame must contain ['input', 'target'] columns.")

        if max_samples and max_samples < len(df):
            df = df.iloc[:max_samples].copy()

        inputs_list = df["input"].tolist()
        references_list = df["target"].tolist()

        predictions = []
        latencies = []

        total_start = time.time()

        for i in range(0, len(inputs_list), batch_size):
            batch_inputs = inputs_list[i : i + batch_size]
            raw_encoded = self.tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            )
            if hasattr(raw_encoded, "to"):
                encoded = raw_encoded.to(self.device)
            else:
                encoded = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in raw_encoded.items()}

            batch_start = time.time()
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=(num_beams > 1),
                )
            batch_time = time.time() - batch_start
            batch_latency = (batch_time / len(batch_inputs)) * 1000  # ms per sample

            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend([d.strip() for d in decoded])
            latencies.extend([batch_latency] * len(batch_inputs))

        total_elapsed = time.time() - total_start
        avg_latency_ms = (total_elapsed / len(inputs_list)) * 1000 if inputs_list else 0.0

        metrics = compute_clinical_metrics(predictions, references_list)
        metrics["avg_latency_ms"] = round(avg_latency_ms, 2)
        metrics["total_eval_time_sec"] = round(total_elapsed, 2)

        results_records = [
            {
                "input": inp,
                "reference": ref,
                "prediction": pred,
                "latency_ms": round(lat, 2),
            }
            for inp, ref, pred, lat in zip(inputs_list, references_list, predictions, latencies)
        ]

        return {
            "metrics": metrics,
            "sample_results": results_records,
        }

    def evaluate_csv(
        self,
        csv_path: Union[str, Path],
        output_results_json: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        max_input_length: int = 512,
        max_new_tokens: int = 128,
        num_beams: int = 2,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """Load a CSV and compute evaluation metrics."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Evaluation CSV not found at: {csv_path}")

        df = pd.read_csv(csv_path)
        eval_output = self.evaluate_dataframe(
            df=df,
            max_samples=max_samples,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            batch_size=batch_size,
        )

        if output_results_json:
            out_path = Path(output_results_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(eval_output, f, indent=2)
            logger.info("Saved evaluation results to %s", out_path)

        return eval_output


def compare_models(
    models_dict: Dict[str, Dict[str, Any]],
    test_csv_path: Union[str, Path],
    max_samples: Optional[int] = 50,
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side on the same evaluation dataset.

    Args:
        models_dict: Dict mapping model_label -> {"model": PreTrainedModel, "tokenizer": PreTrainedTokenizer}
        test_csv_path: Path to evaluation CSV file
        max_samples: Sample limit for quick comparison
    """
    df = pd.read_csv(test_csv_path)
    summary_rows = []

    for label, comp in models_dict.items():
        logger.info("Evaluating model variant: %s...", label)
        evaluator = ClinicalModelEvaluator(
            model=comp["model"],
            tokenizer=comp["tokenizer"],
        )
        res = evaluator.evaluate_dataframe(df=df, max_samples=max_samples)
        metrics = res["metrics"]
        summary_rows.append(
            {
                "Model Variant": label,
                "ROUGE-1": metrics["rouge1"],
                "ROUGE-2": metrics["rouge2"],
                "ROUGE-L": metrics["rougeL"],
                "BLEU": metrics["bleu"],
                "Exact Match": metrics["exact_match"],
                "Avg Latency (ms)": metrics["avg_latency_ms"],
            }
        )

    return pd.DataFrame(summary_rows)
