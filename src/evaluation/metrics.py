"""
Metric computation functions: ROUGE (1/2/L), BLEU, Exact Match, and generation statistics.
"""

import logging
import re
from typing import Dict, List, Union
import numpy as np

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for consistent metric comparison."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s,]", "", text)
    return text.strip()


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Compute normalized exact match percentage."""
    if not predictions or len(predictions) != len(references):
        return 0.0
    matches = sum(
        1 for p, r in zip(predictions, references) if normalize_text(p) == normalize_text(r)
    )
    return float(matches / len(predictions))


def compute_rouge_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores using evaluate or rouge_score."""
    # Attempt 1: evaluate library
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        return {
            "rouge1": round(float(results.get("rouge1", 0.0)), 4),
            "rouge2": round(float(results.get("rouge2", 0.0)), 4),
            "rougeL": round(float(results.get("rougeL", 0.0)), 4),
            "rougeLsum": round(float(results.get("rougeLsum", 0.0)), 4),
        }
    except Exception as e:
        logger.debug("evaluate.load('rouge') not available or failed: %s. Using rouge_score fallback.", e)

    # Attempt 2: rouge_score library directly
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_list, r2_list, rl_list = [], [], []

        for p, r in zip(predictions, references):
            scores = scorer.score(r, p)
            r1_list.append(scores["rouge1"].fmeasure)
            r2_list.append(scores["rouge2"].fmeasure)
            rl_list.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": round(float(np.mean(r1_list)), 4) if r1_list else 0.0,
            "rouge2": round(float(np.mean(r2_list)), 4) if r2_list else 0.0,
            "rougeL": round(float(np.mean(rl_list)), 4) if rl_list else 0.0,
            "rougeLsum": round(float(np.mean(rl_list)), 4) if rl_list else 0.0,
        }
    except Exception as e:
        logger.warning("ROUGE scoring fallback failed: %s", e)
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score using nltk."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        smoother = SmoothingFunction().method1
        bleu_scores = []
        for p, r in zip(predictions, references):
            ref_tokens = [normalize_text(r).split()]
            pred_tokens = normalize_text(p).split()
            if not pred_tokens:
                bleu_scores.append(0.0)
                continue
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoother)
            bleu_scores.append(score)
        return round(float(np.mean(bleu_scores)), 4) if bleu_scores else 0.0
    except Exception as e:
        logger.warning("BLEU computation failed: %s", e)
        return 0.0


def compute_clinical_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, Union[float, int]]:
    """
    Compute full clinical evaluation metrics across predictions and references.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Predictions count ({len(predictions)}) does not match references count ({len(references)})"
        )

    rouge_scores = compute_rouge_metrics(predictions, references)
    bleu_score = compute_bleu_score(predictions, references)
    exact_match = compute_exact_match(predictions, references)

    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]

    return {
        "num_samples": len(predictions),
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "rougeLsum": rouge_scores["rougeLsum"],
        "bleu": bleu_score,
        "exact_match": round(exact_match, 4),
        "avg_pred_length": round(float(np.mean(pred_lengths)), 2) if pred_lengths else 0.0,
        "avg_ref_length": round(float(np.mean(ref_lengths)), 2) if ref_lengths else 0.0,
    }

