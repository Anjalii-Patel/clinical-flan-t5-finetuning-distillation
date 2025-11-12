"""
Evaluation metrics and batch evaluation harness for Clinical Seq2Seq models.
"""

from .evaluator import ClinicalModelEvaluator, compare_models
from .metrics import compute_clinical_metrics

__all__ = [
    "ClinicalModelEvaluator",
    "compare_models",
    "compute_clinical_metrics",
]

