"""
Knowledge Distillation modules for Seq2Seq transformers.
"""

from .distillation_trainer import Seq2SeqDistillationTrainer
from .loss import DistillationLoss

__all__ = [
    "DistillationLoss",
    "Seq2SeqDistillationTrainer",
]

