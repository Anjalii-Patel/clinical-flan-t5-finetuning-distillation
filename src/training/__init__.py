"""
Training components for FLAN-T5 fine-tuning.
"""

from .collator import ClinicalSeq2SeqCollator
from .trainer import build_training_args, train_seq2seq_model

__all__ = [
    "ClinicalSeq2SeqCollator",
    "build_training_args",
    "train_seq2seq_model",
]

