"""
Inference interfaces and pipelines for Clinical FLAN-T5.
"""

from .pipeline import CLINICAL_QUESTIONS, ClinicalQAPipeline

__all__ = [
    "CLINICAL_QUESTIONS",
    "ClinicalQAPipeline",
]

