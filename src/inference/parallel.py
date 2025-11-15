"""
Parallel inference utilities for high-throughput consultation extraction.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from .pipeline import CLINICAL_QUESTIONS, ClinicalQAPipeline

logger = logging.getLogger(__name__)


def batch_extract_consultation(
    pipeline: ClinicalQAPipeline,
    context: str,
    max_input_length: int = 512,
    max_new_tokens: int = 150,
    num_beams: int = 2,
    batch_size: int = 5,
) -> Dict[str, Any]:
    """
    Batched inference across all 10 clinical questions for high throughput.
    """
    keys = [k for k, _ in CLINICAL_QUESTIONS]
    questions = [q for _, q in CLINICAL_QUESTIONS]
    formatted_prompts = [f"Question: {q} Context: {context.strip()}" for q in questions]

    answers = []
    total_start = time.time()

    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i : i + batch_size]
        inputs = pipeline.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        ).to(pipeline.device)

        with torch_inference_mode():
            outputs = pipeline.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )
        decoded = pipeline.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers.extend([d.strip() for d in decoded])

    total_elapsed = time.time() - total_start
    result = dict(zip(keys, answers))
    result["_meta"] = {
        "total_latency_ms": round(total_elapsed * 1000, 2),
        "avg_latency_per_question_ms": round((total_elapsed / len(questions)) * 1000, 2),
    }
    return result


def torch_inference_mode():
    import torch
    return torch.inference_mode()

