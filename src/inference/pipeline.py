"""
Unified inference pipeline for single-question QA and full 10-section clinical consultation extraction.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

CLINICAL_QUESTIONS = [
    ("brief_history", "What is brief history of patient?"),
    ("chief_complaints", "What are chief complaints of patient?"),
    ("past_medical_history", "What is past medical history of patient?"),
    ("hospitalization_surgical_history", "What is Hospitalization & Surgical History of patient?"),
    ("gynecological_history", "What is a Patient's Gynec History?"),
    ("lifestyle_social_activity", "What is a patient's Lifestyle & Social activity?"),
    ("family_history", "What is the Patient's Family history?"),
    ("allergies_hypersensitivities", "What are Patient's Allergies & Hypersensitivity?"),
    ("previous_investigations", "What is Patient's Previous Investigation?"),
    ("review_of_systems", "What is the overall Review of the system of patient?"),
]


def _sanitize_tokenizer_config(directory: Union[str, Path]) -> None:
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


class ClinicalQAPipeline:
    """
    Inference pipeline for extracting structured clinical summaries from consultation dialogues.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Path, PreTrainedModel] = "google/flan-t5-base",
        tokenizer_name_or_path: Optional[Union[str, Path, PreTrainedTokenizerBase]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if isinstance(model_name_or_path, PreTrainedModel):
            self.model = model_name_or_path
        else:
            logger.info("Loading model from: %s", model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_name_or_path),
                torch_dtype=torch_dtype,
            )

        self.model.to(self.device)
        self.model.eval()

        tok_path = tokenizer_name_or_path or model_name_or_path
        if isinstance(tok_path, (str, Path)):
            logger.info("Loading tokenizer from: %s", tok_path)
            if Path(tok_path).is_dir():
                _sanitize_tokenizer_config(tok_path)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
            except AttributeError:
                logger.warning("Tokenizer attribute mismatch in %s; falling back to 'google/flan-t5-base'", tok_path)
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        else:
            self.tokenizer = tok_path

    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        max_input_length: int = 512,
        max_new_tokens: int = 150,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Answer a single clinical question given dialogue context."""
        if context and context.strip():
            input_text = f"Question: {question.strip()} Context: {context.strip()}"
        else:
            input_text = f"Question: {question.strip()}"

        raw_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
        )
        if hasattr(raw_inputs, "to"):
            inputs = raw_inputs.to(self.device)
        else:
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}

        start_time = time.time()
        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "do_sample": do_sample,
            }
            if num_beams > 1:
                gen_kwargs["early_stopping"] = True
            if do_sample:
                gen_kwargs["temperature"] = temperature

            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                **gen_kwargs,
            )

        elapsed_sec = time.time() - start_time
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return {
            "question": question,
            "answer": answer,
            "latency_ms": round(elapsed_sec * 1000, 2),
        }

    def extract_full_consultation(
        self,
        context: str,
        max_input_length: int = 512,
        max_new_tokens: int = 150,
        num_beams: int = 4,
    ) -> Dict[str, Any]:
        """
        Extract all 10 clinical sections from a consultation dialogue.
        """
        results: Dict[str, Any] = {}
        total_start = time.time()

        for key, question in CLINICAL_QUESTIONS:
            res = self.answer_question(
                question=question,
                context=context,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            results[key] = res["answer"]

        total_elapsed = time.time() - total_start
        results["_meta"] = {
            "total_latency_ms": round(total_elapsed * 1000, 2),
            "num_questions": len(CLINICAL_QUESTIONS),
            "avg_latency_per_question_ms": round((total_elapsed / len(CLINICAL_QUESTIONS)) * 1000, 2),
        }
        return results
