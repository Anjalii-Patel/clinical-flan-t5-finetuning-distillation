"""
Custom Seq2SeqTrainer for online Knowledge Distillation.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, Seq2SeqTrainer

from .loss import DistillationLoss

logger = logging.getLogger(__name__)


class Seq2SeqDistillationTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer subclass that performs online knowledge distillation from a pre-trained
    or fine-tuned teacher model into a compact student model.
    """

    def __init__(
        self,
        *args,
        teacher_model: PreTrainedModel,
        temperature: float = 2.0,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.distillation_loss_fn = DistillationLoss(
            temperature=temperature,
            alpha=alpha,
            ignore_index=-100,
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Forward passes for student and teacher, computing combined distillation loss.
        """
        labels = inputs.get("labels", None)

        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Align teacher model device with inputs if needed
        input_device = inputs["input_ids"].device
        if next(self.teacher_model.parameters()).device != input_device:
            self.teacher_model.to(input_device)

        # Teacher forward pass with gradients disabled
        with torch.no_grad():
            teacher_inputs = {
                k: v.to(input_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
            }
            teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

        loss = self.distillation_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
        )

        return (loss, student_outputs) if return_outputs else loss

