"""
Loss functions for sequence-to-sequence knowledge distillation.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Combined Knowledge Distillation loss function:
    Total Loss = alpha * (T^2 * KL(P_student || P_teacher)) + (1 - alpha) * CrossEntropy(student, labels)

    Args:
        temperature (float): Softmax temperature scaling factor. Defaults to 2.0.
        alpha (float): Weight assigned to the distillation (KL) loss vs hard CE loss. Defaults to 0.5.
        ignore_index (int): Label index to ignore in cross-entropy calculation (standard padding index is -100).
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the combined distillation loss.

        Args:
            student_logits: Tensor of shape (batch_size, seq_len, vocab_size)
            teacher_logits: Tensor of shape (batch_size, seq_len, vocab_size)
            labels: Ground truth token labels of shape (batch_size, seq_len)
        """
        # Ensure identical sequence length between teacher and student logits if minor mismatch
        min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_seq_len, :].contiguous()
        teacher_logits = teacher_logits[:, :min_seq_len, :].contiguous()

        # Compute KL Divergence in log space for student and probability space for teacher
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        if labels is not None:
            labels_for_mask = labels[:, :min_seq_len].contiguous() if labels.size(1) >= min_seq_len else labels
            # Mask out padding positions in distillation loss
            mask = (labels_for_mask != self.ignore_index).unsqueeze(-1)
            kl_pointwise = F.kl_div(student_log_probs, teacher_probs, reduction="none") * (self.temperature ** 2)
            valid_tokens = mask.sum().clamp(min=1)
            kl_loss = (kl_pointwise * mask).sum() / valid_tokens
        else:
            kl_loss = self.kl_loss_fn(student_log_probs, teacher_probs) * (self.temperature ** 2)

        if self.alpha == 1.0 or labels is None:
            return kl_loss

        # Compute standard cross-entropy loss against ground truth tokens
        vocab_size = student_logits.size(-1)
        if labels.size(1) > min_seq_len:
            labels = labels[:, :min_seq_len].contiguous()

        ce_loss = F.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=self.ignore_index,
        )

        total_loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
        return total_loss

