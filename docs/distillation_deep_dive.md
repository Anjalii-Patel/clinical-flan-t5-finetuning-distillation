# Knowledge Distillation Deep Dive

This document outlines the mathematical foundation and engineering challenges addressed in our FLAN-T5 knowledge distillation implementation.

## 1. Mathematical Formulation

Knowledge distillation (Hinton et al., 2015) transfers dark knowledge from a large teacher model $T$ to a compact student model $S$.

### Soft Target Loss (Kullback-Leibler Divergence)
The teacher and student logits ($z_T, z_S$) are smoothed using a temperature parameter $\tau > 1$:

$$p_i(z, \tau) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

The soft target loss is the KL divergence between student log-probabilities and teacher probabilities, scaled by $\tau^2$ to keep gradient magnitudes consistent across different values of $\tau$:

$$\mathcal{L}_{soft} = \tau^2 \cdot \mathcal{D}_{KL}(p(z_S, \tau) \parallel p(z_T, \tau)) = \tau^2 \sum_{k} p_k(z_T, \tau) \log \left(\frac{p_k(z_T, \tau)}{p_k(z_S, \tau)}\right)$$

### Hard Target Loss (Cross-Entropy)
Standard cross-entropy loss between student predictions and ground-truth token labels $y$:

$$\mathcal{L}_{hard} = -\sum_{k} y_k \log p_k(z_S, 1)$$

### Total Distillation Objective

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{soft} + (1 - \alpha) \cdot \mathcal{L}_{hard}$$

Where:
- $\alpha \in [0, 1]$ balances teacher mimicry against true label fitting.
- $\tau$ controls logit smoothing (empirically tuned between $2.0$ and $5.0$).

## 2. Engineering Challenges & Solutions

| Challenge | Failure Mode | Solution Implemented |
| :--- | :--- | :--- |
| **Logits Memory Footprint** | Offline precomputation of full vocabulary logits ($512 \times 32128$) for 1,000 samples requires ~65 GB RAM, causing fatal `MemoryError`. | Implemented **Online Distillation** (`Seq2SeqDistillationTrainer`) where teacher logits are computed on-the-fly per mini-batch with `torch.no_grad()`. |
| **Label Padding Masking** | Padded label tokens ($pad\_token\_id=0$) skew cross-entropy loss gradients. | Labels are dynamically masked with `ignore_index = -100` in both `ClinicalSeq2SeqCollator` and `DistillationLoss`. |
| **KL Divergence Direction** | Reversing arguments in PyTorch `F.kl_div(input, target)` produces erroneous gradients because PyTorch expects log-probabilities as `input` and probabilities as `target`. | Passed `F.log_softmax(student_logits / T)` as input and `F.softmax(teacher_logits / T)` as target. |

