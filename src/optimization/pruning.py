"""
Model pruning routines (unstructured and global magnitude pruning) for FLAN-T5.
"""

import logging
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


def calculate_model_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Calculate global and layer-level parameter sparsity (percentage of zero weights).
    """
    total_zeros = 0
    total_elements = 0
    layer_sparsity = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = getattr(module, "weight", None)
            if weight is not None:
                zeros = int(torch.sum(weight == 0).item())
                elements = int(weight.nelement())
                total_zeros += zeros
                total_elements += elements
                layer_sparsity[name] = float(zeros / elements) if elements > 0 else 0.0

    global_sparsity = float(total_zeros / total_elements) if total_elements > 0 else 0.0

    return {
        "global_sparsity": global_sparsity,
        "total_zeros": total_zeros,
        "total_elements": total_elements,
        "layer_sparsity": layer_sparsity,
    }


def apply_magnitude_pruning(
    model: nn.Module,
    pruning_ratio: float = 0.20,
    method: str = "global",
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Apply magnitude-based L1 weight pruning to all Linear layers in the model.

    Args:
        model: PyTorch model (e.g. AutoModelForSeq2SeqLM)
        pruning_ratio: Fraction of weights to prune (between 0.0 and 1.0)
        method: 'global' for global unstructured pruning across all linear layers,
                or 'local' for layer-by-layer L1 unstructured pruning.

    Returns:
        Tuple of (pruned_model, sparsity_stats)
    """
    if not (0.0 < pruning_ratio < 1.0):
        raise ValueError(f"pruning_ratio must be strictly between 0 and 1, got {pruning_ratio}")

    linear_modules: List[Tuple[nn.Module, str]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append((module, "weight"))

    if not linear_modules:
        logger.warning("No Linear layers found in model for pruning.")
        return model, calculate_model_sparsity(model)

    logger.info("Applying %s L1 pruning (ratio=%.2f) across %d Linear modules...", method, pruning_ratio, len(linear_modules))

    if method == "global":
        prune.global_unstructured(
            parameters=linear_modules,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
    elif method == "local":
        for module, param_name in linear_modules:
            prune.l1_unstructured(module, name=param_name, amount=pruning_ratio)
    else:
        raise ValueError(f"Unknown pruning method: {method}. Choose 'global' or 'local'.")

    sparsity = calculate_model_sparsity(model)
    logger.info("Pruning complete. Global model sparsity: %.2f%%", sparsity["global_sparsity"] * 100)
    return model, sparsity


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Remove pruning reparameterization masks to permanently bake zero weights into tensors.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")
                count += 1

    logger.info("Removed pruning reparameterization masks from %d Linear layers.", count)
    return model

