"""
Model quantization utilities for FLAN-T5.
"""

import io
import logging
from typing import Dict, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_model_size_mb(model: nn.Module) -> Dict[str, float]:
    """
    Calculate the memory footprint and parameter count of a PyTorch model.
    """
    param_size_bytes = 0
    buffer_size_bytes = 0
    total_params = 0

    for param in model.parameters():
        param_size_bytes += param.nelement() * param.element_size()
        total_params += param.nelement()

    for buffer in model.buffers():
        buffer_size_bytes += buffer.nelement() * buffer.element_size()

    total_mb = (param_size_bytes + buffer_size_bytes) / (1024 * 1024)

    return {
        "model_size_mb": round(total_mb, 2),
        "total_parameters": total_params,
        "param_size_mb": round(param_size_bytes / (1024 * 1024), 2),
    }


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply Post-Training Dynamic Quantization (INT8) to the model's Linear layers.
    Optimizes CPU memory usage and inference latency for transformer architectures.
    """
    logger.info("Applying PyTorch dynamic quantization (INT8) to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    logger.info("Dynamic quantization applied successfully.")
    return quantized_model

