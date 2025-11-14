"""
Model pruning and quantization optimization tools for FLAN-T5.
"""

from .pruning import (
    apply_magnitude_pruning,
    calculate_model_sparsity,
    make_pruning_permanent,
)
from .quantization import (
    apply_dynamic_quantization,
    get_model_size_mb,
)

__all__ = [
    "apply_magnitude_pruning",
    "calculate_model_sparsity",
    "make_pruning_permanent",
    "apply_dynamic_quantization",
    "get_model_size_mb",
]

