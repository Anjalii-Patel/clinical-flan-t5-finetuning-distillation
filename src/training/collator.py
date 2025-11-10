"""
Data collator for Seq2Seq tasks ensuring dynamic batch padding and label mask handling.
"""

from typing import Any, Dict, List, Optional
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizerBase

class ClinicalSeq2SeqCollator(DataCollatorForSeq2Seq):
    """
    Data collator that dynamically pads inputs and labels to the maximum length in the batch.
    Replaces padding token ids in labels with -100 so cross-entropy loss ignores them.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[PreTrainedModel] = None,
        padding: bool = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
            return_tensors=return_tensors,
        )
