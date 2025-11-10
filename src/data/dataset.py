"""
Dataset loading, prompt formatting, and tokenization utilities for Seq2Seq learning.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def format_prompt(question: str, context: Optional[str] = None) -> str:
    """Format a question and context into a standardized prompt string for FLAN-T5."""
    question = question.strip() if question else ""
    if context and context.strip():
        return f"Question: {question} Context: {context.strip()}"
    return f"Question: {question}"


def prepare_dataset_from_csv(
    csv_path: Union[str, Path],
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int = 512,
    max_target_length: int = 128,
    pad_to_max: bool = False,
) -> Dataset:
    """
    Load a CSV file, standardize column names to ['input', 'target'],
    and tokenize the dataset for Seq2Seq learning.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded CSV from %s with %d rows and columns: %s", csv_path, len(df), list(df.columns))

    # Standardize column structure
    if "input" not in df.columns or "target" not in df.columns:
        if {"Question", "Context", "Answer"}.issubset(df.columns):
            df["input"] = [
                format_prompt(q, c) for q, c in zip(df["Question"], df["Context"])
            ]
            df["target"] = df["Answer"].astype(str)
        elif {"question", "context", "answer"}.issubset(df.columns):
            df["input"] = [
                format_prompt(q, c) for q, c in zip(df["question"], df["context"])
            ]
            df["target"] = df["answer"].astype(str)
        else:
            raise ValueError(
                f"CSV must contain either ['input', 'target'] or ['Question', 'Context', 'Answer']. Found: {list(df.columns)}"
            )

    hf_dataset = Dataset.from_pandas(df[["input", "target"]].dropna())

    def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        padding = "max_length" if pad_to_max else False
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_input_length,
            truncation=True,
            padding=padding,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=max_target_length,
            truncation=True,
            padding=padding,
        )

        # Replace padding token ids in labels with -100 so loss ignores them
        if pad_to_max:
            labels_ids = [
                [
                    label if label != tokenizer.pad_token_id else -100
                    for label in label_seq
                ]
                for label_seq in labels["input_ids"]
            ]
        else:
            labels_ids = labels["input_ids"]

        model_inputs["labels"] = labels_ids
        return model_inputs

    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset",
        remove_columns=["input", "target"],
    )
    return tokenized_dataset


def load_clinical_splits(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    test_path: Optional[Union[str, Path]],
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int = 512,
    max_target_length: int = 128,
    pad_to_max: bool = False,
) -> DatasetDict:
    """Load and tokenize train, validation, and optional test dataset splits."""
    splits = {}
    splits["train"] = prepare_dataset_from_csv(
        train_path, tokenizer, max_input_length, max_target_length, pad_to_max
    )
    splits["validation"] = prepare_dataset_from_csv(
        val_path, tokenizer, max_input_length, max_target_length, pad_to_max
    )
    if test_path and Path(test_path).exists():
        splits["test"] = prepare_dataset_from_csv(
            test_path, tokenizer, max_input_length, max_target_length, pad_to_max
        )
    return DatasetDict(splits)


def split_raw_dataset(
    raw_csv_path: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_by_context: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a raw medical QA CSV into train, validation, and test DataFrames.
    If group_by_context is True, all QA pairs from the same patient dialogue remain in the same split.
    """
    df = pd.read_csv(raw_csv_path)

    if not {"Question", "Context", "Answer"}.issubset(df.columns):
        raise ValueError("Raw CSV must contain ['Question', 'Context', 'Answer'] columns.")

    df["input"] = [format_prompt(q, c) for q, c in zip(df["Question"], df["Context"])]
    df["target"] = df["Answer"].astype(str)

    if group_by_context:
        # Grouped split by Context to prevent context leakage across splits
        gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_idx, temp_idx = next(gss.split(df, groups=df["Context"]))
        train_df = df.iloc[train_idx].copy()
        temp_df = df.iloc[temp_idx].copy()

        # Split remaining into val and test
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, train_size=relative_val_ratio, random_state=seed)
        val_sub_idx, test_sub_idx = next(gss_val.split(temp_df, groups=temp_df["Context"]))
        val_df = temp_df.iloc[val_sub_idx].copy()
        test_df = temp_df.iloc[test_sub_idx].copy()
    else:
        # Standard random split
        train_df, temp_df = train_test_split(
            df, train_size=train_ratio, random_state=seed, shuffle=True
        )
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, train_size=relative_val_ratio, random_state=seed, shuffle=True
        )

    return (
        train_df[["input", "target"]].reset_index(drop=True),
        val_df[["input", "target"]].reset_index(drop=True),
        test_df[["input", "target"]].reset_index(drop=True),
    )


class ClinicalQADataset(TorchDataset):
    """PyTorch Dataset wrapper for pre-tokenized tensors."""

    def __init__(self, tokenized_inputs: Dict[str, torch.Tensor], tokenized_targets: Dict[str, torch.Tensor]):
        self.input_ids = tokenized_inputs["input_ids"]
        self.attention_mask = tokenized_inputs["attention_mask"]
        self.labels = tokenized_targets["input_ids"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

