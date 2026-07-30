"""DuEE Top10 联合标注 PyTorch Dataset (Python 3.12+)"""

import logging

import torch
from torch.utils.data import Dataset

from src.data.alignment import align_char_labels_to_subwords
from src.data.loader import load_duee_file

# from transformers import PreTrainedTokenizerFast
from src.type_defs import TokenizerProtocol

logger = logging.getLogger(__name__)


class DuEEJointDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: TokenizerProtocol,
        label2id: dict[str, int],
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.samples = list(load_duee_file(data_path))
        logger.info(f"DuEEJointDataset loaded: {len(self.samples)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        char_labels = sample["char_labels"]

        aligned = align_char_labels_to_subwords(
            text=text,
            char_labels=char_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label2id=self.label2id,
        )

        return {
            "input_ids": torch.tensor(aligned["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(aligned["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned["labels"], dtype=torch.long),
        }
