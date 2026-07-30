"""动态Padding Collator (Python 3.12+)"""

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.alignment import IGNORE_LABEL_ID


class JointBioDataCollator:
    """
    动态 batch padding collator。
    labels 使用 IGNORE_LABEL_ID (-100) 填充，确保 CRF/Loss 忽略 padding 位置。
    """

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_LABEL_ID),
        }
        return batch
