"""Subword ↔ Char 标签对齐工具 (Python 3.12+)"""

import logging

# from transformers import PreTrainedTokenizerFast
from src.type_defs import TokenizerProtocol

logger = logging.getLogger(__name__)

# CRF / CrossEntropy 忽略索引
IGNORE_LABEL_ID = -100


def align_char_labels_to_subwords(
    text: str,
    char_labels: list[str],
    tokenizer: TokenizerProtocol,
    max_length: int,
    label2id: dict[str, int],
) -> dict[str, list[int]]:
    """
    将字符级联合BIO标签对齐到BERT子词级别。

    策略：首子词承载标签，其余子词/Special/Padding → IGNORE_LABEL_ID

    Returns:
        dict with keys: input_ids, attention_mask, labels
    """
    if len(text) != len(char_labels):
        raise ValueError(f"Text length ({len(text)}) != char_labels length ({len(char_labels)})")

    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_offsets_mapping=True,
    )

    offset_mapping = encoding["offset_mapping"]
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    aligned_labels = [IGNORE_LABEL_ID] * len(input_ids)

    # 更稳健的对齐逻辑：使用 seen_char_indices 跟踪已标注的字符起始位置，
    # 保证每个字符的第一个子词被标注一次（避免因为 offset 精度或相邻边界导致遗漏）。
    seen_char_starts: set[int] = set()
    for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
        # Special tokens ([CLS], [SEP], [PAD]) 的 offset 为 (0, 0)
        if char_start == char_end == 0:
            continue

        if char_start >= len(char_labels):
            continue

        # 如果这个字符的首子词还未被标注，则使用该字符的标签
        if char_start not in seen_char_starts:
            label_str = char_labels[char_start]
            aligned_labels[tok_idx] = label2id.get(label_str, 0)
            seen_char_starts.add(char_start)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": aligned_labels,
    }


if __name__ == "__main__":
    import json

    from transformers import AutoTokenizer

    from src.constants import LABEL2ID_FILE

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    with open(LABEL2ID_FILE, encoding="utf-8") as f:
        label2id = json.load(f)

    text = "消失的外企光环5月份在华裁员900余人"
    char_labels = ["O"] * len(text)
    char_labels[15] = "B-Trigger_组织关系-裁员"
    char_labels[16] = "I-Trigger_组织关系-裁员"
    char_labels[10] = "B-Arg_时间"

    result = align_char_labels_to_subwords(text, char_labels, tokenizer, 128, label2id)

    tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
    print("Token".ljust(12), "Label")
    print("-" * 40)
    for tok, lbl in zip(tokens, result["labels"], strict=True):
        label_name = next((k for k, v in label2id.items() if v == lbl), "IGNORE")
        print(f"{tok:<12s} {label_name}")
