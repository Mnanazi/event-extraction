"""Tokenizer对齐单元测试 (Python 3.12+)"""

import pytest
from src.data.alignment import IGNORE_LABEL_ID, align_char_labels_to_subwords
from transformers import AutoTokenizer
from pathlib import Path


@pytest.fixture(scope="module")
def tokenizer() -> AutoTokenizer:
    # 优先使用仓库中的本地 tokenizer，避免在 CI / 离线环境下尝试下载并卡住
    local_dir = Path(__file__).resolve().parents[2] / "pretrained_models" / "hfl" / "chinese-roberta-wwm-ext"
    return AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)


@pytest.fixture
def simple_label2id() -> dict[str, int]:
    return {"O": 0, "B-Trigger_T1": 1, "I-Trigger_T1": 2, "B-Arg_R1": 3}


class TestAlignment:
    def test_all_o_produces_only_zero_and_ignore(
        self, tokenizer: AutoTokenizer, simple_label2id: dict[str, int]
    ) -> None:
        text = "今天天气不错"
        char_labels = ["O"] * len(text)
        result = align_char_labels_to_subwords(text, char_labels, tokenizer, 128, simple_label2id)

        for tok_id, label in zip(result["input_ids"], result["labels"], strict=True):
            token = tokenizer.convert_ids_to_tokens(tok_id)
            if token in ("[CLS]", "[SEP]") or label == IGNORE_LABEL_ID:
                assert label == IGNORE_LABEL_ID
            else:
                assert label == 0

    def test_trigger_aligned_to_first_subword(
        self, tokenizer: AutoTokenizer, simple_label2id: dict[str, int]
    ) -> None:
        text = "公司宣布裁员"
        char_labels = ["O", "O", "O", "O", "B-Trigger_T1", "I-Trigger_T1"]
        result = align_char_labels_to_subwords(text, char_labels, tokenizer, 128, simple_label2id)

        tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
        label_map = dict(zip(tokens, result["labels"], strict=True))

        # "裁" 的首子词应为 B-Trigger_T1
        assert label_map.get("裁") == 1 or any(
            lbl == 1 for tok, lbl in zip(tokens, result["labels"], strict=True) if "裁" in tok
        )

    def test_special_tokens_are_ignored(
        self, tokenizer: AutoTokenizer, simple_label2id: dict[str, int]
    ) -> None:
        text = "测试"
        char_labels = ["O", "O"]
        result = align_char_labels_to_subwords(text, char_labels, tokenizer, 128, simple_label2id)

        tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
        for tok, lbl in zip(tokens, result["labels"], strict=True):
            if tok in ("[CLS]", "[SEP]"):
                assert lbl == IGNORE_LABEL_ID

    def test_truncation_respects_max_length(
        self, tokenizer: AutoTokenizer, simple_label2id: dict[str, int]
    ) -> None:
        text = "abcdefghijklmnop" * 20  # 320 chars
        char_labels = ["O"] * len(text)
        result = align_char_labels_to_subwords(text, char_labels, tokenizer, 32, simple_label2id)

        assert len(result["input_ids"]) <= 32
        assert len(result["labels"]) == len(result["input_ids"])

    def test_text_label_length_mismatch_raises(
        self, tokenizer: AutoTokenizer, simple_label2id: dict[str, int]
    ) -> None:
        with pytest.raises(ValueError, match="Text length"):
            align_char_labels_to_subwords("abc", ["O", "O"], tokenizer, 128, simple_label2id)

    def test_unknown_label_defaults_to_o(
        self, tokenizer: AutoTokenizer, simple_label2id: dict[str, int]
    ) -> None:
        text = "未知标签测试"
        char_labels = ["B-Unknown_Type", "O", "O", "O", "O", "O"]
        result = align_char_labels_to_subwords(text, char_labels, tokenizer, 128, simple_label2id)

        # 未知标签应 fallback 到 O (id=0)，而非崩溃
        non_ignore_labels = [l for l in result["labels"] if l != IGNORE_LABEL_ID]
        assert all(l == 0 for l in non_ignore_labels)


class TestCollator:
    def test_batch_padding_shapes(self) -> None:
        import torch
        from src.data.collator import JointBioDataCollator

        collator = JointBioDataCollator()
        features = [
            {
                "input_ids": torch.tensor([101, 200, 102]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([-100, 1, -100]),
            },
            {
                "input_ids": torch.tensor([101, 300, 400, 102]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "labels": torch.tensor([-100, 0, 2, -100]),
            },
        ]
        batch = collator(features)

        assert batch["input_ids"].shape == (2, 4)
        assert batch["labels"].shape == (2, 4)
        # 短样本的 padding 位置 labels 应为 -100
        assert batch["labels"][0, 3].item() == IGNORE_LABEL_ID
