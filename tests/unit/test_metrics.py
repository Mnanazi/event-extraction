"""Entity-Level F1 单元测试 (Python 3.12+)"""

import pytest
from src.training.metrics import Entity, compute_entity_f1, decode_bio_tags


@pytest.fixture
def id2label() -> dict[int, str]:
    return {
        -100: "IGNORE",
        0: "O",
        1: "B-Trigger_裁员",
        2: "I-Trigger_裁员",
        3: "B-Arg_时间",
        4: "I-Arg_时间",
    }


class TestDecodeBioTags:
    def test_basic_decoding(self, id2label: dict[int, str]) -> None:
        labels = [0, 0, 1, 2, 0, 3, 4, 4, 0]
        entities = decode_bio_tags(labels, id2label)
        assert entities == {
            Entity(type="Trigger_裁员", start=2, end=4),
            Entity(type="Arg_时间", start=5, end=8),
        }

    def test_ignore_labels_skipped(self, id2label: dict[int, str]) -> None:
        labels = [-100, -100, 1, 2, -100, 0]
        entities = decode_bio_tags(labels, id2label)
        assert entities == {Entity(type="Trigger_裁员", start=2, end=4)}

    def test_mismatched_i_tag_closes_entity(self, id2label: dict[int, str]) -> None:
        # B-Trigger 后紧跟 I-Arg（类型不匹配）→ Trigger 在 idx=1 处闭合
        labels = [1, 4, 0]
        entities = decode_bio_tags(labels, id2label)
        assert Entity(type="Trigger_裁员", start=0, end=1) in entities
        # I-Arg 没有对应 B-，不应产生实体
        assert not any(e.type == "Arg_时间" for e in entities)

    def test_entity_at_end_of_sequence(self, id2label: dict[int, str]) -> None:
        labels = [0, 1, 2]
        entities = decode_bio_tags(labels, id2label)
        assert entities == {Entity(type="Trigger_裁员", start=1, end=3)}

    def test_empty_labels(self, id2label: dict[int, str]) -> None:
        assert decode_bio_tags([], id2label) == set()

    def test_all_o_and_ignore(self, id2label: dict[int, str]) -> None:
        labels = [0, -100, 0, 0]
        assert decode_bio_tags(labels, id2label) == set()


class TestComputeEntityF1:
    def test_perfect_match(self, id2label: dict[int, str]) -> None:
        preds = [[0, 1, 2, 0]]
        golds = [[0, 1, 2, 0]]
        metrics = compute_entity_f1(preds, golds, id2label)
        assert metrics["entity_f1"] == 1.0
        assert metrics["entity_precision"] == 1.0
        assert metrics["entity_recall"] == 1.0

    def test_no_overlap(self, id2label: dict[int, str]) -> None:
        preds = [[0, 1, 2, 0]]
        golds = [[0, 3, 4, 0]]
        metrics = compute_entity_f1(preds, golds, id2label)
        assert metrics["entity_f1"] == 0.0

    def test_partial_overlap(self, id2label: dict[int, str]) -> None:
        preds = [[0, 1, 2, 0, 3, 4, 0]]  # Trigger + Arg
        golds = [[0, 1, 2, 0, 0, 0, 0]]  # Only Trigger
        metrics = compute_entity_f1(preds, golds, id2label)
        assert metrics["entity_precision"] == 0.5  # 1/2
        assert metrics["entity_recall"] == 1.0  # 1/1
        assert metrics["entity_f1"] == pytest.approx(2 / 3, abs=0.001)

    def test_empty_predictions(self, id2label: dict[int, str]) -> None:
        preds = [[0, 0, 0]]
        golds = [[0, 1, 2]]
        metrics = compute_entity_f1(preds, golds, id2label)
        assert metrics["entity_precision"] == 0.0
        assert metrics["entity_recall"] == 0.0
        assert metrics["entity_f1"] == 0.0

    def test_both_empty(self, id2label: dict[int, str]) -> None:
        metrics = compute_entity_f1([[0, 0]], [[0, 0]], id2label)
        assert metrics["entity_f1"] == 0.0
