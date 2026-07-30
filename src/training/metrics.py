"""Entity-Level F1 评估器 (Python 3.12+)"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    """不可变实体表示，支持 hash/set 操作"""

    type: str
    start: int
    end: int  # exclusive


def decode_bio_tags(labels: list[int], id2label: dict[int, str]) -> set[Entity]:
    """
    将 BIO 标签序列解码为 Entity 集合。
    忽略 -100 (IGNORE_LABEL_ID) 和 O 标签。
    """
    entities: set[Entity] = set()
    current_type: str | None = None
    current_start: int = -1

    for idx, label_id in enumerate(labels):
        label = id2label.get(label_id, "O")

        if label.startswith("B-"):
            # 遇到新的 B- 标签，先闭合前一个实体（如果有）
            if current_type is not None and current_start >= 0:
                entities.add(Entity(type=current_type, start=current_start, end=idx))
            current_type = label[2:]
            current_start = idx

        elif label.startswith("I-") and current_type is not None:
            i_type = label[2:]
            if i_type != current_type:
                # I- 类型与当前 B- 不匹配 → 视为非法，闭合前一个并重置
                entities.add(Entity(type=current_type, start=current_start, end=idx))
                current_type = None
                current_start = -1
            # 否则继续延伸当前实体
        else:
            # O / IGNORE / 非法标签 → 闭合当前实体
            if current_type is not None and current_start >= 0:
                entities.add(Entity(type=current_type, start=current_start, end=idx))
            current_type = None
            current_start = -1

    # 处理末尾未闭合的实体
    if current_type is not None and current_start >= 0:
        entities.add(Entity(type=current_type, start=current_start, end=len(labels)))

    return entities


def compute_entity_f1(
    all_preds: list[list[int]],
    all_labels: list[list[int]],
    id2label: dict[int, str],
) -> dict[str, float]:
    """
    计算 Entity-Level Precision / Recall / F1。
    preds 和 labels 中的 -100 位置在解码时自动被忽略。
    """
    total_tp = 0
    total_pred = 0
    total_gold = 0

    for pred_ids, gold_ids in zip(all_preds, all_labels):
        pred_entities = decode_bio_tags(pred_ids, id2label)
        gold_entities = decode_bio_tags(gold_ids, id2label)

        tp = len(pred_entities & gold_entities)
        total_tp += tp
        total_pred += len(pred_entities)
        total_gold += len(gold_entities)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "entity_precision": round(precision, 4),
        "entity_recall": round(recall, 4),
        "entity_f1": round(f1, 4),
    }
