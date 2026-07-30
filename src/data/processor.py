"""DuEE Top10 筛选 + 联合BIO标签生成 (Python 3.12+)"""

import json
import logging
from collections import Counter
from pathlib import Path

from src.constants import MAX_TEXT_LENGTH, TOP_K_EVENTS
from src.data.loader import load_duee_file

logger = logging.getLogger(__name__)


def discover_top_k_event_types(file_path: str | Path, top_k: int = TOP_K_EVENTS) -> list[str]:
    """统计训练集中事件类型频次，返回 Top-K 高频类型"""
    counter: Counter[str] = Counter()
    for sample in load_duee_file(file_path):
        for event in sample.get("event_list", []):
            counter[event["event_type"]] += 1

    top_types = [t for t, _ in counter.most_common(top_k)]
    logger.info(f"Top-{top_k} event types discovered:")
    for rank, (etype, cnt) in enumerate(counter.most_common(top_k), 1):
        logger.info(f"  {rank:>2}. {etype:<30s} ({cnt})")
    return top_types


def build_joint_bio_labels(
    text: str,
    event_list: list[dict],
    valid_types: set[str],
) -> list[str]:
    """
    构建联合BIO标签序列（字符级）
    - Trigger: B-Trigger_{event_type} / I-Trigger_{event_type}
    - Argument: B-Arg_{role} / I-Arg_{role}
    - O: 其他

    ⚠️ 当触发词与论元位置重叠时，优先保留触发词标签
    """
    labels = ["O"] * len(text)

    # 先标记论元（优先级低）
    for event in event_list:
        if event["event_type"] not in valid_types:
            continue
        for arg in event.get("arguments", []):
            a_start = arg["argument_start_index"]
            a_word = arg["argument"]
            role = arg["role"]
            end = min(a_start + len(a_word), len(labels))
            if a_start < len(labels):
                labels[a_start] = f"B-Arg_{role}"
                for i in range(a_start + 1, end):
                    labels[i] = f"I-Arg_{role}"

    # 再标记触发词（优先级高，覆盖重叠区域）
    for event in event_list:
        etype = event["event_type"]
        if etype not in valid_types:
            continue
        t_start = event["trigger_start_index"]
        t_word = event["trigger"]
        end = min(t_start + len(t_word), len(labels))
        if t_start < len(labels):
            labels[t_start] = f"B-Trigger_{etype}"
            for i in range(t_start + 1, end):
                labels[i] = f"I-Trigger_{etype}"

    return labels


def process_and_save(
    input_path: str | Path,
    output_path: str | Path,
    valid_types: set[str],
) -> dict[str, int]:
    """
    处理单个文件：筛选Top10事件 + 生成联合标签 + 保存为JSONL
    返回统计信息
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "kept": 0, "dropped": 0}
    all_labels: set[str] = set()

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in load_duee_file(input_path):
            stats["total"] += 1
            text = sample["text"][:MAX_TEXT_LENGTH]

            # 检查是否包含至少一个有效事件
            has_valid = any(e["event_type"] in valid_types for e in sample.get("event_list", []))
            if not has_valid:
                stats["dropped"] += 1
                continue

            # 构建联合标签
            char_labels = build_joint_bio_labels(text, sample["event_list"], valid_types)

            # 收集所有出现的标签（用于构建 label2id）
            all_labels.update(char_labels)

            record = {
                "id": sample["id"],
                "text": text,
                "char_labels": char_labels[: len(text)],
                "event_list": [e for e in sample["event_list"] if e["event_type"] in valid_types],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    logger.info(
        f"Processed {input_path.name} → {output_path.name}: "  # type: ignore
        f"{stats['kept']}/{stats['total']} kept, {stats['dropped']} dropped"
    )
    return stats


def build_label_vocab(processed_files: list[str | Path]) -> dict[str, int]:
    """从已处理的JSONL文件中提取完整标签词表"""
    all_labels: set[str] = set()
    for fp in processed_files:
        for sample in load_duee_file(fp):
            all_labels.update(sample["char_labels"])

    # O 固定为 0，其余按字母排序保证确定性
    sorted_labels = sorted(all_labels - {"O"})
    label2id = {"O": 0}
    for idx, label in enumerate(sorted_labels, start=1):
        label2id[label] = idx

    logger.info(f"Label vocabulary built: {len(label2id)} tags")
    return label2id


# ========= DEBUG ENTRY POINT =========
# 仅用于本地调试，提交前删除或通过环境变量控制
# 在项目根路径的终端执行命令：uv run python -m src.data.processor
# ========= DEBUG ENTRY POINT =========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from src.constants import (
        DUEE_DEV_FILE,
        DUEE_TRAIN_FILE,
        LABEL2ID_FILE,
        TOP10_DEV_FILE,
        TOP10_TRAIN_FILE,
        TOP10_TYPES_FILE,
    )

    # Step 1: 发现 Top10 事件类型
    print("=" * 60)
    print("Step 1: Discovering Top-10 Event Types")
    print("=" * 60)
    top10_types = discover_top_k_event_types(DUEE_TRAIN_FILE)

    # 保存 Top10 类型列表
    TOP10_TYPES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOP10_TYPES_FILE, "w", encoding="utf-8") as f:
        json.dump(top10_types, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved top10 types to {TOP10_TYPES_FILE}")

    # Step 2: 处理训练集和验证集
    valid_set = set(top10_types)
    print("\n" + "=" * 60)
    print("Step 2: Processing Train & Dev Sets")
    print("=" * 60)
    train_stats = process_and_save(DUEE_TRAIN_FILE, TOP10_TRAIN_FILE, valid_set)
    dev_stats = process_and_save(DUEE_DEV_FILE, TOP10_DEV_FILE, valid_set)

    # Step 3: 构建标签词表
    print("\n" + "=" * 60)
    print("Step 3: Building Label Vocabulary")
    print("=" * 60)
    label2id = build_label_vocab([TOP10_TRAIN_FILE, TOP10_DEV_FILE])

    with open(LABEL2ID_FILE, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved label2id ({len(label2id)} tags) to {LABEL2ID_FILE}")

    # Step 4: 抽样验证联合标签正确性
    print("\n" + "=" * 60)
    print("Step 4: Spot-check Joint BIO Labels")
    print("=" * 60)
    for i, sample in enumerate(load_duee_file(TOP10_TRAIN_FILE)):
        if i >= 3:
            break
        text = sample["text"]
        labels = sample["char_labels"]
        print(f"\n[Sample {i + 1}] {sample['id']}")
        print(f"  Text:   {text}")
        print(f"  Labels: {''.join(labels)}")
        # 打印非O标签及其对应字符
        non_o = [(c, l) for c, l in zip(text, labels, strict=True) if l != "O"]  # noqa: E741
        if non_o:
            print("  Entities:")
            for char, label in non_o:
                print(f"    '{char}' → {label}")
