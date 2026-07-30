"""健壮的数据加载器：自动识别 json/jsonl，容错处理 (Python 3.12+)"""

import json
import logging
from collections.abc import Generator
from pathlib import Path

logger = logging.getLogger(__name__)


def load_duee_file(file_path: str | Path) -> Generator[dict, None, None]:
    """
    自动识别 .json / .jsonl 格式，逐条 yield 样本。
    遇到解析错误时记录日志并跳过，不中断整个流程。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    suffix = file_path.suffix.lower()
    error_count = 0
    total_count = 0

    with open(file_path, encoding="utf-8") as f:
        if suffix == ".json":
            # DuEE train.json 可能是整体列表或 NDJSON
            content = f.read().strip()
            if content.startswith("["):
                try:
                    items = json.loads(content)
                    for item in items:
                        total_count += 1
                        yield item
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON array in {file_path}: {e}")
                    return
            else:
                # 回退到逐行解析
                for line_num, line in enumerate(content.splitlines(), 1):
                    total_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        error_count += 1
                        logger.warning(f"[{file_path.name}:{line_num}] JSON decode error: {e}")

        elif suffix == ".jsonl":
            for line_num, line in enumerate(f, 1):
                total_count += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    error_count += 1
                    logger.warning(f"[{file_path.name}:{line_num}] JSON decode error: {e}")
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Expected .json or .jsonl")

    logger.info(f"Loaded {file_path.name}: {total_count} total, {error_count} errors skipped")


# ========= DEBUG ENTRY POINT =========
# 仅用于本地调试，提交前删除或通过环境变量控制
# 在项目根路径的终端执行命令：uv run python -m src.data.loader
# ========= DEBUG ENTRY POINT =========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.constants import DUEE_TRAIN_FILE

    count = 0
    for sample in load_duee_file(DUEE_TRAIN_FILE):
        count += 1
        if count <= 3:
            print(f"[Sample {count}] id={sample.get('id')}, text={sample['text'][:50]}...")
            print(f"  event_types: {[e['event_type'] for e in sample['event_list']]}")
    print(f"\n✅ Total valid samples loaded: {count}")
