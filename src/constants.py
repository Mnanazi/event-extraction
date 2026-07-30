"""全局常量与路径管理 (Python 3.12+)"""

from pathlib import Path

# ==================== 项目根目录 ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ==================== 本地下载的预训练模型保存路径 ====================
LOCAL_PRETRAINED_MODEL_ROOT = Path("pretrained_models")

# ==================== DuEE 1.0 原始文件路径 ====================
DUEE_TRAIN_FILE = RAW_DATA_DIR / "train.json"
DUEE_DEV_FILE = RAW_DATA_DIR / "dev.json"
DUEE_TEST_FILE = RAW_DATA_DIR / "test1.json"  # DuEE 1.0 测试集通常为 test1.json

# ==================== MVP 输出路径 ====================
TOP10_TRAIN_FILE = PROCESSED_DATA_DIR / "top10_train.jsonl"
TOP10_DEV_FILE = PROCESSED_DATA_DIR / "top10_dev.jsonl"
LABEL2ID_FILE = PROCESSED_DATA_DIR / "label2id.json"
TOP10_TYPES_FILE = PROCESSED_DATA_DIR / "top10_event_types.json"

# ==================== 数据处理常量 ====================
MAX_TEXT_LENGTH = 128  # GTX1650 4GB显存安全上限
PAD_LABEL_ID = -100  # CrossEntropy/CRF ignore_index
TOP_K_EVENTS = 10  # MVP 高频事件数量
