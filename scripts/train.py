"""DuEE Top10 MVP 训练入口 (Python 3.12+)"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.constants import (
    LABEL2ID_FILE,
    LOCAL_PRETRAINED_MODEL_ROOT,
    MAX_TEXT_LENGTH,
    TOP10_DEV_FILE,
    TOP10_TRAIN_FILE,
)
from src.data.collator import JointBioDataCollator
from src.data.dataset import DuEEJointDataset
from src.models.task_model import EventExtractModel
from src.training.trainer import JointBioTrainer

if TYPE_CHECKING:
    from src.type_defs import TokenizerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_tokenizer(model_id: str) -> tuple["TokenizerProtocol", str]:
    local_model_dir = LOCAL_PRETRAINED_MODEL_ROOT / model_id
    load_candidates = [("local", local_model_dir, True), ("remote", model_id, False)]

    for source, path, local_only in load_candidates:
        try:
            if source == "remote":
                logger.info("Trying HuggingFace remote tokenizer for '%s'...", model_id)
                tokenizer = AutoTokenizer.from_pretrained(path)
                logger.info("Loaded tokenizer from HuggingFace remote model '%s'.", model_id)
                return tokenizer, model_id

            logger.info("Trying local tokenizer path '%s'...", path)
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_only)
            logger.info("Loaded tokenizer from local model directory '%s'.", path)
            return tokenizer, str(path)
        except Exception as exc:
            logger.warning("Failed to load tokenizer from %s (%s): %s", source, path, exc)

    raise RuntimeError(
        "无法从本地目录或 HuggingFace 网络加载 tokenizer。"
        f" 请把模型文件保存到 {local_model_dir}，或者检查 HuggingFace 网络是否可用。"
    )


def main() -> None:
    # 🔥 GPU 可用性检查（放在所有模型加载之前）
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available! Please check:\n"
            "  1. nvidia-smi 是否正常\n"
            "  2. torch 是否为 CUDA 版本: python -c 'import torch; print(torch.version.cuda)'\n"
            "  3. uv 安装时是否指定了 --index-url cu121"
        )
    logger.info(
        f"✅ GPU: {torch.cuda.get_device_name(0)} | "
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # 1. 加载配置
    cfg = OmegaConf.load("configs/train.yaml")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # 2. 加载标签词表 & Tokenizer
    with open(LABEL2ID_FILE, encoding="utf-8") as f:
        label2id: dict[str, int] = json.load(f)
    num_tags = len(label2id)

    # 🔥 构建反向映射 id2label
    id2label: dict[int, str] = {v: k for k, v in label2id.items()}

    model_id = cfg.model.encoder.pretrained_model_name_or_path
    tokenizer, resolved_model_path = load_tokenizer(model_id)
    cfg.model.encoder.pretrained_model_name_or_path = resolved_model_path

    # 3. 构建 Dataset + DataLoader
    train_dataset = DuEEJointDataset(str(TOP10_TRAIN_FILE), tokenizer, label2id, MAX_TEXT_LENGTH)
    val_dataset = DuEEJointDataset(str(TOP10_DEV_FILE), tokenizer, label2id, MAX_TEXT_LENGTH)
    collator = JointBioDataCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # 4. 构建模型
    cfg.model.head.num_tags = num_tags
    model = EventExtractModel.from_config(cfg)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 5. 训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 🔥 传入 id2label
    trainer = JointBioTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device=device,
        id2label=id2label,  # 🔥 新增
        output_dir=cfg.train.output_dir,  # 🔥 补上缺失的参数
    )
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.evaluate()
        # 🔥 增强 epoch 结束日志
        logger.info(
            f"Epoch {epoch} Summary | "
            f"TrainLoss: {train_metrics['train_loss']:.4f} | "
            f"ValLoss: {val_metrics['val_loss']:.4f} | "
            f"ValEntityF1: {val_metrics['entity_f1']:.4f} | "
            f"P: {val_metrics['entity_precision']:.4f} | "
            f"R: {val_metrics['entity_recall']:.4f}"
        )

        # 保存 checkpoint
        ckpt_path = Path(cfg.train.output_dir) / f"epoch_{epoch}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()  # 终端运行命令：uv run python -m scripts.train
