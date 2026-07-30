"""企业级手写训练循环：MLflow + Modern AMP + Resource Monitoring (Python 3.12+)"""

import logging
import time
from pathlib import Path

import mlflow
import psutil
import torch
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast  # 🔥 Modern AMP API
from torch.utils.data import DataLoader

from src.training.metrics import compute_entity_f1

logger = logging.getLogger(__name__)


class JointBioTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
        id2label: dict[int, str],
        output_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.id2label = id2label
        self.output_dir = output_dir

        # 参数分组差异化学习率
        bert_params = list(model.encoder.parameters())
        crf_params = list(model.head.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": bert_params, "lr": cfg.train.bert_lr},
                {"params": crf_params, "lr": cfg.train.crf_lr},
            ],
            weight_decay=cfg.train.weight_decay,
        )

        # 🔥 Modern AMP: 指定 device_type
        self.scaler = GradScaler(device_type="cuda", enabled=cfg.train.fp16)
        self.grad_accum: int = cfg.train.gradient_accumulation_steps
        self.max_grad_norm: float = cfg.train.max_grad_norm

        # 状态追踪
        self._train_start_time: float = time.time()
        self._best_f1: float = 0.0
        self._global_step: int = 0

    # ==================== 资源监控工具 ====================

    @staticmethod
    def _get_resource_usage() -> dict[str, float | str]:
        """获取 CPU/RAM/GPU 实时占用"""
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        ram_gb_used = ram.used / (1024**3)
        ram_gb_total = ram.total / (1024**3)

        info: dict[str, float | str] = {
            "cpu_percent": cpu_percent,
            "ram_used_gb": round(ram_gb_used, 2),
            "ram_total_gb": round(ram_gb_total, 2),
        }

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            info["gpu_used_gb"] = round(allocated, 2)
            info["gpu_total_gb"] = round(total, 2)
            info["gpu_str"] = f"{allocated:.1f}GB/{total:.1f}GB"
        else:
            info["gpu_str"] = "N/A"

        return info

    @staticmethod
    def _format_duration(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"

    # ==================== 训练核心 ====================

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        step_count = 0
        epoch_start = time.time()
        num_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, 1):
            batch_start = time.time()
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            # 🔥 Modern AMP autocast
            with autocast("cuda", dtype=torch.float16, enabled=self.cfg.train.fp16):
                loss = self.model(**batch)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if step % self.grad_accum == 0 or step == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # 🔥 set_to_none=True 更省显存

                self._global_step += 1
                total_loss += loss.item() * self.grad_accum
                step_count += 1

                # 每 10 个有效步打印日志
                if step_count % 10 == 0 or step == num_batches:
                    resources = self._get_resource_usage()
                    elapsed_step = time.time() - batch_start
                    elapsed_epoch = time.time() - epoch_start
                    elapsed_total = time.time() - self._train_start_time
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    current_loss = loss.item() * self.grad_accum

                    # Console 日志
                    logger.info(
                        f"Epoch {epoch} | Step {step}/{num_batches} | "
                        f"LR: {current_lr:.2e} | TrainLoss: {current_loss:.4f} | "
                        f"GPU: {resources['gpu_str']} | "
                        f"CPU: {resources['cpu_percent']:.0f}% | "
                        f"RAM: {resources['ram_used_gb']:.1f}/{resources['ram_total_gb']:.1f}GB | "
                        f"Time: {elapsed_step:.3f}s/step | "
                        f"Epoch Time: {self._format_duration(elapsed_epoch)} | "
                        f"Total Time: {self._format_duration(elapsed_total)}"
                    )

                    # 🔥 MLflow 逐步记录
                    mlflow.log_metrics(
                        {
                            "train/loss": current_loss,
                            "train/lr": current_lr,
                            "system/gpu_used_gb": resources.get("gpu_used_gb", 0),
                            "system/cpu_percent": resources["cpu_percent"],
                            "system/ram_used_gb": resources["ram_used_gb"],
                        },
                        step=self._global_step,
                    )

        avg_loss = total_loss / max(step_count, 1)
        return {"train_loss": round(avg_loss, 4)}

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        all_preds: list[list[int]] = []
        all_labels: list[list[int]] = []

        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with autocast("cuda", dtype=torch.float16, enabled=self.cfg.train.fp16):
                loss = self.model(**batch)
                preds = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None,
                )

            total_loss += loss.item()
            count += 1
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())

        avg_loss = total_loss / max(count, 1)
        f1_metrics = compute_entity_f1(all_preds, all_labels, self.id2label)

        result = {
            "val_loss": round(avg_loss, 4),
            "val_entity_f1": f1_metrics["entity_f1"],
            "val_entity_precision": f1_metrics["entity_precision"],
            "val_entity_recall": f1_metrics["entity_recall"],
        }

        # 🔥 MLflow 记录验证指标
        mlflow.log_metrics(
            {f"val/{k}": v for k, v in result.items()},
            step=self._global_step,
        )

        # 🔥 Best Model 保存策略
        if result["val_entity_f1"] > self._best_f1:
            self._best_f1 = result["val_entity_f1"]
            best_path = self.output_dir / "best_model.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), best_path)
            logger.info(f"✅ New best model saved! ValEntityF1: {self._best_f1:.4f} → {best_path}")
            # MLflow 记录最佳模型 artifact
            mlflow.log_artifact(str(best_path))

        return result
