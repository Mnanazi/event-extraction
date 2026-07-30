"""企业级训练循环：MLflow + 资源监控 + 增强日志 (Python 3.12+)"""

import logging
import time

import mlflow
import psutil
import torch
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
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
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.id2label = id2label

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

        self.scaler = GradScaler(enabled=cfg.train.fp16)
        self.grad_accum: int = cfg.train.gradient_accumulation_steps
        self.max_grad_norm: float = cfg.train.max_grad_norm

        # 🔥 全局计时器
        self._run_start_time: float = time.time()

    # ─── 资源监控工具方法 ───────────────────────────────

    def _get_resource_stats(self) -> dict[str, float]:
        """获取当前 GPU + CPU 资源占用"""
        stats: dict[str, float] = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            stats["gpu_allocated_gb"] = round(allocated, 2)
            stats["gpu_total_gb"] = round(total, 2)
            stats["gpu_utilization"] = torch.cuda.utilization(self.device)
        return stats

    def _format_resource_str(self, stats: dict[str, float]) -> str:
        """格式化资源字符串用于控制台日志"""
        parts = [f"CPU: {stats['cpu_percent']:.0f}%"]
        parts.append(f"RAM: {stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f}GB")
        if "gpu_allocated_gb" in stats:
            parts.append(
                f"GPU: {stats['gpu_allocated_gb']:.1f}/{stats['gpu_total_gb']:.1f}GB"
                f"({stats['gpu_utilization']}%)"
            )
        return " | ".join(parts)

    def _get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ─── 训练主循环 ─────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        step_count = 0
        epoch_start = time.time()
        num_batches = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, 1):
            batch_start = time.time()
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast("cuda", dtype=torch.float16, enabled=self.cfg.train.fp16):
                loss = self.model(**batch)
                loss = loss / self.grad_accum

            self.scaler.scale(loss).backward()

            if step % self.grad_accum == 0 or step == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                current_loss = loss.item() * self.grad_accum
                total_loss += current_loss
                step_count += 1

                # 🔥 每 10 个有效步打印 + 上报 MLflow
                if step_count % 10 == 0 or step == num_batches:
                    elapsed_step = time.time() - batch_start
                    elapsed_epoch = time.time() - epoch_start
                    elapsed_total = time.time() - self._run_start_time
                    resource_stats = self._get_resource_stats()

                    # 控制台日志
                    logger.info(
                        f"Epoch {epoch} | Step {step}/{num_batches} | "
                        f"LR: {self._get_current_lr():.2e} | "
                        f"TrainLoss: {current_loss:.4f} | "
                        f"{self._format_resource_str(resource_stats)} | "
                        f"StepTime: {elapsed_step:.3f}s | "
                        f"EpochTime: {int(elapsed_epoch // 60)}m{int(elapsed_epoch % 60):02d}s | "
                        f"TotalTime: {int(elapsed_total // 60)}m{int(elapsed_total % 60):02d}s"
                    )

                    # 🔥 MLflow 逐步指标上报
                    global_step = (epoch - 1) * num_batches + step
                    mlflow.log_metrics(
                        {
                            "train/loss": round(current_loss, 4),
                            "train/lr": self._get_current_lr(),
                            "train/step_time_s": round(elapsed_step, 3),
                            **{f"system/{k}": v for k, v in resource_stats.items()},
                        },
                        step=global_step,
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

        eval_start = time.time()
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
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

        eval_time = time.time() - eval_start
        avg_loss = total_loss / max(count, 1)
        f1_metrics = compute_entity_f1(all_preds, all_labels, self.id2label)

        result = {
            "val_loss": round(avg_loss, 4),
            "val_eval_time_s": round(eval_time, 3),
            **f1_metrics,
        }

        # 🔥 MLflow Epoch 级验证指标上报
        mlflow.log_metrics(
            {f"val/{k}": v for k, v in result.items()},
            step=epoch,
        )
        return result
