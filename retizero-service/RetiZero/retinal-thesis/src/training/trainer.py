"""
Training engine with W&B integration.
Handles: mixed precision, gradient accumulation, early stopping,
differential LR, warmup for ViT/hybrids, periodic Grad-CAM logging.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm

import wandb

from src.training.metrics import compute_all_metrics


class Trainer:
    """
    Full training loop with W&B logging.

    Handles:
    - Mixed precision training (FP16)
    - Gradient accumulation for large models / high resolution
    - Differential learning rates (backbone vs head)
    - Linear warmup + cosine annealing schedule
    - Early stopping on validation macro-AUC
    - Periodic Grad-CAM heatmap logging to W&B
    - Branch freezing for hybrid models
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        label_names: List[str],
        device: torch.device,
        gradcam_fn: Optional[callable] = None,
        gradcam_samples: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.label_names = label_names
        self.device = device
        self.gradcam_fn = gradcam_fn
        self.gradcam_samples = gradcam_samples

        # Training config
        train_cfg = config.get("training", {})
        opt_cfg = config.get("optimizer", {})
        sched_cfg = config.get("scheduler", {})

        self.max_epochs = train_cfg.get("max_epochs", 50)
        self.patience = train_cfg.get("early_stopping_patience", 10)
        self.accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.use_amp = train_cfg.get("mixed_precision", True) and device.type == "cuda"

        # Hybrid-specific: branch freezing
        self.freeze_epochs = 0
        model_cfg = config.get("model", {})
        if model_cfg.get("family") == "hybrid":
            hybrid_cfg = model_cfg.get("hybrid", {})
            self.freeze_epochs = hybrid_cfg.get("freeze_branches_epochs", 0)
            self.unfreeze_lr_factor = hybrid_cfg.get("unfreeze_lr_factor", 0.1)

        # ViT-specific: warmup
        vit_cfg = config.get("vit", {})
        self.warmup_epochs = vit_cfg.get("warmup_epochs", 0)

        # Build optimizer with differential LR
        self.optimizer = self._build_optimizer(opt_cfg)

        # Build scheduler
        self.scheduler = self._build_scheduler(sched_cfg)

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Early stopping state
        self.best_metric = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

        # Checkpointing
        ckpt_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")
        self.checkpoint_dir = Path(ckpt_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self, opt_cfg: Dict) -> AdamW:
        """Build AdamW with differential LR for backbone vs head."""
        backbone_lr = opt_cfg.get("lr_backbone", 1e-4)
        head_lr = opt_cfg.get("lr_head", 5e-4)
        weight_decay = opt_cfg.get("weight_decay", 1e-4)

        # ViT override
        vit_cfg = self.config.get("vit", {})
        if vit_cfg:
            weight_decay = vit_cfg.get("weight_decay", weight_decay)

        # Separate backbone and head parameters
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Head layers: 'head', 'fusion', 'classifier', 'cls_token', 'pos_embed'
            if any(kw in name for kw in ["head", "fusion", "classifier", "cls_token", "pos_embed"]):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": head_lr, "name": "head"},
        ]

        return AdamW(param_groups, weight_decay=weight_decay)

    def _build_scheduler(self, sched_cfg: Dict):
        """Build learning rate scheduler with optional warmup."""
        T_0 = sched_cfg.get("T_0", 10)
        T_mult = sched_cfg.get("T_mult", 2)

        cosine = CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=T_mult)

        if self.warmup_epochs > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )

        return cosine

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Returns dict with best metrics and checkpoint path.
        """
        exp_id = self.config.get("experiment_id", "unknown")
        gradcam_cfg = self.config.get("gradcam", {})
        gradcam_interval = gradcam_cfg.get("log_every_n_epochs", 5)

        # Handle hybrid branch freezing
        if self.freeze_epochs > 0:
            self._freeze_branches()
            print(f"[Hybrid] Branches frozen for first {self.freeze_epochs} epochs")

        for epoch in range(1, self.max_epochs + 1):
            # Unfreeze branches after warmup
            if epoch == self.freeze_epochs + 1 and self.freeze_epochs > 0:
                self._unfreeze_branches()
                print(f"[Hybrid] Branches unfrozen at epoch {epoch}")

            # Train one epoch
            train_loss = self._train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_probs, val_labels = self._validate_epoch(val_loader)
            val_metrics = compute_all_metrics(
                val_labels, val_probs, self.label_names
            )

            # Log to W&B
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/macro_auc": val_metrics["macro_auc"],
                "val/micro_auc": val_metrics["micro_auc"],
                "val/f1_macro": val_metrics["f1_macro"],
                "val/f1_micro": val_metrics["f1_micro"],
                "val/precision_macro": val_metrics["precision_macro"],
                "val/recall_macro": val_metrics["recall_macro"],
                "val/specificity_macro": val_metrics["specificity_macro"],
                "val/brier_score": val_metrics["brier_score_macro"],
                "val/ece": val_metrics["ece"],
                "lr/backbone": self.optimizer.param_groups[0]["lr"],
                "lr/head": self.optimizer.param_groups[1]["lr"],
            }

            # Per-label AUC
            for label_name, auc_val in val_metrics.get("per_label_auc", {}).items():
                if not np.isnan(auc_val):
                    log_dict[f"val/auc_{label_name}"] = auc_val

            wandb.log(log_dict, step=epoch)

            # Grad-CAM logging
            if (
                gradcam_cfg.get("enabled", False)
                and self.gradcam_fn is not None
                and epoch % gradcam_interval == 0
            ):
                self._log_gradcam(epoch)

            # Scheduler step
            self.scheduler.step()

            # Early stopping check
            current_metric = val_metrics["macro_auc"]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                # Save best checkpoint
                ckpt_path = self.checkpoint_dir / f"{exp_id}_best.pt"
                self._save_checkpoint(ckpt_path, epoch, val_metrics)
                wandb.run.summary["best_val_macro_auc"] = self.best_metric
                wandb.run.summary["best_epoch"] = epoch
            else:
                self.epochs_no_improve += 1

            print(
                f"Epoch {epoch}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val mAUC: {val_metrics['macro_auc']:.4f} | "
                f"Best: {self.best_metric:.4f} (ep{self.best_epoch})"
            )

            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break

        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "checkpoint_path": str(self.checkpoint_dir / f"{exp_id}_best.pt"),
        }

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> float:
        """Train one epoch with mixed precision and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Train Ep{epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps
            n_batches += 1
            pbar.set_postfix({"loss": total_loss / n_batches})

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Validate and return loss, predictions, labels."""
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []

        for batch in val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()

        n = len(val_loader)
        return (
            total_loss / max(n, 1),
            np.vstack(all_probs),
            np.vstack(all_labels),
        )

    def _log_gradcam(self, epoch: int) -> None:
        """Generate and log Grad-CAM heatmaps to W&B."""
        if self.gradcam_fn is None or self.gradcam_samples is None:
            return
        try:
            images = self.gradcam_fn(
                self.model, self.gradcam_samples, self.device
            )
            wandb.log({"gradcam/samples": images}, step=epoch)
        except Exception as e:
            print(f"Grad-CAM logging failed at epoch {epoch}: {e}")

    def _freeze_branches(self) -> None:
        """Freeze backbone branches (for hybrid warmup)."""
        if hasattr(self.model, "freeze_branches"):
            self.model.freeze_branches()
        elif hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()
        elif hasattr(self.model, "freeze_cnn"):
            self.model.freeze_cnn()

    def _unfreeze_branches(self) -> None:
        """Unfreeze and rebuild optimizer with lower backbone LR."""
        if hasattr(self.model, "unfreeze_branches"):
            self.model.unfreeze_branches()
        elif hasattr(self.model, "unfreeze_backbone"):
            self.model.unfreeze_backbone()
        elif hasattr(self.model, "unfreeze_cnn"):
            self.model.unfreeze_cnn()

        # Rebuild optimizer with reduced backbone LR
        opt_cfg = self.config.get("optimizer", {})
        old_backbone_lr = opt_cfg.get("lr_backbone", 1e-4)
        opt_cfg_copy = dict(opt_cfg)
        opt_cfg_copy["lr_backbone"] = old_backbone_lr * self.unfreeze_lr_factor
        self.optimizer = self._build_optimizer(opt_cfg_copy)

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict,
    ) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            path,
        )
        # Also save as W&B artifact
        artifact = wandb.Artifact(
            name=f"model-{self.config.get('experiment_id', 'unknown')}",
            type="model",
            metadata={"epoch": epoch, "macro_auc": metrics.get("macro_auc", 0)},
        )
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
