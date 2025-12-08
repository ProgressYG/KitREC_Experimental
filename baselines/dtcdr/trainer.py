"""
DTCDR Trainer

Multi-task 학습 (source + target domains)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import os
from tqdm import tqdm

from .model import DTCDR
from .data_converter import DTCDRSample


class DTCDRDataset(Dataset):
    """DTCDR PyTorch Dataset"""

    def __init__(
        self,
        samples: List[DTCDRSample],
        num_negatives: int = 4,
        num_source_items: int = 10000,
        num_target_items: int = 10000
    ):
        self.samples = samples
        self.num_negatives = num_negatives
        self.num_source_items = num_source_items
        self.num_target_items = num_target_items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Positive items
        pos_source = sample.source_item_ids[-1] if sample.source_item_ids else 0
        pos_target = sample.ground_truth_id

        # Negative items for target domain
        exclude = set(sample.target_item_ids + [pos_target])
        neg_targets = []
        max_attempts = self.num_negatives * 100  # Prevent infinite loop
        attempts = 0
        while len(neg_targets) < self.num_negatives and attempts < max_attempts:
            neg = torch.randint(1, self.num_target_items, (1,)).item()
            if neg not in exclude:
                neg_targets.append(neg)
                exclude.add(neg)
            attempts += 1

        # Fill with zeros if not enough negatives found (edge case)
        while len(neg_targets) < self.num_negatives:
            neg_targets.append(0)

        return {
            "user_id": torch.tensor(sample.user_id, dtype=torch.long),
            "pos_source_item": torch.tensor(pos_source, dtype=torch.long),
            "pos_target_item": torch.tensor(pos_target, dtype=torch.long),
            "neg_target_items": torch.tensor(neg_targets, dtype=torch.long),
        }


class DTCDRTrainer:
    """DTCDR 학습 관리자"""

    def __init__(
        self,
        model: DTCDR,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        multi_task_weight: float = 0.5,
        orthogonal_weight: float = 0.01,
        device: str = "cuda",
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
    ):
        """
        Args:
            model: DTCDR model
            learning_rate: Learning rate
            weight_decay: Weight decay
            multi_task_weight: Weight for source domain loss (1 - weight for target)
            orthogonal_weight: Weight for orthogonality regularization
            device: Device
            scheduler_patience: LR scheduler patience
            scheduler_factor: LR scheduler reduction factor
        """
        self.model = model.to(device)
        self.device = device
        self.multi_task_weight = multi_task_weight
        self.orthogonal_weight = orthogonal_weight

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler for adaptive learning
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """단일 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        total_source_loss = 0.0
        total_target_loss = 0.0
        total_ortho_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            user_ids = batch["user_id"].to(self.device)
            pos_source = batch["pos_source_item"].to(self.device)
            pos_target = batch["pos_target_item"].to(self.device)
            neg_targets = batch["neg_target_items"].to(self.device)

            # Forward
            source_scores, target_pos_scores = self.model(
                user_ids, pos_source, pos_target
            )

            # Target negative scores
            neg_scores_list = []
            for i in range(neg_targets.size(1)):
                _, neg_score = self.model(user_ids, pos_source, neg_targets[:, i])
                neg_scores_list.append(neg_score)
            target_neg_scores = torch.cat(neg_scores_list, dim=1)

            # Source loss (positive only, simplified)
            source_loss = self.criterion(
                source_scores,
                torch.ones_like(source_scores)
            )

            # Target loss (BPR-like)
            target_pos_loss = self.criterion(
                target_pos_scores,
                torch.ones_like(target_pos_scores)
            )
            target_neg_loss = self.criterion(
                target_neg_scores,
                torch.zeros_like(target_neg_scores)
            )
            target_loss = target_pos_loss + target_neg_loss

            # Orthogonality loss
            ortho_loss = self.model.get_orthogonal_loss()

            # Combined loss
            loss = (
                self.multi_task_weight * source_loss +
                (1 - self.multi_task_weight) * target_loss +
                self.orthogonal_weight * ortho_loss
            )

            # Backward with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_source_loss += source_loss.item()
            total_target_loss += target_loss.item()
            total_ortho_loss += ortho_loss.item()

            pbar.set_postfix({
                "loss": loss.item(),
                "target": target_loss.item()
            })

        n = len(dataloader)
        return {
            "total_loss": total_loss / n,
            "source_loss": total_source_loss / n,
            "target_loss": total_target_loss / n,
            "ortho_loss": total_ortho_loss / n,
        }

    def train(
        self,
        train_samples: List[DTCDRSample],
        val_samples: Optional[List[DTCDRSample]] = None,
        epochs: int = 50,
        batch_size: int = 256,
        num_negatives: int = 4,
        early_stopping_patience: int = 5,
        checkpoint_dir: str = "checkpoints/dtcdr",
    ) -> Dict:
        """전체 학습"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get vocab sizes (handle empty source_item_ids safely)
        source_maxes = [max(s.source_item_ids) for s in train_samples if s.source_item_ids]
        num_source = (max(source_maxes) if source_maxes else 0) + 100
        num_target = max(s.ground_truth_id for s in train_samples) + 100

        train_dataset = DTCDRDataset(
            train_samples, num_negatives, num_source, num_target
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "best_val_loss": float('inf')
        }

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(metrics["total_loss"])

            print(f"Epoch {epoch}: Loss = {metrics['total_loss']:.4f} "
                  f"(Source: {metrics['source_loss']:.4f}, "
                  f"Target: {metrics['target_loss']:.4f})")

            # Validation
            if val_samples:
                val_loss = self._evaluate_loss(val_samples, batch_size, num_source, num_target)
                history["val_loss"].append(val_loss)

                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"          Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}")

                if val_loss < history["best_val_loss"]:
                    history["best_val_loss"] = val_loss
                    history["best_epoch"] = epoch
                    patience_counter = 0
                    self.save_checkpoint(os.path.join(checkpoint_dir, "best_model.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        return history

    def _evaluate_loss(self, samples, batch_size, num_source, num_target):
        """검증 loss"""
        self.model.eval()
        dataset = DTCDRDataset(samples, 4, num_source, num_target)
        loader = DataLoader(dataset, batch_size=batch_size)

        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                user_ids = batch["user_id"].to(self.device)
                pos_source = batch["pos_source_item"].to(self.device)
                pos_target = batch["pos_target_item"].to(self.device)
                neg_targets = batch["neg_target_items"].to(self.device)

                _, target_pos = self.model(user_ids, pos_source, pos_target)

                neg_list = []
                for i in range(neg_targets.size(1)):
                    _, neg_score = self.model(user_ids, pos_source, neg_targets[:, i])
                    neg_list.append(neg_score)
                target_neg = torch.cat(neg_list, dim=1)

                loss = (
                    self.criterion(target_pos, torch.ones_like(target_pos)) +
                    self.criterion(target_neg, torch.zeros_like(target_neg))
                )
                total_loss += loss.item()

        return total_loss / len(loader)

    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint with security validation

        Security Note: weights_only=False is required for optimizer state.
        Only load checkpoints from trusted sources.
        """
        # Path validation - prevent directory traversal attacks
        # Only block ".." patterns, allow any absolute path (RunPod uses /workspace/)
        if ".." in path:
            raise ValueError(f"Suspicious checkpoint path detected: {path}")

        # weights_only=False needed for optimizer_state_dict (contains non-tensor objects)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available (backward compatible)
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
