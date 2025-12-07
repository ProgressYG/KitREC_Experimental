"""
CoNet Trainer

학습 루프 및 체크포인트 관리
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import os
from tqdm import tqdm

from .model import CoNet
from .data_converter import CoNetSample


class CoNetDataset(Dataset):
    """CoNet PyTorch Dataset"""

    def __init__(
        self,
        samples: List[CoNetSample],
        num_negatives: int = 4,
        num_target_items: int = 10000
    ):
        """
        Args:
            samples: CoNetSample 리스트
            num_negatives: 네거티브 샘플 수
            num_target_items: 전체 타겟 아이템 수 (네거티브 샘플링용)
        """
        self.samples = samples
        self.num_negatives = num_negatives
        self.num_target_items = num_target_items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Positive sample: ground truth item
        pos_item = sample.ground_truth_id

        # Negative samples: random items (excluding GT and history)
        exclude = set(sample.target_item_ids + [pos_item])
        neg_items = []
        max_attempts = self.num_negatives * 100  # Prevent infinite loop
        attempts = 0
        while len(neg_items) < self.num_negatives and attempts < max_attempts:
            neg = torch.randint(1, self.num_target_items, (1,)).item()
            if neg not in exclude:
                neg_items.append(neg)
                exclude.add(neg)
            attempts += 1

        # Fill with zeros if not enough negatives found (edge case)
        while len(neg_items) < self.num_negatives:
            neg_items.append(0)

        # Use last source item as representative (or padding)
        source_item = sample.source_item_ids[-1] if sample.source_item_ids else 0

        return {
            "user_id": torch.tensor(sample.user_id, dtype=torch.long),
            "source_item_id": torch.tensor(source_item, dtype=torch.long),
            "pos_target_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_target_items": torch.tensor(neg_items, dtype=torch.long),
        }


class CoNetTrainer:
    """CoNet 학습 관리자"""

    def __init__(
        self,
        model: CoNet,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "cuda",
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
    ):
        self.model = model.to(device)
        self.device = device

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
    ) -> float:
        """단일 에포크 학습"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            user_ids = batch["user_id"].to(self.device)
            source_items = batch["source_item_id"].to(self.device)
            pos_items = batch["pos_target_item"].to(self.device)
            neg_items = batch["neg_target_items"].to(self.device)

            batch_size = user_ids.size(0)
            num_neg = neg_items.size(1)

            # Positive scores
            _, pos_scores = self.model(user_ids, source_items, pos_items)

            # Negative scores
            neg_scores_list = []
            for i in range(num_neg):
                _, neg_score = self.model(user_ids, source_items, neg_items[:, i])
                neg_scores_list.append(neg_score)
            neg_scores = torch.cat(neg_scores_list, dim=1)

            # BPR-like loss: positive should score higher than negatives
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)

            pos_loss = self.criterion(pos_scores, pos_labels)
            neg_loss = self.criterion(neg_scores, neg_labels)

            loss = pos_loss + neg_loss

            # Backward with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(dataloader)

    def train(
        self,
        train_samples: List[CoNetSample],
        val_samples: Optional[List[CoNetSample]] = None,
        epochs: int = 50,
        batch_size: int = 256,
        num_negatives: int = 4,
        early_stopping_patience: int = 5,
        checkpoint_dir: str = "checkpoints/conet",
    ) -> Dict:
        """
        전체 학습

        Args:
            train_samples: 학습 샘플
            val_samples: 검증 샘플 (선택)
            epochs: 에포크 수
            batch_size: 배치 크기
            num_negatives: 네거티브 샘플 수
            early_stopping_patience: Early stopping patience
            checkpoint_dir: 체크포인트 저장 디렉토리

        Returns:
            학습 히스토리
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Dataset & Dataloader
        num_target_items = max(s.ground_truth_id for s in train_samples) + 100
        train_dataset = CoNetDataset(train_samples, num_negatives, num_target_items)
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
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            # Validation (if provided)
            if val_samples:
                val_loss = self._evaluate_loss(val_samples, batch_size)
                history["val_loss"].append(val_loss)

                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"          Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}")

                # Early stopping
                if val_loss < history["best_val_loss"]:
                    history["best_val_loss"] = val_loss
                    history["best_epoch"] = epoch
                    patience_counter = 0

                    # Save best model
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, "best_model.pt")
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

        return history

    def _evaluate_loss(
        self,
        samples: List[CoNetSample],
        batch_size: int
    ) -> float:
        """검증 loss 계산"""
        self.model.eval()
        num_target_items = max(s.ground_truth_id for s in samples) + 100
        dataset = CoNetDataset(samples, num_negatives=4, num_target_items=num_target_items)
        loader = DataLoader(dataset, batch_size=batch_size)

        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                user_ids = batch["user_id"].to(self.device)
                source_items = batch["source_item_id"].to(self.device)
                pos_items = batch["pos_target_item"].to(self.device)
                neg_items = batch["neg_target_items"].to(self.device)

                _, pos_scores = self.model(user_ids, source_items, pos_items)

                neg_scores_list = []
                for i in range(neg_items.size(1)):
                    _, neg_score = self.model(user_ids, source_items, neg_items[:, i])
                    neg_scores_list.append(neg_score)
                neg_scores = torch.cat(neg_scores_list, dim=1)

                pos_loss = self.criterion(pos_scores, torch.ones_like(pos_scores))
                neg_loss = self.criterion(neg_scores, torch.zeros_like(neg_scores))
                loss = pos_loss + neg_loss

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
        """체크포인트 로드

        Security Note: weights_only=False is required for optimizer state.
        Only load checkpoints from trusted sources.
        """
        # Path validation - prevent directory traversal
        # Use explicit parentheses for clarity (and binds tighter than or)
        if (".." in path) or (path.startswith("/") and not path.startswith("/Volumes")):
            raise ValueError(f"Suspicious checkpoint path detected: {path}")

        # weights_only=False needed for optimizer_state_dict (contains non-tensor objects)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available (backward compatible)
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
