"""
DTCDR Model Implementation

Dual-Target Cross-Domain Recommendation
- Shared user embeddings
- Domain-specific item embeddings
- Orthogonal mapping for domain adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class OrthogonalMapping(nn.Module):
    """
    Orthogonal mapping for domain adaptation

    Maps embeddings from source to target domain while preserving structure
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.mapping = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Initialize as orthogonal matrix
        nn.init.orthogonal_(self.mapping.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapping(x)

    def get_orthogonal_loss(self) -> torch.Tensor:
        """Orthogonality regularization loss"""
        W = self.mapping.weight
        WtW = torch.mm(W.t(), W)
        I = torch.eye(W.size(1), device=W.device)
        return torch.norm(WtW - I, p='fro')


class DTCDR(nn.Module):
    """
    DTCDR: Dual-Target Cross-Domain Recommendation

    Architecture:
    - Shared user embedding layer
    - Domain-specific item embedding layers
    - Orthogonal mapping for cross-domain transfer
    - MLP for prediction
    """

    def __init__(
        self,
        num_users: int,
        num_items_source: int,
        num_items_target: int,
        embedding_dim: int = 128,
        mlp_layers: list = None,
        dropout: float = 0.1,
        use_mapping: bool = True,
    ):
        """
        Args:
            num_users: Number of users (shared)
            num_items_source: Number of source domain items
            num_items_target: Number of target domain items
            embedding_dim: Embedding dimension
            mlp_layers: MLP hidden dimensions
            dropout: Dropout rate
            use_mapping: Whether to use orthogonal mapping
        """
        super().__init__()

        if mlp_layers is None:
            mlp_layers = [256, 128]

        self.embedding_dim = embedding_dim
        self.use_mapping = use_mapping

        # Shared user embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # Domain-specific item embeddings
        self.item_embedding_source = nn.Embedding(num_items_source, embedding_dim)
        self.item_embedding_target = nn.Embedding(num_items_target, embedding_dim)

        # Orthogonal mapping (source → target)
        if use_mapping:
            self.mapping = OrthogonalMapping(embedding_dim)

        # Prediction MLPs
        self._build_mlp(mlp_layers, dropout)

    def _build_mlp(self, mlp_layers: list, dropout: float):
        """Build MLP layers for prediction"""
        input_dim = self.embedding_dim * 2  # user + item

        # Source domain MLP
        source_layers = []
        prev_dim = input_dim
        for hidden_dim in mlp_layers:
            source_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        source_layers.append(nn.Linear(prev_dim, 1))
        self.mlp_source = nn.Sequential(*source_layers)

        # Target domain MLP
        target_layers = []
        prev_dim = input_dim
        for hidden_dim in mlp_layers:
            target_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        target_layers.append(nn.Linear(prev_dim, 1))
        self.mlp_target = nn.Sequential(*target_layers)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids_source: torch.Tensor,
        item_ids_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both domains

        Args:
            user_ids: User IDs [batch]
            item_ids_source: Source item IDs [batch]
            item_ids_target: Target item IDs [batch]

        Returns:
            (source_scores, target_scores)
        """
        # User embedding
        user_emb = self.user_embedding(user_ids)

        # Source domain
        item_emb_source = self.item_embedding_source(item_ids_source)
        source_input = torch.cat([user_emb, item_emb_source], dim=-1)
        score_source = self.mlp_source(source_input)

        # Target domain (with optional mapping)
        if self.use_mapping:
            user_emb_mapped = self.mapping(user_emb)
        else:
            user_emb_mapped = user_emb

        item_emb_target = self.item_embedding_target(item_ids_target)
        target_input = torch.cat([user_emb_mapped, item_emb_target], dim=-1)
        score_target = self.mlp_target(target_input)

        return score_source, score_target

    def predict_target(
        self,
        user_ids: torch.Tensor,
        item_ids_target: torch.Tensor
    ) -> torch.Tensor:
        """Predict target domain only"""
        user_emb = self.user_embedding(user_ids)

        if self.use_mapping:
            user_emb = self.mapping(user_emb)

        item_emb_target = self.item_embedding_target(item_ids_target)
        target_input = torch.cat([user_emb, item_emb_target], dim=-1)

        return self.mlp_target(target_input)

    def get_candidate_scores(
        self,
        user_id: int,
        candidate_item_ids: torch.Tensor,
        device: str = "cuda"
    ) -> torch.Tensor:
        """Get scores for candidate items

        Note: Uses model's actual device, ignoring device parameter for consistency.
        The device parameter is kept for API compatibility.
        """
        # Use model's actual device to avoid device mismatch
        model_device = next(self.parameters()).device

        num_candidates = len(candidate_item_ids)
        user_ids = torch.tensor([user_id] * num_candidates, device=model_device)
        candidate_item_ids = candidate_item_ids.to(model_device)

        with torch.no_grad():
            scores = self.predict_target(user_ids, candidate_item_ids)

        return scores.squeeze(-1)

    def get_orthogonal_loss(self) -> torch.Tensor:
        """Get orthogonality regularization loss"""
        if self.use_mapping:
            return self.mapping.get_orthogonal_loss()
        # Return 0 on the same device as model parameters to avoid device mismatch
        return torch.tensor(0.0, device=next(self.parameters()).device)
