"""
CoNet Model Implementation

Collaborative Cross Networks for Cross-Domain Recommendation
- Cross-stitch units for knowledge transfer between domains
- MLP-based collaborative filtering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossStitchUnit(nn.Module):
    """
    Cross-stitch unit for knowledge transfer

    Combines hidden representations from source and target domains
    using learnable mixing weights.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Learnable cross-stitch weights (initialized as identity-like)
        self.alpha = nn.Parameter(torch.tensor([[0.9, 0.1], [0.1, 0.9]]))

    def forward(
        self,
        h_source: torch.Tensor,
        h_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-stitch transformation

        Args:
            h_source: Source domain hidden representation [batch, hidden_dim]
            h_target: Target domain hidden representation [batch, hidden_dim]

        Returns:
            Transformed (h_source', h_target')
        """
        # Stack representations
        stacked = torch.stack([h_source, h_target], dim=1)  # [batch, 2, hidden_dim]

        # Apply cross-stitch weights
        alpha = F.softmax(self.alpha, dim=1)  # Normalize
        output = torch.einsum('ij,bjd->bid', alpha, stacked)

        return output[:, 0, :], output[:, 1, :]


class DomainMLP(nn.Module):
    """Domain-specific MLP tower"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CoNet(nn.Module):
    """
    CoNet: Collaborative Cross Networks

    Architecture:
    - Separate embedding layers for users and items in each domain
    - MLP towers for source and target domains
    - Cross-stitch units between layers for knowledge transfer
    - Final prediction layer for target domain
    """

    def __init__(
        self,
        num_users: int,
        num_items_source: int,
        num_items_target: int,
        embedding_dim: int = 128,
        hidden_dims: list = None,
        num_cross_stitch: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            num_users: Number of users (shared across domains)
            num_items_source: Number of items in source domain
            num_items_target: Number of items in target domain
            embedding_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions [256, 128, 64]
            num_cross_stitch: Number of cross-stitch layers
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.embedding_dim = embedding_dim

        # User embeddings (shared representation)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        # Item embeddings (domain-specific)
        self.item_embedding_source = nn.Embedding(num_items_source, embedding_dim)
        self.item_embedding_target = nn.Embedding(num_items_target, embedding_dim)

        # MLP layers for each domain
        input_dim = embedding_dim * 2  # user + item concatenation

        self.source_layers = nn.ModuleList()
        self.target_layers = nn.ModuleList()
        self.cross_stitch_units = nn.ModuleList()

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.source_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            self.target_layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            if i < num_cross_stitch:
                self.cross_stitch_units.append(CrossStitchUnit(hidden_dim))
            prev_dim = hidden_dim

        # Output layers
        self.output_source = nn.Linear(hidden_dims[-1], 1)
        self.output_target = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids_source: torch.Tensor,
        item_ids_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            user_ids: User IDs [batch]
            item_ids_source: Source domain item IDs [batch]
            item_ids_target: Target domain item IDs [batch]

        Returns:
            (source_scores, target_scores) each [batch, 1]
        """
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        item_emb_source = self.item_embedding_source(item_ids_source)
        item_emb_target = self.item_embedding_target(item_ids_target)

        # Concatenate user and item embeddings
        h_source = torch.cat([user_emb, item_emb_source], dim=-1)
        h_target = torch.cat([user_emb, item_emb_target], dim=-1)

        # Forward through MLP layers with cross-stitch
        for i, (source_layer, target_layer) in enumerate(
            zip(self.source_layers, self.target_layers)
        ):
            h_source = source_layer(h_source)
            h_target = target_layer(h_target)

            # Apply cross-stitch
            if i < len(self.cross_stitch_units):
                h_source, h_target = self.cross_stitch_units[i](h_source, h_target)

        # Output scores
        score_source = self.output_source(h_source)
        score_target = self.output_target(h_target)

        return score_source, score_target

    def predict_target(
        self,
        user_ids: torch.Tensor,
        item_ids_target: torch.Tensor,
        item_ids_source: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict target domain scores only

        For evaluation, we only need target scores
        """
        if item_ids_source is None:
            # Use dummy source items if not provided
            item_ids_source = torch.zeros_like(item_ids_target)

        _, target_scores = self.forward(user_ids, item_ids_source, item_ids_target)
        return target_scores

    def get_candidate_scores(
        self,
        user_id: int,
        candidate_item_ids: torch.Tensor,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Get scores for all candidate items

        Args:
            user_id: Single user ID
            candidate_item_ids: Candidate item IDs [num_candidates]
            device: Device (kept for API compatibility, uses model's device)

        Returns:
            Scores [num_candidates]

        Note: Uses model's actual device, ignoring device parameter for consistency.
        """
        # Use model's actual device to avoid device mismatch
        model_device = next(self.parameters()).device

        num_candidates = len(candidate_item_ids)
        user_ids = torch.tensor([user_id] * num_candidates, device=model_device)
        candidate_item_ids = candidate_item_ids.to(model_device)

        with torch.no_grad():
            scores = self.predict_target(user_ids, candidate_item_ids)

        return scores.squeeze(-1)
