"""Grasp success evaluator predicting success probability from grasp and scene feature."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss for class imbalance."""

    def __init__(self, alpha: float = 0.65, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GraspEvaluator(nn.Module):
    """Evaluates 6-DoF grasp candidates conditioned on scene features."""

    def __init__(self, scene_feat_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        grasp_dim = 13  # 9 (R) + 3 (p) + 1 (w)
        in_dim = grasp_dim + scene_feat_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, grasps: torch.Tensor, scene_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            grasps: (B, 13)
            scene_feat: (B, scene_feat_dim)
        Returns:
            logits: (B, 1)
        """
        x = torch.cat([grasps, scene_feat], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def predict_prob(self, grasps: torch.Tensor, scene_feat: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probability."""
        logits = self.forward(grasps, scene_feat)
        return torch.sigmoid(logits)
