"""Global PointNet encoder and Local Point Feature Extractor.

Specification:
- Global PointNet: takes point cloud (B, M, 3), outputs (B, 256) embedding.
- Local feature extractor: takes 64 cloud points nearest to grasp center,
  transforms into noisy grasp frame: p_local = R^T * (p - t),
  passes through MLP to 64 dims, then max + mean pooling to 128 dims.
- Plain-numpy reference function for acceptance test verification.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_local_points_numpy(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    k: int = 64,
) -> np.ndarray:
    """Plain-numpy reference for local point geometry.
    
    Args:
        points: (M, 3) point cloud
        R: (3, 3) grasp rotation matrix
        t: (3,) grasp translation
        k: number of nearest neighbors (64)
    Returns:
        p_local: (k, 3) points expressed in grasp frame: R^T * (p - t)
    """
    diff = points - t[None, :]  # (M, 3)
    dists_sq = np.sum(diff ** 2, axis=-1)  # (M,)
    nearest_idx = np.argsort(dists_sq)[:k]
    p_nearest = points[nearest_idx]  # (k, 3)
    # Transform to grasp local frame: R^T * (p - t)
    p_local = np.matmul(p_nearest - t[None, :], R)  # (k, 3)
    return p_local.astype(np.float32)


def extract_local_points_torch(
    points: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    k: int = 64,
) -> torch.Tensor:
    """Differentiable / vectorized PyTorch extraction of local points in grasp frame.
    
    Args:
        points: (B, M, 3) or (M, 3) point cloud
        R: (B, 3, 3) grasp rotation matrices
        t: (B, 3) grasp translations
        k: 64 nearest neighbors
    Returns:
        p_local: (B, k, 3) points in grasp frame
    """
    if points.ndim == 2:
        points = points.unsqueeze(0).expand(R.shape[0], -1, -1)

    # points: (B, M, 3), t: (B, 3) -> diff: (B, M, 3)
    diff = points - t.unsqueeze(1)  # (B, M, 3)
    dists_sq = torch.sum(diff ** 2, dim=-1)  # (B, M)

    # Select k nearest
    _, idx = torch.topk(dists_sq, k=k, dim=-1, largest=False)  # (B, k)
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)  # (B, k, 3)
    p_nearest = torch.gather(points, dim=1, index=idx_expanded)  # (B, k, 3)

    # Relative vectors: (B, k, 3)
    rel = p_nearest - t.unsqueeze(1)

    # R: (B, 3, 3). In grasp frame: p_local = R^T * rel = rel * R
    # (B, k, 3) bmm (B, 3, 3)
    p_local = torch.bmm(rel, R)  # (B, k, 3)
    return p_local


class GlobalPointNet(nn.Module):
    """Global PointNet encoder mapping (B, M, 3) point clouds to (B, 256) embedding."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Linear(in_channels, 64)
        self.conv2 = nn.Linear(64, 128)
        self.conv3 = nn.Linear(128, out_dim)
        self.act = nn.SiLU()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Args:
            points: (B, M, 3) or (B, 3, M) point cloud
        Returns:
            global_feat: (B, 256)
        """
        if points.shape[1] == 3 and points.ndim == 3 and points.shape[2] != 3:
            points = points.transpose(1, 2)  # (B, M, 3)

        x = self.act(self.conv1(points))
        x = self.act(self.conv2(x))
        x = self.conv3(x)  # (B, M, out_dim)

        # Global max pooling over points
        global_feat, _ = torch.max(x, dim=1)  # (B, out_dim)
        return global_feat


class LocalPointFeatureExtractor(nn.Module):
    """Local feature extractor on the 64 points nearest to grasp.
    
    Transforms points into noisy grasp frame, applies MLP,
    then max + mean pooling to out_dim (default 128 dims).
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        hidden_dim = max(1, out_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        points: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        k: int = 64,
    ) -> torch.Tensor:
        """Args:
            points: (B, M, 3) point cloud
            R: (B, 3, 3) rotation matrices
            t: (B, 3) grasp translations
            k: number of nearest neighbors (64)
        Returns:
            local_feat: (B, out_dim) concatenation of max and mean pooled features
        """
        # 1. Transform to grasp frame: (B, k, 3)
        p_local = extract_local_points_torch(points, R, t, k=k)

        # 2. MLP: (B, k, hidden_dim)
        feat = self.mlp(p_local)

        # 3. Max + Mean pooling: (B, hidden_dim) + (B, hidden_dim) -> (B, out_dim)
        feat_max, _ = torch.max(feat, dim=1)
        feat_mean = torch.mean(feat, dim=1)
        local_feat = torch.cat([feat_max, feat_mean], dim=-1)  # (B, out_dim)
        return local_feat
