"""PointNet scene encoder for point cloud feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """Lightweight PointNet encoder producing global and per-point scene representations."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        # Per-point MLPs
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)

        # Global feature projection
        self.fc1 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)
        
        self.fc2 = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, N, 3) or (B, 3, N) point clouds
        Returns:
            global_feature: (B, feature_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        if x.shape[-1] == 3 and x.shape[1] != 3:
            # Permute (B, N, 3) -> (B, 3, N)
            x = x.transpose(1, 2)

        B, _, N = x.shape

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))  # (B, 256, N)

        # Symmetric max pooling over points
        global_feat, _ = torch.max(h, dim=2)  # (B, 256)

        out = F.relu(self.ln4(self.fc1(global_feat)))
        out = self.fc2(out)  # (B, feature_dim)
        return out
