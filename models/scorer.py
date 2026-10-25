"""Grasp Success Scorer with Temperature Calibration, AUROC, and ECE evaluation.

Specification:
- MLP on grasp (6D rotation, translation, width) + global embedding (256) + local features (128)
- BCE loss.
- Temperature scaling on held-out scene split.
- Reports ECE and AUROC.
"""

from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

from models.so3 import rot_to_6d
from models.pointnet import GlobalPointNet, LocalPointFeatureExtractor


def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = len(probs)
    if total == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < num_bins - 1 else probs <= bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_count / total) * abs(bin_acc - bin_conf)

    return float(ece)


class GraspSuccessScorer(nn.Module):
    """Calibrated Grasp Success Scorer network."""

    def __init__(
        self,
        global_dim: int = 256,
        local_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        in_dim = 6 + 3 + 1 + global_dim + local_dim  # 394 dims

        self.global_encoder = GlobalPointNet(in_channels=3, out_dim=global_dim)
        self.local_encoder = LocalPointFeatureExtractor(in_channels=3, out_dim=local_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Learned or calibrated temperature parameter
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward_logits(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        w: torch.Tensor,
        points: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute uncalibrated logits.
        
        Args:
            R: (B, 3, 3)
            t: (B, 3) (normalized)
            w: (B, 1) (normalized)
            points: (B, M, 3) (normalized)
        Returns:
            logits: (B, 1)
        """
        if global_feat is None:
            global_feat = self.global_encoder(points)

        local_feat = self.local_encoder(points, R, t, k=64)
        rot_6d = rot_to_6d(R)

        x = torch.cat([rot_6d, t, w, global_feat, local_feat], dim=-1)
        return self.mlp(x)

    def forward(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        w: torch.Tensor,
        points: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute temperature-calibrated probabilities."""
        logits = self.forward_logits(R, t, w, points, global_feat=global_feat)
        calibrated_logits = logits / torch.clamp(self.temperature, min=0.01)
        return torch.sigmoid(calibrated_logits)

    @torch.no_grad()
    def score_grasps(
        self,
        R: np.ndarray,
        t: np.ndarray,
        w: np.ndarray,
        points: np.ndarray,
        chunk_size: int = 32,
    ) -> np.ndarray:
        """Score candidate grasps on given point cloud.
        
        Args:
            R: (N, 3, 3) rotation matrices
            t: (N, 3) translation in meters
            w: (N,) widths in meters
            points: (M, 3) point cloud in meters
        Returns:
            scores: (N,) calibrated probabilities in [0, 1]
        """
        self.eval()
        device = next(self.parameters()).device
        N = len(R)

        # Normalize coordinates by 0.15 m
        R_torch = torch.from_numpy(R).float().to(device)
        t_norm = torch.from_numpy(t / 0.15).float().to(device)
        w_norm = torch.from_numpy((w / 0.15)[:, None]).float().to(device)

        pts_norm = torch.from_numpy(points / 0.15).float().unsqueeze(0).to(device)
        global_feat = self.global_encoder(pts_norm)

        scores = []
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            cur_R = R_torch[start_idx:end_idx]
            cur_t = t_norm[start_idx:end_idx]
            cur_w = w_norm[start_idx:end_idx]
            cur_pts = pts_norm.repeat(end_idx - start_idx, 1, 1)
            cur_global = global_feat.repeat(end_idx - start_idx, 1)

            logits = self.forward_logits(cur_R, cur_t, cur_w, cur_pts, global_feat=cur_global)
            probs = torch.sigmoid(logits / torch.clamp(self.temperature, min=0.01))
            scores.append(probs.squeeze(-1).cpu().numpy())

        return np.concatenate(scores, axis=0)

    def fit_temperature(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """Fit temperature parameter T > 0 on held-out scene split by minimizing NLL."""
        logits_tensor = torch.from_numpy(val_logits).float()
        labels_tensor = torch.from_numpy(val_labels).float()

        def nll_obj(T_val):
            T = max(1e-3, float(T_val[0]))
            scaled = logits_tensor / T
            loss = F.binary_cross_entropy_with_logits(scaled, labels_tensor)
            return loss.item()

        res = minimize(nll_obj, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        best_T = float(res.x[0])
        self.temperature.data = torch.tensor([best_T], dtype=torch.float32)
        return best_T
