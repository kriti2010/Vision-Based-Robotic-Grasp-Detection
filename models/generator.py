"""SE(3) Variance-Exploding Diffusion Grasp Generator.

Specification:
- Variance-exploding diffusion on (R, t, w).
- Training sigma log-uniform in [0.01, 1.5].
- Translation noise: sigma * eps_t (eps_t ~ N(0, I)).
- Width noise: 0.3 * sigma * eps_w (eps_w ~ N(0, 1)).
- Rotation noise: R_k = R_0 Exp(2 * sigma * eps_R) (eps_R ~ N(0, I)).
- Denoiser predicts (eps_t, eps_R, eps_w).
- Loss = sum of three MSEs: MSE(eps_t) + MSE(eps_R) + MSE(eps_w).
- Denoiser inputs: 6D rotation (6), translation (3), width (1),
  sigma embedding (64), global PointNet embedding (256), local features (128).
  MLP 3x512 SiLU.
- Sampling: 12 deterministic DDIM-style steps from uniform random rotations
  and Gaussian translations, using the group exponential for rotations.
- Chunk large forward passes to avoid OOM.
"""

from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.so3 import (
    exp_so3_torch, log_so3_torch, rot_to_6d, rot_from_6d,
    sample_uniform_so3_numpy
)
from models.pointnet import GlobalPointNet, LocalPointFeatureExtractor


class SinusoidalSigmaEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous noise level sigma."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """Args:
            sigma: (B,) or (B, 1) continuous noise levels
        Returns:
            emb: (B, embed_dim)
        """
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)  # (B, 1)

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=sigma.device) * -emb)
        args = sigma * freqs.unsqueeze(0)  # (B, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(embedding)


class GraspDiffusionDenoiser(nn.Module):
    """Variance-exploding denoiser for SE(3) grasps."""

    def __init__(
        self,
        global_dim: int = 256,
        local_dim: int = 128,
        sigma_dim: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        # Inputs: 6 (rot6d) + 3 (t) + 1 (w) + 64 (sigma) + 256 (global) + 128 (local) = 468
        in_dim = 6 + 3 + 1 + sigma_dim + global_dim + local_dim

        self.sigma_embed = SinusoidalSigmaEmbedding(embed_dim=sigma_dim)
        self.global_encoder = GlobalPointNet(in_channels=3, out_dim=global_dim)
        self.local_encoder = LocalPointFeatureExtractor(in_channels=3, out_dim=local_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 + 3 + 1),  # eps_t (3), eps_R (3), eps_w (1)
        )

    def forward(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        w: torch.Tensor,
        sigma: torch.Tensor,
        points: torch.Tensor,
        global_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of denoiser.
        
        Args:
            R: (B, 3, 3) noisy rotation matrices
            t: (B, 3) noisy translation (normalized)
            w: (B, 1) noisy width (normalized)
            sigma: (B,) noise levels
            points: (B, M, 3) point cloud (normalized)
            global_feat: optional pre-extracted (B, 256)
        Returns:
            (eps_t_pred, eps_R_pred, eps_w_pred)
        """
        B = R.shape[0]
        if global_feat is None:
            global_feat = self.global_encoder(points)  # (B, 256)

        local_feat = self.local_encoder(points, R, t, k=64)  # (B, 128)
        sigma_feat = self.sigma_embed(sigma)  # (B, 64)
        rot_6d = rot_to_6d(R)  # (B, 6)

        x = torch.cat([rot_6d, t, w, sigma_feat, global_feat, local_feat], dim=-1)
        out = self.mlp(x)  # (B, 7)

        eps_t_pred = out[:, :3]
        eps_R_pred = out[:, 3:6]
        eps_w_pred = out[:, 6:7]

        return eps_t_pred, eps_R_pred, eps_w_pred

    def compute_loss(
        self,
        R_0: torch.Tensor,
        t_0: torch.Tensor,
        w_0: torch.Tensor,
        points: torch.Tensor,
        sigma_min: float = 0.01,
        sigma_max: float = 1.5,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute training loss on ground truth successful grasps."""
        B = R_0.shape[0]
        device = R_0.device

        # Sample sigma log-uniform in [sigma_min, sigma_max]
        log_sigma = torch.rand(B, device=device) * (np.log(sigma_max) - np.log(sigma_min)) + np.log(sigma_min)
        sigma = torch.exp(log_sigma)  # (B,)

        # Sample standard Gaussian noise
        eps_t = torch.randn_like(t_0)
        eps_w = torch.randn_like(w_0)
        eps_R = torch.randn_like(t_0)  # (B, 3) axis-angle perturbation

        # Add variance-exploding perturbations:
        # translation noise: sigma * eps_t
        t_noisy = t_0 + sigma.unsqueeze(-1) * eps_t
        # width noise: 0.3 * sigma * eps_w
        w_noisy = w_0 + 0.3 * sigma.unsqueeze(-1) * eps_w
        # rotation noise: R_k = R_0 * Exp(2 * sigma * eps_R)
        rot_noise = exp_so3_torch(2.0 * sigma.unsqueeze(-1) * eps_R)
        R_noisy = torch.bmm(R_0, rot_noise)

        # Denoiser prediction
        eps_t_pred, eps_R_pred, eps_w_pred = self.forward(
            R_noisy, t_noisy, w_noisy, sigma, points
        )

        # Sum of three MSEs
        loss_t = F.mse_loss(eps_t_pred, eps_t)
        loss_R = F.mse_loss(eps_R_pred, eps_R)
        loss_w = F.mse_loss(eps_w_pred, eps_w)
        loss = loss_t + loss_R + loss_w

        return loss, {
            "loss": loss.item(),
            "loss_t": loss_t.item(),
            "loss_R": loss_R.item(),
            "loss_w": loss_w.item(),
        }

    @torch.no_grad()
    def sample(
        self,
        points: torch.Tensor,
        num_samples: int = 32,
        num_steps: int = 12,
        sigma_min: float = 0.01,
        sigma_max: float = 1.5,
        chunk_size: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample grasps using 12 deterministic DDIM-style steps.
        
        Args:
            points: (1, M, 3) or (M, 3) point cloud (normalized by 0.15 m)
            num_samples: N = 32 samples
            num_steps: K = 12 steps
        Returns:
            R: (N, 3, 3) rotation matrices
            t: (N, 3) grasp centers in meters
            w: (N,) grasp widths in meters
        """
        self.eval()
        device = next(self.parameters()).device

        if points.ndim == 2:
            points = points.unsqueeze(0)
        points = points.to(device)

        # Precompute global scene feature
        global_feat = self.global_encoder(points)  # (1, 256)

        # Geometric noise schedule over 12 steps
        sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), num_steps))
        # Append 0 for final step
        sigmas = np.append(sigmas, 0.0)

        # Process in chunks to avoid OOM
        all_R = []
        all_t = []
        all_w = []

        for start_idx in range(0, num_samples, chunk_size):
            cur_chunk = min(chunk_size, num_samples - start_idx)

            # 1. Initialize from Haar uniform rotations and Gaussian translations
            R_init_np = sample_uniform_so3_numpy(cur_chunk)
            R_cur = torch.from_numpy(R_init_np).float().to(device)
            # Translations centered near point cloud
            pts_mean = torch.mean(points, dim=1)  # (1, 3)
            t_cur = pts_mean.repeat(cur_chunk, 1) + torch.randn(cur_chunk, 3, device=device) * 0.3
            w_cur = torch.empty(cur_chunk, 1, device=device).uniform_(0.02 / 0.15, 0.08 / 0.15)

            points_batch = points.repeat(cur_chunk, 1, 1)
            global_batch = global_feat.repeat(cur_chunk, 1)

            # 2. 12 deterministic DDIM-style steps
            for k in range(num_steps):
                s_cur = sigmas[k]
                s_next = sigmas[k + 1]
                ds = float(s_cur - s_next)

                s_tensor = torch.full((cur_chunk,), s_cur, dtype=torch.float32, device=device)
                eps_t, eps_R, eps_w = self.forward(
                    R_cur, t_cur, w_cur, s_tensor, points_batch, global_feat=global_batch
                )

                # Deterministic drift update
                t_cur = t_cur - ds * eps_t
                w_cur = w_cur - 0.3 * ds * eps_w

                # Lie group exponential update: R_{k+1} = R_k Exp(-2 * ds * eps_R)
                step_R = exp_so3_torch(-2.0 * ds * eps_R)
                R_cur = torch.bmm(R_cur, step_R)

            # Denormalize coordinates by 0.15 m
            t_meters = (t_cur * 0.15).cpu().numpy()
            w_meters = np.clip((w_cur * 0.15).squeeze(-1).cpu().numpy(), 0.015, 0.085)
            R_np = R_cur.cpu().numpy()

            all_R.append(R_np)
            all_t.append(t_meters)
            all_w.append(w_meters)

        R_out = np.concatenate(all_R, axis=0)
        t_out = np.concatenate(all_t, axis=0)
        w_out = np.concatenate(all_w, axis=0)

        return R_out, t_out, w_out
