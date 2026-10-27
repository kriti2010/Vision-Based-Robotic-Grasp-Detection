"""Diffusion generator on SO(3) x R^3 with DDPM training and DDIM sampling.

Rotation noise is applied on the manifold: R_t = R_0 * exp(sigma_t * eps_R).
Translation and width use standard Gaussian noise.
All sampling outputs guaranteed valid SO(3) rotation matrices.
"""

from typing import Tuple, Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.grasp.so3 import exp_so3_torch, project_so3_torch, project_so3_numpy
from src.grasp.representation import Grasp


def get_sinusoidal_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DenoiserMLP(nn.Module):
    """Predicts Lie algebra rotation noise eps_R, translation noise eps_p, and width noise eps_w."""

    def __init__(self, scene_feat_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        # Input: R (9) + p (3) + w (1) = 13
        self.in_dim = 13
        self.time_emb_dim = 64
        
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        total_in = self.in_dim + hidden_dim + scene_feat_dim
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 7)  # 3 (eps_R) + 3 (eps_p) + 1 (eps_w)
        )

    def forward(
        self,
        R_t: torch.Tensor,
        p_t: torch.Tensor,
        w_t: torch.Tensor,
        t: torch.Tensor,
        scene_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            R_t: (B, 3, 3)
            p_t: (B, 3)
            w_t: (B, 1)
            t: (B,) integer timesteps
            scene_feat: (B, scene_feat_dim)
        Returns:
            eps_R: (B, 3), eps_p: (B, 3), eps_w: (B, 1)
        """
        B = R_t.shape[0]
        R_flat = R_t.reshape(B, 9)
        grasp_in = torch.cat([R_flat, p_t, w_t], dim=1)  # (B, 13)

        t_emb = get_sinusoidal_timestep_embedding(t, self.time_emb_dim)
        t_feat = self.time_mlp(t_emb)  # (B, hidden_dim)

        x = torch.cat([grasp_in, t_feat, scene_feat], dim=1)
        out = self.net(x)

        eps_R = out[:, :3]
        eps_p = out[:, 3:6]
        eps_w = out[:, 6:7]
        return eps_R, eps_p, eps_w


class SO3R3Diffusion(nn.Module):
    """Manifold DDPM & DDIM diffusion model on SO(3) x R^3."""

    def __init__(
        self,
        scene_feat_dim: int = 128,
        hidden_dim: int = 256,
        timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        super().__init__()
        self.timesteps = timesteps
        self.denoiser = DenoiserMLP(scene_feat_dim=scene_feat_dim, hidden_dim=hidden_dim)

        # Variance schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    def get_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """Rotation noise scale on the manifold at timestep t."""
        # Scale with sqrt(1 - alphas_bar[t]) * pi so at T it perturbs fully
        return (self.sqrt_one_minus_alphas_bar[t] * math.pi).unsqueeze(-1)  # (B, 1)

    def forward_loss(
        self,
        R_0: torch.Tensor,
        p_0: torch.Tensor,
        w_0: torch.Tensor,
        scene_feat: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss on ground-truth successful grasps."""
        B = R_0.shape[0]
        device = R_0.device
        
        # Normalize translation and width for zero-mean unit-variance diffusion
        p_0_norm = p_0 / 0.10
        w_0_norm = (w_0 - 0.065) / 0.02

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        # Sample noise
        eps_p = torch.randn_like(p_0_norm)
        eps_w = torch.randn_like(w_0_norm)
        eps_R = torch.randn((B, 3), device=device)
        eps_R_unit = eps_R / (torch.norm(eps_R, dim=-1, keepdim=True) + 1e-8)

        # Perturb translation and width in normalized space
        sqrt_ab = self.sqrt_alphas_bar[t].unsqueeze(-1)
        sqrt_1m_ab = self.sqrt_one_minus_alphas_bar[t].unsqueeze(-1)
        
        p_t = sqrt_ab * p_0_norm + sqrt_1m_ab * eps_p
        w_t = sqrt_ab * w_0_norm + sqrt_1m_ab * eps_w

        # Perturb rotation on SO(3) manifold: R_t = R_0 * exp(sigma_t * eps_R)
        sigma_t = self.get_sigma_t(t)  # (B, 1)
        v_rot = sigma_t * eps_R_unit
        R_perturb = exp_so3_torch(v_rot)
        R_t = torch.bmm(R_0, R_perturb)
        R_t = project_so3_torch(R_t)

        # Denoiser predicts noise
        pred_eps_R, pred_eps_p, pred_eps_w = self.denoiser(R_t, p_t, w_t, t, scene_feat)

        target_eps_R = eps_R_unit
        loss_R = F.mse_loss(pred_eps_R, target_eps_R)
        loss_p = F.mse_loss(pred_eps_p, eps_p)
        loss_w = F.mse_loss(pred_eps_w, eps_w)

        total_loss = loss_R + loss_p + loss_w
        return total_loss

    @torch.no_grad()
    def sample_ddim(
        self,
        scene_feat: torch.Tensor,
        num_samples: int = 64,
        ddim_steps: int = 20,
        p_center: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast DDIM sampling on SO(3) x R^3.
        
        Args:
            scene_feat: (1, feat_dim) or (feat_dim,)
            num_samples: K grasp samples
            ddim_steps: number of sampling steps
            p_center: Optional (3,) center of observed object points to guide noise distribution
        Returns:
            (rotations (K, 3, 3), translations (K, 3), widths (K, 1)) as NumPy arrays
        """
        device = self.betas.device
        if scene_feat.dim() == 1:
            scene_feat = scene_feat.unsqueeze(0)
        scene_feat_rep = scene_feat.repeat(num_samples, 1)

        # Timestep schedule for DDIM
        step_indices = np.linspace(0, self.timesteps - 1, ddim_steps, dtype=int)
        
        # Initialize random state at t = T
        # Tabletop prior: approach pointing towards table with random roll and Lie algebra dispersion
        rolls = torch.rand(num_samples, 1, device=device) * (2 * math.pi)
        cos_r = torch.cos(rolls)
        sin_r = torch.sin(rolls)
        zeros = torch.zeros_like(rolls)
        ones = torch.ones_like(rolls)
        
        row0 = torch.cat([cos_r, sin_r, zeros], dim=-1)
        row1 = torch.cat([sin_r, -cos_r, zeros], dim=-1)
        row2 = torch.cat([zeros, zeros, -ones], dim=-1)
        R_base = torch.stack([row0, row1, row2], dim=-2)
        
        # Perturbation on Lie algebra
        rand_v = torch.randn(num_samples, 3, device=device) * 0.45
        R_curr = project_so3_torch(torch.bmm(R_base, exp_so3_torch(rand_v)))
        
        # Translation in normalized space
        if p_center is not None:
            p_c_norm = (p_center.to(device).reshape(1, 3)) / 0.10
            p_curr = p_c_norm + torch.randn(num_samples, 3, device=device) * 0.25
        else:
            p_curr = torch.randn(num_samples, 3, device=device) * 0.5
        
        # Width in normalized space
        w_curr = torch.randn(num_samples, 1, device=device) * 0.4

        for i in reversed(range(len(step_indices))):
            t_idx = step_indices[i]
            t_tensor = torch.full((num_samples,), t_idx, device=device, dtype=torch.long)
            
            # Predict noise
            pred_eps_R, pred_eps_p, pred_eps_w = self.denoiser(R_curr, p_curr, w_curr, t_tensor, scene_feat_rep)

            alpha_bar_t = self.alphas_bar[t_idx]
            sqrt_ab_t = torch.sqrt(alpha_bar_t)
            sqrt_1m_ab_t = torch.sqrt(1.0 - alpha_bar_t)
            sigma_t = self.get_sigma_t(t_tensor)

            # Reconstruct clean target x_0 in normalized space
            pred_p0 = (p_curr - sqrt_1m_ab_t * pred_eps_p) / torch.clamp(sqrt_ab_t, min=1e-6)
            pred_w0 = (w_curr - sqrt_1m_ab_t * pred_eps_w) / torch.clamp(sqrt_ab_t, min=1e-6)
            
            # Rotation on manifold: R_0 = R_curr * exp(-sigma_t * pred_eps_R)
            step_v = -sigma_t * pred_eps_R
            R_0_pred = project_so3_torch(torch.bmm(R_curr, exp_so3_torch(step_v)))

            if i > 0:
                t_prev_idx = step_indices[i - 1]
                alpha_bar_prev = self.alphas_bar[t_prev_idx]
                sqrt_ab_prev = torch.sqrt(alpha_bar_prev)
                sqrt_1m_ab_prev = torch.sqrt(1.0 - alpha_bar_prev)
                sigma_prev = self.get_sigma_t(torch.full((num_samples,), t_prev_idx, device=device, dtype=torch.long))

                # Step forward to t_prev
                p_curr = sqrt_ab_prev * pred_p0 + sqrt_1m_ab_prev * pred_eps_p
                w_curr = sqrt_ab_prev * pred_w0 + sqrt_1m_ab_prev * pred_eps_w
                R_curr = project_so3_torch(torch.bmm(R_0_pred, exp_so3_torch(sigma_prev * pred_eps_R)))
            else:
                p_curr = pred_p0
                w_curr = pred_w0
                R_curr = R_0_pred

        R_final = project_so3_torch(R_curr).cpu().numpy()
        # Unnormalize translation and width back to physical units (meters)
        p_final = (p_curr * 0.10).cpu().numpy()
        p_final[:, 2] = np.maximum(p_final[:, 2], 0.02)
        w_final = np.clip((w_curr * 0.02 + 0.065).cpu().numpy(), 0.02, 0.085)

        return R_final, p_final, w_final
