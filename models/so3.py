"""SO(3) Lie group mathematics, 6D continuous rotation representations,
and uniform Haar sampling for SE(3) diffusion models.

Specification:
- Exponential & Logarithmic maps for SO(3) in NumPy and PyTorch.
- Round-trip error < 1e-5.
- Sampled rotations satisfy |R^T R - I| < 1e-5.
- Continuous 6D rotation representation (Zhou et al.).
- Uniform SO(3) sampling via Haar measure.
"""

from typing import Union, Tuple, Optional
import numpy as np
import torch


# ================= NumPy Implementations =================

def hat_numpy(v: np.ndarray) -> np.ndarray:
    """Map vector in R^3 to 3x3 skew-symmetric matrix in so(3)."""
    v = np.asarray(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = np.zeros_like(x)
    return np.stack([
        np.stack([zeros, -z, y], axis=-1),
        np.stack([z, zeros, -x], axis=-1),
        np.stack([-y, x, zeros], axis=-1),
    ], axis=-2)


def vee_numpy(mat: np.ndarray) -> np.ndarray:
    """Map 3x3 skew-symmetric matrix to vector in R^3."""
    mat = np.asarray(mat)
    return np.stack([mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]], axis=-1)


def exp_so3_numpy(v: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) vector in R^3 to SO(3) using Rodrigues formula."""
    v = np.asarray(v, dtype=np.float64)
    orig_shape = v.shape[:-1]
    v_flat = v.reshape(-1, 3)
    theta = np.linalg.norm(v_flat, axis=-1, keepdims=True)  # (N, 1)

    K = hat_numpy(v_flat)  # (N, 3, 3)
    K2 = np.matmul(K, K)
    I = np.eye(3, dtype=np.float64)[None, :, :]

    # Taylor expansion for small theta:
    # sin(theta)/theta ~ 1 - theta^2 / 6 + theta^4 / 120
    # (1 - cos(theta))/theta^2 ~ 0.5 - theta^2 / 24 + theta^4 / 720
    theta_sq = theta ** 2
    small = theta < 1e-6
    s_div = np.where(small, 1.0 - theta_sq / 6.0 + (theta_sq ** 2) / 120.0, np.sin(theta) / np.maximum(theta, 1e-12))
    c_div = np.where(small, 0.5 - theta_sq / 24.0 + (theta_sq ** 2) / 720.0, (1.0 - np.cos(theta)) / np.maximum(theta_sq, 1e-12))

    s_div = s_div[:, :, None]
    c_div = c_div[:, :, None]

    R = I + s_div * K + c_div * K2
    return R.reshape(orig_shape + (3, 3))


def log_so3_numpy(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) to so(3) vector in R^3."""
    R = np.asarray(R, dtype=np.float64)
    orig_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    # Compute trace and angle theta
    tr = np.trace(R_flat, axis1=1, axis2=2)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (N,)

    skew = 0.5 * (R_flat - np.swapaxes(R_flat, 1, 2))
    v = vee_numpy(skew)  # (N, 3)

    small = theta < 1e-6
    theta_exp = theta[:, None]
    sin_theta = np.sin(theta_exp)

    # For small theta: theta / sin(theta) ~ 1 + theta^2 / 6 + 7 * theta^4 / 360
    scale = np.where(
        small[:, None],
        1.0 + (theta_exp ** 2) / 6.0 + 7.0 * (theta_exp ** 4) / 360.0,
        theta_exp / np.maximum(sin_theta, 1e-12)
    )
    v_out = scale * v

    # Handle near-pi boundary
    near_pi = theta > (np.pi - 1e-4)
    if np.any(near_pi):
        for idx in np.where(near_pi)[0]:
            th = theta[idx]
            # Symmetric part
            sym = 0.5 * (R_flat[idx] + np.eye(3))
            diag = np.clip(np.diag(sym), 0.0, 1.0)
            axis = np.sqrt(diag)
            # Match signs based on off-diagonal entries
            if sym[0, 1] < 0:
                axis[1] = -axis[1]
            if sym[0, 2] < 0:
                axis[2] = -axis[2]
            norm_a = np.linalg.norm(axis)
            if norm_a > 1e-6:
                axis = axis / norm_a
            v_out[idx] = axis * th

    return v_out.reshape(orig_shape + (3,))


def project_so3_numpy(R: np.ndarray) -> np.ndarray:
    """Project 3x3 matrix to closest valid SO(3) rotation via SVD."""
    R = np.asarray(R, dtype=np.float64)
    orig_shape = R.shape
    R_flat = R.reshape(-1, 3, 3)
    U, _, Vt = np.linalg.svd(R_flat)
    d = np.linalg.det(np.matmul(U, Vt))
    D = np.eye(3, dtype=np.float64)[None, :, :].repeat(len(R_flat), axis=0)
    D[:, 2, 2] = d
    R_proj = np.matmul(np.matmul(U, D), Vt)
    return R_proj.reshape(orig_shape)


def sample_uniform_so3_numpy(num_samples: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """Sample uniform rotations in SO(3) using Haar measure (Shoemake algorithm)."""
    if rng is None:
        rng = np.random.RandomState(42)

    u1 = rng.uniform(0.0, 1.0, size=num_samples)
    u2 = rng.uniform(0.0, 2.0 * np.pi, size=num_samples)
    u3 = rng.uniform(0.0, 2.0 * np.pi, size=num_samples)

    q0 = np.sqrt(1.0 - u1) * np.sin(u2)
    q1 = np.sqrt(1.0 - u1) * np.cos(u2)
    q2 = np.sqrt(u1) * np.sin(u3)
    q3 = np.sqrt(u1) * np.cos(u3)

    # Convert unit quaternions to 3x3 rotation matrices
    R = np.zeros((num_samples, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (q2 ** 2 + q3 ** 2)
    R[:, 0, 1] = 2.0 * (q1 * q2 - q0 * q3)
    R[:, 0, 2] = 2.0 * (q1 * q3 + q0 * q2)

    R[:, 1, 0] = 2.0 * (q1 * q2 + q0 * q3)
    R[:, 1, 1] = 1.0 - 2.0 * (q1 ** 2 + q3 ** 2)
    R[:, 1, 2] = 2.0 * (q2 * q3 - q0 * q1)

    R[:, 2, 0] = 2.0 * (q1 * q3 - q0 * q2)
    R[:, 2, 1] = 2.0 * (q2 * q3 + q0 * q1)
    R[:, 2, 2] = 1.0 - 2.0 * (q1 ** 2 + q2 ** 2)

    return R


# ================= PyTorch Implementations =================

def hat_torch(v: torch.Tensor) -> torch.Tensor:
    """Map vector in R^3 to skew-symmetric matrix in so(3)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = torch.zeros_like(x)
    return torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1),
    ], dim=-2)


def vee_torch(mat: torch.Tensor) -> torch.Tensor:
    """Map skew-symmetric matrix to vector in R^3."""
    return torch.stack([mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]], dim=-1)


def exp_so3_torch(v: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map from so(3) vector in R^3 to SO(3)."""
    orig_shape = v.shape[:-1]
    v_flat = v.reshape(-1, 3).to(torch.float64)
    theta = torch.norm(v_flat, dim=-1, keepdim=True)  # (N, 1)

    K = hat_torch(v_flat)
    K2 = torch.bmm(K, K)
    I = torch.eye(3, dtype=torch.float64, device=v.device).unsqueeze(0).repeat(v_flat.shape[0], 1, 1)

    theta_sq = theta ** 2
    small = theta < 1e-6
    s_div = torch.where(small, 1.0 - theta_sq / 6.0 + (theta_sq ** 2) / 120.0, torch.sin(theta) / torch.clamp(theta, min=1e-12))
    c_div = torch.where(small, 0.5 - theta_sq / 24.0 + (theta_sq ** 2) / 720.0, (1.0 - torch.cos(theta)) / torch.clamp(theta_sq, min=1e-12))

    s_div = s_div.unsqueeze(-1)
    c_div = c_div.unsqueeze(-1)

    R = I + s_div * K + c_div * K2
    return R.reshape(*orig_shape, 3, 3).to(v.dtype)


def log_so3_torch(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map from SO(3) to so(3) vector in R^3."""
    orig_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3).to(torch.float64)

    tr = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    skew = 0.5 * (R_flat - R_flat.transpose(1, 2))
    v = vee_torch(skew)

    small = (theta < 1e-6).unsqueeze(-1)
    theta_exp = theta.unsqueeze(-1)
    sin_theta = torch.sin(theta_exp)

    scale = torch.where(
        small,
        1.0 + (theta_exp ** 2) / 6.0 + 7.0 * (theta_exp ** 4) / 360.0,
        theta_exp / torch.clamp(sin_theta, min=1e-12)
    )
    v_out = scale * v
    return v_out.reshape(*orig_shape, 3).to(R.dtype)


def rot_to_6d(R: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 3) rotation matrix to 6D continuous representation (first two columns)."""
    # Columns 0 and 1
    col0 = R[..., :, 0]
    col1 = R[..., :, 1]
    return torch.cat([col0, col1], dim=-1)


def rot_from_6d(d6: torch.Tensor) -> torch.Tensor:
    """Convert (..., 6) continuous representation to valid (..., 3, 3) rotation matrix.
    
    Gram-Schmidt orthogonalization:
    r1 = d6[:3] / ||d6[:3]||
    r2 = (d6[3:6] - (r1 . d6[3:6]) r1) / ||...||
    r3 = r1 x r2
    R = [r1, r2, r3]
    """
    orig_shape = d6.shape[:-1]
    d6_flat = d6.reshape(-1, 6)

    c1 = d6_flat[:, :3]
    c2 = d6_flat[:, 3:]

    r1 = c1 / torch.clamp(torch.norm(c1, dim=-1, keepdim=True), min=1e-8)
    dot = torch.sum(r1 * c2, dim=-1, keepdim=True)
    c2_ortho = c2 - dot * r1
    r2 = c2_ortho / torch.clamp(torch.norm(c2_ortho, dim=-1, keepdim=True), min=1e-8)
    r3 = torch.cross(r1, r2, dim=-1)

    R = torch.stack([r1, r2, r3], dim=-1)  # (N, 3, 3)
    return R.reshape(*orig_shape, 3, 3)
