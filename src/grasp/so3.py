"""SO(3) Lie group and Lie algebra operations.

Provides exponential and logarithmic maps, geodesic distance,
projection onto SO(3) via SVD, and Riemannian Karcher/Frechet mean.
Supports both NumPy and PyTorch implementations.
"""

from typing import Union, List
import numpy as np
import torch


def hat_numpy(v: np.ndarray) -> np.ndarray:
    """Map vector in R^3 to 3x3 skew-symmetric matrix in so(3).
    
    Args:
        v: (..., 3) array
    Returns:
        (..., 3, 3) skew-symmetric matrix
    """
    v = np.asarray(v)
    shape = v.shape[:-1]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = np.zeros_like(x)
    
    mat = np.stack([
        np.stack([zeros, -z, y], axis=-1),
        np.stack([z, zeros, -x], axis=-1),
        np.stack([-y, x, zeros], axis=-1),
    ], axis=-2)
    return mat


def vee_numpy(mat: np.ndarray) -> np.ndarray:
    """Map 3x3 skew-symmetric matrix to vector in R^3."""
    mat = np.asarray(mat)
    return np.stack([mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]], axis=-1)


def exp_so3_numpy(v: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) vector to SO(3) rotation matrix using Rodrigues formula.
    
    Args:
        v: (..., 3) axis-angle vector
    Returns:
        (..., 3, 3) rotation matrix in SO(3)
    """
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)  # (..., 1)
    K = hat_numpy(v)  # (..., 3, 3)
    I = np.eye(3, dtype=np.float64)
    for _ in range(v.ndim - 1):
        I = np.expand_dims(I, axis=0)
    
    theta_sq = theta ** 2
    theta_expanded = np.expand_dims(theta, axis=-1)
    theta_sq_expanded = np.expand_dims(theta_sq, axis=-1)
    
    # Small angle approximation to avoid division by zero
    small_mask = theta_expanded < 1e-7
    
    # Standard Rodrigues formula: I + (sin(theta)/theta) * K + ((1 - cos(theta))/theta^2) * K^2
    K2 = np.matmul(K, K)
    
    # For small angles: sin(theta)/theta ~ 1 - theta^2 / 6
    # (1 - cos(theta))/theta^2 ~ 0.5 - theta^2 / 24
    s_div_theta = np.where(small_mask, 1.0 - theta_sq_expanded / 6.0, np.sin(theta_expanded) / np.maximum(theta_expanded, 1e-9))
    one_minus_c_div_thetasq = np.where(small_mask, 0.5 - theta_sq_expanded / 24.0, (1.0 - np.cos(theta_expanded)) / np.maximum(theta_sq_expanded, 1e-12))
    
    R = I + s_div_theta * K + one_minus_c_div_thetasq * K2
    return project_so3_numpy(R)


def log_so3_numpy(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from SO(3) to so(3) vector in R^3.
    
    Args:
        R: (..., 3, 3) rotation matrix
    Returns:
        (..., 3) axis-angle vector
    """
    R = project_so3_numpy(R)
    trace = np.trace(R, axis1=-2, axis2=-1)  # (...)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (...)
    
    skew = 0.5 * (R - np.swapaxes(R, -1, -2))
    v = vee_numpy(skew)  # (..., 3)
    
    theta_expanded = np.expand_dims(theta, axis=-1)
    small_mask = theta_expanded < 1e-6
    near_pi_mask = theta_expanded > (np.pi - 1e-4)
    
    # Standard factor: theta / sin(theta)
    sin_theta = np.sin(theta_expanded)
    scale = np.where(
        small_mask,
        1.0 + (theta_expanded ** 2) / 6.0,
        theta_expanded / np.maximum(sin_theta, 1e-9)
    )
    res = scale * v
    
    # When theta is close to pi, (R - R^T) vanishes; use symmetric part (R + I) / 2
    if np.any(near_pi_mask):
        flat_R = R.reshape(-1, 3, 3)
        flat_mask = near_pi_mask.reshape(-1)
        flat_theta = theta.reshape(-1)
        flat_res = res.reshape(-1, 3)
        for i in range(len(flat_R)):
            if flat_mask[i]:
                th = flat_theta[i]
                r_sym = 0.5 * (flat_R[i] + np.eye(3))
                diag = np.clip(np.diag(r_sym), 0.0, 1.0)
                axis = np.sqrt(diag)
                # Determine signs
                if r_sym[0, 1] < 0:
                    axis[1] = -axis[1]
                if r_sym[0, 2] < 0:
                    axis[2] = -axis[2]
                norm_axis = np.linalg.norm(axis)
                if norm_axis > 1e-6:
                    axis = axis / norm_axis
                flat_res[i] = axis * th
        res = flat_res.reshape(res.shape)
        
    return res


def project_so3_numpy(R: np.ndarray) -> np.ndarray:
    """Project 3x3 matrix to valid SO(3) rotation matrix via SVD."""
    R = np.asarray(R, dtype=np.float64)
    orig_shape = R.shape
    flat_R = R.reshape(-1, 3, 3)
    U, _, Vt = np.linalg.svd(flat_R)
    
    # Ensure det(U * Vt) = +1
    d = np.linalg.det(np.matmul(U, Vt))
    diag = np.ones((len(flat_R), 3), dtype=np.float64)
    diag[:, 2] = d
    D = np.zeros((len(flat_R), 3, 3), dtype=np.float64)
    for i in range(3):
        D[:, i, i] = diag[:, i]
        
    R_proj = np.matmul(np.matmul(U, D), Vt)
    return R_proj.reshape(orig_shape)


def geodesic_distance_numpy(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Compute Riemannian geodesic distance between two rotation matrices in SO(3).
    
    d(R1, R2) = ||log(R1^T R2)||
    """
    R_rel = np.matmul(np.swapaxes(R1, -1, -2), R2)
    v = log_so3_numpy(R_rel)
    return np.linalg.norm(v, axis=-1)


def karcher_mean_so3_numpy(rotations: np.ndarray, max_iters: int = 50, tol: float = 1e-7) -> np.ndarray:
    """Compute Karcher (Frechet) Riemannian mean of rotations on SO(3).
    
    Args:
        rotations: (N, 3, 3) rotation matrices
    Returns:
        (3, 3) mean rotation matrix
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    assert rotations.ndim == 3 and rotations.shape[1:] == (3, 3)
    N = len(rotations)
    if N == 1:
        return rotations[0].copy()
    
    # Initialize at first rotation
    R_mean = rotations[0].copy()
    
    for _ in range(max_iters):
        # Tangent vector from R_mean to each R_i: v_i = log(R_mean^T * R_i)
        R_rel = np.matmul(R_mean.T, rotations)
        v = log_so3_numpy(R_rel)  # (N, 3)
        v_mean = np.mean(v, axis=0)  # (3,)
        
        norm_v = np.linalg.norm(v_mean)
        if norm_v < tol:
            break
            
        # Update Riemannian mean: R_mean <- R_mean * exp(v_mean)
        R_step = exp_so3_numpy(v_mean)
        R_mean = project_so3_numpy(np.matmul(R_mean, R_step))
        
    return R_mean


# ================= PyTorch Implementations =================

def hat_torch(v: torch.Tensor) -> torch.Tensor:
    """Map vector in R^3 to skew-symmetric matrix in so(3)."""
    shape = v.shape[:-1]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = torch.zeros_like(x)
    row0 = torch.stack([zeros, -z, y], dim=-1)
    row1 = torch.stack([z, zeros, -x], dim=-1)
    row2 = torch.stack([-y, x, zeros], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def vee_torch(mat: torch.Tensor) -> torch.Tensor:
    """Map skew-symmetric matrix to vector in R^3."""
    return torch.stack([mat[..., 2, 1], mat[..., 0, 2], mat[..., 1, 0]], dim=-1)


def exp_so3_torch(v: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map from so(3) vector to SO(3) matrix."""
    theta = torch.norm(v, dim=-1, keepdim=True)  # (..., 1)
    K = hat_torch(v)  # (..., 3, 3)
    
    I = torch.eye(3, dtype=v.dtype, device=v.device)
    for _ in range(v.ndim - 1):
        I = I.unsqueeze(0)
        
    theta_sq = theta ** 2
    theta_exp = theta.unsqueeze(-1)
    theta_sq_exp = theta_sq.unsqueeze(-1)
    
    small_mask = theta_exp < 1e-6
    K2 = torch.matmul(K, K)
    
    s_div_theta = torch.where(
        small_mask,
        1.0 - theta_sq_exp / 6.0,
        torch.sin(theta_exp) / torch.clamp(theta_exp, min=1e-8)
    )
    one_minus_c = torch.where(
        small_mask,
        0.5 - theta_sq_exp / 24.0,
        (1.0 - torch.cos(theta_exp)) / torch.clamp(theta_sq_exp, min=1e-12)
    )
    
    R = I + s_div_theta * K + one_minus_c * K2
    return project_so3_torch(R)


def project_so3_torch(R: torch.Tensor) -> torch.Tensor:
    """Project batch of 3x3 matrices to SO(3) via SVD."""
    orig_shape = R.shape
    flat_R = R.reshape(-1, 3, 3)
    U, _, Vh = torch.linalg.svd(flat_R)
    V = Vh.transpose(-1, -2)
    
    d = torch.det(torch.bmm(U, Vh)).unsqueeze(-1)
    diag = torch.ones((flat_R.shape[0], 3), dtype=R.dtype, device=R.device)
    diag[:, 2] = d.squeeze(-1)
    D = torch.diag_embed(diag)
    
    R_proj = torch.bmm(torch.bmm(U, D), Vh)
    return R_proj.reshape(orig_shape)


def log_so3_torch(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map from SO(3) to so(3) vector in R^3."""
    R = project_so3_torch(R)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    skew = 0.5 * (R - R.transpose(-1, -2))
    v = vee_torch(skew)
    
    theta_exp = theta.unsqueeze(-1)
    small_mask = theta_exp < 1e-6
    scale = torch.where(
        small_mask,
        1.0 + (theta_exp ** 2) / 6.0,
        theta_exp / torch.clamp(torch.sin(theta_exp), min=1e-8)
    )
    return scale * v
