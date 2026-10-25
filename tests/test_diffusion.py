"""Unit test for SO3R3Diffusion model sampling valid rotation matrices."""

import numpy as np
import pytest
import torch

from src.models.diffusion import SO3R3Diffusion


def test_diffusion_sampling_valid_rotations():
    torch.manual_seed(42)
    diffusion = SO3R3Diffusion(
        scene_feat_dim=64,
        hidden_dim=128,
        timesteps=20
    )
    
    scene_feat = torch.randn(1, 64)
    num_samples = 16
    ddim_steps = 5

    R_samples, p_samples, w_samples = diffusion.sample_ddim(
        scene_feat=scene_feat,
        num_samples=num_samples,
        ddim_steps=ddim_steps
    )

    assert R_samples.shape == (num_samples, 3, 3)
    assert p_samples.shape == (num_samples, 3)
    assert w_samples.shape == (num_samples, 1)

    for i in range(num_samples):
        R = R_samples[i]
        # Check orthogonality: R * R^T = I
        assert np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-5), f"Rotation {i} not orthogonal"
        # Check determinant is +1
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5), f"Rotation {i} det is not 1"
