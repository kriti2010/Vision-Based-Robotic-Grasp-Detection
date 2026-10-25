"""Tests for Stage 2 Models: GraspDiffusionDenoiser, GraspSuccessScorer, and calibration."""

import numpy as np
import torch
import pytest
from models.generator import GraspDiffusionDenoiser
from models.scorer import GraspSuccessScorer, compute_ece
from models.so3 import sample_uniform_so3_numpy


def test_generator_forward_and_loss():
    """Verify denoiser shapes and compute_loss execution."""
    B = 4
    M = 512
    device = torch.device("cpu")

    denoiser = GraspDiffusionDenoiser(global_dim=64, local_dim=32, hidden_dim=64)

    R_0_np = sample_uniform_so3_numpy(B)
    R_0 = torch.from_numpy(R_0_np).float().to(device)
    t_0 = torch.randn(B, 3, device=device) * 0.1
    w_0 = torch.empty(B, 1, device=device).uniform_(0.03, 0.08)
    points = torch.randn(B, M, 3, device=device) * 0.15

    loss, metrics = denoiser.compute_loss(R_0, t_0, w_0, points)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "loss_t" in metrics and "loss_R" in metrics and "loss_w" in metrics


def test_generator_deterministic_sampling():
    """Verify 12-step deterministic DDIM sampling produces N=16 valid grasps."""
    denoiser = GraspDiffusionDenoiser(global_dim=64, local_dim=32, hidden_dim=64)
    points = torch.randn(1, 256, 3) * 0.15

    R_out, t_out, w_out = denoiser.sample(points, num_samples=16, num_steps=12, chunk_size=8)
    assert R_out.shape == (16, 3, 3)
    assert t_out.shape == (16, 3)
    assert w_out.shape == (16,)

    # Check rotations satisfy |R^T R - I| < 1e-4
    for R in R_out:
        assert np.max(np.abs(np.dot(R.T, R) - np.eye(3))) < 1e-4
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-4)


def test_scorer_and_temperature_calibration():
    """Verify scorer forward pass and temperature calibration."""
    B = 20
    M = 256
    scorer = GraspSuccessScorer(global_dim=64, local_dim=32, hidden_dim=64)

    R_np = sample_uniform_so3_numpy(B)
    t_np = np.random.randn(B, 3) * 0.05
    w_np = np.random.uniform(0.02, 0.08, size=B)
    points_np = np.random.randn(M, 3) * 0.1

    scores = scorer.score_grasps(R_np, t_np, w_np, points_np, chunk_size=8)
    assert scores.shape == (B,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    # Test ECE calculation
    labels = np.random.randint(0, 2, size=B)
    ece = compute_ece(scores, labels, num_bins=5)
    assert 0.0 <= ece <= 1.0

    # Test temperature fitting
    val_logits = np.random.randn(50)
    val_labels = (val_logits > 0).astype(float)
    init_T = scorer.temperature.item()
    best_T = scorer.fit_temperature(val_logits, val_labels)
    assert best_T > 0.0
