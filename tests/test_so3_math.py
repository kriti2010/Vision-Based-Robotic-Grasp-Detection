"""Acceptance Test 1:
- SO(3) exp/log round trip error < 1e-5.
- Sampled rotations satisfy |R^T R - I| < 1e-5.
"""

import numpy as np
import torch
import pytest
from models.so3 import (
    exp_so3_numpy, log_so3_numpy, project_so3_numpy, sample_uniform_so3_numpy,
    exp_so3_torch, log_so3_torch, rot_to_6d, rot_from_6d
)


def test_so3_exp_log_roundtrip_numpy():
    """Verify numpy SO(3) exp/log round trip error < 1e-5 across diverse angles."""
    rng = np.random.RandomState(42)

    # 1. Zero and tiny angles
    v_zero = np.zeros(3)
    assert np.allclose(log_so3_numpy(exp_so3_numpy(v_zero)), v_zero, atol=1e-8)

    v_small = np.array([1e-5, -2e-5, 3e-5])
    assert np.allclose(log_so3_numpy(exp_so3_numpy(v_small)), v_small, atol=1e-8)

    # 2. Random vectors with norm up to pi - 0.05
    for _ in range(100):
        axis = rng.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = rng.uniform(0.01, np.pi - 0.02)
        v = axis * angle

        R = exp_so3_numpy(v)
        # Check orthogonality
        assert np.max(np.abs(np.dot(R.T, R) - np.eye(3))) < 1e-6
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

        v_rec = log_so3_numpy(R)
        err = np.linalg.norm(v - v_rec)
        assert err < 1e-5, f"Roundtrip error {err} exceeds 1e-5"


def test_so3_exp_log_roundtrip_torch():
    """Verify torch SO(3) exp/log round trip error < 1e-5."""
    torch.manual_seed(42)
    for _ in range(50):
        axis = torch.randn(3, dtype=torch.float64)
        axis = axis / torch.norm(axis)
        angle = torch.rand(1, dtype=torch.float64) * (np.pi - 0.02)
        v = axis * angle

        R = exp_so3_torch(v)
        assert torch.max(torch.abs(torch.matmul(R.T, R) - torch.eye(3, dtype=torch.float64))) < 1e-6
        assert torch.isclose(torch.det(R), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

        v_rec = log_so3_torch(R)
        err = float(torch.norm(v - v_rec))
        assert err < 1e-5, f"Torch roundtrip error {err} exceeds 1e-5"


def test_sampled_rotations_orthonormality():
    """Verify sampled rotations satisfy |R^T R - I| < 1e-5."""
    rotations = sample_uniform_so3_numpy(500, rng=np.random.RandomState(42))
    for R in rotations:
        ortho_err = np.max(np.abs(np.dot(R.T, R) - np.eye(3)))
        assert ortho_err < 1e-5, f"Orthonormality error {ortho_err} exceeds 1e-5"
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_6d_rotation_representation():
    """Verify 6D representation Gram-Schmidt conversion roundtrip."""
    rotations = sample_uniform_so3_numpy(100, rng=np.random.RandomState(123))
    R_torch = torch.from_numpy(rotations).float()

    d6 = rot_to_6d(R_torch)
    R_rec = rot_from_6d(d6)

    diff = torch.max(torch.abs(R_torch - R_rec)).item()
    assert diff < 1e-5, f"6D conversion difference {diff} exceeds 1e-5"
