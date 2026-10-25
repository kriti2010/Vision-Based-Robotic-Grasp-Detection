"""Acceptance Test 2:
Local-feature geometry matches a plain-numpy reference.
"""

import numpy as np
import torch
import pytest
from models.pointnet import extract_local_points_numpy, extract_local_points_torch
from models.so3 import sample_uniform_so3_numpy


def test_local_feature_geometry_matches_plain_numpy():
    """Verify PyTorch local point extraction in grasp frame matches plain-numpy reference."""
    rng = np.random.RandomState(42)

    for trial in range(25):
        num_points = 2048
        points_np = rng.uniform(-0.2, 0.2, size=(num_points, 3)).astype(np.float32)

        # Random grasp R, t
        R_np = sample_uniform_so3_numpy(1, rng=rng)[0].astype(np.float32)
        t_np = rng.uniform(-0.1, 0.1, size=3).astype(np.float32)

        # Plain numpy reference
        local_ref = extract_local_points_numpy(points_np, R_np, t_np, k=64)

        # PyTorch implementation
        points_torch = torch.from_numpy(points_np).unsqueeze(0)  # (1, M, 3)
        R_torch = torch.from_numpy(R_np).unsqueeze(0)            # (1, 3, 3)
        t_torch = torch.from_numpy(t_np).unsqueeze(0)            # (1, 3)

        local_torch = extract_local_points_torch(points_torch, R_torch, t_torch, k=64).squeeze(0).numpy()

        # Both should match closely:
        # Note: topk ordering for identical distances could differ, but on continuous uniform floats they are unique.
        # Check max absolute difference
        diff = np.max(np.abs(local_ref - local_torch))
        assert diff < 1e-5, f"Trial {trial}: Local feature mismatch {diff} >= 1e-5"
