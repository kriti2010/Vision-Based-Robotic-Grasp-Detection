"""Tests for SO(3) operations: exp, log, geodesic distance, Karcher mean, and valid rotation projections."""

import numpy as np
import pytest
import torch
from src.grasp.so3 import (
    exp_so3_numpy, log_so3_numpy, project_so3_numpy,
    geodesic_distance_numpy, karcher_mean_so3_numpy,
    exp_so3_torch, log_so3_torch, project_so3_torch
)


def test_exp_log_roundtrip_numpy():
    np.random.seed(42)
    # Test zero vector
    v_zero = np.zeros(3)
    R_zero = exp_so3_numpy(v_zero)
    assert np.allclose(R_zero, np.eye(3), atol=1e-6)
    assert np.allclose(log_so3_numpy(R_zero), v_zero, atol=1e-6)
    
    # Test small angles
    v_small = np.array([1e-5, -2e-5, 3e-5])
    R_small = exp_so3_numpy(v_small)
    v_rec = log_so3_numpy(R_small)
    assert np.allclose(v_small, v_rec, atol=1e-6)
    
    # Test random rotations
    for _ in range(50):
        # generate random axis-angle with norm < pi
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0.01, np.pi - 0.05)
        v = axis * angle
        
        R = exp_so3_numpy(v)
        # Check orthogonality
        assert np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)
        
        v_recovered = log_so3_numpy(R)
        assert np.allclose(v, v_recovered, atol=1e-4)


def test_exp_log_roundtrip_torch():
    torch.manual_seed(42)
    for _ in range(20):
        axis = torch.randn(3)
        axis = axis / torch.norm(axis)
        angle = torch.rand(1) * (np.pi - 0.05)
        v = axis * angle
        
        R = exp_so3_torch(v)
        # Check orthogonality
        assert torch.allclose(torch.mm(R, R.t()), torch.eye(3), atol=1e-5)
        assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-5)
        
        v_recovered = log_so3_torch(R)
        assert torch.allclose(v, v_recovered, atol=1e-4)


def test_geodesic_distance_symmetry():
    np.random.seed(42)
    for _ in range(25):
        v1 = np.random.randn(3)
        v2 = np.random.randn(3)
        R1 = exp_so3_numpy(v1)
        R2 = exp_so3_numpy(v2)
        
        d12 = geodesic_distance_numpy(R1, R2)
        d21 = geodesic_distance_numpy(R2, R1)
        assert np.isclose(d12, d21, atol=1e-6)
        assert d12 >= 0.0
        
        # Distance to self is zero
        assert np.isclose(geodesic_distance_numpy(R1, R1), 0.0, atol=1e-6)


def test_valid_rotation_matrices_from_sampling():
    # Perturb identity or arbitrary matrix with noise and project
    np.random.seed(123)
    random_matrices = np.random.randn(50, 3, 3)
    R_proj = project_so3_numpy(random_matrices)
    for R in R_proj:
        assert np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_karcher_mean_so3():
    np.random.seed(42)
    # Generate a cluster of rotations around a ground truth center R_true
    v_center = np.array([0.2, -0.3, 0.4])
    R_true = exp_so3_numpy(v_center)
    
    # Perturb with small zero-mean normal noise
    perturbations = np.random.normal(0, 0.05, size=(30, 3))
    rotations = [np.matmul(R_true, exp_so3_numpy(dv)) for dv in perturbations]
    rotations = np.array(rotations)
    
    R_mean = karcher_mean_so3_numpy(rotations)
    
    # The mean should be very close to R_true
    dist = geodesic_distance_numpy(R_true, R_mean)
    assert dist < 0.05
    # Orthogonality of mean
    assert np.allclose(np.dot(R_mean, R_mean.T), np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R_mean), 1.0, atol=1e-5)
