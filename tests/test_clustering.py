"""Unit tests for DBSCAN mode clustering and uncertainty quantification on synthetic grasps."""

import numpy as np
import pytest

from src.grasp.representation import Grasp
from src.grasp.so3 import exp_so3_numpy
from src.uncertainty.clustering import cluster_grasps_dbscan
from src.uncertainty.ambiguity import UncertaintyEstimator


def test_dbscan_mode_clustering_synthetic_data():
    np.random.seed(42)
    # Create two synthetic clusters in SO(3) x R^3
    # Mode 1: centered around R1, p1
    R1 = exp_so3_numpy(np.array([0.0, 0.0, 0.0]))
    p1 = np.array([0.05, 0.05, 0.10])

    # Mode 2: centered around R2, p2 (far from Mode 1)
    R2 = exp_so3_numpy(np.array([0.0, 0.0, np.pi / 2]))
    p2 = np.array([-0.10, -0.10, 0.10])

    cluster1 = []
    for _ in range(15):
        noise_v = np.random.normal(0, 0.04, size=3)
        noise_p = np.random.normal(0, 0.005, size=3)
        g = Grasp(R=np.matmul(R1, exp_so3_numpy(noise_v)), p=p1 + noise_p, score=0.85)
        cluster1.append(g)

    cluster2 = []
    for _ in range(12):
        noise_v = np.random.normal(0, 0.04, size=3)
        noise_p = np.random.normal(0, 0.005, size=3)
        g = Grasp(R=np.matmul(R2, exp_so3_numpy(noise_v)), p=p2 + noise_p, score=0.60)
        cluster2.append(g)

    all_grasps = cluster1 + cluster2
    labels = cluster_grasps_dbscan(all_grasps, eps=0.30, min_samples=3)

    # Must find at least two distinct valid clusters
    unique_labels = [l for l in np.unique(labels) if l != -1]
    assert len(unique_labels) >= 2, f"Expected at least 2 clusters, got {len(unique_labels)}"

    # All members of cluster 1 should share a common label
    labels_c1 = labels[:15]
    labels_c2 = labels[15:]
    assert np.all(labels_c1 == labels_c1[0])
    assert np.all(labels_c2 == labels_c2[0])
    assert labels_c1[0] != labels_c2[0]


def test_uncertainty_estimator_identifies_best_mode():
    np.random.seed(42)
    R1 = exp_so3_numpy(np.array([0.0, 0.0, 0.0]))
    p1 = np.array([0.0, 0.0, 0.10])

    grasps = []
    # 10 grasps with high score (0.9) tightly clustered
    for _ in range(10):
        noise_v = np.random.normal(0, 0.02, size=3)
        noise_p = np.random.normal(0, 0.002, size=3)
        grasps.append(Grasp(R=np.matmul(R1, exp_so3_numpy(noise_v)), p=p1 + noise_p, score=0.90))

    # 10 grasps with lower score (0.55) spread out
    for _ in range(10):
        noise_v = np.random.normal(0, 0.20, size=3)
        noise_p = np.random.normal(0, 0.04, size=3)
        grasps.append(Grasp(R=exp_so3_numpy(noise_v), p=p1 + noise_p, score=0.55))

    estimator = UncertaintyEstimator(tau_s=0.50)
    res = estimator.evaluate_uncertainty(grasps, total_k=20)

    assert res.rho_t == 1.0  # all 20 have score >= 0.50
    assert res.top_grasp is not None
    assert res.top_grasp.score >= 0.89
    assert res.U_t < 0.10  # best mode is the tight cluster
