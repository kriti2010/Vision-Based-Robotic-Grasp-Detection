"""DBSCAN clustering of 6-DoF grasps on SO(3) x R^3."""

from typing import List, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

from src.grasp.representation import Grasp
from src.grasp.so3 import geodesic_distance_numpy


def compute_grasp_distance_matrix(
    grasps: List[Grasp],
    alpha: float = 1.0,
    beta: float = 4.0
) -> np.ndarray:
    """Compute pairwise squared distance matrix on SO(3) x R^3.
    
    d^2 = alpha * geodesic_angle(R_i, R_j)^2 + beta * ||p_i - p_j||^2
    """
    N = len(grasps)
    dist_mat = np.zeros((N, N), dtype=np.float64)
    if N <= 1:
        return dist_mat

    rotations = np.array([g.R for g in grasps])  # (N, 3, 3)
    positions = np.array([g.p for g in grasps])  # (N, 3)

    for i in range(N):
        # Geodesic distances from grasp i to all grasps
        # Broadcast R_i with all rotations
        R_i = rotations[i]
        geo_dist = geodesic_distance_numpy(R_i[None, :, :], rotations)  # (N,)
        pos_dist_sq = np.sum((positions[i] - positions) ** 2, axis=-1)  # (N,)
        
        d_sq = alpha * (geo_dist ** 2) + beta * pos_dist_sq
        dist = np.sqrt(np.maximum(d_sq, 0.0))
        dist_mat[i] = dist

    # Symmetrize
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def cluster_grasps_dbscan(
    grasps: List[Grasp],
    eps: float = 0.30,
    min_samples: int = 2,
    alpha: float = 1.0,
    beta: float = 4.0
) -> np.ndarray:
    """Cluster accepted grasps with DBSCAN using hybrid manifold metric.
    
    Returns:
        labels: (N,) array of cluster assignments (-1 indicates noise)
    """
    if len(grasps) == 0:
        return np.array([], dtype=int)
    if len(grasps) == 1:
        return np.array([0], dtype=int)

    dist_mat = compute_grasp_distance_matrix(grasps, alpha=alpha, beta=beta)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_mat)
    return labels
