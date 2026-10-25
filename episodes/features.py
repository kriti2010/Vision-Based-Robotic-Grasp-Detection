"""Feature extraction for episode risk models: symmetry-aware distance,
DBSCAN sample dispersion, and cross-view progression features.

Specification:
- Symmetry-aware distance:
  d = |t_i - t_j| / 0.03 + min(angle(R_i, R_j), angle(R_i * Rz(pi), R_j)) / 0.5 + |w_i - w_j| / 0.02
- DBSCAN eps 1.0 on top 50% of samples (16 samples).
- Score-weighted cluster masses M_c = sum_{i in c} s_i*.
- Largest-cluster mass and normalized entropy.
- Cross-view features: view_idx / (T - 1), distance between consecutive best grasps
  (capped at 10 then / 10), top-1 minus top-2 score gap.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from sklearn.cluster import DBSCAN
from models.so3 import log_so3_numpy


# 180 degree rotation around approach axis Z: Rz(pi) = diag([-1, -1, 1])
RZ_PI = np.diag([-1.0, -1.0, 1.0])


def symmetry_aware_distance(
    R1: np.ndarray, t1: np.ndarray, w1: float,
    R2: np.ndarray, t2: np.ndarray, w2: float,
) -> float:
    """Compute symmetry-aware distance between two grasps.
    
    d = ||t1 - t2|| / 0.03 + min(angle(R1, R2), angle(R1 * Rz(pi), R2)) / 0.5 + |w1 - w2| / 0.02
    """
    dt = float(np.linalg.norm(t1 - t2)) / 0.03
    dw = abs(float(w1) - float(w2)) / 0.02

    # Relative rotation without flip
    R_rel1 = np.matmul(R1.T, R2)
    ang1 = float(np.linalg.norm(log_so3_numpy(R_rel1)))

    # Relative rotation with 180-deg flip around approach axis
    R1_flipped = np.matmul(R1, RZ_PI)
    R_rel2 = np.matmul(R1_flipped.T, R2)
    ang2 = float(np.linalg.norm(log_so3_numpy(R_rel2)))

    d_ang = min(ang1, ang2) / 0.5
    return dt + d_ang + dw


def compute_dispersion_features(
    R_samples: np.ndarray,
    t_samples: np.ndarray,
    w_samples: np.ndarray,
    scores: np.ndarray,
    top_fraction: float = 0.5,
    eps: float = 1.0,
) -> Dict[str, float]:
    """Compute DBSCAN dispersion features on top 50% of samples.
    
    Returns:
        dict containing:
        - largest_cluster_mass
        - normalized_entropy
        - num_clusters
    """
    N = len(scores)
    k_top = max(2, int(N * top_fraction))

    # Top k_top indices sorted by descending score
    top_idx = np.argsort(scores)[::-1][:k_top]
    R_top = R_samples[top_idx]
    t_top = t_samples[top_idx]
    w_top = w_samples[top_idx]
    s_top = scores[top_idx]

    # Compute pairwise distance matrix
    dist_matrix = np.zeros((k_top, k_top), dtype=np.float64)
    for i in range(k_top):
        for j in range(i + 1, k_top):
            d = symmetry_aware_distance(
                R_top[i], t_top[i], w_top[i],
                R_top[j], t_top[j], w_top[j],
            )
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # DBSCAN with eps = 1.0 on precomputed distances
    db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    # Score-weighted cluster masses (excluding noise label -1, or treating noise as separate singletons)
    unique_labels = [lbl for lbl in np.unique(labels) if lbl >= 0]
    total_score = float(np.sum(s_top))
    if total_score <= 1e-8:
        total_score = 1.0

    if len(unique_labels) == 0:
        # All noise points: each is its own cluster
        masses = [float(s) for s in s_top]
    else:
        masses = []
        for lbl in unique_labels:
            masses.append(float(np.sum(s_top[labels == lbl])))
        # Add noise points as individual singleton masses
        for i in range(k_top):
            if labels[i] == -1:
                masses.append(float(s_top[i]))

    masses = np.array(masses, dtype=np.float64)
    sum_masses = np.sum(masses)
    if sum_masses <= 1e-8:
        sum_masses = 1.0

    probs = masses / sum_masses
    largest_cluster_mass = float(np.max(probs))

    # Normalized entropy: - sum(p * ln(p)) / ln(max(len(probs), 2))
    entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    norm_denom = np.log(max(len(probs), 2))
    norm_entropy = float(entropy / norm_denom)

    return {
        "largest_cluster_mass": largest_cluster_mass,
        "normalized_entropy": norm_entropy,
        "num_clusters": float(len(unique_labels)),
    }


def compute_cross_view_features(
    view_idx: int,
    T: int,
    best_grasp_curr: Tuple[np.ndarray, np.ndarray, float],
    best_grasp_prev: Optional[Tuple[np.ndarray, np.ndarray, float]],
    scores: np.ndarray,
) -> Dict[str, float]:
    """Compute cross-view progression features."""
    # 1. view index / (T - 1)
    norm_view_idx = float(view_idx) / max(1.0, float(T - 1))

    # 2. Distance between consecutive best grasps capped at 10 then / 10
    if best_grasp_prev is None:
        norm_dist_prev = 0.0
    else:
        R_curr, t_curr, w_curr = best_grasp_curr
        R_prev, t_prev, w_prev = best_grasp_prev
        d = symmetry_aware_distance(R_curr, t_curr, w_curr, R_prev, t_prev, w_prev)
        norm_dist_prev = float(min(d, 10.0) / 10.0)

    # 3. Top-1 minus top-2 score gap
    sorted_scores = np.sort(scores)[::-1]
    if len(sorted_scores) >= 2:
        score_gap = float(sorted_scores[0] - sorted_scores[1])
    else:
        score_gap = float(sorted_scores[0])

    return {
        "norm_view_idx": norm_view_idx,
        "norm_dist_prev": norm_dist_prev,
        "score_gap": score_gap,
    }
