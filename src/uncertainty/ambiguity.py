"""Mode identification, Riemannian Karcher mean, and ambiguity score U_t."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from src.grasp.representation import Grasp
from src.grasp.so3 import karcher_mean_so3_numpy, geodesic_distance_numpy
from src.uncertainty.clustering import cluster_grasps_dbscan


@dataclass
class UncertaintyResult:
    """Full uncertainty quantification result for a viewpoint."""
    rho_t: float                    # acceptance fraction (accepted / K)
    U_t: float                      # ambiguity score (mean squared distance to manifold mean)
    mode_entropy: float             # entropy of cluster distribution (ablation)
    best_mode_grasps: List[Grasp]   # grasps belonging to best mode m*
    top_grasp: Optional[Grasp]      # highest scoring grasp in best mode
    num_clusters: int               # total identified DBSCAN clusters


class UncertaintyEstimator:
    """Quantifies grasp uncertainty via accepted fraction rho_t and mode spread U_t."""

    def __init__(
        self,
        tau_s: float = 0.50,
        eps: float = 0.30,
        min_samples: int = 2,
        alpha: float = 1.0,
        beta: float = 4.0
    ):
        self.tau_s = tau_s
        self.eps = eps
        self.min_samples = min_samples
        self.alpha = alpha
        self.beta = beta

    def evaluate_uncertainty(
        self,
        all_samples: List[Grasp],
        total_k: int
    ) -> UncertaintyResult:
        """Compute rho_t, DBSCAN clusters, best mode m*, and manifold spread U_t.
        
        Args:
            all_samples: K generated grasp samples with .score set by evaluator
            total_k: Total number of sampled grasps K
        """
        # Filter accepted grasps: score >= tau_s
        accepted = [g for g in all_samples if g.score >= self.tau_s]
        rho_t = len(accepted) / max(1, total_k)

        if len(accepted) == 0:
            # Fallback: if none pass tau_s, take top scoring sample but report maximum uncertainty
            best_sample = max(all_samples, key=lambda g: g.score) if len(all_samples) > 0 else None
            return UncertaintyResult(
                rho_t=0.0,
                U_t=1.0,
                mode_entropy=0.0,
                best_mode_grasps=[best_sample] if best_sample else [],
                top_grasp=best_sample,
                num_clusters=0
            )

        # Cluster accepted grasps with DBSCAN
        labels = cluster_grasps_dbscan(
            accepted,
            eps=self.eps,
            min_samples=self.min_samples,
            alpha=self.alpha,
            beta=self.beta
        )

        unique_labels = [l for l in np.unique(labels) if l != -1]
        
        # Calculate cluster sizes and mode entropy
        if len(unique_labels) > 0:
            cluster_sizes = [np.sum(labels == l) for l in unique_labels]
            probs = np.array(cluster_sizes, dtype=np.float64) / np.sum(cluster_sizes)
            mode_entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
            
            # Best mode m* = cluster with highest mean evaluator score
            best_cluster_label = None
            best_mean_score = -1.0
            
            for l in unique_labels:
                m_grasps = [accepted[i] for i in range(len(accepted)) if labels[i] == l]
                mean_sc = np.mean([g.score for g in m_grasps])
                if mean_sc > best_mean_score:
                    best_mean_score = mean_sc
                    best_cluster_label = l

            best_mode_grasps = [accepted[i] for i in range(len(accepted)) if labels[i] == best_cluster_label]
            num_clusters = len(unique_labels)
        else:
            # All samples are considered noise or small cluster
            mode_entropy = 0.0
            best_mode_grasps = accepted
            num_clusters = 1

        # Calculate Karcher mean on SO(3) and Euclidean mean on R^3 for best mode
        rotations = np.array([g.R for g in best_mode_grasps])
        positions = np.array([g.p for g in best_mode_grasps])

        R_mean = karcher_mean_so3_numpy(rotations)
        p_mean = np.mean(positions, axis=0)

        # U_t = mean squared distance of its members to the manifold mean
        geo_dists = geodesic_distance_numpy(R_mean[None, :, :], rotations)  # (M,)
        pos_dists_sq = np.sum((positions - p_mean) ** 2, axis=-1)  # (M,)

        d_sq = self.alpha * (geo_dists ** 2) + self.beta * pos_dists_sq
        U_t = float(np.mean(d_sq))

        # Top-scoring grasp in best mode
        top_grasp = max(best_mode_grasps, key=lambda g: g.score)

        return UncertaintyResult(
            rho_t=rho_t,
            U_t=U_t,
            mode_entropy=mode_entropy,
            best_mode_grasps=best_mode_grasps,
            top_grasp=top_grasp,
            num_clusters=num_clusters
        )
