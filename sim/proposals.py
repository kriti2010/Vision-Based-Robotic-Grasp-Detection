"""Antipodal grasp proposal generation and negative perturbation.

Specification:
- Antipodal grasps from the fused full-view cloud.
- PCA normals from 16 neighbours.
- Two ray casts or opposing contact search with width in [0.015, 0.075] m.
- Approach axis perpendicular to the closing axis and pointing down with a_z < -0.3.
- Grasp = (R, t, w) with closing axis as R's first column, approach axis as third column.
- Perturbed negatives: rotation noise 0.6 rad, translation 2 cm, width 1 cm.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
import pybullet as p


def compute_pca_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Compute surface normals using PCA on k nearest neighbors.
    
    Args:
        points: (N, 3) array
        k: number of nearest neighbors (16)
    Returns:
        normals: (N, 3) unit normal vectors
    """
    N = len(points)
    if N < k:
        return np.tile(np.array([0.0, 0.0, 1.0]), (N, 1))

    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)

    normals = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        neighbor_pts = points[indices[i]]  # (k, 3)
        centered = neighbor_pts - np.mean(neighbor_pts, axis=0)
        cov = np.matmul(centered.T, centered) / float(k)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # eigenvector for smallest eigenvalue
        # Orient outward / upward (towards +z)
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(norms > 1e-6, normals / np.maximum(norms, 1e-6), np.array([0.0, 0.0, 1.0]))
    return normals.astype(np.float32)


def sample_antipodal_grasps(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    num_grasps: int = 24,
    min_width: float = 0.015,
    max_width: float = 0.075,
    max_candidates_to_try: int = 500,
    rng: Optional[np.random.RandomState] = None,
    physics_client: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Sample antipodal grasps from point cloud.
    
    Returns list of (R, t, w):
    - R: (3, 3) rotation matrix: col 0 = closing axis, col 1 = binormal, col 2 = approach
    - t: (3,) grasp centre in meters
    - w: float grasp width in meters
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if normals is None:
        normals = compute_pca_normals(points, k=16)

    tree = cKDTree(points)
    grasps = []

    # Permute candidate indices
    cand_indices = rng.permutation(len(points))

    for idx in cand_indices:
        if len(grasps) >= num_grasps:
            break

        p1 = points[idx]
        n1 = normals[idx]

        # Search for opposing points along inward normal -n1 within [min_width, max_width]
        # Opposing surface normal should have n1 . n2 < -0.3
        search_radius = max_width
        neighbor_indices = tree.query_ball_point(p1, r=search_radius)

        valid_pairs = []
        for n_idx in neighbor_indices:
            if n_idx == idx:
                continue
            p2 = points[n_idx]
            diff = p2 - p1
            dist = float(np.linalg.norm(diff))
            if dist < min_width or dist > max_width:
                continue

            # Direction from p1 to p2
            dir_12 = diff / dist
            # Must align somewhat with inward normal -n1
            cos_align = float(np.dot(-n1, dir_12))
            if cos_align < 0.3:
                continue

            n2 = normals[n_idx]
            # Antipodal condition: opposing normals
            cos_oppose = float(np.dot(n1, n2))
            if cos_oppose < -0.2:
                valid_pairs.append((p2, dist, dir_12))

        if not valid_pairs:
            # Fallback ray-cast if physics_client is provided
            if physics_client is not None:
                ray_from = (p1 - 0.005 * n1).tolist()
                ray_to = (p1 - (max_width + 0.01) * n1).tolist()
                res = p.rayTest(ray_from, ray_to, physicsClientId=physics_client)
                if res and res[0][0] >= 0:
                    hit_pos = np.array(res[0][3])
                    hit_dist = float(np.linalg.norm(hit_pos - p1))
                    if min_width <= hit_dist <= max_width:
                        valid_pairs.append((hit_pos, hit_dist, (hit_pos - p1) / hit_dist))

        if not valid_pairs:
            continue

        # Pick best or random valid pair
        p2, width, closing = valid_pairs[rng.randint(len(valid_pairs))]

        # Grasp center
        t = (p1 + p2) / 2.0

        # Closing axis is R's first column (unit vector)
        c = closing / np.linalg.norm(closing)

        # Approach axis must be perpendicular to closing axis and pointing down with a_z < -0.3
        # Start from downward vector [0, 0, -1]
        down = np.array([0.0, 0.0, -1.0])
        # Project downward vector onto plane orthogonal to c: a = down - (down . c) * c
        a = down - float(np.dot(down, c)) * c
        norm_a = np.linalg.norm(a)
        if norm_a < 1e-4:
            continue
        a = a / norm_a

        # If a_z >= -0.3, check if tilting can satisfy a_z < -0.3
        if a[2] >= -0.3:
            continue

        # Invert if necessary to ensure a[2] < -0.3
        if a[2] > 0:
            a = -a

        if a[2] >= -0.3:
            continue

        # Binormal b = a x c so that det([c, b, a]) = +1
        b = np.cross(a, c)
        norm_b = np.linalg.norm(b)
        if norm_b < 1e-4:
            continue
        b = b / norm_b

        # Re-orthogonalize approach to ensure exact SO(3)
        a = np.cross(c, b)
        a = a / np.linalg.norm(a)

        # Ensure a_z < -0.3
        if a[2] >= -0.3:
            continue

        R = np.column_stack([c, b, a])
        # Ensure det(R) == 1
        if np.linalg.det(R) < 0:
            b = -b
            R = np.column_stack([c, b, a])

        grasps.append((R.astype(np.float32), t.astype(np.float32), float(width)))

    # If fewer than num_grasps found, augment with slight variations
    while len(grasps) < num_grasps and len(grasps) > 0:
        base_R, base_t, base_w = grasps[rng.randint(len(grasps))]
        # Small jitter
        d_roll = rng.uniform(-0.15, 0.15)
        c, b, a = base_R[:, 0], base_R[:, 1], base_R[:, 2]
        # Rotate c and b around a
        c_new = np.cos(d_roll) * c + np.sin(d_roll) * b
        b_new = np.cross(a, c_new)
        R_new = np.column_stack([c_new, b_new, a])
        t_new = base_t + rng.normal(0, 0.003, size=3)
        w_new = float(np.clip(base_w + rng.uniform(-0.005, 0.005), min_width, max_width))
        grasps.append((R_new.astype(np.float32), t_new.astype(np.float32), w_new))

    return grasps[:num_grasps]


def perturb_grasps_negative(
    grasps: List[Tuple[np.ndarray, np.ndarray, float]],
    rng: Optional[np.random.RandomState] = None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Perturb grasps to create hard negatives:
    - rotation noise 0.6 rad
    - translation 2 cm (0.02 m)
    - width 1 cm (0.01 m)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    negatives = []
    for R, t, w in grasps:
        # 1. Rotation noise 0.6 rad
        axis = rng.randn(3)
        axis = axis / np.maximum(np.linalg.norm(axis), 1e-6)
        angle = 0.6  # 0.6 rad
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        # Rodrigues formula
        R_noise = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * np.matmul(K, K)
        R_neg = np.matmul(R, R_noise)

        # 2. Translation noise 2 cm
        t_dir = rng.randn(3)
        t_dir = t_dir / np.maximum(np.linalg.norm(t_dir), 1e-6)
        t_neg = t + 0.02 * t_dir

        # 3. Width noise 1 cm
        w_delta = rng.choice([-0.01, 0.01])
        w_neg = float(np.clip(w + w_delta, 0.015, 0.085))

        negatives.append((R_neg.astype(np.float32), t_neg.astype(np.float32), w_neg))

    return negatives
