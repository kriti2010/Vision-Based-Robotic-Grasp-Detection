"""Multi-view fusion and point cloud subsampling to M=2048 points (xyz + rgb)."""

from typing import List, Tuple, Optional
import numpy as np
from sim.camera import CameraRig, CameraView


def fuse_and_subsample(
    rig: CameraRig,
    view_indices: List[int],
    plane_id: int = 0,
    num_points: int = 2048,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse point clouds from given camera view indices and subsample to M points.
    
    Returns:
        points: (M, 3) float32 in world frame (m)
        colors: (M, 3) float32 in [0, 1]
    """
    if rng is None:
        rng = np.random.RandomState(42)

    all_pts = []
    all_colors = []

    for idx in view_indices:
        view = rig.candidate_views[idx]
        rgb, depth, seg = rig.render(view)
        pts, colors = rig.backproject(rgb, depth, seg, view, plane_id=plane_id)
        if len(pts) > 0:
            all_pts.append(pts)
            all_colors.append(colors)

    if len(all_pts) == 0:
        # Fallback dummy cloud if occluded
        pts_fused = np.zeros((num_points, 3), dtype=np.float32)
        cols_fused = np.ones((num_points, 3), dtype=np.float32) * 0.5
        return pts_fused, cols_fused

    pts_fused = np.concatenate(all_pts, axis=0)
    cols_fused = np.concatenate(all_colors, axis=0)

    n_total = len(pts_fused)
    if n_total >= num_points:
        idx_sample = rng.choice(n_total, size=num_points, replace=False)
    else:
        idx_sample = rng.choice(n_total, size=num_points, replace=True)

    return pts_fused[idx_sample].astype(np.float32), cols_fused[idx_sample].astype(np.float32)
