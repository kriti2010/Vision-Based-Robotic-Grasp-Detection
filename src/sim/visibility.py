"""Visibility checker via ray casting through 3-state voxel occupancy grid."""

from typing import List, Tuple
import numpy as np
from src.sim.fusion import VoxelOccupancyGrid


class VisibilityChecker:
    """Ray-casts through voxel occupancy grid to determine visibility from camera positions."""

    def __init__(self, voxel_grid: VoxelOccupancyGrid):
        self.grid = voxel_grid

    def is_point_visible(
        self,
        camera_eye: np.ndarray,
        target_point: np.ndarray,
        num_steps: int = 30
    ) -> bool:
        """Check if target_point has unoccluded line of sight from camera_eye.
        
        Returns False if ray hits an OCCUPIED voxel before reaching target_point.
        """
        # Linear steps from eye to target point (stopping just before target)
        t = np.linspace(0.05, 0.95, num_steps).reshape(-1, 1)
        ray_pts = camera_eye + t * (target_point - camera_eye)
        
        indices = self.grid.point_to_idx(ray_pts)
        valid = self.grid.is_inside(indices)
        indices = indices[valid]

        if len(indices) == 0:
            return True

        states = self.grid.grid[indices[:, 0], indices[:, 1], indices[:, 2]]
        # If any traversed voxel is OCCUPIED, ray is blocked
        if np.any(states == VoxelOccupancyGrid.OCCUPIED):
            return False
        return True

    def count_visible_unknown_in_points(
        self,
        camera_eye: np.ndarray,
        query_points: np.ndarray,
        num_steps: int = 20
    ) -> int:
        """Count how many query points (which are in UNKNOWN voxels) are visible from camera_eye."""
        if len(query_points) == 0:
            return 0

        # Check which query points lie in UNKNOWN voxels
        idx = self.grid.point_to_idx(query_points)
        valid = self.grid.is_inside(idx)
        unk_mask = np.zeros(len(query_points), dtype=bool)
        unk_mask[valid] = (self.grid.grid[idx[valid, 0], idx[valid, 1], idx[valid, 2]] == VoxelOccupancyGrid.UNKNOWN)

        unk_pts = query_points[unk_mask]
        if len(unk_pts) == 0:
            return 0

        # Vectorized ray marching for unk_pts
        visible_count = 0
        t_vals = np.linspace(0.1, 0.92, num_steps)  # (S,)
        
        for pt in unk_pts:
            # Ray from eye to pt
            pts_along_ray = camera_eye + t_vals[:, None] * (pt - camera_eye)
            r_idx = self.grid.point_to_idx(pts_along_ray)
            in_bounds = self.grid.is_inside(r_idx)
            r_idx = r_idx[in_bounds]
            if len(r_idx) > 0:
                states = self.grid.grid[r_idx[:, 0], r_idx[:, 1], r_idx[:, 2]]
                if not np.any(states == VoxelOccupancyGrid.OCCUPIED):
                    visible_count += 1
            else:
                visible_count += 1

        return visible_count

    def count_visible_unknown_in_region(
        self,
        camera_eye: np.ndarray,
        center: np.ndarray,
        extents: np.ndarray,
        rotation: np.ndarray,
        sample_density: int = 3
    ) -> int:
        """Sample points inside an oriented bounding box and count visible UNKNOWN voxels."""
        half_ext = extents / 2.0
        # Create small regular 3D grid in local frame
        xs = np.linspace(-half_ext[0], half_ext[0], sample_density)
        ys = np.linspace(-half_ext[1], half_ext[1], sample_density)
        zs = np.linspace(-half_ext[2], half_ext[2], sample_density)
        grid_local = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3)

        # Transform to world
        grid_world = np.matmul(grid_local, rotation.T) + center
        return self.count_visible_unknown_in_points(camera_eye, grid_world)
