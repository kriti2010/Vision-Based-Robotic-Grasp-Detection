"""Multi-view point cloud fusion and 3-state voxel occupancy grid."""

from typing import Tuple, List, Optional
import numpy as np
import open3d as o3d


class VoxelOccupancyGrid:
    """3-state voxel occupancy grid: UNKNOWN (-1), FREE (0), OCCUPIED (1)."""
    UNKNOWN: int = -1
    FREE: int = 0
    OCCUPIED: int = 1

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        voxel_size: float = 0.015
    ):
        self.min_bound = np.asarray(bounds[0], dtype=np.float32)
        self.max_bound = np.asarray(bounds[1], dtype=np.float32)
        self.voxel_size = float(voxel_size)

        self.dims = np.ceil((self.max_bound - self.min_bound) / self.voxel_size).astype(int)
        # Initialize all voxels as UNKNOWN
        self.grid = np.full(self.dims, fill_value=self.UNKNOWN, dtype=np.int8)

    def point_to_idx(self, pts: np.ndarray) -> np.ndarray:
        """Convert (N, 3) 3D world coordinates to (N, 3) integer voxel indices."""
        indices = np.floor((pts - self.min_bound) / self.voxel_size).astype(int)
        return indices

    def idx_to_point(self, indices: np.ndarray) -> np.ndarray:
        """Convert (N, 3) voxel indices to (N, 3) voxel center coordinates."""
        centers = self.min_bound + (indices + 0.5) * self.voxel_size
        return centers

    def is_inside(self, indices: np.ndarray) -> np.ndarray:
        """Check whether indices are within grid bounds."""
        return np.all((indices >= 0) & (indices < self.dims), axis=-1)

    def update_view(
        self,
        camera_eye: np.ndarray,
        surface_points: np.ndarray,
        max_ray_steps: int = 40
    ):
        """Update occupancy grid given camera position and observed surface points.
        
        Marks voxels containing surface points as OCCUPIED.
        Marks voxels along lines of sight between camera and surface points as FREE.
        """
        if len(surface_points) == 0:
            return

        # Mark surface voxels as OCCUPIED
        surf_idx = self.point_to_idx(surface_points)
        valid_surf = self.is_inside(surf_idx)
        surf_idx = surf_idx[valid_surf]
        self.grid[surf_idx[:, 0], surf_idx[:, 1], surf_idx[:, 2]] = self.OCCUPIED

        # Ray-march to carve FREE space
        # Sample points along ray from eye to surface point
        sub_pts = surface_points[::max(1, len(surface_points) // 250)]  # sub-sample rays for fast CPU execution
        t_vals = np.linspace(0.1, 0.95, max_ray_steps).reshape(-1, 1, 1)  # (S, 1, 1)
        
        # Ray points: eye + t * (pt - eye)
        dirs = sub_pts - camera_eye  # (M, 3)
        ray_pts = camera_eye + t_vals * dirs  # (S, M, 3)
        ray_pts = ray_pts.reshape(-1, 3)

        ray_idx = self.point_to_idx(ray_pts)
        valid_ray = self.is_inside(ray_idx)
        ray_idx = ray_idx[valid_ray]

        # Update ray voxels: if currently UNKNOWN, mark as FREE. (Do not overwrite OCCUPIED)
        curr_state = self.grid[ray_idx[:, 0], ray_idx[:, 1], ray_idx[:, 2]]
        free_mask = curr_state == self.UNKNOWN
        free_idx = ray_idx[free_mask]
        self.grid[free_idx[:, 0], free_idx[:, 1], free_idx[:, 2]] = self.FREE

    def count_unknown_in_box(
        self,
        center: np.ndarray,
        extents: np.ndarray,
        rotation: np.ndarray
    ) -> int:
        """Count number of UNKNOWN voxels inside an oriented bounding box."""
        # Find candidate voxels in AABB enclosing the OBB
        half_ext = extents / 2.0
        # Corners of local box
        corners_local = np.array([
            [sx * half_ext[0], sy * half_ext[1], sz * half_ext[2]]
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ])
        corners_world = np.matmul(corners_local, rotation.T) + center
        
        aabb_min = np.clip(np.min(corners_world, axis=0), self.min_bound, self.max_bound)
        aabb_max = np.clip(np.max(corners_world, axis=0), self.min_bound, self.max_bound)

        idx_min = self.point_to_idx(aabb_min.reshape(1, 3))[0]
        idx_max = self.point_to_idx(aabb_max.reshape(1, 3))[0] + 1
        
        idx_min = np.clip(idx_min, 0, self.dims)
        idx_max = np.clip(idx_max, 0, self.dims)

        if np.any(idx_min >= idx_max):
            return 0

        # Sub-volume
        sub_grid = self.grid[
            idx_min[0]:idx_max[0],
            idx_min[1]:idx_max[1],
            idx_min[2]:idx_max[2]
        ]
        
        # Get coordinates of UNKNOWN voxels in sub-volume
        unk_coords = np.argwhere(sub_grid == self.UNKNOWN)
        if len(unk_coords) == 0:
            return 0

        # Global indices
        unk_global_idx = unk_coords + idx_min
        pts = self.idx_to_point(unk_global_idx)

        # Transform to local box frame
        pts_local = np.matmul(pts - center, rotation)
        inside = np.all(np.abs(pts_local) <= half_ext, axis=1)
        return int(np.sum(inside))


def fuse_point_clouds(clouds: List[np.ndarray], voxel_size: float = 0.008) -> np.ndarray:
    """Merge multiple point clouds and voxel-downsample with Open3D."""
    if not clouds:
        return np.empty((0, 3), dtype=np.float32)
        
    all_pts = np.vstack(clouds)
    if len(all_pts) == 0:
        return np.empty((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down_pcd.points, dtype=np.float32)
