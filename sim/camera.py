"""RGB-D Camera simulation and Golden-Spiral Hemisphere views.

128x128 resolution, FOV 60 deg, near 0.02 m, far 1.5 m.
16 candidate views on a golden-spiral hemisphere (radius 0.5 m, elevation 30-70 deg)
looking at (0, 0, 0.03).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pybullet as p


@dataclass
class CameraView:
    view_index: int
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    view_matrix: np.ndarray  # 4x4 matrix
    proj_matrix: np.ndarray  # 4x4 matrix
    c2w_rotation: np.ndarray  # 3x3
    c2w_translation: np.ndarray  # 3


def compute_golden_spiral_views(
    num_views: int = 16,
    radius: float = 0.5,
    min_elev_deg: float = 30.0,
    max_elev_deg: float = 70.0,
    target: Optional[np.ndarray] = None,
    fov: float = 60.0,
    aspect: float = 1.0,
    near: float = 0.02,
    far: float = 1.5,
    physics_client: int = 0,
) -> List[CameraView]:
    """Generate 16 golden-spiral candidate viewpoints on a hemisphere."""
    if target is None:
        target = np.array([0.0, 0.0, 0.03])

    views = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.39996 rad (~137.5 deg)

    for i in range(num_views):
        t = float(i) / max(1.0, float(num_views - 1))
        elevation_deg = min_elev_deg + (max_elev_deg - min_elev_deg) * t
        theta = np.radians(elevation_deg)
        phi = (i * golden_angle) % (2.0 * np.pi)

        eye = np.array([
            radius * np.cos(theta) * np.cos(phi) + target[0],
            radius * np.cos(theta) * np.sin(phi) + target[1],
            radius * np.sin(theta) + target[2],
        ])
        up = np.array([0.0, 0.0, 1.0])

        v_mat_list = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=physics_client,
        )
        p_mat_list = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far,
            physicsClientId=physics_client,
        )

        V = np.array(v_mat_list, dtype=np.float64).reshape((4, 4), order="F")
        P = np.array(p_mat_list, dtype=np.float64).reshape((4, 4), order="F")

        # Invert view matrix to get camera-to-world transform
        V_inv = np.linalg.inv(V)
        c2w_R = V_inv[:3, :3]
        c2w_t = V_inv[:3, 3]

        views.append(
            CameraView(
                view_index=i,
                eye=eye,
                target=target,
                up=up,
                view_matrix=V,
                proj_matrix=P,
                c2w_rotation=c2w_R,
                c2w_translation=c2w_t,
            )
        )
    return views


class CameraRig:
    """Renders 128x128 RGB-D images and back-projects partial point clouds."""

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        fov: float = 60.0,
        near: float = 0.02,
        far: float = 1.5,
        physics_client: int = 0,
    ):
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.physics_client = physics_client

        aspect = float(width) / float(height)
        fov_rad = np.radians(fov)
        self.fy = (height / 2.0) / np.tan(fov_rad / 2.0)
        self.fx = (width / 2.0) / (np.tan(fov_rad / 2.0) * aspect)
        self.cx = width / 2.0
        self.cy = height / 2.0

        self.candidate_views = compute_golden_spiral_views(
            num_views=16,
            radius=0.5,
            min_elev_deg=30.0,
            max_elev_deg=70.0,
            target=np.array([0.0, 0.0, 0.03]),
            fov=fov,
            aspect=aspect,
            near=near,
            far=far,
            physics_client=physics_client,
        )

    def render(self, view: CameraView) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render RGB, metric depth (m), and segmentation mask."""
        _, _, rgb, depth_buf, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view.view_matrix.flatten(order="F").tolist(),
            projectionMatrix=view.proj_matrix.flatten(order="F").tolist(),
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client,
        )

        rgb = np.array(rgb, dtype=np.uint8).reshape((self.height, self.width, 4))[:, :, :3]
        depth_buf = np.array(depth_buf, dtype=np.float32).reshape((self.height, self.width))
        seg = np.array(seg, dtype=np.int32).reshape((self.height, self.width))

        # True metric depth
        metric_depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buf)
        return rgb, metric_depth, seg

    def backproject(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg: np.ndarray,
        view: CameraView,
        plane_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backproject single camera image to world points (N, 3) and RGB (N, 3).
        
        Drops plane pixels (seg == plane_id or world_z <= 0.005).
        """
        v_coords, u_coords = np.indices((self.height, self.width))
        valid = (depth > self.near + 0.005) & (depth < self.far - 0.05) & (seg != plane_id)

        u_valid = u_coords[valid]
        v_valid = v_coords[valid]
        z_valid = depth[valid]
        rgb_valid = rgb[valid].astype(np.float32) / 255.0

        # OpenGL camera frame: x right, y up, -z into scene
        x_cam = (u_valid - self.cx) * z_valid / self.fx
        y_cam = -(v_valid - self.cy) * z_valid / self.fy
        z_cam = -z_valid

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        pts_world = np.matmul(pts_cam, view.c2w_rotation.T) + view.c2w_translation

        # Drop plane points by height
        table_mask = pts_world[:, 2] > 0.005
        pts_world = pts_world[table_mask]
        rgb_valid = rgb_valid[table_mask]

        return pts_world, rgb_valid
