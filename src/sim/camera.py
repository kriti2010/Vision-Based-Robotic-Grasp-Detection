"""Simulated RGB-D camera on a hemisphere around the tabletop workspace."""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pybullet as p


@dataclass
class CameraPose:
    """Spherical camera viewpoint and world pose."""
    radius: float
    elevation_deg: float
    azimuth_deg: float
    eye: np.ndarray
    target: np.ndarray
    view_matrix: np.ndarray       # 4x4 matrix
    proj_matrix: np.ndarray       # 4x4 matrix
    c2w_rotation: np.ndarray      # 3x3 rotation from cam to world
    c2w_translation: np.ndarray   # 3D eye position in world


class DepthCamera:
    """Renders RGB-D views and back-projects partial point clouds."""

    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        fov: float = 60.0,
        near: float = 0.1,
        far: float = 1.5,
        physics_client: int = 0
    ):
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.physics_client = physics_client

        # Focal lengths and principal points
        aspect = float(width) / float(height)
        fov_rad = np.radians(fov)
        self.fy = (height / 2.0) / np.tan(fov_rad / 2.0)
        self.fx = (width / 2.0) / (np.tan(fov_rad / 2.0) * aspect)
        self.cx = width / 2.0
        self.cy = height / 2.0

    def compute_pose(
        self,
        elevation_deg: float,
        azimuth_deg: float,
        radius: float = 0.60,
        target: Optional[np.ndarray] = None
    ) -> CameraPose:
        """Compute camera matrices from spherical coordinates."""
        if target is None:
            target = np.array([0.0, 0.0, 0.05])
            
        el_rad = np.radians(elevation_deg)
        az_rad = np.radians(azimuth_deg)
        
        # Position on upper hemisphere
        eye = np.array([
            radius * np.cos(el_rad) * np.cos(az_rad),
            radius * np.cos(el_rad) * np.sin(az_rad),
            radius * np.sin(el_rad)
        ]) + np.array([target[0], target[1], 0.0])

        up = np.array([0.0, 0.0, 1.0])
        view_mat_list = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
            physicsClientId=self.physics_client
        )
        proj_mat_list = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far,
            physicsClientId=self.physics_client
        )
        
        # PyBullet view matrix is column-major 4x4
        V = np.array(view_mat_list).reshape((4, 4), order='F')
        P = np.array(proj_mat_list).reshape((4, 4), order='F')
        
        # Camera-to-world transform is inv(V)
        # Note: in OpenGL camera frame: +X right, +Y up, -Z forward (into scene)
        V_inv = np.linalg.inv(V)
        c2w_R = V_inv[:3, :3]
        c2w_t = V_inv[:3, 3]

        return CameraPose(
            radius=radius,
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg,
            eye=eye,
            target=target,
            view_matrix=V,
            proj_matrix=P,
            c2w_rotation=c2w_R,
            c2w_translation=c2w_t
        )

    def render(self, pose: CameraPose) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render RGB, linear depth (meters), and segmentation mask."""
        _, _, rgb, depth_buf, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=pose.view_matrix.flatten(order='F').tolist(),
            projectionMatrix=pose.proj_matrix.flatten(order='F').tolist(),
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client
        )

        rgb = np.array(rgb, dtype=np.uint8).reshape((self.height, self.width, 4))[:, :, :3]
        depth_buf = np.array(depth_buf, dtype=np.float32).reshape((self.height, self.width))
        seg = np.array(seg, dtype=np.int32).reshape((self.height, self.width))

        # Convert OpenGL normalized depth buffer to true metric depth in camera frame
        metric_depth = self.far * self.near / (self.far - (self.far - self.near) * depth_buf)
        return rgb, metric_depth, seg

    def backproject_pointcloud(
        self,
        depth: np.ndarray,
        pose: CameraPose,
        seg_mask: Optional[np.ndarray] = None,
        crop_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """Back-project depth image to 3D point cloud in world coordinates.
        
        Args:
            depth: (H, W) metric depth in meters
            pose: CameraPose
            seg_mask: Optional object segmentation mask (ignoring table/background if > 0)
            crop_bounds: (min_bound, max_bound) 3D world bounding box
        Returns:
            (N, 3) points in world coordinates
        """
        v_coords, u_coords = np.indices((self.height, self.width))
        valid = (depth > self.near + 0.01) & (depth < self.far - 0.05)

        if seg_mask is not None:
            # Table is ID 0 in PyBullet scene
            valid = valid & (seg_mask > 0)

        u_valid = u_coords[valid]
        v_valid = v_coords[valid]
        z_valid = depth[valid]

        # OpenGL camera frame: x right, y up, -z forward
        x_cam = (u_valid - self.cx) * z_valid / self.fx
        y_cam = -(v_valid - self.cy) * z_valid / self.fy
        z_cam = -z_valid

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

        # Transform to world coordinates: P_world = R_c2w * P_cam + t_c2w
        pts_world = np.matmul(pts_cam, pose.c2w_rotation.T) + pose.c2w_translation

        if crop_bounds is not None:
            min_b, max_b = crop_bounds
            inside = np.all((pts_world >= min_b) & (pts_world <= max_b), axis=1)
            pts_world = pts_world[inside]

        return pts_world
