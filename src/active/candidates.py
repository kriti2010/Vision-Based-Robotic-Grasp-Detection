"""Candidate viewpoint generator on the hemisphere around the scene."""

from typing import List, Tuple
import numpy as np
from src.sim.camera import CameraPose, DepthCamera


class ViewCandidateGenerator:
    """Generates candidate camera viewpoints on the upper hemisphere."""

    def __init__(
        self,
        num_candidates: int = 16,
        radius: float = 0.60,
        elevation_range: Tuple[float, float] = (30.0, 75.0),
        azimuth_range: Tuple[float, float] = (0.0, 360.0)
    ):
        self.num_candidates = num_candidates
        self.radius = radius
        self.min_el, self.max_el = elevation_range
        self.min_az, self.max_az = azimuth_range

    def generate_candidates(self, camera: DepthCamera) -> List[CameraPose]:
        """Generate N candidate camera poses evenly spaced on the hemisphere."""
        poses = []
        # Fibonacci spiral on hemisphere for uniform coverage
        for i in range(self.num_candidates):
            phi = (i / max(1, self.num_candidates - 1)) * (self.max_az - self.min_az) + self.min_az
            # Vary elevation
            theta = self.min_el + (self.max_el - self.min_el) * (0.5 + 0.5 * np.sin(2.5 * i))
            
            pose = camera.compute_pose(
                elevation_deg=float(theta),
                azimuth_deg=float(phi),
                radius=self.radius
            )
            poses.append(pose)
            
        return poses
