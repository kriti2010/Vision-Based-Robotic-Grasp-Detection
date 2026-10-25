"""Unit test for voxel visibility on a hand-made scene."""

import numpy as np
import pytest
from src.sim.fusion import VoxelOccupancyGrid
from src.sim.visibility import VisibilityChecker


def test_voxel_visibility_on_handmade_scene():
    # Create grid spanning [-0.2, 0.2] in X and Y, [0.0, 0.2] in Z
    bounds = (np.array([-0.2, -0.2, 0.0]), np.array([0.2, 0.2, 0.2]))
    grid = VoxelOccupancyGrid(bounds=bounds, voxel_size=0.02)
    
    # Hand-made scene:
    # An occluder box placed at (0, 0, 0.1)
    # Camera at (0, -0.5, 0.1) looking along +Y
    camera_eye = np.array([0.0, -0.4, 0.1])
    
    # Target point 1: in front of occluder (0, -0.1, 0.1) -> should be VISIBLE
    target_visible = np.array([0.0, -0.1, 0.1])
    
    # Target point 2: behind occluder (0, 0.15, 0.1) -> should be OCCLUDED
    target_occluded = np.array([0.0, 0.15, 0.1])
    
    # Mark occluder voxels as OCCUPIED
    occluder_pts = []
    for x in np.linspace(-0.04, 0.04, 5):
        for z in np.linspace(0.06, 0.14, 5):
            occluder_pts.append([x, 0.0, z])
    occluder_pts = np.array(occluder_pts)
    
    grid.update_view(camera_eye=camera_eye, surface_points=occluder_pts)
    
    checker = VisibilityChecker(grid)
    
    # Verify line-of-sight
    assert checker.is_point_visible(camera_eye, target_visible) is True
    assert checker.is_point_visible(camera_eye, target_occluded) is False
