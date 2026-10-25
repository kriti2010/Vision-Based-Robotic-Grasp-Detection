"""Unit tests for AnalyticGraspChecker on known geometric objects."""

import numpy as np
import pytest
from src.grasp.representation import Grasp, GripperParams
from src.grasp.checker import AnalyticGraspChecker
from src.sim.scene import ObjectGeometry


def test_analytic_grasp_on_known_box():
    # Define a known box centered at (0, 0, 0.05) with half-extents (0.025, 0.025, 0.025)
    # Box width along X is 0.05, Y is 0.05, Z is 0.05
    box = ObjectGeometry(
        body_id=1,
        obj_type="box",
        position=np.array([0.0, 0.0, 0.025]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # identity quaternion
        dimensions=np.array([0.025, 0.025, 0.025]),
        color=(0.8, 0.2, 0.2, 1.0)
    )
    
    checker = AnalyticGraspChecker(GripperParams())

    # 1. Valid top-down grasp:
    # Approach direction: -Z (pointing down toward table), so R col 2 is [0, 0, -1]
    # Closing direction: +X (pinching across the box X faces), so R col 0 is [1, 0, 0]
    # Binormal: Y = Z_app x X = [0, -1, 0]
    # Rotation matrix: [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    R_valid = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])
    p_valid = np.array([0.0, 0.0, 0.025])  # centered on the box
    grasp_valid = Grasp(R=R_valid, p=p_valid, width=0.065)

    is_success, reason = checker.check_grasp(grasp_valid, [box])
    assert is_success, f"Expected valid grasp to succeed, but got: {reason}"

    # 2. Width out of limits (too narrow to fit box)
    grasp_narrow = Grasp(R=R_valid, p=p_valid, width=0.005)
    is_success, reason = checker.check_grasp(grasp_narrow, [box])
    assert not is_success
    assert reason == "width_out_of_limits"

    # 3. Collision with table (grasp center below table surface)
    p_below = np.array([0.0, 0.0, -0.05])
    grasp_below = Grasp(R=R_valid, p=p_below, width=0.065)
    is_success, reason = checker.check_grasp(grasp_below, [box])
    assert not is_success
    assert reason == "table_collision"

    # 4. Grasps in empty space (far away from box)
    p_empty = np.array([0.5, 0.5, 0.1])
    grasp_empty = Grasp(R=R_valid, p=p_empty, width=0.065)
    is_success, reason = checker.check_grasp(grasp_empty, [box])
    assert not is_success
    assert reason == "no_object_in_jaws"
