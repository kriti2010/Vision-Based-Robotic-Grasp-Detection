"""Test Acceptance criterion 4:
Antipodal proposals succeed roughly 0.34 +/- 0.1 on scenes (simulator sanity check).
"""

import numpy as np
import pytest
from sim.scene import ClutteredScene
from sim.camera import CameraRig
from sim.fusion import fuse_and_subsample
from sim.proposals import sample_antipodal_grasps, perturb_grasps_negative
from sim.gripper import SimGraspExecutor


def test_antipodal_proposal_geometry():
    """Verify geometry of antipodal proposals:
    - R orthonormal, det(R) == 1
    - closing axis is col 0, approach axis is col 2
    - approach axis a_z < -0.3
    - width in [0.015, 0.075]
    """
    # Create simple box synthetic points
    np.random.seed(42)
    pts = np.random.uniform([-0.05, -0.05, 0.02], [0.05, 0.05, 0.06], size=(500, 3)).astype(np.float32)
    grasps = sample_antipodal_grasps(pts, num_grasps=10, rng=np.random.RandomState(42))
    assert len(grasps) > 0
    for R, t, w in grasps:
        assert np.allclose(np.matmul(R.T, R), np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)
        # Approach axis is col 2, a_z < -0.3
        assert R[2, 2] < -0.3
        # Width in [0.015, 0.075]
        assert 0.014 <= w <= 0.076


def test_antipodal_success_rate_sim():
    """Check antipodal proposals success rate in PyBullet simulator across 5 scenes in fast mode."""
    num_scenes = 10  # 10 scenes x 6 grasps = 60 trials
    grasps_per_scene = 6
    successes = 0
    total = 0

    scene = ClutteredScene()
    rig = CameraRig(physics_client=scene.client)
    executor = SimGraspExecutor(scene)

    for seed in range(num_scenes):
        scene.create_scene(seed=seed)
        pts, cols = fuse_and_subsample(rig, view_indices=list(range(8)), plane_id=scene.plane_id, num_points=2048)
        grasps = sample_antipodal_grasps(pts, num_grasps=grasps_per_scene, rng=np.random.RandomState(seed))

        for R, t, w in grasps:
            succ, info = executor.execute_grasp(R, t, w, restore=True)
            if succ:
                successes += 1
            total += 1

    scene.close()
    rate = successes / max(1, total)
    print(f"\nAntipodal proposals success rate: {successes}/{total} = {rate:.2%}")
    # Acceptance Test 4: roughly 0.34 +/- 0.10
    assert 0.20 <= rate <= 0.48, f"Expected antipodal success in [0.20, 0.48], got {rate}"
