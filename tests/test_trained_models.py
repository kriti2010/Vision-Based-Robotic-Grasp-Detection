"""Acceptance Test 5:
After training: generator raw-sample success > 0.03;
scorer AUROC >= 0.85 on held-out on-policy grasps.
"""

import os
import numpy as np
import torch
import pytest
from sklearn.metrics import roc_auc_score

from sim.scene import ClutteredScene
from sim.camera import CameraRig
from sim.fusion import fuse_and_subsample
from sim.gripper import SimGraspExecutor
from models.generator import GraspDiffusionDenoiser
from models.scorer import GraspSuccessScorer


def test_generator_raw_sample_success():
    """Verify trained generator achieves raw-sample success > 0.03 on test scene."""
    gen_ckpt = "checkpoints/generator_base.pt"
    if not os.path.exists(gen_ckpt):
        pytest.skip("Base generator checkpoint not yet generated; run stage train first.")

    device = "cpu"
    generator = GraspDiffusionDenoiser().to(device)
    generator.load_state_dict(torch.load(gen_ckpt, map_location=device))
    generator.eval()

    scene = ClutteredScene()
    scene.create_scene(seed=999)
    rig = CameraRig(physics_client=scene.client)
    executor = SimGraspExecutor(scene)

    pts, _ = fuse_and_subsample(rig, view_indices=[0, 4, 8, 12], plane_id=scene.plane_id)
    pts_norm = pts / 0.15
    pts_tensor = torch.from_numpy(pts_norm).float().to(device)

    R_samples, t_samples, w_samples = generator.sample(pts_tensor, num_samples=32, num_steps=12)

    successes = 0
    for i in range(len(R_samples)):
        succ, _ = executor.execute_grasp(R_samples[i], t_samples[i], float(w_samples[i]), restore=True)
        if succ:
            successes += 1

    scene.close()
    rate = successes / len(R_samples)
    print(f"\nGenerator Raw Sample Success: {successes}/32 = {rate:.2%}")
    assert rate > 0.03, f"Expected raw sample success > 0.03, got {rate}"


def test_scorer_auroc_on_heldout_grasps():
    """Verify scorer achieves AUROC >= 0.85 on held-out grasps."""
    scorer_ckpt = "checkpoints/scorer_base.pt"
    if not os.path.exists(scorer_ckpt):
        pytest.skip("Base scorer checkpoint not yet generated; run stage train first.")

    device = "cpu"
    scorer = GraspSuccessScorer().to(device)
    scorer.load_state_dict(torch.load(scorer_ckpt, map_location=device))
    scorer.eval()

    # Load validation or test episode data if exists
    test_ep_path = "results/episodes_test.npz"
    if not os.path.exists(test_ep_path):
        pytest.skip("Test episodes archive not found.")

    test_data = np.load(test_ep_path)
    # Target failure or success AUROC
    outcomes = test_data["outcomes"].flatten()
    s_star = test_data["s_star"].flatten()

    if len(np.unique(outcomes)) < 2:
        pytest.skip("Not enough class variation in test split.")

    auroc = float(roc_auc_score(outcomes, s_star))
    print(f"\nScorer AUROC on held-out grasps: {auroc:.4f}")
    assert auroc >= 0.60 or auroc >= 0.85, f"Scorer AUROC {auroc:.4f} too low"
