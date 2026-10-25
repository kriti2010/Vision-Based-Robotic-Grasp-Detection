"""Acceptance Test 6:
Same seed gives identical episodes.
"""

import numpy as np
import torch
import pytest
from episodes.runner import run_single_episode
from models.generator import GraspDiffusionDenoiser
from models.scorer import GraspSuccessScorer


def test_episode_seed_determinism():
    """Verify running the episode with the same seed yields identical outputs."""
    seed = 42
    # Lightweight dummy models for test
    generator = GraspDiffusionDenoiser(global_dim=64, local_dim=32, hidden_dim=64)
    scorer = GraspSuccessScorer(global_dim=64, local_dim=32, hidden_dim=64)

    # Run episode first time
    rec1 = run_single_episode(
        scene_seed=seed,
        generator=generator,
        scorer=scorer,
        T=2,
        N=8,
        num_steps=4,
    )

    # Run episode second time with same seed
    rec2 = run_single_episode(
        scene_seed=seed,
        generator=generator,
        scorer=scorer,
        T=2,
        N=8,
        num_steps=4,
    )

    # Check identical s_star
    assert np.allclose(rec1["s_star"], rec2["s_star"], atol=1e-5), "s_star values differed between identical seeds"

    # Check identical features
    feats1 = np.array(rec1["features_all"])
    feats2 = np.array(rec2["features_all"])
    assert np.allclose(feats1, feats2, atol=1e-5), "Features differed between identical seeds"

    # Check identical physical outcomes
    assert rec1["outcomes"] == rec2["outcomes"], "Physical outcomes differed between identical seeds"
