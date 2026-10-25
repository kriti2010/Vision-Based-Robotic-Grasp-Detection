"""Multi-view active grasp episode runner.

Specification:
- Episode = one scene, T = 4 views in fixed order, N = 32 samples, K = 12 steps.
- At each view records:
  - Calibrated score s* of best sample.
  - Dispersion features (DBSCAN eps 1.0 on top 50%, largest-cluster mass, normalized entropy).
  - Cross-view features (view_idx / (T-1), distance between consecutive best grasps, score gap).
  - Physical outcome of executing best grasp in simulator (restore state afterwards).
- Per-episode torch and numpy seeds for reproducibility.
- Splits: fusion (100000+), calibration (200000+), test (300000+).
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm

from sim.scene import ClutteredScene
from sim.camera import CameraRig
from sim.fusion import fuse_and_subsample
from sim.gripper import SimGraspExecutor
from models.generator import GraspDiffusionDenoiser
from models.scorer import GraspSuccessScorer
from episodes.features import (
    compute_dispersion_features, compute_cross_view_features
)


FIXED_VIEW_ORDER = [0, 4, 8, 12]


def run_single_episode(
    scene_seed: int,
    generator: GraspDiffusionDenoiser,
    scorer: GraspSuccessScorer,
    T: int = 4,
    N: int = 32,
    num_steps: int = 12,
    device: str = "cpu",
    client_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a single multi-view grasp episode and record all view features and outcomes."""
    # Set per-episode random seed
    torch.manual_seed(scene_seed)
    np.random.seed(scene_seed)

    scene = ClutteredScene(physics_client=client_id)
    scene.create_scene(seed=scene_seed)
    rig = CameraRig(physics_client=scene.client)
    executor = SimGraspExecutor(scene)

    views_to_use = FIXED_VIEW_ORDER[:T]
    prev_best_grasp = None

    records = {
        "scene_seed": scene_seed,
        "s_star": [],
        "dispersion": [],
        "cross_view": [],
        "features_all": [],
        "outcomes": [],
        "best_grasps": [],
    }

    accumulated_views = []

    for v_idx, view_id in enumerate(views_to_use):
        accumulated_views.append(view_id)

        # 1. Fuse camera views up to current view
        pts, cols = fuse_and_subsample(
            rig, view_indices=accumulated_views, plane_id=scene.plane_id, num_points=2048
        )

        # 2. Generator samples N = 32 grasps
        pts_norm = pts / 0.15
        pts_tensor = torch.from_numpy(pts_norm).float().to(device)

        R_samples, t_samples, w_samples = generator.sample(
            pts_tensor, num_samples=N, num_steps=num_steps, chunk_size=16
        )

        # 3. Scorer predicts calibrated probabilities
        scores = scorer.score_grasps(R_samples, t_samples, w_samples, pts, chunk_size=16)

        # 4. Best sample pick
        best_idx = int(np.argmax(scores))
        s_star = float(scores[best_idx])
        curr_best_grasp = (R_samples[best_idx], t_samples[best_idx], float(w_samples[best_idx]))

        # 5. Dispersion features on top 50%
        disp_feats = compute_dispersion_features(R_samples, t_samples, w_samples, scores)

        # 6. Cross-view features
        cross_feats = compute_cross_view_features(
            view_idx=v_idx,
            T=T,
            best_grasp_curr=curr_best_grasp,
            best_grasp_prev=prev_best_grasp,
            scores=scores,
        )

        # 7. Physical execution in simulator
        succ, info = executor.execute_grasp(
            curr_best_grasp[0], curr_best_grasp[1], curr_best_grasp[2], restore=True
        )
        outcome = 1 if succ else 0

        # Feature vector concatenation:
        # [1 - s*, largest_cluster_mass, norm_entropy, norm_view_idx, norm_dist_prev, score_gap]
        feat_vector = np.array([
            1.0 - s_star,
            disp_feats["largest_cluster_mass"],
            disp_feats["normalized_entropy"],
            cross_feats["norm_view_idx"],
            cross_feats["norm_dist_prev"],
            cross_feats["score_gap"],
        ], dtype=np.float32)

        records["s_star"].append(s_star)
        records["dispersion"].append(disp_feats)
        records["cross_view"].append(cross_feats)
        records["features_all"].append(feat_vector)
        records["outcomes"].append(outcome)
        records["best_grasps"].append(curr_best_grasp)

        prev_best_grasp = curr_best_grasp

    scene.close()
    return records


def run_episodes_split(
    seeds: List[int],
    generator: GraspDiffusionDenoiser,
    scorer: GraspSuccessScorer,
    output_path: str,
    T: int = 4,
    N: int = 32,
    num_steps: int = 12,
    device: str = "cpu",
    desc: str = "Episodes",
) -> str:
    """Run episodes over seeds and save compressed .npz archive."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        print(f"Loaded existing episodes archive from {output_path}")
        return output_path

    all_s_star = []
    all_features = []
    all_outcomes = []
    all_seeds = []

    print(f"\nRunning {desc}: {len(seeds)} episodes...")
    for seed in tqdm(seeds, desc=desc):
        rec = run_single_episode(
            scene_seed=seed,
            generator=generator,
            scorer=scorer,
            T=T,
            N=N,
            num_steps=num_steps,
            device=device,
        )
        all_seeds.append(rec["scene_seed"])
        all_s_star.append(rec["s_star"])          # (T,)
        all_features.append(rec["features_all"])  # (T, 6)
        all_outcomes.append(rec["outcomes"])      # (T,)

    np.savez_compressed(
        output_path,
        seeds=np.array(all_seeds, dtype=np.int32),
        s_star=np.array(all_s_star, dtype=np.float32),      # (NumEpisodes, T)
        features=np.array(all_features, dtype=np.float32),  # (NumEpisodes, T, 6)
        outcomes=np.array(all_outcomes, dtype=np.int32),    # (NumEpisodes, T)
    )
    print(f"Saved {len(seeds)} episodes to {output_path}")
    return output_path
