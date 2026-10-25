"""Dataset generation and PyTorch Dataset loading for grasp diffusion and scoring.

Specification:
- Base data: scenes with fused point clouds from random 1-4 view subsets.
- 48 grasps per scene: 24 antipodal proposals + 24 perturbed negatives.
- Grasp execution via SimGraspExecutor with physics ground-truth labels.
- Scene-level splitting (never split by grasp).
- Dataset support for generator (successful grasps only) and scorer (all grasps).
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sim.scene import ClutteredScene
from sim.camera import CameraRig
from sim.fusion import fuse_and_subsample
from sim.proposals import sample_antipodal_grasps, perturb_grasps_negative
from sim.gripper import SimGraspExecutor


def generate_scene_data(
    scene_seed: int,
    num_antipodal: int = 24,
    num_perturbed: int = 24,
    min_views: int = 1,
    max_views: int = 4,
    num_points: int = 2048,
    client_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate all data for a single scene."""
    rng = np.random.RandomState(scene_seed)
    scene = ClutteredScene(physics_client=client_id)
    scene.create_scene(seed=scene_seed)

    rig = CameraRig(physics_client=scene.client)
    executor = SimGraspExecutor(scene)

    # 1. Random 1-4 view subset from 16 golden-spiral viewpoints
    num_v = rng.randint(min_views, max_views + 1)
    view_indices = rng.choice(16, size=num_v, replace=False).tolist()

    pts, cols = fuse_and_subsample(
        rig, view_indices=view_indices, plane_id=scene.plane_id, num_points=num_points, rng=rng
    )

    # 2. Antipodal grasps
    antipodal = sample_antipodal_grasps(
        pts, num_grasps=num_antipodal, rng=rng, physics_client=scene.client
    )

    # 3. Perturbed negatives
    perturbed = perturb_grasps_negative(antipodal[:num_perturbed], rng=rng)

    all_grasps = antipodal + perturbed
    R_list = []
    t_list = []
    w_list = []
    labels = []

    # 4. Label all grasps by physics execution
    for R, t, w in all_grasps:
        succ, info = executor.execute_grasp(R, t, w, restore=True)
        R_list.append(R)
        t_list.append(t)
        w_list.append(w)
        labels.append(1.0 if succ else 0.0)

    scene.close()

    return {
        "scene_seed": scene_seed,
        "points": pts.astype(np.float32),
        "colors": cols.astype(np.float32),
        "R": np.array(R_list, dtype=np.float32),
        "t": np.array(t_list, dtype=np.float32),
        "w": np.array(w_list, dtype=np.float32),
        "labels": np.array(labels, dtype=np.float32),
    }


def generate_base_dataset(
    output_dir: str,
    num_scenes: int = 200,
    seed_start: int = 0,
    num_antipodal: int = 24,
    num_perturbed: int = 24,
    desc: str = "Base Data",
) -> List[str]:
    """Generate and save base scene dataset."""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    print(f"\nGenerating {desc}: {num_scenes} scenes (seeds {seed_start}..{seed_start + num_scenes - 1})")
    for i in tqdm(range(num_scenes), desc=desc):
        seed = seed_start + i
        file_path = os.path.join(output_dir, f"scene_{seed:07d}.npz")
        if not os.path.exists(file_path):
            data = generate_scene_data(
                scene_seed=seed,
                num_antipodal=num_antipodal,
                num_perturbed=num_perturbed,
            )
            np.savez_compressed(file_path, **data)
        saved_files.append(file_path)

    return saved_files


class GraspDataset(Dataset):
    """PyTorch Dataset loading scene grasps with 0.15 m coordinate normalization."""

    def __init__(
        self,
        file_paths: List[str],
        successful_only: bool = False,
        normalize_scale: float = 0.15,
        max_samples_per_scene: Optional[int] = None,
    ):
        self.successful_only = successful_only
        self.normalize_scale = normalize_scale
        self.records = []

        for p in file_paths:
            data = np.load(p)
            pts = data["points"]
            R_all = data["R"]
            t_all = data["t"]
            w_all = data["w"]
            labels_all = data["labels"]

            if successful_only:
                mask = labels_all > 0.5
                if not np.any(mask):
                    continue
                R_all = R_all[mask]
                t_all = t_all[mask]
                w_all = w_all[mask]
                labels_all = labels_all[mask]

            if max_samples_per_scene is not None and len(labels_all) > max_samples_per_scene:
                sub = np.random.choice(len(labels_all), size=max_samples_per_scene, replace=False)
                R_all = R_all[sub]
                t_all = t_all[sub]
                w_all = w_all[sub]
                labels_all = labels_all[sub]

            for g_idx in range(len(labels_all)):
                self.records.append((
                    pts,
                    R_all[g_idx],
                    t_all[g_idx],
                    w_all[g_idx],
                    labels_all[g_idx],
                ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        pts, R, t, w, label = self.records[idx]

        # Normalise coordinates by 0.15 m
        pts_norm = pts / self.normalize_scale
        t_norm = t / self.normalize_scale
        w_norm = np.array([w / self.normalize_scale], dtype=np.float32)

        return {
            "points": torch.from_numpy(pts_norm).float(),
            "R": torch.from_numpy(R).float(),
            "t": torch.from_numpy(t_norm).float(),
            "w": torch.from_numpy(w_norm).float(),
            "label": torch.tensor(label, dtype=torch.float32),
        }
