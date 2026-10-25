"""Self-training rounds: Round 1 (on-generator fine-tuning) and Round 2 (joint retraining).

Specification:
- Round 1: 300 scenes (seeds 600000+) x 32 generator samples, labelled in simulator.
  Fine-tune scorer on 70/30 mix with base data; refit temperature.
- Round 2: 1,200 scenes (seeds 700000+). Warm-start both models.
  Generator trained on successful grasps (80% own samples, 20% base data).
  Scorer trained on all labelled grasps. 12k steps, lr 2e-4; refit temperature.
- Split by SCENE, never by grasp.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sim.scene import ClutteredScene
from sim.camera import CameraRig
from sim.fusion import fuse_and_subsample
from sim.gripper import SimGraspExecutor
from models.generator import GraspDiffusionDenoiser
from models.scorer import GraspSuccessScorer
from train.dataset import GraspDataset
from train.train_base import train_generator, train_scorer


def collect_on_generator_data(
    generator: GraspDiffusionDenoiser,
    num_scenes: int = 300,
    seed_start: int = 600000,
    output_dir: str = "data/round1",
    samples_per_scene: int = 32,
    device: str = "cpu",
    desc: str = "Round 1 On-Generator Data",
) -> List[str]:
    """Generate scenes and label generator samples in PyBullet simulator."""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    generator.eval()

    print(f"\nCollecting {desc}: {num_scenes} scenes (seeds {seed_start}..{seed_start + num_scenes - 1})")
    for i in tqdm(range(num_scenes), desc=desc):
        seed = seed_start + i
        file_path = os.path.join(output_dir, f"scene_{seed:07d}.npz")
        if os.path.exists(file_path):
            saved_files.append(file_path)
            continue

        rng = np.random.RandomState(seed)
        scene = ClutteredScene()
        scene.create_scene(seed=seed)
        rig = CameraRig(physics_client=scene.client)
        executor = SimGraspExecutor(scene)

        # Random 1-4 view subset
        num_v = rng.randint(1, 5)
        view_indices = rng.choice(16, size=num_v, replace=False).tolist()
        pts, cols = fuse_and_subsample(rig, view_indices=view_indices, plane_id=scene.plane_id, rng=rng)

        # Sample grasps using generator
        pts_norm = pts / 0.15
        pts_tensor = torch.from_numpy(pts_norm).float().to(device)
        R_samples, t_samples, w_samples = generator.sample(
            pts_tensor, num_samples=samples_per_scene, num_steps=12, chunk_size=16
        )

        labels = []
        for g_idx in range(len(R_samples)):
            succ, info = executor.execute_grasp(
                R_samples[g_idx], t_samples[g_idx], float(w_samples[g_idx]), restore=True
            )
            labels.append(1.0 if succ else 0.0)

        scene.close()

        np.savez_compressed(
            file_path,
            scene_seed=seed,
            points=pts.astype(np.float32),
            colors=cols.astype(np.float32),
            R=R_samples.astype(np.float32),
            t=t_samples.astype(np.float32),
            w=w_samples.astype(np.float32),
            labels=np.array(labels, dtype=np.float32),
        )
        saved_files.append(file_path)

    return saved_files


def run_round1(
    base_train_files: List[str],
    base_val_files: List[str],
    generator_ckpt: str = "checkpoints/generator_base.pt",
    scorer_ckpt: str = "checkpoints/scorer_base.pt",
    num_scenes: int = 300,
    seed_start: int = 600000,
    steps: int = 12000,
    lr: float = 2e-4,
    device: str = "cpu",
    output_dir: str = "data/round1",
    output_scorer_ckpt: str = "checkpoints/scorer_round1.pt",
) -> Tuple[GraspSuccessScorer, dict]:
    """Execute Round 1: On-generator data collection + 70/30 scorer fine-tuning."""
    # 1. Load generator
    generator = GraspDiffusionDenoiser().to(device)
    generator.load_state_dict(torch.load(generator_ckpt, map_location=device))

    # 2. Collect on-generator data
    r1_files = collect_on_generator_data(
        generator,
        num_scenes=num_scenes,
        seed_start=seed_start,
        output_dir=output_dir,
        device=device,
        desc="Round 1 Data",
    )

    # Split r1 by scene (80/20)
    split_idx = int(0.8 * len(r1_files))
    r1_train = r1_files[:split_idx]
    r1_val = r1_files[split_idx:] if split_idx < len(r1_files) else r1_files[:1]

    # Combine 70% round 1 files and 30% base data files
    n_r1 = len(r1_train)
    n_base = max(1, int(n_r1 * (30.0 / 70.0)))
    combined_train = r1_train + list(np.random.choice(base_train_files, size=min(n_base, len(base_train_files)), replace=False))

    # 3. Fine-tune scorer
    scorer, metrics = train_scorer(
        train_files=combined_train,
        val_files=r1_val,
        steps=steps,
        lr=lr,
        device=device,
        output_ckpt=output_scorer_ckpt,
        warm_start_ckpt=scorer_ckpt,
    )
    return scorer, metrics


def run_round2(
    base_train_files: List[str],
    generator_ckpt: str = "checkpoints/generator_base.pt",
    scorer_ckpt: str = "checkpoints/scorer_round1.pt",
    num_scenes: int = 1200,
    seed_start: int = 700000,
    steps: int = 12000,
    lr: float = 2e-4,
    device: str = "cpu",
    output_dir: str = "data/round2",
    output_gen_ckpt: str = "checkpoints/generator_round2.pt",
    output_scorer_ckpt: str = "checkpoints/scorer_round2.pt",
) -> Tuple[GraspDiffusionDenoiser, GraspSuccessScorer, dict]:
    """Execute Round 2: 1200 scenes, warm-start both models (80/20 mix for generator)."""
    # 1. Load generator to generate round 2 data
    generator = GraspDiffusionDenoiser().to(device)
    generator.load_state_dict(torch.load(generator_ckpt, map_location=device))

    # 2. Collect round 2 data
    r2_files = collect_on_generator_data(
        generator,
        num_scenes=num_scenes,
        seed_start=seed_start,
        output_dir=output_dir,
        device=device,
        desc="Round 2 Data",
    )

    split_idx = int(0.85 * len(r2_files))
    r2_train = r2_files[:split_idx]
    r2_val = r2_files[split_idx:] if split_idx < len(r2_files) else r2_files[:1]

    # Generator: 80% own samples, 20% base data
    n_r2 = len(r2_train)
    n_base_gen = max(1, int(n_r2 * (20.0 / 80.0)))
    gen_train_files = r2_train + list(np.random.choice(base_train_files, size=min(n_base_gen, len(base_train_files)), replace=False))

    # Retrain generator
    gen_trained = train_generator(
        train_files=gen_train_files,
        steps=steps,
        lr=lr,
        device=device,
        output_ckpt=output_gen_ckpt,
        warm_start_ckpt=generator_ckpt,
    )

    # Retrain scorer on all labelled grasps
    scorer_trained, metrics = train_scorer(
        train_files=r2_train,
        val_files=r2_val,
        steps=steps,
        lr=lr,
        device=device,
        output_ckpt=output_scorer_ckpt,
        warm_start_ckpt=scorer_ckpt,
    )

    return gen_trained, scorer_trained, metrics
