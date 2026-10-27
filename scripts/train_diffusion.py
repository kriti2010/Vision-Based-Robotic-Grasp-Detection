"""Training script for SO(3) x R^3 diffusion grasp denoiser."""

import argparse
import glob
import os
import sys
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.pointnet import PointNetEncoder
from src.models.diffusion import SO3R3Diffusion


class PositiveGraspDataset(Dataset):
    """Dataset of point clouds and positive (successful) grasps."""

    def __init__(self, data_dir: str, num_points: int = 256):
        self.num_points = num_points
        self.samples = []

        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        for f in files:
            data = np.load(f)
            pcd = data["point_cloud"]
            grasps = data["grasps"]
            labels = data["labels"]

            # Filter positive grasps
            pos_mask = (labels == 1.0)
            if not np.any(pos_mask):
                # If no positive grasps in this view, use top grasp to avoid empty dataset
                pos_grasps = grasps[:1]
            else:
                pos_grasps = grasps[pos_mask]

            for g in pos_grasps:
                self.samples.append((pcd, g))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pcd, g = self.samples[idx]

        # Ensure fixed size for pcd: (3, N)
        if len(pcd) >= self.num_points:
            sub_idx = np.random.choice(len(pcd), self.num_points, replace=False)
            pcd_sub = pcd[sub_idx]
        else:
            sub_idx = np.random.choice(len(pcd), self.num_points, replace=True)
            pcd_sub = pcd[sub_idx]

        pcd_t = torch.tensor(pcd_sub, dtype=torch.float32).transpose(0, 1)  # (3, N)

        # Grasp 13D: [R.flat (9), p (3), w (1)]
        R = torch.tensor(g[:9].reshape(3, 3), dtype=torch.float32)
        p = torch.tensor(g[9:12], dtype=torch.float32)
        w = torch.tensor([g[12]], dtype=torch.float32)

        return pcd_t, R, p, w


def main():
    parser = argparse.ArgumentParser(description="Train diffusion grasp generator.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with .npz files")
    parser.add_argument("--output_path", type=str, default="checkpoints/diffusion.pt", help="Checkpoint save path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training diffusion model on: {device}")

    m_cfg = config.get("models", {})
    pn_cfg = m_cfg.get("pointnet", {})
    diff_cfg = m_cfg.get("diffusion", {})

    feat_dim = pn_cfg.get("feature_dim", 64)
    hidden_dim = diff_cfg.get("hidden_dim", 128)
    timesteps = diff_cfg.get("timesteps", 30)
    batch_size = diff_cfg.get("batch_size", 16)
    epochs = diff_cfg.get("epochs", 5)
    lr = float(diff_cfg.get("lr", 0.002))

    dataset = PositiveGraspDataset(args.data_dir, num_points=config.get("sim", {}).get("num_points_sampled", 256))
    if len(dataset) == 0:
        raise ValueError(f"No samples found in {args.data_dir}. Run scripts/generate_data.py first.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"Loaded {len(dataset)} positive grasp samples for training.")

    encoder = PointNetEncoder(in_channels=3, feature_dim=feat_dim).to(device)
    diffusion = SO3R3Diffusion(
        scene_feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        timesteps=timesteps,
        beta_start=float(diff_cfg.get("beta_start", 0.0001)),
        beta_end=float(diff_cfg.get("beta_end", 0.02))
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(diffusion.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    encoder.train()
    diffusion.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for pcd, R, p, w in dataloader:
            pcd = pcd.to(device)
            R = R.to(device)
            p = p.to(device)
            w = w.to(device)

            optimizer.zero_grad()
            scene_feat = encoder(pcd)
            loss = diffusion.forward_loss(R_0=R, p_0=p, w_0=w, scene_feat=scene_feat)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        print(f"Epoch {epoch:02d}/{epochs:02d} - Diffusion Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "diffusion": diffusion.state_dict(),
        "config": config
    }, args.output_path)
    print(f"Saved diffusion checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
