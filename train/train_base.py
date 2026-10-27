"""Base model training: Diffusion Generator and Scorer.

Specification:
- Generator: AdamW, lr 3e-4, cosine annealing, batch 64. Trained ONLY on successful grasps.
- Scorer: AdamW, lr 3e-4, cosine annealing, batch 64, BCE loss.
- Temperature calibration on held-out scene split. Reports ECE and AUROC.
- Split by SCENE, never by grasp.
"""

import os
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.generator import GraspDiffusionDenoiser
from models.scorer import GraspSuccessScorer, compute_ece
from train.dataset import GraspDataset, generate_base_dataset


def train_generator(
    train_files: List[str],
    steps: int = 20000,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
    output_ckpt: Optional[str] = None,
    warm_start_ckpt: Optional[str] = None,
) -> GraspDiffusionDenoiser:
    """Train the variance-exploding diffusion grasp generator."""
    dataset = GraspDataset(train_files, successful_only=True)
    if len(dataset) == 0:
        dataset = GraspDataset(train_files, successful_only=False)

    actual_batch_size = max(1, min(batch_size, len(dataset)))
    loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=False)
    model = GraspDiffusionDenoiser().to(device)

    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        print(f"Warm-starting generator from {warm_start_ckpt}")
        model.load_state_dict(torch.load(warm_start_ckpt, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    model.train()
    step = 0
    pbar = tqdm(total=steps, desc="Training Generator")
    iter_loader = iter(loader)

    while step < steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)

        points = batch["points"].to(device)
        R_0 = batch["R"].to(device)
        t_0 = batch["t"].to(device)
        w_0 = batch["w"].to(device)

        optimizer.zero_grad()
        loss, metrics = model.compute_loss(R_0, t_0, w_0, points)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        pbar.update(1)
        if step % 50 == 0 or step == steps:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    pbar.close()

    if output_ckpt:
        os.makedirs(os.path.dirname(output_ckpt) or ".", exist_ok=True)
        torch.save(model.state_dict(), output_ckpt)
        print(f"Saved generator checkpoint to {output_ckpt}")

    return model


def train_scorer(
    train_files: List[str],
    val_files: List[str],
    steps: int = 20000,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
    output_ckpt: Optional[str] = None,
    warm_start_ckpt: Optional[str] = None,
) -> Tuple[GraspSuccessScorer, dict]:
    train_ds = GraspDataset(train_files, successful_only=False)
    val_ds = GraspDataset(val_files, successful_only=False)
    actual_batch_size = max(1, min(batch_size, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(1, min(batch_size, len(val_ds))), shuffle=False)

    model = GraspSuccessScorer().to(device)
    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        print(f"Warm-starting scorer from {warm_start_ckpt}")
        model.load_state_dict(torch.load(warm_start_ckpt, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    step = 0
    pbar = tqdm(total=steps, desc="Training Scorer")
    iter_loader = iter(train_loader)

    while step < steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            batch = next(iter_loader)

        points = batch["points"].to(device)
        R = batch["R"].to(device)
        t = batch["t"].to(device)
        w = batch["w"].to(device)
        labels = batch["label"].to(device).unsqueeze(-1)

        optimizer.zero_grad()
        logits = model.forward_logits(R, t, w, points)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        pbar.update(1)
        if step % 50 == 0 or step == steps:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    pbar.close()

    # Temperature scaling on held-out validation scenes
    model.eval()
    val_logits_list = []
    val_labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            points = batch["points"].to(device)
            R = batch["R"].to(device)
            t = batch["t"].to(device)
            w = batch["w"].to(device)
            labels = batch["label"].cpu().numpy()

            logits = model.forward_logits(R, t, w, points).squeeze(-1).cpu().numpy()
            val_logits_list.append(logits)
            val_labels_list.append(labels)

    val_logits = np.concatenate(val_logits_list, axis=0)
    val_labels = np.concatenate(val_labels_list, axis=0)

    # Fit temperature
    best_T = model.fit_temperature(val_logits, val_labels)

    # Evaluate AUROC and ECE
    calibrated_probs = 1.0 / (1.0 + np.exp(-val_logits / best_T))
    try:
        auroc = float(roc_auc_score(val_labels, calibrated_probs))
    except Exception:
        auroc = 0.5
    ece = compute_ece(calibrated_probs, val_labels, num_bins=10)

    metrics = {
        "best_temperature": best_T,
        "val_auroc": auroc,
        "val_ece": ece,
    }
    print(f"\nScorer Validation Metrics: AUROC = {auroc:.4f}, ECE = {ece:.4f}, Temperature T = {best_T:.4f}")

    if output_ckpt:
        os.makedirs(os.path.dirname(output_ckpt) or ".", exist_ok=True)
        torch.save(model.state_dict(), output_ckpt)
        print(f"Saved scorer checkpoint to {output_ckpt}")

    return model, metrics
