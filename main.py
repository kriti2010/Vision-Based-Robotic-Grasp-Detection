"""Master CLI entry point for CSE471 Research Prototype Reproduction.

Supports all pipeline stages and Makefile targets:
- smoke: Rapid end-to-end test (< 3 min on CPU)
- data: Procedural base scene dataset generation
- train: Base SO(3) x R^3 diffusion generator and calibrated scorer training
- rounds: Round 1 (on-generator) and Round 2 (joint retraining)
- episodes: Multi-view active grasp episode collection (fusion, calib, test)
- analyze: Risk model fitting, conformal calibration, tables, and figures
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import pandas as pd

from models.generator import GraspDiffusionDenoiser
from models.scorer import GraspSuccessScorer
from train.dataset import generate_base_dataset
from train.train_base import train_generator, train_scorer
from train.rounds import run_round1, run_round2
from episodes.runner import run_episodes_split
from risk.models import RiskModelSuite
from risk.conformal import calibrate_conformal_risk, evaluate_policy_offline
from analysis.tables import (
    compute_auroc_table, compute_fixed_budget_table,
    compute_policy_table, compute_bound_check
)
from analysis.figures import (
    plot_pipeline_diagram, plot_success_vs_views, plot_bound_validation
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def stage_data(cfg: dict):
    print("\n" + "=" * 60)
    print("STAGE: DATA GENERATION")
    print("=" * 60)
    data_cfg = cfg["data"]
    saved = generate_base_dataset(
        output_dir=data_cfg["data_dir"],
        num_scenes=data_cfg["base_scenes"],
        seed_start=data_cfg["base_seed_start"],
        num_antipodal=data_cfg["antipodal_per_scene"],
        num_perturbed=data_cfg["perturbed_per_scene"],
        desc="Base Data",
    )
    print(f"Generated {len(saved)} scenes in {data_cfg['data_dir']}")
    return saved


def stage_train(cfg: dict, data_files: list[str] = None):
    print("\n" + "=" * 60)
    print("STAGE: BASE TRAINING (GENERATOR & SCORER)")
    print("=" * 60)
    if not data_files:
        data_dir = cfg["data"]["data_dir"]
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]

    split_idx = max(1, int(0.8 * len(data_files)))
    train_f = data_files[:split_idx]
    val_f = data_files[split_idx:] if split_idx < len(data_files) else data_files[:1]

    device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    steps = cfg["train"]["base_steps"]

    # 1. Train generator
    gen = train_generator(
        train_files=train_f,
        steps=steps,
        batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"],
        device=device,
        output_ckpt="checkpoints/generator_base.pt",
    )

    # 2. Train scorer
    scorer, metrics = train_scorer(
        train_files=train_f,
        val_files=val_f,
        steps=steps,
        batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"],
        device=device,
        output_ckpt="checkpoints/scorer_base.pt",
    )
    return gen, scorer


def stage_rounds(cfg: dict, data_files: list[str] = None):
    print("\n" + "=" * 60)
    print("STAGE: SELF-TRAINING ROUNDS (ROUND 1 & 2)")
    print("=" * 60)
    if not data_files:
        data_dir = cfg["data"]["data_dir"]
        data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]

    device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    r_cfg = cfg["rounds"]

    # Round 1
    scorer_r1, m1 = run_round1(
        base_train_files=data_files,
        base_val_files=data_files[:max(1, int(0.2 * len(data_files)))],
        generator_ckpt="checkpoints/generator_base.pt",
        scorer_ckpt="checkpoints/scorer_base.pt",
        num_scenes=r_cfg["round1_scenes"],
        seed_start=r_cfg["round1_seed_start"],
        steps=r_cfg["round1_steps"],
        device=device,
        output_dir="data/round1",
        output_scorer_ckpt="checkpoints/scorer_round1.pt",
    )

    # Round 2
    gen_r2, scorer_r2, m2 = run_round2(
        base_train_files=data_files,
        generator_ckpt="checkpoints/generator_base.pt",
        scorer_ckpt="checkpoints/scorer_round1.pt",
        num_scenes=r_cfg["round2_scenes"],
        seed_start=r_cfg["round2_seed_start"],
        steps=r_cfg["round2_steps"],
        device=device,
        output_dir="data/round2",
        output_gen_ckpt="checkpoints/generator_round2.pt",
        output_scorer_ckpt="checkpoints/scorer_round2.pt",
    )
    return gen_r2, scorer_r2


def stage_episodes(cfg: dict):
    print("\n" + "=" * 60)
    print("STAGE: MULTI-VIEW EPISODES COLLECTION")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"
    ep_cfg = cfg["episodes"]

    # Load generator & scorer
    gen_ckpt = "checkpoints/generator_round2.pt" if os.path.exists("checkpoints/generator_round2.pt") else "checkpoints/generator_base.pt"
    scorer_ckpt = "checkpoints/scorer_round2.pt" if os.path.exists("checkpoints/scorer_round2.pt") else "checkpoints/scorer_base.pt"

    generator = GraspDiffusionDenoiser().to(device)
    generator.load_state_dict(torch.load(gen_ckpt, map_location=device))
    scorer = GraspSuccessScorer().to(device)
    scorer.load_state_dict(torch.load(scorer_ckpt, map_location=device))

    # 1. Fusion split
    fusion_seeds = list(range(ep_cfg["fusion_seed_start"], ep_cfg["fusion_seed_start"] + ep_cfg["fusion_episodes"]))
    run_episodes_split(
        seeds=fusion_seeds,
        generator=generator,
        scorer=scorer,
        output_path="results/episodes_fusion.npz",
        T=ep_cfg["views_T"],
        N=ep_cfg["samples_N"],
        num_steps=ep_cfg["steps_K"],
        device=device,
        desc="Fusion Episodes",
    )

    # 2. Calibration split
    calib_seeds = list(range(ep_cfg["calib_seed_start"], ep_cfg["calib_seed_start"] + ep_cfg["calib_episodes"]))
    run_episodes_split(
        seeds=calib_seeds,
        generator=generator,
        scorer=scorer,
        output_path="results/episodes_calib.npz",
        T=ep_cfg["views_T"],
        N=ep_cfg["samples_N"],
        num_steps=ep_cfg["steps_K"],
        device=device,
        desc="Calibration Episodes",
    )

    # 3. Test split
    test_seeds = list(range(ep_cfg["test_seed_start"], ep_cfg["test_seed_start"] + ep_cfg["test_episodes"]))
    run_episodes_split(
        seeds=test_seeds,
        generator=generator,
        scorer=scorer,
        output_path="results/episodes_test.npz",
        T=ep_cfg["views_T"],
        N=ep_cfg["samples_N"],
        num_steps=ep_cfg["steps_K"],
        device=device,
        desc="Test Episodes",
    )


def stage_analyze(cfg: dict):
    print("\n" + "=" * 60)
    print("STAGE: ANALYSIS, TABLES, AND FIGURES")
    print("=" * 60)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    fusion_data = np.load("results/episodes_fusion.npz")
    calib_data = np.load("results/episodes_calib.npz")
    test_data = np.load("results/episodes_test.npz")

    # 1. Fit Risk Model Suite on Fusion Split
    print("\n--- Fitting 4 Risk Models on Fusion Split ---")
    risk_suite = RiskModelSuite()
    risk_suite.fit(fusion_data["features"], fusion_data["outcomes"])

    # Predict risks on calibration and test splits
    calib_risks = risk_suite.predict_risk(calib_data["features"])
    test_risks = risk_suite.predict_risk(test_data["features"])

    # 2. Table 1: AUROC with 95% Bootstrap CI and AURC
    print("\n--- Table 1: AUROC & AURC Comparison ---")
    auroc_df = compute_auroc_table(test_data["features"], test_data["outcomes"], risk_suite, n_bootstrap=200)
    print(auroc_df.to_string(index=False))
    auroc_df.to_csv("results/tables/table1_auroc_aurc.csv", index=False)

    # 3. Table 2: Fixed-Budget Success vs Views
    print("\n--- Table 2: Fixed-Budget Success vs Views ---")
    fixed_df = compute_fixed_budget_table(test_data["outcomes"])
    print(fixed_df.to_string(index=False))
    fixed_df.to_csv("results/tables/table2_fixed_budget.csv", index=False)

    # 4. Table 3: Conformal Policy Table (alpha in 0.05, 0.10, 0.20)
    print("\n--- Table 3: Conformal Policy Performance (Score-Only) ---")
    policy_df = compute_policy_table(
        calib_risks["score_only"], calib_data["outcomes"],
        test_risks["score_only"], test_data["outcomes"],
        alphas=[0.05, 0.10, 0.20],
    )
    print(policy_df.to_string(index=False))
    policy_df.to_csv("results/tables/table3_policy_alphas.csv", index=False)

    # 5. Table 4: Bound Check across Random Re-splits
    print("\n--- Table 4: Conformal Bound Check across Re-splits ---")
    # Pool calibration + test episodes
    pooled_outcomes = np.concatenate([calib_data["outcomes"], test_data["outcomes"]], axis=0)
    pooled_risks = {
        m: np.concatenate([calib_risks[m], test_risks[m]], axis=0) for m in calib_risks
    }

    n_resplits = cfg["risk"].get("bound_check_resplits", 100)
    bound_df = compute_bound_check(
        all_risks=pooled_risks,
        all_outcomes=pooled_outcomes,
        alpha=0.10,
        n_resplits=n_resplits,
        calib_size=len(calib_data["outcomes"]),
    )
    print(bound_df.to_string(index=False))
    bound_df.to_csv("results/tables/table4_bound_check.csv", index=False)

    # 6. Figures
    print("\n--- Generating Publication Figures ---")
    plot_pipeline_diagram("results/figures/pipeline_diagram.png")
    plot_success_vs_views(fixed_df["Raw Mean"].tolist(), fixed_df["Std Error"].tolist(), "results/figures/success_vs_views.png")

    # Bound validation figure data
    resplit_losses = {}
    rng = np.random.RandomState(42)
    for m in ["score_only", "dispersion", "cross_view", "all_features"]:
        m_r = pooled_risks[m]
        losses = []
        for _ in range(50):
            p = rng.permutation(len(pooled_outcomes))
            c_idx, t_idx = p[:len(calib_data["outcomes"])], p[len(calib_data["outcomes"]):]
            lam, _, _ = calibrate_conformal_risk(m_r[c_idx], pooled_outcomes[c_idx], alpha=0.10)
            res = evaluate_policy_offline(m_r[t_idx], pooled_outcomes[t_idx], lam)
            losses.append(res["mean_loss"])
        resplit_losses[m] = np.array(losses)

    plot_bound_validation(resplit_losses, alpha=0.10, output_path="results/figures/bound_validation.png")
    print("\nAnalysis successfully completed. Artifacts saved in results/tables/ and results/figures/.")


def main():
    parser = argparse.ArgumentParser(description="CSE471 Research Prototype Pipeline.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--stage", type=str, default="smoke", choices=["smoke", "data", "train", "rounds", "episodes", "analyze", "all"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"\nRunning CSE471 Pipeline | Mode: {cfg.get('mode', 'smoke')} | Stage: {args.stage}")

    if args.stage == "smoke" or args.stage == "all":
        data_files = stage_data(cfg)
        stage_train(cfg, data_files)
        stage_rounds(cfg, data_files)
        stage_episodes(cfg)
        stage_analyze(cfg)
    elif args.stage == "data":
        stage_data(cfg)
    elif args.stage == "train":
        stage_train(cfg)
    elif args.stage == "rounds":
        stage_rounds(cfg)
    elif args.stage == "episodes":
        stage_episodes(cfg)
    elif args.stage == "analyze":
        stage_analyze(cfg)


if __name__ == "__main__":
    main()
