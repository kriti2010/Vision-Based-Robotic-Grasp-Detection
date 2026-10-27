"""Run interactive active view selection session and demonstrate stopping decisions."""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sim.scene import ProceduralScene
from src.sim.camera import DepthCamera
from src.models.pointnet import PointNetEncoder
from src.models.diffusion import SO3R3Diffusion
from src.models.evaluator import GraspEvaluator
from src.grasp.checker import AnalyticGraspChecker
from src.uncertainty.ambiguity import UncertaintyEstimator
from src.active.candidates import ViewCandidateGenerator
from src.active.scoring import ViewScorer
from src.active.pipeline import ActiveGraspPipeline


def main():
    parser = argparse.ArgumentParser(description="Run active grasp detection episode.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config YAML")
    parser.add_argument("--diff_ckpt", type=str, default="checkpoints/diffusion.pt", help="Diffusion checkpoint")
    parser.add_argument("--eval_ckpt", type=str, default="checkpoints/evaluator.pt", help="Evaluator checkpoint")
    parser.add_argument("--calib_yaml", type=str, default="configs/calibration.yaml", help="Calibration thresholds YAML")
    parser.add_argument("--scene_seed", type=int, default=101, help="Random seed for procedural scene")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load calibration if available
    tau_U = config.get("uncertainty", {}).get("tau_U", 0.15)
    tau_rho = config.get("uncertainty", {}).get("tau_rho", 0.25)
    if os.path.exists(args.calib_yaml):
        with open(args.calib_yaml, "r") as f:
            calib = yaml.safe_load(f)
            tau_U = calib.get("tau_U", tau_U)
            tau_rho = calib.get("tau_rho", tau_rho)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diff_data = torch.load(args.diff_ckpt, map_location=device)
    eval_data = torch.load(args.eval_ckpt, map_location=device)

    m_cfg = config.get("models", {})
    feat_dim = m_cfg.get("pointnet", {}).get("feature_dim", 64)
    hidden_dim = m_cfg.get("diffusion", {}).get("hidden_dim", 128)
    timesteps = m_cfg.get("diffusion", {}).get("timesteps", 30)

    encoder = PointNetEncoder(in_channels=3, feature_dim=feat_dim).to(device)
    encoder.load_state_dict(diff_data["encoder"])
    encoder.eval()

    diffusion = SO3R3Diffusion(scene_feat_dim=feat_dim, hidden_dim=hidden_dim, timesteps=timesteps).to(device)
    diffusion.load_state_dict(diff_data["diffusion"])
    diffusion.eval()

    evaluator = GraspEvaluator(scene_feat_dim=feat_dim, hidden_dim=hidden_dim).to(device)
    evaluator.load_state_dict(eval_data["evaluator"])
    evaluator.eval()

    unc_cfg = config.get("uncertainty", {})
    unc_est = UncertaintyEstimator(
        tau_s=float(unc_cfg.get("tau_s", 0.45)),
        eps=float(unc_cfg.get("dbscan", {}).get("eps", 0.35)),
        min_samples=int(unc_cfg.get("dbscan", {}).get("min_samples", 2))
    )

    scene_sim = ProceduralScene(seed=args.scene_seed)
    scene_sim.reset()
    camera = DepthCamera(width=128, height=128, physics_client=scene_sim.client_id)
    checker = AnalyticGraspChecker()

    pipeline = ActiveGraspPipeline(
        scene=scene_sim,
        camera=camera,
        encoder=encoder,
        diffusion=diffusion,
        evaluator=evaluator,
        uncertainty_estimator=unc_est,
        view_scorer=ViewScorer(cost_weight=float(config.get("active", {}).get("travel_cost_weight", 1.0))),
        candidate_gen=ViewCandidateGenerator(num_candidates=int(config.get("active", {}).get("num_candidate_views", 6))),
        checker=checker,
        tau_U=tau_U,
        tau_rho=tau_rho,
        max_views=int(config.get("active", {}).get("max_views", 3)),
        num_samples_k=int(unc_cfg.get("num_samples_k", 16)),
        ddim_steps=int(m_cfg.get("diffusion", {}).get("ddim_steps", 6)),
        device=device
    )

    print(f"\n--- Running Active Grasp Detection on Scene Seed {args.scene_seed} ---")
    print(f"Stopping Thresholds: tau_U = {tau_U:.4f}, tau_rho = {tau_rho:.4f}")

    res = pipeline.run_episode(scene_id=args.scene_seed, verify_physics=True)

    print("\nEpisode Step Execution Trace:")
    for r in res.history:
        print(f"  Step {r.step}: Points={r.num_points:4d} | U_t={r.U_t:.4f} | rho_t={r.rho_t:.2f} | Action={r.decision.upper():5s} | TopScore={r.top_score:.3f} | Time={r.step_duration:.2f}s")

    print("\nExecution Summary:")
    print(f"  Total Views Taken: {res.num_views}")
    print(f"  Analytic Grasp Success: {res.analytic_success}")
    print(f"  Physics Grasp Success:  {res.physics_success}")
    print(f"  Total Episode Time:     {res.total_time:.2f}s")
    
    scene_sim.close()


if __name__ == "__main__":
    main()
