"""Calibration script: searches for optimal stopping thresholds and computes risk-coverage curve."""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sim.scene import ProceduralScene
from src.sim.camera import DepthCamera
from src.models.pointnet import PointNetEncoder
from src.models.diffusion import SO3R3Diffusion
from src.models.evaluator import GraspEvaluator
from src.grasp.checker import AnalyticGraspChecker
from src.grasp.representation import Grasp, GripperParams
from src.uncertainty.ambiguity import UncertaintyEstimator
from src.uncertainty.calibration import (
    CalibrationRecord,
    calibrate_stopping_thresholds,
    compute_risk_coverage_curve
)


def main():
    parser = argparse.ArgumentParser(description="Calibrate uncertainty stopping thresholds.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to config YAML")
    parser.add_argument("--diff_ckpt", type=str, default="checkpoints/diffusion.pt", help="Diffusion checkpoint")
    parser.add_argument("--eval_ckpt", type=str, default="checkpoints/evaluator.pt", help="Evaluator checkpoint")
    parser.add_argument("--num_val_scenes", type=int, default=5, help="Number of validation scenes")
    parser.add_argument("--output_yaml", type=str, default="configs/calibration.yaml", help="Saved calibration config")
    parser.add_argument("--curve_path", type=str, default="results/risk_coverage_data.npz", help="Risk-coverage data path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running calibration on device: {device}")

    # Load checkpoints
    diff_data = torch.load(args.diff_ckpt, map_location=device)
    eval_data = torch.load(args.eval_ckpt, map_location=device)

    m_cfg = config.get("models", {})
    feat_dim = m_cfg.get("pointnet", {}).get("feature_dim", 64)
    hidden_dim = m_cfg.get("diffusion", {}).get("hidden_dim", 128)
    timesteps = m_cfg.get("diffusion", {}).get("timesteps", 30)

    encoder = PointNetEncoder(in_channels=3, feature_dim=feat_dim).to(device)
    encoder.load_state_dict(diff_data["encoder"])
    encoder.eval()

    diffusion = SO3R3Diffusion(
        scene_feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        timesteps=timesteps
    ).to(device)
    diffusion.load_state_dict(diff_data["diffusion"])
    diffusion.eval()

    evaluator = GraspEvaluator(scene_feat_dim=feat_dim, hidden_dim=hidden_dim).to(device)
    evaluator.load_state_dict(eval_data["evaluator"])
    evaluator.eval()

    unc_cfg = config.get("uncertainty", {})
    target_success = float(unc_cfg.get("target_success", 0.80))
    k_samples = unc_cfg.get("num_samples_k", 16)
    ddim_steps = config.get("models", {}).get("diffusion", {}).get("ddim_steps", 6)

    unc_estimator = UncertaintyEstimator(
        tau_s=float(unc_cfg.get("tau_s", 0.45)),
        eps=float(unc_cfg.get("dbscan", {}).get("eps", 0.35)),
        min_samples=int(unc_cfg.get("dbscan", {}).get("min_samples", 2))
    )

    checker = AnalyticGraspChecker()
    scene_sim = ProceduralScene(seed=777)
    camera = DepthCamera(width=128, height=128, physics_client=scene_sim.client_id)

    records: list[CalibrationRecord] = []
    print(f"Collecting validation records across {args.num_val_scenes} scenes...")

    for s_idx in tqdm(range(args.num_val_scenes), desc="Calibration"):
        objects = scene_sim.reset()
        # Viewpoints
        for el, az in [(45.0, 0.0), (55.0, 120.0)]:
            pose = camera.compute_pose(elevation_deg=el, azimuth_deg=az, radius=0.60)
            _, depth, seg = camera.render(pose)
            pcd = camera.backproject_pointcloud(depth, pose, seg_mask=seg)
            
            n_pts = 256
            if len(pcd) == 0:
                pts_tensor = torch.zeros((1, 3, n_pts), dtype=torch.float32, device=device)
            else:
                idx = np.random.choice(len(pcd), n_pts, replace=(len(pcd) < n_pts))
                pts_tensor = torch.tensor(pcd[idx], dtype=torch.float32, device=device).transpose(0, 1).unsqueeze(0)

            with torch.no_grad():
                scene_feat = encoder(pts_tensor)

            R_s, p_s, w_s = diffusion.sample_ddim(scene_feat, num_samples=k_samples, ddim_steps=ddim_steps)

            grasps = [Grasp(R=R_s[i], p=p_s[i], width=float(np.squeeze(w_s[i]))) for i in range(k_samples)]
            g_t = torch.tensor(np.array([g.to_array() for g in grasps]), dtype=torch.float32, device=device)
            s_rep = scene_feat.repeat(k_samples, 1)

            with torch.no_grad():
                probs = evaluator.predict_prob(g_t, s_rep).cpu().numpy().flatten()

            for i, g in enumerate(grasps):
                g.score = float(probs[i])

            unc_res = unc_estimator.evaluate_uncertainty(grasps, total_k=k_samples)
            top_g = unc_res.top_grasp

            if top_g is not None:
                is_success, _ = checker.check_grasp(top_g, objects)
                records.append(
                    CalibrationRecord(
                        U_t=unc_res.U_t,
                        rho_t=unc_res.rho_t,
                        success=is_success,
                        top_score=top_g.score
                    )
                )

    scene_sim.close()
    print(f"Collected {len(records)} validation records.")

    # Calibrate thresholds
    calib_result = calibrate_stopping_thresholds(records, target_success=target_success)
    coverages, risks, rc_auc = compute_risk_coverage_curve(records)

    print("\n[CALIBRATION RESULTS]")
    print(f"Target success rate: {target_success * 100:.1f}%")
    print(f"Optimal tau_U:       {calib_result['tau_U']:.4f}")
    print(f"Optimal tau_rho:     {calib_result['tau_rho']:.4f}")
    print(f"Empirical Coverage:  {calib_result['coverage'] * 100:.1f}%")
    print(f"Empirical Success:   {calib_result['success_rate'] * 100:.1f}%")
    print(f"Risk-Coverage AUC:   {rc_auc:.4f}")

    os.makedirs(os.path.dirname(args.output_yaml), exist_ok=True)
    with open(args.output_yaml, "w") as f:
        yaml.dump(calib_result, f)

    os.makedirs(os.path.dirname(args.curve_path), exist_ok=True)
    np.savez_compressed(
        args.curve_path,
        coverages=coverages,
        risks=risks,
        rc_auc=rc_auc,
        tau_U=calib_result["tau_U"],
        tau_rho=calib_result["tau_rho"]
    )
    print(f"Saved calibration to {args.output_yaml} and curve to {args.curve_path}")


if __name__ == "__main__":
    main()
