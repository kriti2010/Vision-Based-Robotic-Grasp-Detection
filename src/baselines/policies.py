"""Baseline policies: Single-view, Random-view, Fixed-sweep, Volumetric-occlusion, Evaluator-only, Deep Ensemble, and Mode Entropy ablation."""

from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import torch

from src.grasp.representation import Grasp
from src.grasp.checker import AnalyticGraspChecker, verify_grasp_physics
from src.sim.scene import ProceduralScene
from src.sim.camera import DepthCamera, CameraPose
from src.sim.fusion import VoxelOccupancyGrid, fuse_point_clouds
from src.sim.visibility import VisibilityChecker
from src.models.pointnet import PointNetEncoder
from src.models.diffusion import SO3R3Diffusion
from src.models.evaluator import GraspEvaluator
from src.uncertainty.ambiguity import UncertaintyEstimator, UncertaintyResult
from src.active.candidates import ViewCandidateGenerator
from src.active.pipeline import EpisodeResult, ActiveStepRecord


class BaselineRunner:
    """Runs different baseline active grasping policies on the same test scenes."""

    def __init__(
        self,
        scene: ProceduralScene,
        camera: DepthCamera,
        encoder: PointNetEncoder,
        diffusion: SO3R3Diffusion,
        evaluator: GraspEvaluator,
        ensemble_evaluators: Optional[List[GraspEvaluator]] = None,
        checker: Optional[AnalyticGraspChecker] = None,
        tau_U: float = 0.05,
        tau_rho: float = 0.30,
        max_views: int = 4,
        num_samples_k: int = 64,
        ddim_steps: int = 15,
        device: str = "cpu"
    ):
        self.scene = scene
        self.camera = camera
        self.encoder = encoder.to(device)
        self.diffusion = diffusion.to(device)
        self.evaluator = evaluator.to(device)
        self.ensemble = [e.to(device) for e in (ensemble_evaluators or [])]
        self.checker = checker or AnalyticGraspChecker()
        self.tau_U = tau_U
        self.tau_rho = tau_rho
        self.max_views = max_views
        self.num_samples_k = num_samples_k
        self.ddim_steps = ddim_steps
        self.device = device
        self.unc_est = UncertaintyEstimator()
        self.candidate_gen = ViewCandidateGenerator(num_candidates=12)

    def _observe_and_sample(
        self,
        pose: CameraPose,
        observed_clouds: List[np.ndarray],
        voxel_grid: VoxelOccupancyGrid,
        bounds: Any
    ) -> Tuple[List[Grasp], np.ndarray, np.ndarray]:
        """Render, fuse, encode scene and sample grasps via diffusion."""
        _, depth, seg = self.camera.render(pose)
        pcd = self.camera.backproject_pointcloud(depth, pose, seg_mask=seg, crop_bounds=bounds)
        observed_clouds.append(pcd)
        voxel_grid.update_view(camera_eye=pose.eye, surface_points=pcd)
        fused_pcd = fuse_point_clouds(observed_clouds, voxel_size=0.008)

        n_pts = 256
        if len(fused_pcd) == 0:
            pts_tensor = torch.zeros((1, 3, n_pts), dtype=torch.float32, device=self.device)
        else:
            if len(fused_pcd) >= n_pts:
                idx = np.random.choice(len(fused_pcd), n_pts, replace=False)
                sub_pts = fused_pcd[idx]
            else:
                idx = np.random.choice(len(fused_pcd), n_pts, replace=True)
                sub_pts = fused_pcd[idx]
            pts_tensor = torch.tensor(sub_pts, dtype=torch.float32, device=self.device).transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            scene_feat = self.encoder(pts_tensor)

        p_center_t = torch.tensor(np.mean(sub_pts, axis=0), dtype=torch.float32, device=self.device) if len(sub_pts) > 0 else None
        R_samples, p_samples, w_samples = self.diffusion.sample_ddim(
            scene_feat=scene_feat,
            num_samples=self.num_samples_k,
            ddim_steps=self.ddim_steps,
            p_center=p_center_t
        )

        grasps = [Grasp(R=R_samples[k], p=p_samples[k], width=float(np.squeeze(w_samples[k]))) for k in range(self.num_samples_k)]
        grasps_tensor = torch.tensor(np.array([g.to_array() for g in grasps]), dtype=torch.float32, device=self.device)
        scene_feat_rep = scene_feat.repeat(self.num_samples_k, 1)

        with torch.no_grad():
            probs = self.evaluator.predict_prob(grasps_tensor, scene_feat_rep).cpu().numpy().flatten()

        for k, g in enumerate(grasps):
            g.score = float(probs[k])

        # Also get ensemble predictions if available
        ensemble_vars = np.zeros(self.num_samples_k)
        if len(self.ensemble) > 0:
            ens_preds = []
            for ens_eval in self.ensemble:
                with torch.no_grad():
                    ep = ens_eval.predict_prob(grasps_tensor, scene_feat_rep).cpu().numpy().flatten()
                    ens_preds.append(ep)
            ensemble_vars = np.var(np.array(ens_preds), axis=0)

        return grasps, ensemble_vars, fused_pcd

    def run_policy(
        self,
        policy_name: str,
        scene_id: int,
        initial_pose: Optional[CameraPose] = None,
        verify_physics: bool = True
    ) -> EpisodeResult:
        """Execute one of the baseline policies:
        - 'single_view'
        - 'random_view'
        - 'fixed_sweep'
        - 'volumetric_occlusion'
        - 'evaluator_only'
        - 'deep_ensemble'
        - 'mode_entropy'
        """
        start_time = time.time()
        objects = self.scene.get_object_geometries()
        bounds = (np.array([-0.22, -0.22, 0.0]), np.array([0.22, 0.22, 0.35]))
        voxel_grid = VoxelOccupancyGrid(bounds=bounds, voxel_size=0.015)

        observed_clouds = []
        history = []
        current_pose = initial_pose or self.camera.compute_pose(elevation_deg=45.0, azimuth_deg=0.0, radius=0.60)
        
        final_grasp = None
        final_U_t = 1.0
        final_rho_t = 0.0

        fixed_azimuths = [0.0, 90.0, 180.0, 270.0]

        for step in range(1, self.max_views + 1):
            step_start = time.time()
            grasps, ens_vars, fused_pcd = self._observe_and_sample(current_pose, observed_clouds, voxel_grid, bounds)

            unc_res = self.unc_est.evaluate_uncertainty(grasps, total_k=self.num_samples_k)
            final_U_t = unc_res.U_t
            final_rho_t = unc_res.rho_t
            final_grasp = unc_res.top_grasp
            is_last_step = (step >= self.max_views)

            # Determine stopping rule per baseline
            should_stop = False

            if policy_name == "single_view":
                should_stop = True
            elif policy_name == "evaluator_only":
                # Stops when top evaluator score >= 0.70 or last step
                max_score = max([g.score for g in grasps]) if grasps else 0.0
                should_stop = (max_score >= 0.70) or is_last_step
            elif policy_name == "deep_ensemble":
                # Stops when mean ensemble variance of accepted grasps is low (< 0.02)
                mean_var = float(np.mean(ens_vars)) if len(ens_vars) > 0 else 0.0
                should_stop = (mean_var < 0.02 and unc_res.rho_t >= self.tau_rho) or is_last_step
            elif policy_name == "mode_entropy":
                # Stops when mode entropy is low (< 0.35) and accepted fraction high
                should_stop = (unc_res.mode_entropy < 0.35 and unc_res.rho_t >= self.tau_rho) or is_last_step
            elif policy_name in ("random_view", "fixed_sweep", "volumetric_occlusion"):
                # Standard uncertainty stopping
                should_stop = (unc_res.U_t <= self.tau_U and unc_res.rho_t >= self.tau_rho) or is_last_step

            step_duration = time.time() - step_start

            if should_stop:
                history.append(
                    ActiveStepRecord(
                        step=step, pose=current_pose, num_points=len(fused_pcd),
                        rho_t=unc_res.rho_t, U_t=unc_res.U_t, decision="grasp",
                        top_score=final_grasp.score if final_grasp else 0.0,
                        step_duration=step_duration
                    )
                )
                break
            else:
                history.append(
                    ActiveStepRecord(
                        step=step, pose=current_pose, num_points=len(fused_pcd),
                        rho_t=unc_res.rho_t, U_t=unc_res.U_t, decision="move",
                        top_score=final_grasp.score if final_grasp else 0.0,
                        step_duration=step_duration
                    )
                )

                # Next view selection logic per baseline
                candidates = self.candidate_gen.generate_candidates(self.camera)

                if policy_name == "random_view":
                    idx = np.random.choice(len(candidates))
                    current_pose = candidates[idx]

                elif policy_name == "fixed_sweep":
                    next_az = fixed_azimuths[step % len(fixed_azimuths)]
                    current_pose = self.camera.compute_pose(elevation_deg=45.0, azimuth_deg=next_az, radius=0.60)

                elif policy_name == "volumetric_occlusion":
                    # Maximize visible unknown voxels in whole workspace bounding box
                    checker = VisibilityChecker(voxel_grid)
                    best_cand = candidates[0]
                    max_scene_unk = -1
                    
                    # Center of workspace
                    ws_c = np.array([0.0, 0.0, 0.10])
                    ws_ext = np.array([0.35, 0.35, 0.20])
                    ws_rot = np.eye(3)

                    for cand in candidates:
                        v_cnt = checker.count_visible_unknown_in_region(
                            camera_eye=cand.eye, center=ws_c, extents=ws_ext, rotation=ws_rot, sample_density=2
                        )
                        if v_cnt > max_scene_unk:
                            max_scene_unk = v_cnt
                            best_cand = cand
                    current_pose = best_cand

                else:
                    # Random next view fallback
                    idx = np.random.choice(len(candidates))
                    current_pose = candidates[idx]

        total_time = time.time() - start_time
        analytic_success = False
        physics_success = False

        if final_grasp is not None:
            analytic_success, _ = self.checker.check_grasp(final_grasp, objects)
            if verify_physics:
                physics_success = verify_grasp_physics(
                    final_grasp, client_id=self.scene.client_id, gripper_params=self.checker.params
                )

        return EpisodeResult(
            scene_id=scene_id,
            executed_grasp=final_grasp,
            analytic_success=analytic_success,
            physics_success=physics_success,
            num_views=len(history),
            total_time=total_time,
            history=history,
            final_U_t=final_U_t,
            final_rho_t=final_rho_t
        )
