"""Active view selection pipeline orchestrator."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import torch

from src.grasp.representation import Grasp, GripperParams
from src.grasp.checker import AnalyticGraspChecker, verify_grasp_physics
from src.sim.scene import ProceduralScene
from src.sim.camera import DepthCamera, CameraPose
from src.sim.fusion import VoxelOccupancyGrid, fuse_point_clouds
from src.models.pointnet import PointNetEncoder
from src.models.diffusion import SO3R3Diffusion
from src.models.evaluator import GraspEvaluator
from src.uncertainty.ambiguity import UncertaintyEstimator, UncertaintyResult
from src.active.candidates import ViewCandidateGenerator
from src.active.scoring import ViewScorer


@dataclass
class ActiveStepRecord:
    step: int
    pose: CameraPose
    num_points: int
    rho_t: float
    U_t: float
    decision: str  # "grasp" or "move"
    top_score: float
    step_duration: float


@dataclass
class EpisodeResult:
    scene_id: int
    executed_grasp: Optional[Grasp]
    analytic_success: bool
    physics_success: bool
    num_views: int
    total_time: float
    history: List[ActiveStepRecord]
    final_U_t: float
    final_rho_t: float


class ActiveGraspPipeline:
    """Orchestrates perception, diffusion sampling, uncertainty evaluation, and active view selection."""

    def __init__(
        self,
        scene: ProceduralScene,
        camera: DepthCamera,
        encoder: PointNetEncoder,
        diffusion: SO3R3Diffusion,
        evaluator: GraspEvaluator,
        uncertainty_estimator: UncertaintyEstimator,
        view_scorer: ViewScorer,
        candidate_gen: ViewCandidateGenerator,
        checker: AnalyticGraspChecker,
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
        self.unc_est = uncertainty_estimator
        self.scorer = view_scorer
        self.candidate_gen = candidate_gen
        self.checker = checker
        
        self.tau_U = tau_U
        self.tau_rho = tau_rho
        self.max_views = max_views
        self.num_samples_k = num_samples_k
        self.ddim_steps = ddim_steps
        self.device = device

    def run_episode(
        self,
        scene_id: int,
        initial_pose: Optional[CameraPose] = None,
        verify_physics: bool = True
    ) -> EpisodeResult:
        """Run an active grasp detection episode on the current scene."""
        start_time = time.time()
        objects = self.scene.get_object_geometries()

        # Voxel grid covering workspace bounds
        bounds = (np.array([-0.22, -0.22, 0.0]), np.array([0.22, 0.22, 0.35]))
        voxel_grid = VoxelOccupancyGrid(bounds=bounds, voxel_size=0.015)
        
        observed_clouds = []
        history = []
        
        # Initial view
        current_pose = initial_pose or self.camera.compute_pose(
            elevation_deg=45.0,
            azimuth_deg=0.0,
            radius=0.60
        )

        final_grasp = None
        final_unc = None

        for step in range(1, self.max_views + 1):
            step_start = time.time()
            
            # 1. Render and backproject
            _, depth, seg = self.camera.render(current_pose)
            pcd = self.camera.backproject_pointcloud(
                depth, current_pose, seg_mask=seg,
                crop_bounds=bounds
            )
            
            observed_clouds.append(pcd)
            voxel_grid.update_view(camera_eye=current_pose.eye, surface_points=pcd)
            fused_pcd = fuse_point_clouds(observed_clouds, voxel_size=0.008)

            # Ensure fixed size for PointNet
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

            # 2. Scene encoding
            with torch.no_grad():
                scene_feat = self.encoder(pts_tensor)  # (1, feat_dim)

            # 3. Diffusion sampling on SO(3) x R^3
            p_center_t = torch.tensor(np.mean(sub_pts, axis=0), dtype=torch.float32, device=self.device) if len(sub_pts) > 0 else None
            R_samples, p_samples, w_samples = self.diffusion.sample_ddim(
                scene_feat=scene_feat,
                num_samples=self.num_samples_k,
                ddim_steps=self.ddim_steps,
                p_center=p_center_t
            )

            # 4. Evaluate grasps with evaluator network
            grasps_list = []
            for k in range(self.num_samples_k):
                g = Grasp(R=R_samples[k], p=p_samples[k], width=float(np.squeeze(w_samples[k])))
                grasps_list.append(g)

            grasps_tensor = torch.tensor(
                np.array([g.to_array() for g in grasps_list]),
                dtype=torch.float32,
                device=self.device
            )
            scene_feat_rep = scene_feat.repeat(self.num_samples_k, 1)
            with torch.no_grad():
                probs = self.evaluator.predict_prob(grasps_tensor, scene_feat_rep).cpu().numpy().flatten()

            for k, g in enumerate(grasps_list):
                g.score = float(probs[k])

            # 5. Uncertainty quantification
            unc_res: UncertaintyResult = self.unc_est.evaluate_uncertainty(
                all_samples=grasps_list,
                total_k=self.num_samples_k
            )
            final_unc = unc_res
            final_grasp = unc_res.top_grasp

            # 6. Stopping condition check
            # Execute if U_t <= tau_U and rho_t >= tau_rho or view budget exhausted
            stop_condition = (unc_res.U_t <= self.tau_U and unc_res.rho_t >= self.tau_rho)
            is_last_step = (step >= self.max_views)

            step_duration = time.time() - step_start

            if stop_condition or is_last_step:
                decision = "grasp"
                history.append(
                    ActiveStepRecord(
                        step=step,
                        pose=current_pose,
                        num_points=len(fused_pcd),
                        rho_t=unc_res.rho_t,
                        U_t=unc_res.U_t,
                        decision=decision,
                        top_score=final_grasp.score if final_grasp else 0.0,
                        step_duration=step_duration
                    )
                )
                break
            else:
                decision = "move"
                history.append(
                    ActiveStepRecord(
                        step=step,
                        pose=current_pose,
                        num_points=len(fused_pcd),
                        rho_t=unc_res.rho_t,
                        U_t=unc_res.U_t,
                        decision=decision,
                        top_score=final_grasp.score if final_grasp else 0.0,
                        step_duration=step_duration
                    )
                )
                # 7. Select next viewpoint
                candidates = self.candidate_gen.generate_candidates(self.camera)
                next_pose = self.scorer.select_best_view(
                    candidates=candidates,
                    current_pose=current_pose,
                    accepted_grasps=unc_res.best_mode_grasps,
                    voxel_grid=voxel_grid
                )
                current_pose = next_pose

        total_time = time.time() - start_time

        # Execute grasp and verify success
        analytic_success = False
        physics_success = False

        if final_grasp is not None:
            analytic_success, _ = self.checker.check_grasp(final_grasp, objects)
            if verify_physics:
                physics_success = verify_grasp_physics(
                    final_grasp,
                    client_id=self.scene.client_id,
                    gripper_params=self.checker.params
                )

        return EpisodeResult(
            scene_id=scene_id,
            executed_grasp=final_grasp,
            analytic_success=analytic_success,
            physics_success=physics_success,
            num_views=len(history),
            total_time=total_time,
            history=history,
            final_U_t=final_unc.U_t if final_unc else 1.0,
            final_rho_t=final_unc.rho_t if final_unc else 0.0
        )
