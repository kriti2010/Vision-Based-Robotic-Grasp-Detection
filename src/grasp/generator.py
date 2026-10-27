"""Grasp candidate sampling and labelled dataset generator."""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from src.grasp.representation import Grasp, GripperParams
from src.grasp.so3 import exp_so3_numpy, project_so3_numpy
from src.grasp.checker import AnalyticGraspChecker
from src.sim.scene import ProceduralScene, ObjectGeometry
from src.sim.camera import DepthCamera, CameraPose


class GraspCandidateSampler:
    """Samples 6-DoF grasp candidates around point clouds and objects."""

    def __init__(self, gripper_params: GripperParams, rng: np.random.RandomState):
        self.params = gripper_params
        self.rng = rng

    def sample_candidates_around_points(
        self,
        points: np.ndarray,
        num_candidates: int = 32
    ) -> List[Grasp]:
        """Sample candidate grasps centered near observed points."""
        if len(points) == 0:
            # Fallback to random workspace positions
            pts = self.rng.uniform([-0.1, -0.1, 0.05], [0.1, 0.1, 0.15], size=(num_candidates, 3))
        else:
            indices = self.rng.choice(len(points), size=min(len(points), num_candidates), replace=True)
            pts = points[indices]

        grasps = []
        for pt in pts:
            # 70% top-down oriented grasps with small tilt, 30% random tilt
            if self.rng.rand() < 0.70:
                theta = self.rng.uniform(0.0, np.radians(30))  # mostly downward
                phi = self.rng.uniform(0, 2 * np.pi)
                approach = -np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
            else:
                theta = self.rng.uniform(np.radians(20), np.radians(60))
                phi = self.rng.uniform(0, 2 * np.pi)
                approach = -np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
            approach = approach / np.linalg.norm(approach)

            # Random roll around approach vector
            arb = np.array([0.0, 0.0, 1.0]) if abs(approach[2]) < 0.85 else np.array([1.0, 0.0, 0.0])
            closing_base = np.cross(approach, arb)
            closing_base = closing_base / np.linalg.norm(closing_base)

            roll = self.rng.uniform(0, np.pi)
            closing = np.cos(roll) * closing_base + np.sin(roll) * np.cross(approach, closing_base)
            closing = closing / np.linalg.norm(closing)

            binormal = np.cross(approach, closing)
            binormal = binormal / np.linalg.norm(binormal)

            R = np.column_stack([closing, binormal, approach])
            R = project_so3_numpy(R)

            # Center position near surface point
            p_cand = pt + self.rng.normal(0, 0.005, size=3)
            p_cand[2] = max(0.020, p_cand[2])

            width = float(self.rng.uniform(0.065, self.params.max_width))
            grasps.append(Grasp(R=R, p=p_cand, width=width))

        return grasps


def generate_dataset(
    config: Dict[str, Any],
    output_dir: str,
    num_scenes: int,
    views_per_scene: int,
    grasps_per_view: int,
    seed: int = 42
) -> str:
    """Generate and save labelled dataset of partial point clouds and grasps."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    
    sim_cfg = config.get("sim", {})
    grasp_cfg = config.get("grasp", {})
    grip_cfg = grasp_cfg.get("gripper", {})

    gripper_params = GripperParams(
        finger_length=grip_cfg.get("finger_length", 0.06),
        finger_width=grip_cfg.get("finger_width", 0.012),
        finger_depth=grip_cfg.get("finger_depth", 0.015),
        base_height=grip_cfg.get("base_height", 0.035),
        base_width=grip_cfg.get("base_width", 0.09),
        max_width=grip_cfg.get("max_width", 0.085),
        min_width=grip_cfg.get("min_width", 0.01),
        friction_coef=grip_cfg.get("friction_coef", 0.5),
        antipodal_cos_thresh=grip_cfg.get("antipodal_cos_thresh", 0.70)
    )

    checker = AnalyticGraspChecker(gripper_params=gripper_params, table_z=sim_cfg.get("table_z", 0.0))
    sampler = GraspCandidateSampler(gripper_params=gripper_params, rng=rng)
    scene_sim = ProceduralScene(seed=seed)
    
    cam_cfg = sim_cfg.get("camera", {})
    camera = DepthCamera(
        width=cam_cfg.get("width", 128),
        height=cam_cfg.get("height", 128),
        fov=cam_cfg.get("fov", 60.0),
        near=cam_cfg.get("near", 0.1),
        far=cam_cfg.get("far", 1.5),
        physics_client=scene_sim.client_id
    )

    dataset_records = []
    total_grasps = 0
    total_positives = 0

    print(f"Generating dataset: {num_scenes} scenes, {views_per_scene} views/scene, {grasps_per_view} grasps/view...")
    for s_idx in tqdm(range(num_scenes), desc="Scenes"):
        objects = scene_sim.reset()

        for v_idx in range(views_per_scene):
            # Sample spherical camera pose
            el = float(rng.uniform(30.0, 70.0))
            az = float(rng.uniform(0.0, 360.0))
            rad = float(cam_cfg.get("radius", 0.60))
            pose = camera.compute_pose(elevation_deg=el, azimuth_deg=az, radius=rad)

            rgb, depth, seg = camera.render(pose)
            
            # Partial point cloud
            ws = np.array(sim_cfg.get("workspace_bounds", [[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.35]]), dtype=np.float32)
            crop_bounds = (ws[:, 0], ws[:, 1])
            pcd = camera.backproject_pointcloud(depth, pose, seg_mask=seg, crop_bounds=crop_bounds)
            
            # Subsample point cloud to fixed count
            n_pts = sim_cfg.get("num_points_sampled", 256)
            if len(pcd) > n_pts:
                sub_idx = rng.choice(len(pcd), size=n_pts, replace=False)
                pcd = pcd[sub_idx]
            elif len(pcd) > 0 and len(pcd) < n_pts:
                sub_idx = rng.choice(len(pcd), size=n_pts, replace=True)
                pcd = pcd[sub_idx]
            elif len(pcd) == 0:
                pcd = np.zeros((n_pts, 3), dtype=np.float32)

            # Sample candidates
            candidates = sampler.sample_candidates_around_points(pcd, num_candidates=grasps_per_view)
            
            # Also sample negative/perturbation grasps to balance classes
            grasps_arr = []
            labels_arr = []

            for g in candidates:
                success, _ = checker.check_grasp(g, objects)
                label = 1.0 if success else 0.0
                grasps_arr.append(g.to_array())
                labels_arr.append(label)
                if success:
                    total_positives += 1

            total_grasps += len(grasps_arr)
            
            record_path = os.path.join(output_dir, f"scene_{s_idx:04d}_view_{v_idx:02d}.npz")
            np.savez_compressed(
                record_path,
                scene_id=s_idx,
                view_id=v_idx,
                camera_eye=pose.eye,
                point_cloud=pcd.astype(np.float32),
                grasps=np.array(grasps_arr, dtype=np.float32),
                labels=np.array(labels_arr, dtype=np.float32)
            )
            dataset_records.append(record_path)

    scene_sim.close()
    pos_rate = (total_positives / total_grasps) if total_grasps > 0 else 0.0
    print(f"Generated {len(dataset_records)} records, {total_grasps} total grasps, {total_positives} positive ({pos_rate*100:.1f}%).")
    return output_dir
