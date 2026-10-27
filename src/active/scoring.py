"""Scoring candidate viewpoints via information gain G(v) and travel cost c(v)."""

from typing import List, Tuple
import numpy as np

from src.grasp.representation import Grasp
from src.sim.camera import CameraPose
from src.sim.fusion import VoxelOccupancyGrid
from src.sim.visibility import VisibilityChecker


class ViewScorer:
    """Computes G(v) / c(v) for candidate viewpoints."""

    def __init__(self, cost_weight: float = 1.0, min_travel_cost: float = 0.05):
        self.cost_weight = cost_weight
        self.min_travel_cost = min_travel_cost

    def score_view(
        self,
        candidate_pose: CameraPose,
        current_pose: CameraPose,
        accepted_grasps: List[Grasp],
        voxel_grid: VoxelOccupancyGrid
    ) -> Tuple[float, int, float]:
        """Compute objective G(v) / c(v).
        
        Args:
            candidate_pose: Candidate view to evaluate
            current_pose: Current robot camera pose
            accepted_grasps: Grasps accepted by evaluator
            voxel_grid: Current 3-state voxel occupancy grid
        Returns:
            (score, G_v, c_v)
        """
        checker = VisibilityChecker(voxel_grid)
        total_unknown_visible = 0

        # Sample points within closing volumes of accepted grasps
        for g in accepted_grasps[:10]:  # focus on top 10 accepted grasps for speed
            c, ext, R = g.get_closing_volume_bounds()
            vis_unk = checker.count_visible_unknown_in_region(
                camera_eye=candidate_pose.eye,
                center=c,
                extents=ext,
                rotation=R,
                sample_density=3
            )
            total_unknown_visible += vis_unk

        G_v = total_unknown_visible

        # Travel distance between camera eye positions
        dist = np.linalg.norm(candidate_pose.eye - current_pose.eye)
        c_v = max(self.min_travel_cost, float(dist) * self.cost_weight)

        score = float(G_v) / c_v
        return score, G_v, c_v

    def select_best_view(
        self,
        candidates: List[CameraPose],
        current_pose: CameraPose,
        accepted_grasps: List[Grasp],
        voxel_grid: VoxelOccupancyGrid
    ) -> CameraPose:
        """Select candidate maximizing G(v) / c(v)."""
        best_score = -1.0
        best_pose = candidates[0]

        for cand in candidates:
            score, _, _ = self.score_view(cand, current_pose, accepted_grasps, voxel_grid)
            if score > best_score:
                best_score = score
                best_pose = cand

        return best_pose
