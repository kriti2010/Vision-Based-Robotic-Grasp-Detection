"""Floating two-finger multibody gripper and physical grasp execution in PyBullet DIRECT mode.

Specifications:
- Palm: half-extents (0.05, 0.012, 0.012)
- Fingers: (0.005, 0.012, 0.03) on prismatic joints along x; opening = width + 0.03.
- Pre-closing collision check: reject if penetration depth > 3 mm (-0.003 m).
- Close with position control: force 40, 120 steps.
- Lift: 0.15 m over 240 steps.
- Success criterion: object nearest grasp centre rises > 0.10 m and stays within 0.06 m of gripper centre.
- Restores saved state after trial.
"""

from typing import Tuple, List, Optional
import numpy as np
import pybullet as p
from sim.scene import ClutteredScene


def create_floating_gripper(
    physics_client: int,
    initial_opening: float = 0.08,
) -> int:
    """Create a floating two-finger gripper with prismatic finger joints.
    
    Returns body_id of the gripper multibody.
    Link 0: Left finger (prismatic along +x)
    Link 1: Right finger (prismatic along -x)
    """
    palm_half = [0.05, 0.012, 0.012]
    finger_half = [0.005, 0.012, 0.03]

    # Collision and visual shapes
    palm_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=palm_half, physicsClientId=physics_client
    )
    palm_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=palm_half, rgbaColor=[0.2, 0.2, 0.2, 1.0], physicsClientId=physics_client
    )

    finger_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=finger_half, physicsClientId=physics_client
    )
    finger_vis_left = p.createVisualShape(
        p.GEOM_BOX, halfExtents=finger_half, rgbaColor=[0.8, 0.2, 0.2, 1.0], physicsClientId=physics_client
    )
    finger_vis_right = p.createVisualShape(
        p.GEOM_BOX, halfExtents=finger_half, rgbaColor=[0.2, 0.2, 0.8, 1.0], physicsClientId=physics_client
    )

    # Fingers are positioned at +/- half_opening along X, and shifted +Z by finger_half[2] + palm_half[2]
    # Palm is at base frame. Grasp centre will be at z = palm_half[2] + finger_half[2] = 0.042
    # In link parent frame (palm):
    joint_positions = [
        [-(initial_opening / 2.0), 0.0, palm_half[2] + finger_half[2]],
        [+(initial_opening / 2.0), 0.0, palm_half[2] + finger_half[2]],
    ]
    joint_orientations = [
        [0, 0, 0, 1],
        [0, 0, 0, 1],
    ]
    joint_types = [p.JOINT_PRISMATIC, p.JOINT_PRISMATIC]
    joint_axes = [
        [1, 0, 0],   # left finger moves along x
        [-1, 0, 0],  # right finger moves along -x
    ]

    gripper_id = p.createMultiBody(
        baseMass=0.0,  # Kinematically driven floating base
        baseCollisionShapeIndex=palm_col,
        baseVisualShapeIndex=palm_vis,
        basePosition=[0, 0, 1.0],
        baseOrientation=[0, 0, 0, 1],
        linkMasses=[0.1, 0.1],
        linkCollisionShapeIndices=[finger_col, finger_col],
        linkVisualShapeIndices=[finger_vis_left, finger_vis_right],
        linkPositions=joint_positions,
        linkOrientations=joint_orientations,
        linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
        linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
        linkParentIndices=[0, 0],
        linkJointTypes=joint_types,
        linkJointAxis=joint_axes,
        physicsClientId=physics_client,
    )

    # Set joint limits and friction
    for joint_idx in [0, 1]:
        p.changeDynamics(
            gripper_id,
            joint_idx,
            lateralFriction=1.0,
            spinningFriction=0.01,
            rollingFriction=0.01,
            physicsClientId=physics_client,
        )

    return gripper_id


class SimGraspExecutor:
    """Executes grasps in PyBullet according to Stage 1 specification."""

    def __init__(self, scene: ClutteredScene):
        self.scene = scene
        self.client = scene.client
        self.gripper_id: Optional[int] = None

    def _ensure_gripper(self, initial_opening: float = 0.08):
        if self.gripper_id is not None:
            try:
                p.removeBody(self.gripper_id, physicsClientId=self.client)
            except Exception:
                pass
            self.gripper_id = None
        self.gripper_id = create_floating_gripper(self.client, initial_opening=initial_opening)

    def restore(self):
        """Clean up gripper body and restore scene state."""
        if self.gripper_id is not None:
            try:
                p.removeBody(self.gripper_id, physicsClientId=self.client)
            except Exception:
                pass
            self.gripper_id = None
        self.scene.restore_state()

    def execute_grasp(
        self,
        R: np.ndarray,
        t: np.ndarray,
        w: float,
        restore: bool = True,
    ) -> Tuple[bool, dict]:
        """Execute grasp (R, t, w) in simulation.
        
        R: (3, 3) rotation matrix where col 0 is closing axis (+x), col 2 is approach (+z)
        t: (3,) grasp centre in world frame
        w: grasp width in meters
        
        Returns:
            (success, info_dict)
        """
        info = {
            "pre_collision": False,
            "max_penetration": 0.0,
            "lifted_dist": 0.0,
            "dist_to_gripper": 0.0,
            "target_obj_id": -1,
            "success": False,
        }

        # Grasp opening = width + 0.03
        opening = float(np.clip(w + 0.03, 0.02, 0.12))
        self._ensure_gripper(initial_opening=opening)

        # Convert R, t to base pose of palm
        # Palm centre is offset along -approach axis (-col 2) by finger_half[2] + palm_half[2] = 0.03 + 0.012 = 0.042
        palm_offset = 0.042
        approach_axis = R[:, 2]
        palm_pos = t - palm_offset * approach_axis

        # PyBullet quaternion from 3x3 rotation matrix
        # R columns: col 0 is X, col 1 is Y, col 2 is Z
        orn_quat = self._rot_matrix_to_quat(R)

        # Teleport gripper to pre-grasp pose
        p.resetBasePositionAndOrientation(
            self.gripper_id, palm_pos.tolist(), orn_quat, physicsClientId=self.client
        )
        p.resetJointState(self.gripper_id, 0, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.client)
        p.resetJointState(self.gripper_id, 1, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.client)

        # 1. Pre-closing penetration check: reject if penetration depth > 3 mm (-0.003 m)
        # Check closest points with all scene objects
        p.stepSimulation(physicsClientId=self.client)
        max_penetration = 0.0
        for obj in self.scene.objects:
            cps = p.getClosestPoints(
                self.gripper_id, obj.body_id, distance=0.05, physicsClientId=self.client
            )
            for cp in cps:
                # cp[8] is contactDistance (< 0 means penetration)
                dist = cp[8]
                if dist < -0.003:
                    penetration = -dist
                    if penetration > max_penetration:
                        max_penetration = penetration

        info["max_penetration"] = max_penetration
        if max_penetration > 0.003:
            info["pre_collision"] = True
            info["success"] = False
            if restore:
                self.restore()
            return False, info

        # Identify object nearest the grasp centre before grasping
        nearest_obj_id = -1
        min_obj_dist = float("inf")
        for obj in self.scene.objects:
            pos, _ = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=self.client)
            d = float(np.linalg.norm(np.array(pos) - t))
            if d < min_obj_dist:
                min_obj_dist = d
                nearest_obj_id = obj.body_id

        info["target_obj_id"] = nearest_obj_id
        if nearest_obj_id == -1:
            if restore:
                self.restore()
            return False, info

        init_obj_pos, _ = p.getBasePositionAndOrientation(nearest_obj_id, physicsClientId=self.client)
        init_obj_z = init_obj_pos[2]

        # 2. Close with position control (force 40, 120 steps)
        # Target position moves fingers inward by (opening - 0.005) / 2
        close_target = opening / 2.0
        p.setJointMotorControlArray(
            self.gripper_id,
            jointIndices=[0, 1],
            controlMode=p.POSITION_CONTROL,
            targetPositions=[close_target, close_target],
            forces=[40.0, 40.0],
            physicsClientId=self.client,
        )
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.client)

        # 3. Lift 0.15 m over 240 steps
        lift_height = 0.15
        lift_steps = 240
        start_palm_pos = np.array(palm_pos)
        dt = 1.0 / 240.0
        vz = lift_height / (lift_steps * dt)  # 0.15 m/s

        for step in range(lift_steps):
            frac = float(step + 1) / float(lift_steps)
            cur_palm_pos = start_palm_pos + np.array([0.0, 0.0, frac * lift_height])
            p.resetBasePositionAndOrientation(
                self.gripper_id, cur_palm_pos.tolist(), orn_quat, physicsClientId=self.client
            )
            p.resetBaseVelocity(
                self.gripper_id, linearVelocity=[0.0, 0.0, vz], angularVelocity=[0.0, 0.0, 0.0], physicsClientId=self.client
            )
            # Maintain grip force
            p.setJointMotorControlArray(
                self.gripper_id,
                jointIndices=[0, 1],
                controlMode=p.POSITION_CONTROL,
                targetPositions=[close_target, close_target],
                forces=[40.0, 40.0],
                physicsClientId=self.client,
            )
            p.stepSimulation(physicsClientId=self.client)

        p.resetBaseVelocity(
            self.gripper_id, linearVelocity=[0.0, 0.0, 0.0], angularVelocity=[0.0, 0.0, 0.0], physicsClientId=self.client
        )

        # 4. Success evaluation:
        # Object nearest grasp centre rises > 0.10 m and stays within 0.06 m of gripper centre
        final_obj_pos, _ = p.getBasePositionAndOrientation(nearest_obj_id, physicsClientId=self.client)
        final_gripper_center = cur_palm_pos + palm_offset * approach_axis

        lifted_z = final_obj_pos[2] - init_obj_z
        dist_to_gripper = float(np.linalg.norm(np.array(final_obj_pos) - final_gripper_center))

        info["lifted_dist"] = lifted_z
        info["dist_to_gripper"] = dist_to_gripper

        is_success = bool((lifted_z > 0.10) and (dist_to_gripper <= 0.06))
        info["success"] = is_success

        # 5. Restore saved state
        if restore:
            self.restore()

        return is_success, info

    def _rot_matrix_to_quat(self, R: np.ndarray) -> List[float]:
        """Convert 3x3 rotation matrix to PyBullet quaternion [x, y, z, w]."""
        # Ensure orthonormal
        U, _, Vt = np.linalg.svd(R)
        R_ortho = np.matmul(U, Vt)
        if np.linalg.det(R_ortho) < 0:
            U[:, 2] = -U[:, 2]
            R_ortho = np.matmul(U, Vt)

        tr = float(R_ortho[0, 0] + R_ortho[1, 1] + R_ortho[2, 2])
        if tr > 0.0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R_ortho[2, 1] - R_ortho[1, 2]) * s
            y = (R_ortho[0, 2] - R_ortho[2, 0]) * s
            z = (R_ortho[1, 0] - R_ortho[0, 1]) * s
        elif (R_ortho[0, 0] > R_ortho[1, 1]) and (R_ortho[0, 0] > R_ortho[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R_ortho[0, 0] - R_ortho[1, 1] - R_ortho[2, 2])
            w = (R_ortho[2, 1] - R_ortho[1, 2]) / s
            x = 0.25 * s
            y = (R_ortho[0, 1] + R_ortho[1, 0]) / s
            z = (R_ortho[0, 2] + R_ortho[2, 0]) / s
        elif R_ortho[1, 1] > R_ortho[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_ortho[1, 1] - R_ortho[0, 0] - R_ortho[2, 2])
            w = (R_ortho[0, 2] - R_ortho[2, 0]) / s
            x = (R_ortho[0, 1] + R_ortho[1, 0]) / s
            y = 0.25 * s
            z = (R_ortho[1, 2] + R_ortho[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R_ortho[2, 2] - R_ortho[0, 0] - R_ortho[1, 1])
            w = (R_ortho[1, 0] - R_ortho[0, 1]) / s
            x = (R_ortho[0, 2] + R_ortho[2, 0]) / s
            y = (R_ortho[1, 2] + R_ortho[2, 1]) / s
            z = 0.25 * s

        quat = [float(x), float(y), float(z), float(w)]
        norm = np.linalg.norm(quat)
        return [q / norm for q in quat]
