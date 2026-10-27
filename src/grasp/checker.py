"""Analytic grasp success checker with friction cone antipodal contact,
collision testing, width limits, and optional PyBullet physics verification.
"""

from typing import List, Tuple, Optional
import numpy as np
import pybullet as p
from src.grasp.representation import Grasp, GripperParams
from src.sim.scene import ObjectGeometry


class AnalyticGraspChecker:
    """Fast analytic grasp checker based on ground truth object geometry."""

    def __init__(self, gripper_params: Optional[GripperParams] = None, table_z: float = 0.0):
        self.params = gripper_params or GripperParams()
        self.table_z = table_z

    def check_grasp(
        self,
        grasp: Grasp,
        objects: List[ObjectGeometry]
    ) -> Tuple[bool, str]:
        """Check if a grasp succeeds analytically.
        
        Returns:
            (is_success, reason)
        """
        # 1. Width limits check
        if grasp.width < self.params.min_width or grasp.width > self.params.max_width:
            return False, "width_out_of_limits"

        # 2. Table clearance check
        # Check palm and finger tip heights
        p_left, p_right = grasp.get_finger_pad_centers()
        palm_center = grasp.p - (self.params.finger_length) * grasp.approach_axis
        
        min_z = min(grasp.p[2], p_left[2], p_right[2], palm_center[2])
        if min_z < self.table_z + 0.005:
            return False, "table_collision"

        # 3. Collision test: Gripper body (palm + finger solids) against all objects
        if self._check_gripper_collision(grasp, objects):
            return False, "gripper_body_collision"

        # 4. Antipodal contact test:
        # Check if closing jaws pinch an object with opposing contact normals inside friction cone
        has_contact, contact_reason = self._check_antipodal_contact(grasp, objects)
        if not has_contact:
            return False, contact_reason

        return True, "success"

    def _check_gripper_collision(self, grasp: Grasp, objects: List[ObjectGeometry]) -> bool:
        """Check if solid parts of the gripper collide with any scene object."""
        rx = grasp.closing_axis
        ry = grasp.binormal_axis
        rz = grasp.approach_axis
        
        half_w = grasp.width / 2.0
        fw = self.params.finger_width
        fl = self.params.finger_length
        fd = self.params.finger_depth

        # Centers of left and right finger solids (displaced outward from pad center by fw / 2)
        c_left_solid = grasp.p + (half_w + fw / 2.0) * rx - (fl * 0.25) * rz
        c_right_solid = grasp.p - (half_w + fw / 2.0) * rx - (fl * 0.25) * rz
        
        # Palm base center behind fingers
        c_palm = grasp.p - (fl * 0.75 + self.params.base_height * 0.5) * rz

        gripper_solids = [
            (c_left_solid, np.array([fw, fd, fl])),
            (c_right_solid, np.array([fw, fd, fl])),
            (c_palm, np.array([self.params.base_width, fd, self.params.base_height]))
        ]

        for obj in objects:
            for g_center, g_ext in gripper_solids:
                if self._box_object_collision(g_center, g_ext, grasp.R, obj):
                    return True
        return False

    def _sat_box_box(
        self,
        cA: np.ndarray, extA: np.ndarray, RA: np.ndarray,
        cB: np.ndarray, extB: np.ndarray, RB: np.ndarray
    ) -> bool:
        """Separating Axis Theorem (SAT) for two 3D oriented bounding boxes.
        Returns True if colliding, False if a separating axis exists.
        """
        halfA = extA / 2.0
        halfB = extB / 2.0

        # Translation vector from A to B in world
        D = cB - cA
        # In frame A
        T_A = np.matmul(D, RA)  # (3,)
        # Rotation matrix of B in frame A: RA^T * RB
        R_AB = np.matmul(RA.T, RB)
        Q = np.abs(R_AB) + 1e-7

        # 3 axes of Box A
        for i in range(3):
            ra = halfA[i]
            rb = np.sum(Q[i, :] * halfB)
            if abs(T_A[i]) > (ra + rb):
                return False  # Separating axis found

        # 3 axes of Box B
        T_B = np.matmul(D, RB)
        for j in range(3):
            ra = np.sum(Q[:, j] * halfA)
            rb = halfB[j]
            if abs(T_B[j]) > (ra + rb):
                return False  # Separating axis found

        return True  # Overlapping on all primary axes

    def _box_object_collision(
        self,
        box_center: np.ndarray,
        box_extents: np.ndarray,
        box_rot: np.ndarray,
        obj: ObjectGeometry
    ) -> bool:
        """Collision check between a gripper solid box and an object."""
        rel_pos = obj.position - box_center
        local_pos = np.matmul(rel_pos, box_rot)
        half_ext = box_extents / 2.0

        if obj.obj_type == "sphere":
            radius = float(obj.dimensions[0])
            closest = np.clip(local_pos, -half_ext, half_ext)
            dist_sq = np.sum((local_pos - closest) ** 2)
            return dist_sq < (radius ** 2)
            
        elif obj.obj_type == "cylinder":
            radius = float(obj.dimensions[0])
            half_h = float(obj.dimensions[1]) / 2.0
            obj_rot = np.array(p.getMatrixFromQuaternion(obj.orientation.tolist())).reshape(3, 3)
            cyl_box_ext = np.array([radius * 2, radius * 2, half_h * 2])
            return self._sat_box_box(box_center, box_extents, box_rot, obj.position, cyl_box_ext, obj_rot)
            
        else:  # box
            obj_half = np.asarray(obj.dimensions)
            obj_ext = obj_half * 2.0
            obj_rot = np.array(p.getMatrixFromQuaternion(obj.orientation.tolist())).reshape(3, 3)
            return self._sat_box_box(box_center, box_extents, box_rot, obj.position, obj_ext, obj_rot)

    def _check_antipodal_contact(
        self,
        grasp: Grasp,
        objects: List[ObjectGeometry]
    ) -> Tuple[bool, str]:
        """Test antipodal contact between jaws:
        Left and right finger pads must pinch an object with opposing contact normals inside friction cone.
        """
        rx = grasp.closing_axis
        p_left, p_right = grasp.get_finger_pad_centers()

        # Find which object is in the closing volume
        contacted_obj = None
        for obj in objects:
            # Check if object center or bounding shape is between pads
            rel = obj.position - grasp.p
            local_rel = np.matmul(rel, grasp.R)
            # Along closing axis (x): must be between [-w/2, +w/2]
            # Along binormal (y): within pad depth
            # Along approach (z): within finger length
            half_w = grasp.width / 2.0
            if (abs(local_rel[0]) <= (half_w + 0.02) and
                abs(local_rel[1]) <= 0.05 and
                abs(local_rel[2]) <= 0.06):
                contacted_obj = obj
                break

        if contacted_obj is None:
            return False, "no_object_in_jaws"

        # Compute surface normal at contact points for the contacted object
        n_left, valid_l = self._get_surface_normal(contacted_obj, p_left, inward_dir=-rx)
        n_right, valid_r = self._get_surface_normal(contacted_obj, p_right, inward_dir=rx)

        if not (valid_l and valid_r):
            return False, "incomplete_contact"

        # Friction cone check:
        # Left finger pushes along -rx: cos_l = (-rx) . (-n_left) = rx . n_left
        # Right finger pushes along +rx: cos_r = (rx) . (-n_right) = -rx . n_right
        cos_l = np.dot(rx, n_left)
        cos_r = -np.dot(rx, n_right)

        thresh = self.params.antipodal_cos_thresh
        if cos_l < thresh or cos_r < thresh:
            return False, "friction_cone_violation"

        return True, "valid_antipodal"

    def _get_surface_normal(
        self,
        obj: ObjectGeometry,
        probe_point: np.ndarray,
        inward_dir: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Compute outward surface normal of object near probe_point."""
        rel = probe_point - obj.position
        
        if obj.obj_type == "sphere":
            dist = np.linalg.norm(rel)
            if dist < 1e-6:
                return inward_dir, True
            return rel / dist, True

        elif obj.obj_type == "box":
            # Rotate into box local frame
            # Object orientation is quaternion [x, y, z, w]
            rot_matrix = np.array(p.getMatrixFromQuaternion(obj.orientation.tolist())).reshape(3, 3)
            local_rel = np.matmul(rel, rot_matrix)
            half = obj.dimensions
            
            # Find face with smallest relative distance
            dist_faces = np.abs(local_rel) - half
            dominant_axis = np.argmax(dist_faces)
            normal_local = np.zeros(3)
            normal_local[dominant_axis] = np.sign(local_rel[dominant_axis])
            normal_world = np.matmul(rot_matrix, normal_local)
            return normal_world, True

        else:  # cylinder
            # Approximate along radial direction in XY plane of cylinder
            rot_matrix = np.array(p.getMatrixFromQuaternion(obj.orientation.tolist())).reshape(3, 3)
            local_rel = np.matmul(rel, rot_matrix)
            # Cylinder along local Z axis: normal in local XY
            r_xy = np.linalg.norm(local_rel[:2])
            if r_xy > 1e-4:
                norm_local = np.array([local_rel[0] / r_xy, local_rel[1] / r_xy, 0.0])
            else:
                norm_local = np.array([0.0, 0.0, np.sign(local_rel[2])])
            norm_world = np.matmul(rot_matrix, norm_local)
            return norm_world, True


def verify_grasp_physics(
    grasp: Grasp,
    client_id: int,
    gripper_params: Optional[GripperParams] = None,
    lift_height: float = 0.08
) -> bool:
    """Dynamic physics verification in PyBullet:
    Spawns a temporary two-jaw gripper at the grasp pose, closes the fingers with force,
    lifts along +Z, and checks if an object remains lifted off the table.
    """
    params = gripper_params or GripperParams()
    
    # Create simple two-finger gripper multibody
    fw = params.finger_width
    fl = params.finger_length
    fd = params.finger_depth
    
    # Palm body
    col_palm = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[params.base_width / 2.0, fd / 2.0, params.base_height / 2.0],
        physicsClientId=client_id
    )
    col_left = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[fw / 2.0, fd / 2.0, fl / 2.0],
        physicsClientId=client_id
    )
    col_right = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[fw / 2.0, fd / 2.0, fl / 2.0],
        physicsClientId=client_id
    )

    half_w = grasp.width / 2.0
    rx = grasp.closing_axis
    rz = grasp.approach_axis
    
    pos_palm = grasp.p - (fl + params.base_height / 2.0) * rz
    pos_left = grasp.p + (half_w + fw / 2.0) * rx - (fl / 2.0) * rz
    pos_right = grasp.p - (half_w + fw / 2.0) * rx - (fl / 2.0) * rz

    orn = p.getQuaternionFromEuler([0, 0, 0])
    
    b_palm = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_palm, basePosition=pos_palm.tolist(), physicsClientId=client_id)
    b_left = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=col_left, basePosition=pos_left.tolist(), physicsClientId=client_id)
    b_right = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=col_right, basePosition=pos_right.tolist(), physicsClientId=client_id)

    # Record initial max object z
    num_bodies = p.getNumBodies(physicsClientId=client_id)
    initial_z = []
    for bid in range(num_bodies):
        if bid not in (b_palm, b_left, b_right, 0):  # exclude table and gripper
            pz = p.getBasePositionAndOrientation(bid, physicsClientId=client_id)[0][2]
            initial_z.append((bid, pz))

    # Close jaws by moving left and right fingers inward
    close_steps = 15
    for s in range(close_steps):
        delta = (s + 1) / close_steps * 0.02
        p.resetBasePositionAndOrientation(b_left, (pos_left - delta * rx).tolist(), orn, physicsClientId=client_id)
        p.resetBasePositionAndOrientation(b_right, (pos_right + delta * rx).tolist(), orn, physicsClientId=client_id)
        p.stepSimulation(physicsClientId=client_id)

    # Lift
    lift_steps = 20
    for s in range(lift_steps):
        dz = (s + 1) / lift_steps * lift_height
        curr_l = pos_left - 0.02 * rx + np.array([0, 0, dz])
        curr_r = pos_right + 0.02 * rx + np.array([0, 0, dz])
        p.resetBasePositionAndOrientation(b_left, curr_l.tolist(), orn, physicsClientId=client_id)
        p.resetBasePositionAndOrientation(b_right, curr_r.tolist(), orn, physicsClientId=client_id)
        p.stepSimulation(physicsClientId=client_id)

    # Check if any object was successfully lifted by > 0.03m
    success = False
    for bid, init_z in initial_z:
        curr_z = p.getBasePositionAndOrientation(bid, physicsClientId=client_id)[0][2]
        if (curr_z - init_z) > 0.025:
            success = True
            break

    # Clean up gripper bodies
    p.removeBody(b_palm, physicsClientId=client_id)
    p.removeBody(b_left, physicsClientId=client_id)
    p.removeBody(b_right, physicsClientId=client_id)

    return success
