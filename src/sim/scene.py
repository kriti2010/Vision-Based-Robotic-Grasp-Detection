"""Procedural cluttered scenes in PyBullet (headless DIRECT mode).

Generates tabletop scenes with 3 to 8 geometric objects (boxes, cylinders, spheres)
with random shapes, orientations, and mild stacking.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class ObjectGeometry:
    """Ground truth geometric description of a scene object."""
    body_id: int
    obj_type: str  # "box", "cylinder", "sphere"
    position: np.ndarray
    orientation: np.ndarray  # quaternion [x, y, z, w]
    dimensions: np.ndarray   # half-extents for box, [radius, height] for cylinder, [radius] for sphere
    color: Tuple[float, float, float, float]


class ProceduralScene:
    """Manages PyBullet procedural scene creation, physics simulation, and queries."""

    def __init__(self, seed: Optional[int] = None):
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        
        self.table_id = -1
        self.objects: List[ObjectGeometry] = []
        self.rng = np.random.RandomState(seed)

    def reset(self, num_objects: Optional[int] = None, bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> List[ObjectGeometry]:
        """Clear and create a new procedural scene on the table."""
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        
        # Create table plane
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client_id)
        self.table_id = p.createMultiBody(0, plane_shape, physicsClientId=self.client_id)
        
        # Workspace bounds: x in [-0.18, 0.18], y in [-0.18, 0.18]
        if bounds is None:
            min_xy, max_xy = np.array([-0.16, -0.16]), np.array([0.16, 0.16])
        else:
            min_xy, max_xy = bounds[0][:2], bounds[1][:2]

        if num_objects is None:
            num_objects = self.rng.randint(3, 8)
            
        self.objects = []
        obj_types = ["box", "cylinder", "sphere"]
        
        for _ in range(num_objects):
            obj_type = self.rng.choice(obj_types)
            pos_xy = self.rng.uniform(min_xy, max_xy)
            pos_z = self.rng.uniform(0.04, 0.12)  # drop height
            
            # Random yaw/roll/pitch
            euler = self.rng.uniform(-np.pi, np.pi, size=3)
            orn = p.getQuaternionFromEuler(euler)
            
            color = (
                float(self.rng.uniform(0.2, 0.9)),
                float(self.rng.uniform(0.2, 0.9)),
                float(self.rng.uniform(0.2, 0.9)),
                1.0
            )
            
            if obj_type == "box":
                half_extents = self.rng.uniform([0.02, 0.02, 0.02], [0.045, 0.045, 0.045])
                col_id = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id
                )
                vis_id = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client_id
                )
                dims = half_extents
            elif obj_type == "cylinder":
                radius = float(self.rng.uniform(0.02, 0.035))
                height = float(self.rng.uniform(0.04, 0.08))
                col_id = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=self.client_id
                )
                vis_id = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color, physicsClientId=self.client_id
                )
                dims = np.array([radius, height])
            else:  # sphere
                radius = float(self.rng.uniform(0.025, 0.045))
                col_id = p.createCollisionShape(
                    p.GEOM_SPHERE, radius=radius, physicsClientId=self.client_id
                )
                vis_id = p.createVisualShape(
                    p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=self.client_id
                )
                dims = np.array([radius])
                
            mass = 0.2
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[pos_xy[0], pos_xy[1], pos_z],
                baseOrientation=orn,
                physicsClientId=self.client_id
            )
            
            self.objects.append(
                ObjectGeometry(
                    body_id=body_id,
                    obj_type=obj_type,
                    position=np.array([pos_xy[0], pos_xy[1], pos_z]),
                    orientation=np.array(orn),
                    dimensions=dims,
                    color=color
                )
            )

        # Step simulation to let objects settle on the table
        for _ in range(80):
            p.stepSimulation(physicsClientId=self.client_id)

        # Update settled positions and orientations
        settled_objects = []
        for obj in self.objects:
            pos, orn = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=self.client_id)
            pos = np.array(pos)
            # Filter objects that fell off the table
            if pos[2] > 0.005:
                settled_objects.append(
                    ObjectGeometry(
                        body_id=obj.body_id,
                        obj_type=obj.obj_type,
                        position=pos,
                        orientation=np.array(orn),
                        dimensions=obj.dimensions,
                        color=obj.color
                    )
                )
            else:
                p.removeBody(obj.body_id, physicsClientId=self.client_id)
                
        self.objects = settled_objects
        return self.objects

    def get_object_geometries(self) -> List[ObjectGeometry]:
        """Return list of settled object geometries in world frame."""
        return self.objects

    def close(self):
        """Disconnect PyBullet client."""
        if p.isConnected(physicsClientId=self.client_id):
            p.disconnect(physicsClientId=self.client_id)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
