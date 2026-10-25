"""Tabletop cluttered scene generator in PyBullet DIRECT mode.

Simulates 8 random geometric objects (boxes and cylinders) dropped in a pile,
settled for 480 simulation steps, with state saving and restoration.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class ObjectInfo:
    body_id: int
    obj_type: str  # "box" or "cylinder"
    dimensions: np.ndarray  # half-extents (3,) for box, [radius, height] (2,) for cylinder
    color: Tuple[float, float, float, float]
    initial_z: float = 0.0


class ClutteredScene:
    """Manages PyBullet procedural cluttered scene generation and state restoration."""

    def __init__(self, physics_client: Optional[int] = None):
        if physics_client is None:
            self.client = p.connect(p.DIRECT)
        else:
            self.client = physics_client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.plane_id = -1
        self.objects: List[ObjectInfo] = []
        self.saved_state_id = -1

    def create_scene(self, seed: int) -> int:
        """Create cluttered scene with seed.
        
        Scene:
        - plane + 8 random boxes (half-extents 0.012-0.03) and cylinders
          (radius 0.012-0.028, height 0.04-0.09), mass 0.1, friction 1.0.
        - Dropped in a pile: xy within +/-0.1, z = 0.05 + 0.07 * i, random orientation.
        - Settle 480 steps, save state.
        
        Returns:
            saved_state_id
        """
        rng = np.random.RandomState(seed)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0, physicsClientId=self.client)

        # Plane at z = 0
        plane_col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client)
        self.plane_id = p.createMultiBody(0, plane_col, physicsClientId=self.client)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=self.client)

        self.objects = []
        num_objects = 8
        for i in range(num_objects):
            obj_type = rng.choice(["box", "cylinder"])
            color = (
                float(rng.uniform(0.1, 0.9)),
                float(rng.uniform(0.1, 0.9)),
                float(rng.uniform(0.1, 0.9)),
                1.0,
            )
            mass = 0.1
            friction = 1.0

            if obj_type == "box":
                half_extents = rng.uniform(0.012, 0.030, size=3)
                col_id = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client
                )
                vis_id = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=self.client
                )
                dims = half_extents
            else:  # cylinder
                radius = float(rng.uniform(0.012, 0.028))
                height = float(rng.uniform(0.040, 0.090))
                col_id = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=self.client
                )
                vis_id = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color, physicsClientId=self.client
                )
                dims = np.array([radius, height])

            # Position in pile: xy within +/-0.1, z = 0.05 + 0.07 * i
            pos_x = float(rng.uniform(-0.1, 0.1))
            pos_y = float(rng.uniform(-0.1, 0.1))
            pos_z = 0.05 + 0.07 * i

            # Random orientation
            euler = rng.uniform(-np.pi, np.pi, size=3)
            orn = p.getQuaternionFromEuler(euler)

            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[pos_x, pos_y, pos_z],
                baseOrientation=orn,
                physicsClientId=self.client,
            )
            p.changeDynamics(body_id, -1, lateralFriction=friction, physicsClientId=self.client)

            self.objects.append(
                ObjectInfo(
                    body_id=body_id,
                    obj_type=obj_type,
                    dimensions=dims,
                    color=color,
                    initial_z=pos_z,
                )
            )

        # Settle for 480 steps
        for _ in range(480):
            p.stepSimulation(physicsClientId=self.client)

        # Update settled initial z
        for obj in self.objects:
            pos, _ = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=self.client)
            obj.initial_z = pos[2]

        # Save settled state
        if self.saved_state_id >= 0:
            p.removeState(self.saved_state_id, physicsClientId=self.client)
        self.saved_state_id = p.saveState(physicsClientId=self.client)
        return self.saved_state_id

    def restore_state(self):
        """Restore the simulation to the settled post-drop state."""
        if self.saved_state_id >= 0:
            p.restoreState(self.saved_state_id, physicsClientId=self.client)

    def get_object_positions(self) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Return list of (body_id, position, orientation) for all objects."""
        res = []
        for obj in self.objects:
            pos, orn = p.getBasePositionAndOrientation(obj.body_id, physicsClientId=self.client)
            res.append((obj.body_id, np.array(pos), np.array(orn)))
        return res

    def close(self):
        if p.isConnected(physicsClientId=self.client):
            p.disconnect(physicsClientId=self.client)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
