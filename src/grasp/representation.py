"""6-DoF parallel-jaw grasp representation and gripper geometry."""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from src.grasp.so3 import project_so3_numpy


@dataclass
class GripperParams:
    """Dimensions of the parallel-jaw gripper."""
    finger_length: float = 0.06      # length along approach direction (z)
    finger_width: float = 0.012      # finger thickness along closing direction (x)
    finger_depth: float = 0.015      # finger dimension along binormal (y)
    base_height: float = 0.035       # palm/base thickness along approach direction (z)
    base_width: float = 0.09         # palm total width along closing direction (x)
    max_width: float = 0.085         # max jaw opening width
    min_width: float = 0.010         # min jaw opening width
    friction_coef: float = 0.5       # friction coefficient mu
    antipodal_cos_thresh: float = 0.70  # cos(theta) threshold for friction cone


@dataclass
class Grasp:
    """A 6-DoF parallel-jaw grasp.
    
    Attributes:
        R: 3x3 rotation matrix in SO(3).
           Col 0 (x): closing direction (finger motion axis)
           Col 1 (y): binormal axis
           Col 2 (z): approach direction (pointing into the scene toward target)
        p: 3D grasp center position (between finger pads) in world coordinates.
        width: Jaw opening width w (meters).
        score: Evaluator or predicted confidence score (float).
    """
    R: np.ndarray
    p: np.ndarray
    width: float = 0.08
    score: float = 0.0

    def __post_init__(self):
        self.R = project_so3_numpy(np.asarray(self.R, dtype=np.float64).reshape(3, 3))
        self.p = np.asarray(self.p, dtype=np.float64).reshape(3)
        self.width = float(np.clip(self.width, 0.005, 0.12))
        self.score = float(self.score)

    @property
    def closing_axis(self) -> np.ndarray:
        """Unit vector along finger closing direction (local x-axis in world frame)."""
        return self.R[:, 0]

    @property
    def binormal_axis(self) -> np.ndarray:
        """Unit vector along binormal axis (local y-axis in world frame)."""
        return self.R[:, 1]

    @property
    def approach_axis(self) -> np.ndarray:
        """Unit vector along grasp approach axis (local z-axis in world frame)."""
        return self.R[:, 2]

    def get_finger_pad_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 3D world positions of the left and right finger contact centers."""
        offset = (self.width / 2.0) * self.closing_axis
        p_left = self.p + offset
        p_right = self.p - offset
        return p_left, p_right

    def get_closing_volume_bounds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (center, extents, R) of the closing region between gripper jaws.
        
        Extents are (dx, dy, dz) in local gripper frame.
        """
        center = self.p.copy()
        extents = np.array([self.width, 0.02, 0.04])
        return center, extents, self.R

    def to_array(self) -> np.ndarray:
        """Serialize grasp to 13D array: [R.flatten (9), p (3), width (1)]."""
        return np.concatenate([self.R.flatten(), self.p, [self.width]])

    @classmethod
    def from_array(cls, arr: np.ndarray, score: float = 0.0) -> "Grasp":
        arr = np.asarray(arr, dtype=np.float64)
        R = arr[:9].reshape(3, 3)
        p = arr[9:12]
        width = float(arr[12]) if len(arr) > 12 else 0.08
        return cls(R=R, p=p, width=width, score=score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "R": self.R.tolist(),
            "p": self.p.tolist(),
            "width": self.width,
            "score": self.score
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Grasp":
        return cls(
            R=np.array(d["R"]),
            p=np.array(d["p"]),
            width=d.get("width", 0.08),
            score=d.get("score", 0.0)
        )
