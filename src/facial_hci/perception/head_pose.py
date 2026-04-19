"""Decompose MediaPipe's 4x4 transformation matrix -> pitch/yaw/roll (deg)."""
from typing import Dict
import numpy as np


def head_pose_from_matrix(matrix) -> Dict[str, float]:
    R = np.array(matrix).reshape(4, 4)[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = 0.0
    return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


def head_stable(pose: Dict[str, float], yaw_tol=15.0, pitch_tol=15.0) -> bool:
    return abs(pose["yaw"]) < yaw_tol and abs(pose["pitch"]) < pitch_tol
