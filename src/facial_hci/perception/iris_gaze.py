"""Gaze estimation from MediaPipe Face Mesh landmarks.

Uses iris landmarks (468-477) relative to eye corners.
Returns normalized gaze in [-1, 1] range for x (horizontal) and y (vertical).
"""
from typing import Dict, List
import numpy as np

# MediaPipe 478-landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = (33, 133)   # outer, inner
RIGHT_EYE_CORNERS = (362, 263)
LEFT_EYE_TOP_BOT = (159, 145)
RIGHT_EYE_TOP_BOT = (386, 374)


def _center(landmarks, idxs: List[int]) -> np.ndarray:
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in idxs])
    return pts.mean(axis=0)


def estimate_gaze(landmarks) -> Dict[str, float]:
    if len(landmarks) < 478:
        return {"gaze_x": 0.0, "gaze_y": 0.0, "available": False}

    l_iris = _center(landmarks, LEFT_IRIS)
    r_iris = _center(landmarks, RIGHT_IRIS)
    l_out, l_in = np.array([landmarks[LEFT_EYE_CORNERS[0]].x, landmarks[LEFT_EYE_CORNERS[0]].y]), \
                  np.array([landmarks[LEFT_EYE_CORNERS[1]].x, landmarks[LEFT_EYE_CORNERS[1]].y])
    r_in, r_out = np.array([landmarks[RIGHT_EYE_CORNERS[0]].x, landmarks[RIGHT_EYE_CORNERS[0]].y]), \
                  np.array([landmarks[RIGHT_EYE_CORNERS[1]].x, landmarks[RIGHT_EYE_CORNERS[1]].y])
    l_top, l_bot = np.array([landmarks[LEFT_EYE_TOP_BOT[0]].x, landmarks[LEFT_EYE_TOP_BOT[0]].y]), \
                   np.array([landmarks[LEFT_EYE_TOP_BOT[1]].x, landmarks[LEFT_EYE_TOP_BOT[1]].y])

    # horizontal: iris position along eye corner axis, normalized
    def norm_x(iris, out, inn):
        axis = inn - out
        axis_len = np.linalg.norm(axis) + 1e-6
        proj = np.dot(iris - out, axis) / (axis_len ** 2)
        return 2 * proj - 1    # -1 (out) to +1 (in)

    gx = 0.5 * (norm_x(l_iris, l_out, l_in) + norm_x(r_iris, r_in, r_out))

    # vertical
    eye_h = np.linalg.norm(l_top - l_bot) + 1e-6
    gy = 2 * (l_iris[1] - l_top[1]) / eye_h - 1

    return {"gaze_x": float(np.clip(gx, -1, 1)),
            "gaze_y": float(np.clip(gy, -1, 1)),
            "available": True}


def gaze_aversion(gaze: Dict[str, float], thresh: float = 0.4) -> bool:
    if not gaze.get("available"):
        return False
    return abs(gaze["gaze_x"]) > thresh or abs(gaze["gaze_y"]) > thresh
