import numpy as np
from constants import get_angle_indices

ANGLE_IDX = get_angle_indices()

def calc_interior_angles_2d(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute interior angles for each frame and each joint triplet using 2D vectors (x, y only),
    via the Law of Cosines for greater numerical stability.
    keypoints: (T, J, 3)
    Returns angles: (T, M) in radians [0, π]
    """
    T, J, C = keypoints.shape
    M = len(ANGLE_IDX)
    angles = np.zeros((T, M), dtype=np.float32)
    for t in range(T):
        for i, (a, b, c) in enumerate(ANGLE_IDX):
            p1 = keypoints[t, a, :2]
            p2 = keypoints[t, b, :2]
            p3 = keypoints[t, c, :2]
            # 세 점 사이의 거리
            d1 = np.linalg.norm(p1 - p2)
            d2 = np.linalg.norm(p3 - p2)
            d3 = np.linalg.norm(p1 - p3)
            if d1 < 1e-6 or d2 < 1e-6:
                angles[t, i] = 0.0
            else:
                # Law of Cosines: cosθ = (d1²  d2² - d3²) / (2·d1·d2)
                cosθ = (d1*d1 + d2*d2 - d3*d3) / (2 * d1 * d2)
                angles[t, i] = np.arccos(np.clip(cosθ, -1.0, 1.0))
    return angles

def calc_signed_bend_angles_2d(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute signed bend angles = π - interior_angle using 2D vectors, sign from 2D cross.
    Returns bends: (T, M) in radians [-π, π]
    """
    T, J, C = keypoints.shape
    M = len(ANGLE_IDX)
    bends = np.zeros((T, M), dtype=np.float32)
    for t in range(T):
        for i, (a, b, c) in enumerate(ANGLE_IDX):
            p1 = keypoints[t, a, :2]
            p2 = keypoints[t, b, :2]
            p3 = keypoints[t, c, :2]
            v1 = p1 - p2
            v2 = p3 - p2
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                bends[t, i] = 0.0
            else:
                cosang = np.dot(v1, v2) / (norm1 * norm2)
                theta = np.arccos(np.clip(cosang, -1.0, 1.0))
                bend = np.pi - theta
                # sign from 2D cross-product z component
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                sign = np.sign(cross_z)
                bends[t, i] = sign * bend
    return bends

def angle_diff(a: float, b: float) -> float:
    """
    Minimal difference between two angles: result in [-π, π]
    """
    return (a - b + np.pi) % (2 * np.pi) - np.pi

