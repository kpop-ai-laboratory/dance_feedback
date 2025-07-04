import numpy as np
from typing import Tuple, List
from constants import JOINT_NAMES

def extract_root_sequence(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract mid-hip root trajectory from keypoints (T, J, 3).
    """
    left_hip = JOINT_NAMES.index('left_hip')
    right_hip = JOINT_NAMES.index('right_hip')
    return (keypoints[:, left_hip] + keypoints[:, right_hip]) * 0.5


def dtw_distance(ts1: np.ndarray, ts2: np.ndarray) -> tuple[float, tuple[list[int], list[int]]]:
    """
    Compute DTW distance and alignment path between ts1 and ts2.
    ts1, ts2: (T1, D), (T2, D)
    Returns: (distance, (path1, path2))
    """
    T1, T2 = ts1.shape[0], ts2.shape[0]
    dp = np.full((T1+1, T2+1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, T1+1):
        for j in range(1, T2+1):
            cost = np.linalg.norm(ts1[i-1] - ts2[j-1])
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    i, j = T1, T2
    path1, path2 = [], []
    while i > 0 and j > 0:
        path1.append(i-1)
        path2.append(j-1)
        steps = [(dp[i-1,j-1], i-1, j-1), (dp[i-1,j], i-1, j), (dp[i,j-1], i, j-1)]
        _, i, j = min(steps, key=lambda x: x[0])
    return dp[T1, T2], (path1[::-1], path2[::-1])
