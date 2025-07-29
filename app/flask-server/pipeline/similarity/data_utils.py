import json
import numpy as np
from typing import Tuple
from .constants import JOINT_NAMES

def load_mediapipe_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a MediaPipe JSON file into keypoints and visibility arrays.
    Returns:
      kp: float32 array of shape (T, J, 3)
      vis: float32 array of shape (T, J)
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        frames = list(raw.values())
    elif isinstance(raw, list):
        frames = raw
    else:
        raise ValueError("Unsupported JSON structure")
    T = len(frames)
    J = 33
    kp = np.zeros((T, J, 3), dtype=np.float32)
    vis = np.zeros((T, J), dtype=np.float32)
    for t, frame in enumerate(frames):
        for j in range(J):
            x = frame.get(f'x{j}', 0.0)
            y = frame.get(f'y{j}', 0.0)
            z = frame.get(f'z{j}', 0.0)
            v = frame.get(f'v{j}', 0.0)
            if isinstance(x, list): x = x[0] if x else 0.0
            if isinstance(y, list): y = y[0] if y else 0.0
            if isinstance(z, list): z = z[0] if z else 0.0
            if isinstance(v, list): v = v[0] if v else 0.0
            kp[t, j] = [float(x), float(y), float(z)]
            vis[t, j] = float(v)
    return kp, vis


def interpolate_missing(kp: np.ndarray, vis: np.ndarray) -> np.ndarray:
    T, J, C = kp.shape
    kp_interp = kp.copy()
    for j in range(J):
        valid = vis[:, j] > 0
        if valid.sum() < 2:
            continue
        times = np.arange(T)
        for c in range(C):
            kp_interp[:, j, c] = np.interp(times, times[valid], kp[valid, j, c])
    return kp_interp

# def interpolate_missing_with_gap_limit(
#     kp: np.ndarray,
#     vis: np.ndarray,
#     max_gap: int = 30
# ) -> np.ndarray:
#     """
#     kp: (T, J, C) keypoints array  
#     vis: (T, J) visibility mask (1=valid, 0=missing)  
#     max_gap: 연속 결손(frames) 이 max_gap 이상이면 보간하지 않음  
#     """
#     T, J, C = kp.shape
#     kp_interp = kp.copy()
#     times = np.arange(T)

#     for j in range(J):
#         valid = np.where(vis[:, j] > 0)[0]
#         if valid.size < 2:
#             # 너무 적은 valid → 보간하지 않음
#             continue

#         # 전체 구간 선형 보간
#         for c in range(C):
#             yi = np.interp(times, valid, kp[valid, j, c])

#             # 이제 gap이 큰 구간은 NaN으로 복원
#             # valid 에서 연속하지 않은 인접 인덱스를 찾아
#             prev = valid[0]
#             for idx in valid[1:]:
#                 gap = idx - prev
#                 if gap > max_gap:
#                     # prev ~ idx 사이는 보간하지 말고 NaN
#                     yi[prev:idx] = np.nan
#                 prev = idx

#             # (선택) 구간의 끝 부분—첫 valid 이전, 마지막 valid 이후도 NaN 처리
#             if valid[0] > 0:
#                 yi[:valid[0]] = np.nan
#             if valid[-1] < T-1:
#                 yi[valid[-1]+1:] = np.nan

#             kp_interp[:, j, c] = yi

#     return kp_interp


def smooth_keypoints(kp: np.ndarray, window: int = 5) -> np.ndarray:
    T, J, C = kp.shape
    kp_smooth = np.zeros_like(kp)
    for j in range(J):
        for c in range(C):
            series = kp[:, j, c]
            kernel = np.ones(window) / window
            kp_smooth[:, j, c] = np.convolve(series, kernel, mode='same')
    return kp_smooth

def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    left_hip = JOINT_NAMES.index('left_hip')
    right_hip = JOINT_NAMES.index('right_hip')
    left_shoulder = JOINT_NAMES.index('left_shoulder')
    right_shoulder = JOINT_NAMES.index('right_shoulder')
    T, J, C = kp.shape
    kp_norm = np.zeros_like(kp)
    for t in range(T):
        root = (kp[t, left_hip] + kp[t, right_hip]) * 0.5
        shoulder = (kp[t, left_shoulder] + kp[t, right_shoulder]) * 0.5
        height = np.linalg.norm(shoulder - root) + 1e-6
        kp_norm[t] = (kp[t] - root) / height
    return kp_norm
