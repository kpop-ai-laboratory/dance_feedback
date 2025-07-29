import numpy as np
from numpy.linalg import LinAlgError
from .angle_utils import calc_interior_angles_2d, calc_signed_bend_angles_2d, angle_diff
from .procrustes_utils import procrustes_frame_dist, compute_procrustes_transform
from .trajectory_utils import extract_root_sequence, dtw_distance

def compute_frame_similarities(
    kp_ref: np.ndarray,
    kp_user: np.ndarray,
    angle_weight: float = 0.6
) -> dict:
    """
    Compute per-frame pose/move/final similarity between reference and user keypoints.
    Returns dict with 'pose', 'move', 'final', 'angle_diffs', 'proc_dists'.
    per-frame 유사도 계산.
    NaN 또는 SVD 실패 프레임은 바로 final=0으로 처리.
    """
    T = kp_ref.shape[0]
    # 0) 프레임 수가 다르면 짧은 쪽에 맞춰 슬라이스
    n_ref, n_usr = kp_ref.shape[0], kp_user.shape[0]
    T = min(n_ref, n_usr)
    kp_ref  = kp_ref[:T]
    kp_user = kp_user[:T]   

    M = calc_interior_angles_2d(kp_ref).shape[1]
    angle_diffs = np.zeros((T, M))
    proc_dists  = np.zeros(T)
    pose_scores = np.zeros(T)
    move_scores = np.zeros(T)

    roots_ref  = extract_root_sequence(kp_ref)
    roots_user = extract_root_sequence(kp_user)
    max_root   = np.linalg.norm(roots_ref - roots_user, axis=1).max() + 1e-6

    for t in range(T):
        # Procrustes align
        R = compute_procrustes_transform(kp_ref[t], kp_user[t])
        aligned = (kp_user[t] - kp_user[t].mean(axis=0)).dot(R)

        # Angle diffs
        ang_ref = calc_interior_angles_2d(kp_ref[t][None])[0]
        ang_usr = calc_interior_angles_2d(aligned[None])[0]
        bend_ref = calc_signed_bend_angles_2d(kp_ref[t][None])[0]
        bend_usr = calc_signed_bend_angles_2d(aligned[None])[0]
                # Angle diffs: use interior for most, bend for elbows/knees
        diff = np.abs(ang_ref - ang_usr)
        bend_diff = np.abs(bend_usr - bend_ref)
        # bend joints: 0,1=elbows; 4,5=knees
        bend_idxs = [0, 1, 4, 5]
        for i in bend_idxs:
            diff[i] = bend_diff[i]
        angle_diffs[t] = diff

        # Procrustes distance
        _, pd = procrustes_frame_dist(kp_ref[t], aligned)
        proc_dists[t] = pd

        # Scores
        angle_sim = 1.0 - diff.mean() / np.pi
        proc_sim  = 1.0 - pd
        pose_scores[t] = angle_weight * angle_sim + (1 - angle_weight) * proc_sim
        move_scores[t] = 1.0 - np.linalg.norm(roots_ref[t] - roots_user[t]) / max_root

    final_scores = 0.5 * pose_scores + 0.5 * move_scores

    return {
        "pose": pose_scores,
        "move": move_scores,
        "final": final_scores,
        "angle_diffs": angle_diffs,
        "proc_dists": proc_dists
    }

# def compute_frame_similarities(
#     kp_ref: np.ndarray,
#     kp_user: np.ndarray,
#     angle_weight: float = 0.6
# ) -> dict:
#     """
#     per-frame 유사도 계산.
#     NaN 또는 SVD 실패 프레임은 바로 final=0으로 처리합니다.
#     """
#     T = min(kp_ref.shape[0], kp_user.shape[0])
#     kp_ref  = kp_ref[:T]
#     kp_user = kp_user[:T]

    # 사전 할당
    # M = calc_interior_angles_2d(kp_ref).shape[1]
    # angle_diffs = np.zeros((T, M))
    # proc_dists  = np.zeros(T)
    # pose_scores = np.zeros(T)
    # move_scores = np.zeros(T)

    # roots_ref  = extract_root_sequence(kp_ref)
    # roots_user = extract_root_sequence(kp_user)
    # max_root   = np.linalg.norm(roots_ref - roots_user, axis=1).max() + 1e-6

    # for t in range(T):
    #     X = kp_ref[t]
    #     Y = kp_user[t]

    #     # — NaN 체크 —
    #     if np.isnan(X).any() or np.isnan(Y).any():
    #         # 데이터가 불완전한 프레임: 0점 처리
    #         pose_scores[t] = 0.0
    #         move_scores[t] = 0.0
    #         angle_diffs[t] = 0.0
    #         proc_dists[t]  = 0.0
    #         continue

    #     #  Procrustes align (내부에서 SVD 실패 시 identity 리턴)
    #     try:
    #         R = compute_procrustes_transform(X, Y)
    #     except LinAlgError:
    #         pose_scores[t] = 0.0
    #         move_scores[t] = 0.0
    #         continue

    #     Y_aligned = (Y - Y.mean(axis=0)).dot(R)

    #     # — Angle diff 계산 —
    #     ang_ref  = calc_interior_angles_2d(X[None])[0]
    #     ang_usr  = calc_interior_angles_2d(Y_aligned[None])[0]
    #     bend_ref = calc_signed_bend_angles_2d(X[None])[0]
    #     bend_usr = calc_signed_bend_angles_2d(Y_aligned[None])[0]
    #     diff = np.abs(ang_ref - ang_usr)
    #     for i in [0,1,4,5]:  # elbow/knee indices
    #         diff[i] = abs(bend_usr[i] - bend_ref[i])
    #     angle_diffs[t] = diff

    #     # — Procrustes distance —
    #     _, pd = procrustes_frame_dist(X, Y_aligned)
    #     proc_dists[t] = pd

    #     # — pose/move score —
    #     angle_sim = 1.0 - diff.mean() / np.pi
    #     proc_sim  = 1.0 - pd
    #     pose_scores[t] = angle_weight * angle_sim + (1 - angle_weight) * proc_sim
    #     move_scores[t] = 1.0 - np.linalg.norm(roots_ref[t] - roots_user[t]) / max_root

    # final_scores = 0.5 * pose_scores + 0.5 * move_scores

    # return {
    #     "pose": pose_scores,
    #     "move": move_scores,
    #     "final": final_scores,
    #     "angle_diffs": angle_diffs,
    #     "proc_dists": proc_dists
    # }


def aggregate_per_second(frame_scores: np.ndarray, fps: float) -> np.ndarray:
    """
    frame_scores: (T,) 길이의 프레임별 점수 배열
    fps: 초당 프레임 수 (예: 30)
    Returns:
      (num_seconds,) 길이의 배열로, 각 원소는
      frame_scores[i*fps : (i+1)*fps] 구간의 평균값
    """
    # T = len(frame_scores)  # 전체 동영상 길이를 초 단위로 내림
    # num = T // fps
    # return np.array([frame_scores[i*fps:(i+1)*fps].mean() for i in range(num)])

    # fps 가 float 로 들어와도 안전하게 int 로 변환
    fps_i = int(round(fps))
    T     = len(frame_scores)
    num   = T // fps_i
    return np.array([
        frame_scores[i*fps_i : (i+1)*fps_i].mean()
        for i in range(num)
    ])


def identify_misaligned_joints(
    angle_diffs: np.ndarray,
    proc_dists: np.ndarray,
    angle_thresh: float,
    proc_thresh: float
) -> tuple[list[int], dict[int, list[str]]]:
    """
    Identify frames and joints where diffs exceed thresholds.
    """
    bad_frames = []
    bad_joints = {}
    T, M = angle_diffs.shape
    for t in range(T):
        joints = []
        for i in range(M):
            if angle_diffs[t, i] > angle_thresh:
                joints.append(f"angle_joint_{i}")
        if proc_dists[t] > proc_thresh:
            joints.append("shape_misaligned")
        if joints:
            bad_frames.append(t)
            bad_joints[t] = joints
    return bad_frames, bad_joints
