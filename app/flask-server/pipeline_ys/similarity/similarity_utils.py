import numpy as np
from pipeline_ys.similarity.angle_utils import calc_interior_angles_2d, calc_signed_bend_angles_2d, angle_diff
from pipeline_ys.similarity.procrustes_utils import procrustes_frame_dist, compute_procrustes_transform
from pipeline_ys.similarity.trajectory_utils import extract_root_sequence, dtw_distance

def compute_bodypart_similarity(kp_ref: np.ndarray, kp_user: np.ndarray, joint_indices: list[int]) -> np.ndarray:
    """
    관절 인덱스 리스트를 기반으로 reference/user의 거리 차이를 계산하고
    0~1 사이 유사도로 정규화하여 반환 (T,)
    """
    T = kp_ref.shape[0]
    diffs = np.zeros((T, len(joint_indices)))

    for i, j in enumerate(joint_indices):
        ref_joint = kp_ref[:, j, :2]  # (T, 2)
        usr_joint = kp_user[:, j, :2]
        diffs[:, i] = np.linalg.norm(ref_joint - usr_joint, axis=1)

    max_val = np.max(diffs) + 1e-6
    normed = 1.0 - (diffs / max_val)
    return normed.mean(axis=1)  # (T,)

def compute_frame_similarities(
    kp_ref: np.ndarray,
    kp_user: np.ndarray,
    angle_weight: float = 0.6
) -> dict:
    """
    Compute per-frame pose/move/final similarity between reference and user keypoints.
    Returns dict with 'pose', 'move', 'final', 'angle_diffs', 'proc_dists', 'hand_sim', 'foot_sim'.
    """
    T = kp_ref.shape[0]
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
        diff = np.abs(ang_ref - ang_usr)
        bend_diff = np.abs(bend_usr - bend_ref)
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

    # 손/발 유사도 추가
    hand_sim = compute_bodypart_similarity(kp_ref, kp_user, [11, 13, 15])  # 어깨, 팔꿈치, 손목
    foot_sim = compute_bodypart_similarity(kp_ref, kp_user, [23, 25, 27])  # 엉덩이, 무릎, 발목

    return {
        "pose": pose_scores,
        "move": move_scores,
        "final": final_scores,
        "angle_diffs": angle_diffs,
        "proc_dists": proc_dists,
        "hand_sim": hand_sim,
        "foot_sim": foot_sim
    }

def aggregate_per_second(frame_scores: np.ndarray, fps: int) -> np.ndarray:
    """
    Aggregate frame-level scores into per-second averages.
    """
    T = len(frame_scores)
    num = T // fps
    return np.array([frame_scores[i*fps:(i+1)*fps].mean() for i in range(num)])

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

# ─────────────────────────────────────────────────────────────
# 🔽 추가 기능: 속도 및 회전 변화량 기반 비교
# ─────────────────────────────────────────────────────────────

def compute_joint_speed(keypoints_seq: np.ndarray, joint_indices: list[int]) -> float:
    """
    주어진 관절 인덱스들의 평균 속도 계산 (전체 시퀀스 기준)
    """
    speeds = []
    for i in range(1, len(keypoints_seq)):
        total = 0
        for idx in joint_indices:
            prev = keypoints_seq[i - 1][idx]
            curr = keypoints_seq[i][idx]
            dist = np.linalg.norm(np.array(curr) - np.array(prev))
            total += dist
        speeds.append(total / len(joint_indices))
    return np.mean(speeds)


def compute_angle_change_rate(keypoints_seq: np.ndarray, joint_triplets: list[tuple[int, int, int]]) -> float:
    """
    관절 각도 변화량(프레임당 평균 변화율) 계산
    """
    angle_changes = []
    for i in range(1, len(keypoints_seq)):
        total = 0
        for a, b, c in joint_triplets:
            def get_angle(p1, p2, p3):
                v1 = np.array(p1) - np.array(p2)
                v2 = np.array(p3) - np.array(p2)
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                return np.arccos(np.clip(cos_theta, -1.0, 1.0))

            prev_angle = get_angle(*[keypoints_seq[i - 1][j] for j in (a, b, c)])
            curr_angle = get_angle(*[keypoints_seq[i][j] for j in (a, b, c)])
            total += abs(curr_angle - prev_angle)
        angle_changes.append(total / len(joint_triplets))
    return np.mean(angle_changes)


def compare_motion_patterns(ref_seq: np.ndarray, usr_seq: np.ndarray) -> dict:
    """
    손/발의 속도와 회전 변화량을 비교하여 피드백 조건 생성용 수치 반환
    """
    hand_joints = [15, 16]
    foot_joints = [27, 28]
    hand_triplets = [(11, 13, 15), (12, 14, 16)]  # 어깨-팔꿈치-손목
    foot_triplets = [(23, 25, 27), (24, 26, 28)]  # 엉덩이-무릎-발목

    return {
        "hand_speed_diff": compute_joint_speed(usr_seq, hand_joints) - compute_joint_speed(ref_seq, hand_joints),
        "foot_speed_diff": compute_joint_speed(usr_seq, foot_joints) - compute_joint_speed(ref_seq, foot_joints),
        "hand_angle_diff": compute_angle_change_rate(usr_seq, hand_triplets) - compute_angle_change_rate(ref_seq, hand_triplets),
        "foot_angle_diff": compute_angle_change_rate(usr_seq, foot_triplets) - compute_angle_change_rate(ref_seq, foot_triplets)
    }
