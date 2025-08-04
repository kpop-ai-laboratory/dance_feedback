# sequence_analyzer.py

import numpy as np
from typing import List, Dict


def compute_speed(ref_kps, usr_kps):
    """
    ref/user 전체 keypoints를 기반으로 프레임 간 속도 유사도 계산 (Cosine 유사도 기반)
    """
    ref_diff = np.diff(ref_kps, axis=0)
    usr_diff = np.diff(usr_kps, axis=0)

    ref_vec = ref_diff.reshape(ref_diff.shape[0], -1)
    usr_vec = usr_diff.reshape(usr_diff.shape[0], -1)

    norm_r = np.linalg.norm(ref_vec, axis=1)
    norm_u = np.linalg.norm(usr_vec, axis=1)

    norm_r[norm_r == 0] = 1e-6
    norm_u[norm_u == 0] = 1e-6

    ref_normed = ref_vec / norm_r[:, None]
    usr_normed = usr_vec / norm_u[:, None]

    cosine = np.sum(ref_normed * usr_normed, axis=1)
    cosine = np.clip(cosine, -1, 1)

    return float(np.mean((cosine + 1) / 2))  # 0~1


def compute_center_movement(kps):
    """
    엉덩이 중심 좌표의 프레임 간 이동 거리 평균
    """
    if len(kps) < 2:
        return 0.0

    centers = (kps[:, 23, :2] + kps[:, 24, :2]) / 2
    diffs = np.diff(centers, axis=0)
    dists = np.linalg.norm(diffs, axis=1)

    return float(np.mean(dists))


def compute_part_similarity(ref_chunk, usr_chunk, joint_indices):
    """
    특정 관절군(손, 발 등)에 대한 위치 유사도 계산 (지수 기반 유사도)
    """
    if len(ref_chunk) != len(usr_chunk):
        return 0.0

    part_diffs = ref_chunk[:, joint_indices, :2] - usr_chunk[:, joint_indices, :2]
    part_dist = np.linalg.norm(part_diffs, axis=2)  # (T, J)
    part_mse = np.mean(part_dist ** 2)
    similarity = np.exp(-part_mse)
    return float(round(similarity, 3))


def compute_part_speed(kps, joint_indices):
    """
    특정 관절군의 평균 속도 계산 (프레임 간 이동 거리 평균)
    """
    diffs = np.diff(kps[:, joint_indices, :2], axis=0)
    dists = np.linalg.norm(diffs, axis=2)
    return float(np.mean(dists))


def compute_rotation_delta(kps, left_idx, right_idx):
    """
    양 손목 또는 어깨 등 양측 관절의 각도 차이 변화량 계산
    ex) 손목 회전량, 몸통 방향 전환량 등에 사용
    """
    vecs = kps[:, right_idx, :2] - kps[:, left_idx, :2]  # (T, 2)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])  # 라디안 각도
    delta = np.diff(angles)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi] 범위 정규화
    return float(np.mean(np.abs(delta)))  # 평균 회전 변화량


def analyze_sequence(kp_ref: np.ndarray, kp_usr: np.ndarray, fps: int = 30) -> List[Dict]:
    """
    3초 단위 시퀀스로 나누어 다양한 정량 지표 분석
    """
    T = min(len(kp_ref), len(kp_usr))
    seq_length = int(3 * fps)
    num_seq = T // seq_length

    output = []

    for i in range(num_seq):
        start = i * seq_length
        end   = start + seq_length

        ref_chunk = kp_ref[start:end]
        usr_chunk = kp_usr[start:end]
        if len(ref_chunk) < 2 or len(usr_chunk) < 2:
            continue

        # 원래 MSE 기반 sim
        sim = np.mean(np.sum((ref_chunk[:, :, :2] - usr_chunk[:, :, :2]) ** 2, axis=(1, 2)))
        # exp(-sim) 대신 1/(1+sim) 사용 → sim_score ∈ (0,1]
        sim_score = 1.0 / (1.0 + sim)

        speed  = compute_speed(ref_chunk, usr_chunk)
        center = compute_center_movement(usr_chunk)

        hand_joints = [15,16,17,18]
        foot_joints = [27,28,29,30]
        hand_sim = compute_part_similarity(ref_chunk, usr_chunk, hand_joints)
        foot_sim = compute_part_similarity(ref_chunk, usr_chunk, foot_joints)

        # 추가 지표
        hand_speed    = compute_part_speed(usr_chunk, hand_joints)
        foot_speed    = compute_part_speed(usr_chunk, foot_joints)
        hand_rotation = compute_rotation_delta(usr_chunk, 15, 16)
        torso_rotation= compute_rotation_delta(usr_chunk, 11, 12)

        output.append({
            "start_time":         round(start / fps, 3),
            "end_time":           round(end   / fps, 3),
            # 0~100 점수로 환산
            "score":              round(sim_score * 100, 2),
            "hand_similarity":    hand_sim,
            "foot_similarity":    foot_sim,
            "speed_similarity":   round(speed, 3),
            "center_movement":    round(center, 3),
            "hand_speed":         round(hand_speed, 3),
            "foot_speed":         round(foot_speed, 3),
            "hand_rotation_delta":round(hand_rotation, 3),
            "torso_turn_delta":   round(torso_rotation, 3),
            "ref_kps":            ref_chunk,
            "usr_kps":            usr_chunk,
            "similarity_scores":  list((1 - sim / (sim + 1e-6)) * np.ones(len(ref_chunk)))
        })

    return output