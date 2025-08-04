# feedback_sequence.py
import numpy as np
import random  # 🔹 랜덤 문구 선택을 위해 추가

def extract_body_tilt_sequence(keypoints_seq):
    """
    keypoints_seq: (frames, 33, 3)
    return: body tilt 각도 시퀀스 (frames,)
    """
    tilts = []
    for frame in keypoints_seq:
        shoulder = (frame[11][:2] + frame[12][:2]) / 2
        hip = (frame[23][:2] + frame[24][:2]) / 2
        vec = hip - shoulder
        angle = np.degrees(np.arctan2(vec[0], vec[1]))  # X/Y → 좌우 기울기
        tilts.append(angle)
    return np.array(tilts)

def feedback_body_tilt_movement(ref_seq, usr_seq, thresh_deg=10):
    """
    기준 대비 사용자의 몸통 기울기 변화폭이 작으면 피드백 생성
    - 기준 대비 70% 이하이며
    - 절대차가 thresh_deg 이상일 경우
    """
    ref_tilt = extract_body_tilt_sequence(ref_seq)
    usr_tilt = extract_body_tilt_sequence(usr_seq)

    delta_ref = np.max(ref_tilt) - np.min(ref_tilt)
    delta_usr = np.max(usr_tilt) - np.min(usr_tilt)

    if delta_usr < delta_ref * 0.7 and (delta_ref - delta_usr) > thresh_deg:
        options = [
            "몸통을 더 깊이 숙이면서 동작에 파워를 실어보세요. 현재는 기준보다 기울기 변화가 작아 약해 보입니다.",
            "몸의 기울기 변화가 작아 동작의 임팩트와 속도감이 부족해 보입니다. 더 확실하게 움직여 주세요.",
            "기준에 비해 몸의 기울기 변화 폭이 작아, 강약 대비가 부족해 보입니다."
        ]
        return random.choice(options)
    return None  # 🔹 조건에 맞지 않으면 피드백 없음