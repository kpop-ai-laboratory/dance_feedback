import numpy as np
from angle_utils import calc_interior_angles_2d, calc_signed_bend_angles_2d
from constants import ANGLE_JOINTS

# Korean labels corresponding to ANGLE_JOINTS order
JOINT_LABELS = [
    "왼쪽 팔꿈치 각도",
    "오른쪽 팔꿈치 각도",
    "왼쪽 손목 각도",
    "오른쪽 손목 각도",
    "왼쪽 무릎 각도",
    "오른쪽 무릎 각도",
    "왼쪽 발목 각도",
    "오른쪽 발목 각도",
    "오른쪽 어깨 각도",
    "왼쪽 어깨 각도",
    "왼쪽 골반 각도",
    "오른쪽 골반 각도",
]

def generate_frame_feedback(
    kp_ref: np.ndarray,
    kp_user: np.ndarray,
    angle_thresh: float = np.deg2rad(5)
) -> list[str]:
    """
    Generate human-readable feedback for a single frame.
    """
    ref_ang   = calc_interior_angles_2d(kp_ref[None])[0]
    user_ang  = calc_interior_angles_2d(kp_user[None])[0]
    ref_bend  = calc_signed_bend_angles_2d(kp_ref[None])[0]
    user_bend = calc_signed_bend_angles_2d(kp_user[None])[0]

    msgs = []
    for i, label in enumerate(JOINT_LABELS):
        # use bend for elbows/knees; interior otherwise
        # elbows and knees use bend, others use interior-angle
        # if i in [0, 1, 4, 5]:  # indices for bend joints
        #     delta = user_bend[i] - ref_bend[i]
        # else:
        #     delta = user_ang[i] - ref_ang[i]

        # 모든 관절을 interior‐angle 차이로만 계산
        delta = user_ang[i] - ref_ang[i]
        
        # wrap to [-π,+π]
        delta = (delta + np.pi) % (2*np.pi) - np.pi
        if abs(delta) < angle_thresh:
            continue
        deg = np.degrees(abs(delta))
        msgs.append(f"{label}가 {deg:.1f}° 차이납니다")
    return msgs
