import numpy as np
from .angle_utils import calc_interior_angles_2d, calc_signed_bend_angles_2d
from .angle_utils import ANGLE_IDX

# Korean labels corresponding to ANGLE_JOINTS order
JOINT_LABELS = [
    "왼쪽 팔꿈치 각도",    # 0
    "오른쪽 팔꿈치 각도",  # 1
    "왼쪽 손목 각도",      # 2 (skip)
    "오른쪽 손목 각도",    # 3 (skip)
    "왼쪽 무릎 각도",      # 4
    "오른쪽 무릎 각도",    # 5
    "왼쪽 발목 각도",      # 6 (skip)
    "오른쪽 발목 각도",    # 7 (skip)
    "오른쪽 팔",    # 8
    "왼쪽 팔",      # 9
    "왼쪽 골반 각도",      # 10
    "오른쪽 골반 각도",    # 11
]

def generate_frame_feedback(
    kp_ref: np.ndarray,
    kp_user: np.ndarray,
    angle_thresh: float = np.deg2rad(5),
    direction_thresh: float = 0.2,
    reverse_thresh_deg: float = 20.0

) -> list[str]:
    """
    1) 손목(2,3), 발목(6,7) 제외
    2) 팔꿈치(0,1), 무릎(4,5), 어깨(8,9), 골반(10,11) 각각 매핑:
       - 팔꿈치/무릎: user_ang<ref_ang→펴세요, >→굽히세요, 벡터차 크면 '(방향 조절이 필요합니다.)'
       - 어깨:     user_ang<ref_ang→팔을 올리세요, >→팔을 내리세요
       - 골반:     user_ang<ref_ang→골반 각도가 넓습니다, >→골반 각도가 좁습니다
    3) 메시지 형식: "{label}를 {action} – 선생님 {teacher_deg:.1f}° / 학생 {student_deg:.1f}°"
    """

    # interior angles & signed bend (rad)
    ref_ang   = calc_interior_angles_2d(kp_ref[None])[0]
    user_ang  = calc_interior_angles_2d(kp_user[None])[0]
    # (signed bend는 벡터 방향 차이 확인용, interior만 매핑하므로 사실 미사용)
    ref_bend  = calc_signed_bend_angles_2d(kp_ref[None])[0]
    user_bend = calc_signed_bend_angles_2d(kp_user[None])[0]

    feedback = []

    for i, (a, b, c) in enumerate(ANGLE_IDX):
        # 1) skip 손목/발목
        if i in {2, 3, 6, 7}:
            continue

        # 2) angle diff threshold
        angle_diff = abs(user_ang[i] - ref_ang[i])
        if angle_diff <= angle_thresh:
                continue

        # common vars
        teacher_deg = np.degrees(ref_ang[i])
        student_deg = np.degrees(user_ang[i])
        label       = JOINT_LABELS[i]

        stu_1dp = round(student_deg, 1)
        tea_1dp = round(teacher_deg, 1)

        if student_deg<teacher_deg: 
                diff_angle= tea_1dp - stu_1dp
        else:
                diff_angle= stu_1dp - tea_1dp

        # 팔꿈치 / 무릎
        if i in {0, 1, 4, 5}:
            # 벡터 방향 차이
            # vec_ref  = kp_ref[c, :2]  - kp_ref[b, :2]
            # vec_user = kp_user[c, :2] - kp_user[b, :2]
            # nr, nu   = np.linalg.norm(vec_ref), np.linalg.norm(vec_user)
            # suffix = ""

            # # signed bend raw (rad)
            # ref_b = ref_bend[i]
            # usr_b = user_bend[i]
            # # 안쪽(+) vs 바깥쪽(-) 완전 반전 감지
            # if ref_b * usr_b < 0:
            #     # 부호가 다르면 무조건 반전으로 간주
            #     suffix = " (방향 조절이 필요합니다.)"
            # else:
            suffix = ""

            # action 결정: user_ang<ref_ang→펴세요, else→굽히세요

            action = "펴세요" if user_ang[i] < ref_ang[i] else "굽히세요"
            feedback.append(
                f"{label}를 {action}{suffix} – "
                f"선생님 {teacher_deg:.1f}° / 학생 {student_deg:.1f}° / 각도 차이 {diff_angle:.1f}°"
            )

        # 어깨
        elif i in {8, 9}:
            # user_ang<ref_ang→올리세요, else→내리세요
            action = f"{label}을 올리세요" if user_ang[i] < ref_ang[i] else f"{label}을 내리세요"
            suffix = ""
            feedback.append(
                f"{action}{suffix} – "
                f"선생님 {teacher_deg:.1f}° / 학생 {student_deg:.1f}° / 각도 차이 {diff_angle:.1f}°"
            )

        # 골반
        elif i in {10, 11}:
            # user_ang<ref_ang→넓습니다, else→좁습니다
            action = f"{label} 각도가 넓습니다" if user_ang[i] < ref_ang[i] else f"{label} 각도가 좁습니다"
            suffix = ""
            feedback.append(
                f"{action}{suffix} – "
                f"선생님 {teacher_deg:.1f}° / 학생 {student_deg:.1f}° / 각도 차이 {diff_angle:.1f}°"
            )

    return feedback