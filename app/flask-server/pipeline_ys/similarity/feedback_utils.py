import numpy as np
from pipeline_ys.similarity.angle_utils import calc_interior_angles_2d, calc_signed_bend_angles_2d
from pipeline_ys.similarity.angle_utils import ANGLE_IDX

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

# 각도 차이에 따라 “조금 굽히세요”~“완전히 펴세요” 문장 반환
def generate_joint_angle_feedback(ref_kps: np.ndarray,
                                  usr_kps: np.ndarray,
                                  low_thresh=10,
                                  mid_thresh=20,
                                  high_thresh=35,
                                  max_thresh=50) -> list[str]:

    # ref_ang  = calc_interior_angles_2d(ref_kps[None])[0]
    # usr_ang  = calc_interior_angles_2d(usr_kps[None])[0]

    """
    시퀀스 단위로:
     1) 프레임별 관절 각도 계산
     2) 시퀀스 전체 프레임 각도 평균화
     3) 기준 vs 사용자 평균 각도 차이에 따라
        '조금/더/많이/완전히 굽히세요/펴세요' 메시지 생성
    """

    # 1) 시퀀스 전체 프레임에 대한 interior angles (shape: T x A)
    ref_ang_seq = calc_interior_angles_2d(ref_kps)   # 이제 (T, num_angles)
    usr_ang_seq = calc_interior_angles_2d(usr_kps)

    # 2) 프레임 차원 평균 (shape: A,)
    ref_ang = np.mean(ref_ang_seq, axis=0)
    usr_ang = np.mean(usr_ang_seq, axis=0)

    feedback = []

    # 관심 관절 인덱스: 팔꿈치(0,1), 무릎(4,5)
    joints = [0, 1, 4, 5]
    labels = ["왼쪽 팔꿈치","오른쪽 팔꿈치","왼쪽 무릎","오른쪽 무릎"]

    for idx, joint in zip(joints, labels):
        r_deg = np.degrees(ref_ang[idx])
        u_deg = np.degrees(usr_ang[idx])
        diff  = r_deg - u_deg

        if abs(diff) < low_thresh:
            continue

        # 방향
        action = "굽히세요" if diff > 0 else "펴세요"
        mag = abs(diff)

        # 정도
        if   mag < mid_thresh:  adv = "조금 "
        elif mag < high_thresh: adv = ""
        elif mag < max_thresh:  adv = "많이 "
        else:                    adv = "완전히 "

        feedback.append(
            f"{joint}를 {adv}{action} – 기준 {r_deg:.1f}°, 사용자 {u_deg:.1f}°, 차이 {mag:.1f}°"
        )

    return feedback

    # for idx in [0, 1, 4, 5]:
    #     r_deg = np.degrees(ref_ang[idx])
    #     u_deg = np.degrees(usr_ang[idx])
    #     diff  = r_deg - u_deg

    #     if abs(diff) < low_thresh:
    #         continue  # 너무 미미하면 패스

    #     joint = ["왼쪽 팔꿈치","오른쪽 팔꿈치","왼쪽 무릎","오른쪽 무릎"][
    #         [0,1,4,5].index(idx)
    #     ]
    #     # diff > 0 → 사용자가 덜 굽힘 → “굽히세요”
    #     # diff < 0 → 사용자가 과도하게 굽힘 → “펴세요”
    #     action = "굽히세요" if diff > 0 else "펴세요"
    #     magnitude = abs(diff)

    #     if magnitude < mid_thresh:
    #         adv = "조금 "
    #     elif magnitude < high_thresh:
    #         adv = ""
    #     elif magnitude < max_thresh:
    #         adv = "많이 "
    #     else:
    #         adv = "완전히 "

    #     feedback.append(f"{joint}를 {adv}{action} – 기준 {r_deg:.1f}°, 사용자 {u_deg:.1f}°, 차이 {abs(diff):.1f}°")

    # return feedback

def compare_transition_timing(ref_kps: np.ndarray,
                              usr_kps: np.ndarray,
                              fps: int = 30,
                              angle_thresh: float = 15.0) -> str | None:
    """
    기준과 사용자 키포인트에서 각도 변화량을 기준으로
    '첫 전환 시점' 차이를 비교하여 타이밍 피드백을 반환합니다.
    """

    def detect_transitions(kps: np.ndarray) -> list[int]:
        angles = np.linalg.norm(np.diff(kps, axis=0), axis=(1, 2))  # 프레임 간 변화량
        peaks = np.where(angles > np.deg2rad(angle_thresh))[0]  # 변화량 큰 프레임
        return peaks.tolist()

    ref_peaks = detect_transitions(ref_kps)
    usr_peaks = detect_transitions(usr_kps)

    if not ref_peaks or not usr_peaks:
        return None  # 전환 시점이 없다면 패스

    diff_sec = (usr_peaks[0] - ref_peaks[0]) / fps

    if abs(diff_sec) < 0.15:
        return None  # 0.15초 이내면 비슷하다고 봄

    if diff_sec > 0:
        return f"동작 전환이 기준보다 {diff_sec:.2f}초 느립니다"
    else:
        return f"동작 전환이 기준보다 {abs(diff_sec):.2f}초 빠릅니다"