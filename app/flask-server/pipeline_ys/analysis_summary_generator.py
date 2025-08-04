# analysis_summary_generator.py

import numpy as np
from typing import List, Dict


def generate_sequence_summary(sequence_data: List[Dict]) -> List[str]:
    """
    시퀀스 분석 결과로부터 요약 피드백 문장을 생성
    - 유사도/속도/중심 이동 정보를 기반으로 요약 문장 생성

    Args:
        sequence_data: analyze_sequence()에서 나온 시퀀스 리스트

    Returns:
        List[str]: 시퀀스별 요약 피드백 문장
    """
    summary = []

    for i, seq in enumerate(sequence_data):
        start = seq["start_time"]
        end = seq["end_time"]
        score = seq["score"]
        speed = seq["speed_similarity"]
        center = seq["center_movement"]

        feedback = f"{start:.2f}초 ~ {end:.2f}초 구간에서: "

        # 전체 유사도 점수 기반
        if score > 85:
            feedback += "전반적으로 동작 유사도가 매우 높아요. "
        elif score > 70:
            feedback += "동작 유사도는 무난하지만 약간의 차이가 있어요. "
        else:
            feedback += "동작 유사도가 낮아 보완이 필요해요. "

        # 속도 비교
        if speed > 0.9:
            feedback += "동작 전환 속도도 잘 맞습니다. "
        elif speed > 0.75:
            feedback += "속도는 대체로 비슷하지만 간혹 어긋납니다. "
        else:
            feedback += "속도 타이밍이 많이 다릅니다. 좀 더 리듬에 맞춰보세요. "

        # 중심 이동
        if center < 5:
            feedback += "중심 이동은 다소 부족해요. 움직임에 더 적극성이 필요해요."
        elif center < 10:
            feedback += "중심 이동은 적당했어요."
        else:
            feedback += "중심 이동이 활발해요. 에너지 넘치는 표현 좋습니다!"

        summary.append(feedback)

    return summary


def generate_analysis_summary(sequence_data: List[Dict]) -> str:
    """
    전체 시퀀스 요약을 하나의 문자열로 통합
    - LLM 프롬프트 등에 넣을 수 있도록 텍스트 구성

    Args:
        sequence_data: analyze_sequence()에서 나온 시퀀스 리스트

    Returns:
        str: 전체 요약 문자열
    """
    lines = generate_sequence_summary(sequence_data)
    return "\n".join(f"- {line}" for line in lines)


# # analysis_summary_generator.py

# import numpy as np
# from pipeline_ys.similarity.feedback_utils import generate_frame_feedback, compare_transition_timing


# def summarize_timed_issues(ref_kps, usr_kps, fps, similarity_scores, sim_thresh=0.8, top_k=3):
#     """
#     유사도가 낮은 프레임을 기준으로 '시간대별 주요 이슈'를 추출하는 함수
#     - 관절 피드백을 프레임 단위로 뽑아 시간값과 함께 출력
#     - 너무 많은 프레임이 걸리면 top_k개만 선택

#     Args:
#         ref_kps: 기준 영상 키포인트 배열 (T, 33, 3)
#         usr_kps: 사용자 영상 키포인트 배열
#         fps: 초당 프레임 수
#         similarity_scores: 프레임별 유사도 리스트
#         sim_thresh: 유사도 기준 (이보다 낮으면 '문제 있음'으로 판단)
#         top_k: 최대 몇 개 시간 포인트 추출할지

#     Returns:
#         list[str]: "1.3초: 오른팔을 더 펴세요" 같은 문장 리스트
#     """
#     T = len(similarity_scores)

#     # 1. 유사도가 낮은 프레임 인덱스 추출
#     low_frames = [i for i, s in enumerate(similarity_scores) if s < sim_thresh]

#     # 2. 초 단위 시간으로 변환 (중복 제거 + 최대 top_k개 제한)
#     timestamps = sorted(set(round(f / fps, 2) for f in low_frames))[:top_k]

#     output = []
#     for t_sec in timestamps:
#         frame_idx = int(t_sec * fps)

#         # 범위를 벗어난 프레임은 스킵
#         if frame_idx < 1 or frame_idx >= T - 1:
#             continue

#         ref_f = ref_kps[frame_idx]
#         usr_f = usr_kps[frame_idx]

#         # 관절 피드백 1줄 생성
#         feedbacks = generate_frame_feedback(ref_f, usr_f)

#         if feedbacks:
#             output.append(f"{t_sec}초: {feedbacks[0]}")
#         else:
#             output.append(f"{t_sec}초: 관절 자세 차이는 크지 않음")

#     return output


# def generate_analysis_summary(seq_data, ref_kps_chunk, usr_kps_chunk, fps, similarity_scores,
#                               speed_scores=None, center_movement=None):
#     """
#     LLM 프롬프트에 넘길 '시퀀스 분석 요약 문장'을 생성하는 함수

#     Args:
#         seq_data: 시퀀스별 점수와 유사도 스탯이 담긴 딕셔너리
#         ref_kps_chunk: 기준자 keypoints (시퀀스 구간)
#         usr_kps_chunk: 사용자 keypoints (시퀀스 구간)
#         fps: 프레임레이트
#         similarity_scores: 시퀀스 구간 프레임별 유사도 리스트
#         speed_scores: (선택) 속도 유사도 리스트
#         center_movement: (선택) 중심 이동량 (seq_data에서 가져올 수도 있음)

#     Returns:
#         str: LLM에게 넘길 요약 텍스트 (줄바꿈 포함)
#     """
#     lines = []

#     # ① 점수, 손/발 유사도, 중심 이동량 가져오기 (sequence_analyzer.py에서 가져 옴)
#     score  = seq_data["score"]
#     hand   = seq_data["hand_similarity"]
#     foot   = seq_data["foot_similarity"]
#     center = seq_data["center_movement"]

#     # ② 동작 전환 타이밍 오차 계산
#     timing = compare_transition_timing(ref_kps_chunk, usr_kps_chunk, fps=fps)

#     # ③ 점수 출력
#     lines.append(f"- 전체 유사도: {score}점")

#     # ④ 손 유사도 해석
#     if hand < 0.5:
#         lines.append("- 손 유사도: 낮음")
#     elif hand < 0.7:
#         lines.append("- 손 유사도: 보통")
#     else:
#         lines.append("- 손 유사도: 높음")

#     # ⑤ 발 유사도 해석
#     if foot < 0.5:
#         lines.append("- 발 유사도: 낮음")
#     elif foot < 0.7:
#         lines.append("- 발 유사도: 보통")
#     else:
#         lines.append("- 발 유사도: 높음")

#     # ⑥ 중심 이동 해석
#     if isinstance(center, (float, int)) and center < 0.2:
#         lines.append("- 중심 이동량: 부족함")
#     elif isinstance(center, str):
#         lines.append(f"- 중심 이동량: {center}")
#     else:
#         lines.append(f"- 중심 이동량: {round(center, 3)}")

#     # ⑦ 타이밍 오차 출력
#     if timing:
#         lines.append(f"- 타이밍 오차: {timing}")

#     # ⑧ 시간대별 관절 피드백
#     lines.append("- 시간대별 이슈:")
#     lines += summarize_timed_issues(ref_kps_chunk, usr_kps_chunk, fps, similarity_scores)

#     return "\n".join(lines)
