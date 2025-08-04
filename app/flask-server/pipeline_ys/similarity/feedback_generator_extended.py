# pipeline_ys/similarity/feedback_generator_extended.py

import re
import random
# 관절 각도 피드백 불필요하므로 import는 남겨도 무방하지만 제거해도 됨
# from .feedback_utils import generate_joint_angle_feedback

def format_feedback_text(feedback: str) -> str:
    """
    이모지(💃, 🖐️, 🦵, ⚡, 🔄) 앞에서 줄바꿈을 추가하여
    자연스러운 멀티라인 피드백 텍스트를 반환합니다.
    """
    pattern = r"(💃|🖐️|🦵|⚡|🔄)"
    return re.sub(pattern, r"\n\1", feedback).strip()


def generate_sequence_feedback(seq_data, ref_kps_chunk, usr_kps_chunk):
    """
    시퀀스 단위로 정성적 피드백을 생성합니다.
    - 점수 및 관절 각도 피드백 제거됨
    - 줄바꿈을 유지해 React에서 멀티라인 지원
    """

    center_move   = seq_data.get("center_movement", 0.0)
    timing_offset = seq_data.get("timing_offset", 0.0)
    hand          = seq_data.get("hand_similarity", 0.0)
    foot          = seq_data.get("foot_similarity", 0.0)
    hand_speed    = seq_data.get("hand_speed", 0.0)
    foot_speed    = seq_data.get("foot_speed", 0.0)
    hand_rotation = seq_data.get("hand_rotation_delta", 0.0)
    torso_turn    = seq_data.get("torso_turn_delta", 0.0)

    lines = []

    # ─── ⏱️ 타이밍 오차 ─────────────────
    if timing_offset > 0.3:
        lines.append(random.choice([
            "⏱️ 다음 동작으로 전환이 늦어요. 리듬을 빠르게 타보세요.",
            "⏱️ 반 박자 느리게 움직이고 있어요. 템포를 올려보세요.",
            "⏱️ 동작 전환이 느려서 흐름이 끊겨 보여요."
        ]))
    elif timing_offset < -0.3:
        lines.append(random.choice([
            "⏱️ 급하게 넘어가요. 한 박자 여유를 가져보세요.",
            "⏱️ 기준보다 빨라요. 다음 동작 전에 호흡을 넣어보세요.",
            "⏱️ 리듬이 앞서 있어요. 박자를 다시 맞춰보세요."
        ]))

    # ─── 💃 중심 이동 ─────────────────
    if center_move < 0.05:
        lines.append(random.choice([
            "💃 움직임이 작아 보여요. 공간을 더 넓게 써보세요.",
            "💃 중심 이동이 적어요. 몸을 더 크게 흔들어보세요.",
            "💃 동작이 고정된 느낌이에요. 방향을 과감히 바꿔보세요."
        ]))
    elif center_move > 0.2:
        lines.append(random.choice([
            "💃 무대 위에서 시원시원한 움직임이 좋았어요!",
            "💃 중심이 크게 흔들려서 몰입감이 있어요!",
            "💃 과감한 이동이 좋았어요. 계속 유지해보세요!"
        ]))

    # ─── 🖐️ 손 유사도 ─────────────────
    if hand < 0.5:
        lines.append(random.choice([
            "🖐️ 손끝이 흐릿해요. 더 뚜렷하게 뻗어주세요.",
            "🖐️ 손의 표현이 부족해요. 라인을 확실히 그려주세요.",
            "🖐️ 손이 멈춰 있어요. 계속 흔들며 연결해보세요."
        ]))
    elif hand < 0.7:
        lines.append(random.choice([
            "🖐️ 손 동작이 조금 약해요. 선을 크게 그려보세요.",
            "🖐️ 손의 방향이 자주 바뀌어요. 집중해서 뻗어보세요.",
            "🖐️ 손끝에 힘을 더 주세요. 기준은 더 날렵했어요."
        ]))
    else:
        lines.append(random.choice([
            "🖐️ 손 표현이 잘 살아 있어요!",
            "🖐️ 손끝까지 흐름이 이어져서 좋아요!",
            "🖐️ 손의 선이 예쁘게 유지되고 있어요."
        ]))

    # ─── 🦵 발 유사도 ─────────────────
    if foot < 0.5:
        lines.append(random.choice([
            "🦵 하체가 불안정해요. 무게 중심을 더 단단히 잡아보세요.",
            "🦵 하체가 흔들려요. 다리의 축을 잡고 움직여보세요.",
            "🦵 발의 위치가 많이 달라요. 기준 동작을 따라가보세요."
        ]))
    elif foot < 0.7:
        lines.append(random.choice([
            "🦵 하체가 살짝 흔들려요. 스탠스를 점검해보세요.",
            "🦵 무릎을 조금 더 유연하게 써보세요.",
            "🦵 스텝이 다소 좁아요. 더 과감히 움직여보세요."
        ]))
    else:
        lines.append(random.choice([
            "🦵 하체가 안정적이에요!",
            "🦵 하체 중심이 잘 유지되고 있어요!",
            "🦵 발의 리듬이 잘 맞아요."
        ]))

    # ─── ⚡ 속도감 (손, 발 별도) ─────────────────
    if hand_speed > 15:
        lines.append(random.choice([
            "⚡ 손을 너무 빠르게 움직여요. 흐름을 더 살려보세요.",
            "⚡ 손이 너무 급해 보여요. 템포를 조금 낮춰보세요.",
            "⚡ 동작이 튀어요. 느긋하게 연결감을 높여보세요."
        ]))
    elif hand_speed < 5:
        lines.append(random.choice([
            "⚡ 손이 정지된 느낌이에요. 더 리듬감 있게 써보세요.",
            "⚡ 손이 머뭇거려요. 자연스럽게 템포를 당겨서 움직이세요.",
            "⚡ 손이 느려서 전달력이 약해요. 선명하게 움직여보세요."
        ]))

    if foot_speed > 20:
        lines.append(random.choice([
            "⚡ 하체가 너무 급해요. 속도 조절이 필요해요.",
            "⚡ 발동작이 튀어요. 부드럽게 연결해보세요.",
            "⚡ 템포가 빠르다 보니 전체 흐름이 깨질 수 있어요."
        ]))
    elif foot_speed < 7:
        lines.append(random.choice([
            "⚡ 하체 동작이 작아요. 스텝을 더 크게 써보세요.",
            "⚡ 하체가 느려서 박자가 밀려요.",
            "⚡ 스텝이 망설여져 보여요. 자신감 있게 딛어보세요."
        ]))

    # ─── 🔄 회전 분석 ─────────────────
    if hand_rotation > 1.0:
        lines.append(random.choice([
            "🔄 손의 회전이 커서 동작이 거칠어 보일 수 있어요.",
            "🔄 손이 너무 많이 돌아가요. 각도를 줄여보세요.",
            "🔄 회전이 커서 불안정해 보여요."
        ]))
    elif hand_rotation < 0.2:
        lines.append(random.choice([
            "🔄 손의 방향 전환이 적어요. 디테일을 살려보세요.",
            "🔄 손의 꺾임이 부족해요. 선을 명확히 그려보세요.",
            "🔄 손이 평면적으로 움직이고 있어요. 손 끝에 힘을 줘서 동작을 취하세요"
        ]))

    if torso_turn > 0.8:
        lines.append(random.choice([
            "🔄 상체 회전이 크고 역동적이에요!",
            "🔄 몸통을 잘 써서 무대가 꽉 차 보여요!",
            "🔄 상체가 크게 돌아가면서 시원한 느낌을 줘요!"
        ]))
    elif torso_turn < 0.2:
        lines.append(random.choice([
            "🔄 상체가 고정된 느낌이에요. 방향 전환을 더 해보세요.",
            "🔄 상체가 너무 고요해요. 동작에 생기를 주세요.",
            "🔄 몸통이 멈춰 있어요. 역동적으로 동작 전환을 하세요."
        ]))

    return "\n".join(lines)
