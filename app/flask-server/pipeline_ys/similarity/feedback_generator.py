# feedback_generator.py

def generate_text_feedback(seq):
    score = seq.get("score", 0)
    hand_sim = seq.get("hand_similarity", 0)
    foot_sim = seq.get("foot_similarity", 0)
    center = seq.get("center_movement", "알 수 없음")
    offset = seq.get("timing_offset", 0)

    # 1. 좋은 점
    if hand_sim >= 0.9:
        pos = "✔️ 손의 움직임이 매우 정교해요"
    elif hand_sim >= 0.8:
        pos = "✔️ 손끝의 흐름이 자연스러워요"
    elif foot_sim >= 0.85:
        pos = "✔️ 하체 중심이 안정적이에요"
    else:
        pos = "✔️ 전체적인 동작이 무난해요"

    # 2. 아쉬운 점
    if center == "많음":
        neg = "⚠️ 중심 이동이 다소 과해요"
    elif center == "적음":
        neg = "⚠️ 고정돼 있는 느낌이에요"
    elif offset > 0.1:
        neg = "⚠️ 박자보다 약간 느려요"
    elif offset < -0.1:
        neg = "⚠️ 박자를 조금 앞서요"
    else:
        neg = "⚠️ 약간의 디테일 조정이 필요해요"

    # 3. 총평
    if score >= 90:
        summary = "🧾 안정감과 표현력이 돋보여요"
    elif score >= 80:
        summary = "🧾 흐름은 좋지만 세부 조정 필요해요"
    elif score >= 70:
        summary = "🧾 기본기는 안정적이에요"
    else:
        summary = "🧾 전체적으로 더 연습이 필요해요"

    return {
        "positive": pos,
        "negative": neg,
        "summary": summary
    }
