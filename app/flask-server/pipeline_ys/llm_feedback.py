# llm_feedback.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

# ✅ .env에서 API 키 불러오기
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Gemini 모델 준비
model = genai.GenerativeModel("gemini-1.5-flash")

def generate_llm_feedback(analysis_summary: str) -> str:
    # ✅ 프롬프트 구성 (너가 작성한 그대로)
    prompt = f"""
너는 K-pop 안무 트레이너이자 심사위원이야.
분석 요약을 기반으로, 시간대별로 어떤 동작을 어떻게 수정해야 하는지 구체적으로 작성해줘.

목표는 학생이 기준 영상(레퍼런스)의 동작 흐름과 최대한 일치하도록 교정하는 것이야.

[분석 요약]에는 유사도 점수 외에도 각 관절의 각도 차이, 속도 유사도, 중심 이동량 부족 여부 등이 포함되어 있어.

### 지시사항:
- 각 시간대에 대해 정확한 동작 교정 피드백을 작성해줘.
- 동작의 방향, 범위, 속도, 이동 여부까지 포함해 말해줘.
- 예를 들어, "왼팔을 더 위로 빠르게 뻗어야 한다", "빠르게 숙이면서 다리를 쭉 뻗어야 한다"처럼 표현할 것.
- 단순히 "굽히세요"가 아니라, 어떤 동작을 하려는 상황인지 추론해서 알려줄 것.
- 문장 수는 시간대별로 1~2줄 이내로 유지할 것.
- 레퍼런스 영상이라는 표현 지양할 것.
- 중심을 앞으로 이동시킨다는 표현 대신 몸을 숙인다던지 젖힌다던지의 표현으로 말할 것.

[분석 요약]
{analysis_summary}

### 예시 출력:
- 1.3초: 왼팔을 더 높이 빠르게 들어올리세요.
- 1.8초: 상체를 더 빠르게 숙이며 왼다리를 쭉 뻗어야 합니다.
- 2.7초: 전체 동작을 더 빠르게 시작하고 중심을 앞으로 이동시키세요.

시퀀스 당 피드백이 끝나고 나면 엔터해서 그 다음 줄에 출력되도록해줘.
"""

    # ✅ Gemini API 호출 방식 (주의!)
    response = model.generate_content(prompt)
    return response.text.strip()
