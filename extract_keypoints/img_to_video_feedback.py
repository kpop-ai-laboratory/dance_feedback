import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────
# 1) 설정
# ──────────────────────────────────────────────────────────
FEEDBACK_JSON = r"D:\kpop\feedback_test\feedback.json"
TEACHER_DIR   = r"D:\kpop\real_kpop\data\results1\frames"
STUDENT_DIR   = r"D:\kpop\real_kpop\data\results2\frames"
OUT_DIR       = r"D:\kpop\feedback_test\results"
os.makedirs(OUT_DIR, exist_ok=True)

# 한글 폰트(.ttf) 경로를 자신의 환경에 맞게 지정하세요.
# Windows 기본 폰트: Malgun Gothic
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_SIZE = 24

total_frames = max(
    len(os.listdir(TEACHER_DIR)),
    len(os.listdir(STUDENT_DIR))
)

# ──────────────────────────────────────────────────────────
# 2) 피드백 로드
# ──────────────────────────────────────────────────────────
with open(FEEDBACK_JSON, 'r', encoding='utf-8') as f:
    raw = json.load(f)
    # feedback 예: { "13": ["어깨 각도가 …", "골반이 …"], "14": […], … }

feedback = {
    str(i): raw.get(str(i), [])
    for i in range(total_frames)
}

# ──────────────────────────────────────────────────────────
# 3) 프레임별로 처리
# ──────────────────────────────────────────────────────────
for frame_str, msgs in tqdm(feedback.items(), desc="Visualizing"):
    t_idx = int(frame_str)
    s_idx = t_idx   # student 인덱스가 t_idx 와 다르면 별도 매핑 필요

    # 3-1) 이미지 불러오기
    t_path = os.path.join(TEACHER_DIR, f"frame_{t_idx:06d}.jpg")
    s_path = os.path.join(STUDENT_DIR, f"frame_{s_idx:06d}.jpg")
    t_img  = cv2.imread(t_path)
    s_img  = cv2.imread(s_path)
    if t_img is None or s_img is None:
        print(f"⚠️ 못 불러옴: {t_path} or {s_path}")
        continue

    # 3-2) 크기 통일 (두 영상이 원본 해상도가 같다면 생략 가능)
    h, w = t_img.shape[:2]
    s_img = cv2.resize(s_img, (w, h))

    # 3-3) 캔버스 생성 (위아래로 각각 h씩, 총 2h)
    canvas = np.zeros((h*2, w, 3), dtype=np.uint8)
    canvas[:h]     = t_img  # 위쪽
    canvas[h:h*2]  = s_img  # 아래쪽

    # ──────────────────────────────────────────────────────────
    # 4) PIL로 변환해서 한글 텍스트 그리기
    # ──────────────────────────────────────────────────────────
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    lines = msgs  # 한 줄씩 출력할 텍스트
    line_spacing = 7

    # 0. 피드백 메시지(empty list)가 없으면 텍스트 박스 없이 바로 저장
    if not msgs:
        out_path = os.path.join(OUT_DIR, f"frame_{t_idx:06d}.jpg")
        cv2.imwrite(out_path, canvas)
        continue

    # ✏️ 변경: draw.textsize → draw.textbbox 으로 크기 계산
    line_sizes = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w_line = bbox[2] - bbox[0]
        h_line = bbox[3] - bbox[1]
        line_sizes.append((w_line, h_line))

   # 수정: 좌우/상하 padding 을 따로 지정
    padding_x = 15   # 좌우 여백 (기존 절반 → 15px씩)
    padding_y = 15   # 상하 여백 (기존 절반 → 15px씩)
    text_w = max((w for w, h in line_sizes), default=0) + padding_x * 2
    text_h  = sum(h for w, h in line_sizes) \
              + padding_y * 2 \
              + line_spacing * (len(lines) - 1)

    # 박스 좌표
    margin = 10
    # 오른쪽 아래 좌표
    # x1 = w - margin
    # y1 = 2*h - margin
    # x0 = x1 - text_w
    # y0 = y1 - text_h

    # 중앙 살짝 아래 좌표
    # ★ 가로 중앙 정렬, 세로는 화면 높이의 75% 지점(원하는 비율로 조정) 사용
    cx = w // 2
    x0 = cx - text_w // 2
    x1 = x0 + text_w
    y1 = int(h * 1.2)    # 화면 높이의 60~80% 사이로 조절 가능
    y0 = y1 - text_h

    # 흰 박스
    draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 255))

    # 줄 단위로 y를 증가시킬 때 line_spacing 추가
    y = y0 + 10
    for (line, (_, h_line)) in zip(lines, line_sizes):
        draw.text((x0 + 10, y), line, font=font, fill=(0, 0, 0))
        y += h_line + line_spacing

    # 4-5) 다시 numpy 배열로
    canvas = np.array(pil)

    # ──────────────────────────────────────────────────────────
    # 5) 결과 저장
    # ──────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, f"frame_{t_idx:06d}.jpg")
    cv2.imwrite(out_path, canvas)
