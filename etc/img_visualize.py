# 이미지들이 모여있는 폴더의 내용을 mediaPipe 진행한 csv 파일과 매칭시켜서 시각화 
import os
import cv2
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 1) 환경 설정
# ──────────────────────────────────────────────────────────────
# FRAME_DIR = r"D:\dance_feedbacker\test\data\next_level\teacher\frames\channel\k0FUo9ItOO8"
# CSV_PATH  = r"D:\dance_feedbacker\test\data\next_level\teacher\pose_csv\k0FUo9ItOO8.csv"
# OUT_DIR   = r"D:\dance_feedbacker\test\data\next_level\teacher\vis_skeleton_mediapipe\k0FUo9ItOO8"
FRAME_DIR = r"D:\dance_feedbacker\test\data\next_level\teacher\crops"
CSV_PATH  = r"D:\dance_feedbacker\test\data\next_level\teacher\pose_crop_csv_mediapipe\teacher.csv"
OUT_DIR   = r"D:\dance_feedbacker\test\data\next_level\teacher\vis_crop_skeleton_mediapipe"
os.makedirs(OUT_DIR, exist_ok=True)

# 2) 사용자 정의 관절 이름 & 연결 쌍
JOINT_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"
]
SKELETON = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),
    (12,14),(14,16),(16,18),(16,20),(16,22),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(29,31),
    (24,26),(26,28),(28,30),(30,32)
]
# ──────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
for _, row in df.iterrows():
    fname = row["frame"]
    img = cv2.imread(os.path.join(FRAME_DIR, fname))
    if img is None:
        continue
    h, w = img.shape[:2]

    # 3) normalized → pixel 좌표로 변환
    points = {}
    for idx, name in enumerate(JOINT_NAMES):
        x_norm = row.get(f"x{idx}", None)
        y_norm = row.get(f"y{idx}", None)
        if pd.isna(x_norm) or pd.isna(y_norm):
            continue
        x_px = int(x_norm * w)
        y_px = int(y_norm * h)
        points[idx] = (x_px, y_px)

    # 4) 사용자 정의 SKELETON으로 뼈대 선 그리기
    for i,j in SKELETON:
        if i in points and j in points:
            cv2.line(img, points[i], points[j], color=(0,255,0), thickness=2)

    # 5) 관절점 찍기 (원)
    for (x,y) in points.values():
        cv2.circle(img, (x,y), radius=3, color=(0,0,255), thickness=-1)

    # 선택: 관절명 레이블 추가
    # for idx, (x,y) in points.items():
    #     cv2.putText(img, JOINT_NAMES[idx], (x+5,y-5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # 6) 저장
    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, img)

print("✔ Custom skeleton visualizations saved to:", OUT_DIR)
