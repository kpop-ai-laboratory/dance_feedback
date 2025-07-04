import os, logging, warnings
import cv2, json, csv
from ultralytics import YOLO
import mediapipe as mp
from tqdm import tqdm

# TensorFlow/absl 로그 끄기
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # ERROR 레벨 이하 숨김
# 파이썬 warnings 모듈 경고 끄기
warnings.filterwarnings("ignore")
# Ultralytics YOLO 로거 최소 WARNING 레벨
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# 1) 환경 설정
VIDEO_PATH = r"C:\Users\human\Desktop\real_kpop\data\nextlevel_2_synced.mp4"
OUTPUT_DIR = r"C:\Users\human\Desktop\real_kpop\data\results2"
CSV_PATH   = os.path.join(OUTPUT_DIR, "keypoints2.csv")
JSON_PATH  = os.path.join(OUTPUT_DIR, "keypoints2.json")
IMAGE_DIR  = os.path.join(OUTPUT_DIR, "frames")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# 2) 모델 초기화
yolo = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3) 프레임별 처리
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
pbar = tqdm(total=total, desc="Processing frames")

frame_idx = 0
records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    annotated = frame.copy()

    # 매 프레임 rec 초기화
    rec = {"frame": frame_idx}

    # 3.1) YOLO 사람 박스 검출
    results = yolo(frame)[0]
    person_boxes = [b for b in results.boxes if int(b.cls) == 0]

    if person_boxes:
        # 최고 신뢰도 박스 선택
        best = max(person_boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)

        # 패딩
        pad = 20
        x1m, y1m = max(0, x1 - pad), max(0, y1 - pad)
        x2m, y2m = min(w, x2 + pad), min(h, y2 + pad)

        # ROI 크롭 및 MediaPipe 처리
        roi = frame[y1m:y2m, x1m:x2m]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            # ROI 위에 랜드마크 그리기
            annotated_roi = roi.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_roi,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
            annotated[y1m:y2m, x1m:x2m] = annotated_roi

            # keypoints 채우기
            for j in range(33):
                lm = res.pose_landmarks.landmark[j]
                ox = x1m + lm.x * (x2m - x1m)
                oy = y1m + lm.y * (y2m - y1m)
                rec[f"x{j}"] = float(ox)
                rec[f"y{j}"] = float(oy)
                rec[f"z{j}"] = float(lm.z)
                rec[f"v{j}"] = float(lm.visibility)
        else:
            # landmark 없으면 빈 배열
            for j in range(33):
                rec[f"x{j}"] = []
                rec[f"y{j}"] = []
                rec[f"z{j}"] = []
                rec[f"v{j}"] = []
    else:
        # 사람 못 찾으면 모든 keypoints 빈 배열
        for j in range(33):
            rec[f"x{j}"] = []
            rec[f"y{j}"] = []
            rec[f"z{j}"] = []
            rec[f"v{j}"] = []

    # records에 무조건 추가
    records.append(rec)

    # 3.3) 어노테이션 프레임 저장
    cv2.imwrite(os.path.join(IMAGE_DIR, f"frame_{frame_idx:06d}.jpg"), annotated)

    frame_idx += 1
    pbar.update(1)

pbar.close()
cap.release()
pose.close()

# 4) CSV & JSON 저장
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"✔ Saved {len(records)} records, frames and annotations.")