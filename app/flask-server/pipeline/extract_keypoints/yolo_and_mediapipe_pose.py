# flask-server/pipeline/extract_keypoints/yolo_and_mediapipe_pose.py

import os
import logging
import warnings
import cv2
import json
import csv
from ultralytics import YOLO
import mediapipe as mp
from tqdm import tqdm

def extract_keypoints(video_path: str, output_dir: str):
    """
    video_path: 싱크된 동영상 경로
    output_dir: keypoints CSV/JSON, annotated frames를 저장할 디렉토리
    Returns: (csv_path, json_path, frames_dir)
    """
    # 1) 출력 폴더 준비
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    csv_path   = os.path.join(output_dir, "keypoints.csv")
    json_path  = os.path.join(output_dir, "keypoints.json")
    os.makedirs(frames_dir, exist_ok=True)

    # 2) 로그/경고 숨기기
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # 3) 모델 로드
    yolo = YOLO("yolov8n.pt")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 4) 프레임별 처리
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total, desc="Extracting keypoints")

    records = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        annotated = frame.copy()
        rec = {"frame": frame_idx}

        # 4.1) YOLO로 사람 박스 찾기
        results = yolo(frame)[0]
        person_boxes = [b for b in results.boxes if int(b.cls) == 0]
        if person_boxes:
            best = max(person_boxes, key=lambda b: float(b.conf))
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)

            pad = 20
            x1m, y1m = max(0, x1 - pad), max(0, y1 - pad)
            x2m, y2m = min(w, x2 + pad), min(h, y2 + pad)

            roi = frame[y1m:y2m, x1m:x2m]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                # 어노테이션
                annotated_roi = roi.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_roi,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                annotated[y1m:y2m, x1m:x2m] = annotated_roi

                # keypoint 기록
                for j, lm in enumerate(res.pose_landmarks.landmark):
                    ox = x1m + lm.x * (x2m - x1m)
                    oy = y1m + lm.y * (y2m - y1m)
                    rec[f"x{j}"] = float(ox)
                    rec[f"y{j}"] = float(oy)
                    rec[f"z{j}"] = float(lm.z)
                    rec[f"v{j}"] = float(lm.visibility)
            else:
                for j in range(33):
                    rec[f"x{j}"] = []
                    rec[f"y{j}"] = []
                    rec[f"z{j}"] = []
                    rec[f"v{j}"] = []
        else:
            for j in range(33):
                rec[f"x{j}"] = []
                rec[f"y{j}"] = []
                rec[f"z{j}"] = []
                rec[f"v{j}"] = []

        records.append(rec)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg"), annotated)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    pose.close()

    # 5) CSV & JSON 저장
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return csv_path, json_path, frames_dir
