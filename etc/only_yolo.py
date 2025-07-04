# Yolo 만 이용해서 크롭 이미지, bbox 좌표, bbox 친 이미지 추출

# video_frame_extraction_and_bbox.py
import os
import cv2
import json
import argparse
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="영상에서 프레임 추출 → YOLO 객체 검출 → 결과 저장")
    parser.add_argument('--video', '-i', required=True, help="입력 비디오 파일 경로")
    parser.add_argument('--output', '-o', required=True, help="결과물 저장 디렉토리")
    parser.add_argument('--conf', '-c', type=float, default=0.25, help="검출 신뢰도 임계값")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) 출력 디렉토리 구조 생성
    frame_dir = os.path.join(args.output, 'frames')
    ann_dir   = os.path.join(args.output, 'annotated_frames')
    crop_dir  = os.path.join(args.output, 'crops')
    meta_dir  = os.path.join(args.output, 'metadata')
    for d in (frame_dir, ann_dir, crop_dir, meta_dir):
        os.makedirs(d, exist_ok=True)

    # 2) YOLO 모델 로드
    model = YOLO("yolov8n.pt")

    # 3) 비디오 열기 및 전체 프레임 수 가져오기
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] 비디오를 열 수 없습니다: {args.video}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"INFO: 열린 비디오 → 총 프레임 수 = {total_frames}")

    # 4) 프레임 단위 반복 처리 (tqdm 으로 진행 상황 표시)
    for idx in tqdm(range(total_frames), desc='프레임 처리'):
        ret, frame = cap.read()
        if not ret:
            break

        # 4.1) 원본 프레임 저장
        fname = f"{idx:06d}.jpg"
        cv2.imwrite(os.path.join(frame_dir, fname), frame)

        # 4.2) YOLO 객체 검출
        results = model(frame, conf=args.conf)[0]

        # 4.3) 메타데이터 구성 ------- 0.5 밑의 박스를 제외하고 돌아가는 코드
        # bboxes = []
        # for i, box in enumerate(results.boxes):
        #     cls   = int(box.cls.cpu().numpy())                # 클래스 ID
        #     conf  = float(box.conf.cpu().numpy())             # 신뢰도

        #     if conf < 0.5:
        #         continue  # 신뢰도 0.5 미만은 무시            

        #     x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())    # 바운딩박스 좌표

        #     # 메타 정보 리스트에 추가
        #     bboxes.append({
        #         'class': cls,
        #         'conf': conf,
        #         'bbox': [x1, y1, x2, y2]
        #     })

        #     # 4.4) 크롭 이미지 저장
        #     crop = frame[y1:y2, x1:x2]
        #     crop_name = f"{idx:06d}_{i:02d}.jpg"
        #     cv2.imwrite(os.path.join(crop_dir, crop_name), crop)

        #     # 4.5) 어노테이션 프레임에 bbox, 레이블 그리기
        #     cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        #     label = f"{cls}:{conf:.2f}"
        #     cv2.putText(frame, label, (x1, y1-5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 4.3) 메타데이터 구성 - 최고 신뢰도 박스 1개만 저장
        bboxes = []
        if len(results.boxes) > 0:
            best_box = max(results.boxes, key=lambda b: b.conf)
            cls   = int(best_box.cls.cpu().numpy())
            conf  = float(best_box.conf.cpu().numpy())
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].cpu())

            # 메타 정보 리스트에 추가
            bboxes.append({
                'class': cls,
                'conf': conf,
                'bbox': [x1, y1, x2, y2]
            })

            # 4.4) 크롭 이미지 저장
            crop = frame[y1:y2, x1:x2]
            crop_name = f"{idx:06d}_00.jpg"
            cv2.imwrite(os.path.join(crop_dir, crop_name), crop)

            # 4.5) 어노테이션 프레임에 bbox, 레이블 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls}:{conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
                # 박스가 없더라도 빈 리스트 포함해서 저장
        meta_path = os.path.join(meta_dir, f"{idx:06d}.json")
        with open(meta_path, 'w', encoding='utf-8') as fp:
            json.dump({
                'frame': idx,
                'bboxes': bboxes  # 없으면 빈 리스트
            }, fp, ensure_ascii=False, indent=2)


        # 4.6) 어노테이션 프레임 저장
        cv2.imwrite(os.path.join(ann_dir, fname), frame)

        # 4.7) JSON 메타데이터 저장
        meta_path = os.path.join(meta_dir, f"{idx:06d}.json")
        with open(meta_path, 'w', encoding='utf-8') as fp:
            json.dump({
                'frame': idx,
                'bboxes': bboxes
            }, fp, ensure_ascii=False, indent=2)

    cap.release()
    print("모든 작업 완료!")

if __name__ == '__main__':
    main()

# 실행 코드
# python bbox_pose.py --video data-files/nextlevel_1_synced.mp4 --output output_dir --conf 0.5
