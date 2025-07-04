## 📑 목차

데이터 준비 (Data Preparation)
스켈레톤 추출 (Skeleton Extraction)
유사도 모듈 (Similarity Module)
추가 개발 항목 (Next Steps)
라이선스 (License)

📂 폴더 구조

├─ data_preparation/             # 1단계: 데이터 준비
│   ├─ down_youtube.py           # YouTube 영상 다운로드
│   └─ corr_sync.py              # 영상 싱크 동기화
├─ skeleton_extraction/          # 2단계: 스켈레톤 추출
│   ├─ yolo_and_mediapipe.py     # 랜드마크(CSV/JSON) 추출 및 시각화
│   └─ imgs_to_video.py          # 프레임 → 비디오 재조합
├─ similarity_module/            # 3단계: 유사도 계산 및 피드백
│   ├─ main.py                   # 파이프라인 메인 스크립트
│   ├─ constants.py              # 관절 및 각도 정의
│   ├─ angle_utils.py            # 각도 계산 유틸
│   ├─ data_utils.py             # JSON → 배열 로딩
│   ├─ procrustes_utils.py       # Procrustes 거리 계산
│   ├─ trajectory_utils.py       # DTW 및 궤적 추출
│   ├─ similarity_utils.py       # 프레임별 유사도 계산
│   └─ feedback_utils.py         # 피드백 메시지 생성
└─ requirements.txt              # 필수 패키지 목록

1. 데이터 준비

1.1 YouTube 동영상 다운로드

down_youtube.py

import os
from yt_dlp import YoutubeDL

VIDEO_DIR    = r'C:\Users\human\Desktop\real_kpop\data'
PLAYLIST_URL = 'https://www.youtube.com/watch?v=XNObbV0AjvM'

os.makedirs(VIDEO_DIR, exist_ok=True)

ytdl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
    'outtmpl': os.path.join(VIDEO_DIR, '%(id)s.%(ext)s'),
    'ignoreerrors': True,
    'quiet': False,
    'noplaylist': True,
}

print("📥 다운로드 시작...")
with YoutubeDL(ytdl_opts) as ydl:
    ydl.download([PLAYLIST_URL])
print("✅ 다운로드 완료!")

설정: VIDEO_DIR, PLAYLIST_URL

옵션: MP4 + M4A 자동 병합

1.2 영상 싱크 동기화

corr_sync.py

import os
import subprocess
import numpy as np
import librosa
import cv2
from scipy.signal import correlate
from tqdm import tqdm

# 설정
DATA_DIR = './data-files'
SR       = 22050
OUT_DIR  = os.path.join(DATA_DIR, 'out-corr-files')
os.makedirs(OUT_DIR, exist_ok=True)

# 페어 검색
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.mp4')])
pairs = [(f, f.replace('_1.mp4', '_2.mp4')) for f in files if f.endswith('_1.mp4')]

# 동기화 함수
# ... 이하 생략 (코드 참고) ...

순서:

ffmpeg로 WAV 추출

librosa FFT 기반 크로스-상관으로 지연(lag) 계산

ffmpeg로 시작점 보정 및 자르기

영상+오디오 재결합

출력: data-files/out-corr-files

2. 스켈레톤 추출

2.1 랜드마크 추출

yolo_and_mediapipe.py

import os, logging, warnings
import cv2, json, csv
from ultralytics import YOLO
import mediapipe as mp
from tqdm import tqdm

# 설정
VIDEO_PATH = r'C:\Users\human\Desktop\...\last20s.mp4'
OUTPUT_DIR = r'C:\Users\human\Desktop\...\result3'

# MediaPipe Pose 초기화
# ... 이하 생략 (코드 참고) ...

결과: keypoints.csv, keypoints.json, 어노테이션된 프레임 이미지

2.2 이미지 → 비디오 재조합

imgs_to_video.py

import os, cv2

IMAGE_DIR    = r".../frames"
OUTPUT_VIDEO = r".../stitched_video.mp4"

# 이미지 목록 → VideoWriter 사용
# ... 이하 생략 (코드 참고) ...

결과: MP4 비디오 파일

3. 유사도 모듈

main.py --ref_json path/to/ref_keypoints.json \
         --user_json path/to/user_keypoints.json \
         --fps 30 --angle_report_thresh 10.0 --proc_thresh 0.1

JSON 로드 → 전처리

compute_frame_similarities로 프레임별 pose, move, final 점수 계산

aggregate_per_second로 초별 평균 유사도

identify_misaligned_joints로 어긋난 프레임/관절 식별

generate_frame_feedback로 텍스트 피드백 생성

similarity.json, feedback.json 저장

Next Steps

캡션 자동 생성

웹/GUI 대시보드 (Flask/React)

대량 배치 처리 및 멀티 유저 지원

