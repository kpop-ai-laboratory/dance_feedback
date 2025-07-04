K-pop 안무 피드백 파이프라인

이 저장소는 K-pop 안무 피드백을 자동화하기 위한 파이프라인을 제공합니다. 전체 과정은 세 가지 주요 단계로 나뉩니다:

데이터 준비 (Data Preparation)

YouTube 영상 다운로드: down_youtube.py

영상 싱크 동기화: corr_sync.py

스켈레톤 추출 (Skeleton Extraction)

YOLOv8 + MediaPipe Pose로 랜드마크 추출: yolo_and_mediapipe.py

이미지 프레임 → 비디오 재조합: imgs_to_video.py

유사도 모듈 (Similarity Module)

DTW 및 피드백 생성: main.py + 유사도 관련 모듈 7개 (constants.py, angle_utils.py, data_utils.py, procrustes_utils.py, trajectory_utils.py, similarity_utils.py, feedback_utils.py)

목차

구조

설치

1. 데이터 준비

2. 스켈레톤 추출

3. 유사도 모듈

추가 개발 항목

라이선스

구조

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

설치

# Python 3.8 이상
pip install -r requirements.txt

requirements.txt 예시:

yt-dlp
ffmpeg
librosa
opencv-python
scipy
tqdm
ultralytics
mediapipe
numpy

1. 데이터 준비

1.1 YouTube 영상 다운로드

python data_preparation/down_youtube.py

VIDEO_DIR와 PLAYLIST_URL 환경 변수 수정

MP4+M4A 자동 병합 옵션 포함

1.2 영상 싱크 동기화

python data_preparation/corr_sync.py

ffmpeg로 WAV 추출

librosa FFT 기반 크로스-상관으로 지연(lag) 계산

ffmpeg로 시작 지점 보정 및 자르기

영상+오디오 재결합

출력: data-files/out-corr-files

2. 스켈레톤 추출

2.1 랜드마크 추출

python skeleton_extraction/yolo_and_mediapipe.py

VIDEO_PATH와 OUTPUT_DIR 설정

결과: keypoints.csv, keypoints.json, 어노테이션 프레임 이미지

2.2 이미지 → 비디오 재조합

python skeleton_extraction/imgs_to_video.py

IMAGE_DIR, OUTPUT_VIDEO 설정

3. 유사도 모듈

python similarity_module/main.py \
  --ref_json path/to/ref_keypoints.json \
  --user_json path/to/user_keypoints.json \
  --fps 30 \
  --angle_report_thresh 10.0 \
  --proc_thresh 0.1

원본 JSON 로드 및 전처리

compute_frame_similarities로 각 프레임 pose, move, final 점수 계산

aggregate_per_second로 초별 평균 유사도

identify_misaligned_joints로 기준 초과 관절 식별

generate_frame_feedback로 픽셀 단위 피드백 메시지 생성

similarity.json, feedback.json 출력

추가 개발 항목

캡션 자동 생성

웹/GUI 대시보드 통합 (Flask/React)

배치 처리 및 멀티 사용자 지원
