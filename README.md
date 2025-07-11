# Kpop Dance Feedback Project PipeLine

#### 📖 프로젝트 개요

엔터테인먼트 내에서 안무 선생님의 동작을 고정밀도 실시간 모션 캡처하고, <br>
이를 연습생 및 연예인에게 즉각적으로 제공하는 피드백 시스템을 개발합니다. <br>
음악의 리듬과 안무 동작을 분석하여 자세, 각도, 동작 유사도를 수치화하고, <br>
자동 캡션으로 구체적인 교정 포인트를 제안합니다. <br>

#### 📑 목차

##### 1. 데이터 준비<br><br>
2개의 동영상 또는 youtube 영상 추출 <br>
youtube 영상 추출 - <etc/ down_youtube.py> <br>

##### 2. 영상 음성 싱크 동기화<br><br>
2개의 동영상의 음성 파형을 기준으로 짧은 영상 길이에 맞게 자동 동기화<br>
<extract_keypoints / sound_sync.py><br>

##### 3. 스켈레톤 추출<br><br>
2개의 동영상에서 프레임을 추출 후 yolov8n을 통해서 사람 영역 bbox로 인식 <br>
-> media pipe로 key points 추출 후 원래 이미지 사이즈로 북구<br>
<yolo_and_media_pose.py>
##### 4. 유사도 모듈<br><br>
Procrustes 정렬, 각도 비교, 궤적 분석 등을 통해 동작 유사도 산출<br>
기준 안무(선생님) vs 연습생 영상 비교 후 자동 피드백 캡션 생성 <similarity/><br>

##### 5. [to-be] Caption 생성 및 웹구현<br><br>
자동 교정 문구 생성 : VLM gpt-4o 사용 or VLM 모델 학습<br>

## 📂 Folder Structure

```text
etc/
├─blender_2d.py
├─blender_3d.py
├─down_youtube.py
├─img_visualize.py
├─multi_play.py
└─only_yolo.py

extract_keypoints/
├─ sound_sync.py
├─ yolo_and_mediapipe_pose.py
└─ img_to_video.py

similarity/
├─ main.py
├─ constants.py
├─ angle_utils.py
├─ data_utils.py
├─ procrustes_utils.py
├─ trajectory_utils.py
├─ similarity_utils.py
└─ feedback_utils.py
requirements.txt
````

### 기술 스택

AI·모델: MediaPipe Pose, PyTorch<br>
영상 처리: OpenCV, NumPy<br>
데이터 처리: pandas<br>
스크립팅: Python<br>
영상 다운로드: yt-dlp<br>
(추후) : VLM 모델 GPT-4o / VLM 모델<br>

### 기대 효과

연습생 및 연예인의 안무 정확도 및 퍼포먼스 향상<br>
실시간 피드백 제공으로 학습 효율성 증대<br>
안무 데이터의 체계적 관리 및 분석 인사이트 확보<br>

