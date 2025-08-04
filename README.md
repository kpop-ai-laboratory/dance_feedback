
# Move AI Scan Project

K-POP 댄스 연습생과 아티스트를 위한 **실시간 동작 분석 및 정밀한 피드백**을 제공하는 인공지능 기반 솔루션입니다. <br />
연습생들이 안무의 잘못된 부분을 정확히 파악하여 효과적으로 개선할 수 있도록 도와줍니다.

<img width="800" height="450" alt="index" src="https://github.com/user-attachments/assets/5c8d556e-b29e-4270-a767-905a8691a3af" />

<br/> <br/> <br/> 

## 📌 주요 기능

### 🎥 영상 업로드

  <img width="800" height="450" alt="index" src="https://github.com/user-attachments/assets/379f4d4b-cb64-472c-8982-4049a1f0edc7" />


* 사용자 영상 및 기준 안무 영상 손쉬운 드래그 앤 드롭 업로드

<br/> <br/> 
### 🔊 오디오 싱크

* **Librosa 및 크로스-상관 분석**으로 음악과 영상의 정밀한 동기화

<br/> <br/> 
### 🦴 스켈레톤(키포인트) 추출

* **YOLOv8n**으로 사람 영역 탐지 → **MediaPipe Pose**로 정확한 관절 키포인트 33개 추출

<br/> <br/> 
### 📐 유사도 측정

* **DTW(동적 시간 워핑)**: 시계열 동작 패턴 비교
* **각도 분석**: 관절 상대 각도 차이 계산
* **Procrustes 분석**: 공간적 형태의 유사성 측정

<br/> <br/> 
### 📊 실시간 피드백 및 결과 확인

* **맞춤형 캡션 생성**과 **유사도 점수** 제공
* React 기반 직관적인 UI로 프레임별 분석 결과 제공

<br/> <br/> 
#### 🖼️ 결과 예시 화면

  <img width="800" height="400" alt="FrameFeedBackHistory" src="https://github.com/user-attachments/assets/8d98b850-b71c-4419-9586-9d27fd7b8820" />

<br/> <br/> 

## 🤖 인공지능 모델 구성

* **객체 탐지**: Ultralytics YOLOv8n (BBox 생성 및 Crop)
* **포즈 추정**: MediaPipe Pose (정밀한 관절 인식)
* **깊이 보정**: MiDaS & ResNet50 (실험적 적용)

<br/> <br/> 

## 📹 데이터 확보 및 전처리

* **유튜브 API 및 yt-dlp**로 고화질 안무 영상 확보
* **FFmpeg**를 통한 프레임 추출 및 영상 전처리

<br/> <br/> 
## 🌟 차별화 포인트

### 🎵 음악 리듬과 완벽한 동기화

* 음악의 박자와 안무 타이밍을 정밀 분석하여 정확한 피드백 제공

### 🚨 즉각적 실시간 AI 피드백

* MediaPipe와 DTW 결합으로 프레임 단위 동작 오류 즉시 감지
* 연습생이 문제를 빠르게 파악하고 수정 가능

<br/> <br/> 
### 📈 다차원 분석 시스템

* 각도, 형태, 시간축을 통합한 정밀 분석 및 종합 점수 제공

<br/> <br/> 
## 🌐 웹 애플리케이션 개발

* **Frontend**: React.js (Tailwind CSS)
* **Backend**: Flask (Python)
* **API 통신**: Axios를 통한 간편한 분석 요청 및 결과 송수신
* **사용자 중심 UI**: 직관적 타임라인 네비게이션, 문제 구간 자동 하이라이트

<br/> <br/> 
## 💻 시스템 아키텍처

```
[React Frontend]
    ↑        ↓
 [Axios]   [Flask API]
    ↑        ↓
[영상 업로드] → [FFmpeg/OpenCV] → [Librosa] → [AI 분석 파이프라인] → [분석 결과]
```


<br/> <br/> 
## 🚩 기대 효과 및 활용 방안

### 💃 안무 정확도 향상

* 미세한 동작 차이까지 정확히 분석해 전문적인 피드백 제공
* 주관적인 평가가 아닌 AI 기반의 객관적 피드백

### ⏳ 시간 및 비용 절감

* 실시간 자동 피드백으로 트레이너 의존도 감소
* 연습생 개인 스케줄 맞춤형 연습 환경 제공

### 📚 모션 데이터 관리

* 지속적인 안무 데이터 축적 및 맞춤형 트레이닝 데이터 구축

### ⚙️ 훈련 프로세스 최적화

* 데이터 기반 객관적 평가로 기획사 훈련 체계 효율화
* 정량적 발전 관리 및 효율적인 리소스 배분


<div align="center">

✨ **안무 실력을 빠르고 정확하게 성장시키는 Move AI Scan과 함께하세요!** ✨

</div>
