# flask-server/pipeline_ys/extract_keypoints/sound_sync.py

import os
import subprocess
import numpy as np
import librosa
import cv2
from scipy.signal import correlate  # FFT 기반 상호 상관 연산

# ─── 기본 설정 ─────────────────────────────────────────────────
# 데이터 디렉토리 및 샘플링 레이트
DATA_DIR = './data'
SR = 22050  # 오디오 샘플링 레이트
# 출력 디렉토리 생성 (모듈 임포트 시에도 존재해야 함)
OUT_DIR = os.path.join(DATA_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 주요 함수: 두 비디오 파일 오디오 싱크 수행 ─────────────────────
def sync_pair(video1_path: str, video2_path: str, out_dir: str = DATA_DIR, sr: int = SR):
    """
    두 비디오 파일을 오디오 크로스-상관으로 동기화하고, 동일 길이로 잘라서
    '*_synced.mp4' 파일을 out_dir에 생성 후 경로를 반환합니다.

    Args:
      video1_path: 첫 번째 비디오 파일 경로
      video2_path: 두 번째 비디오 파일 경로
      out_dir: 결과물을 저장할 디렉토리
      sr: 오디오 샘플링 레이트
    Returns:
      (synced1_path, synced2_path)
    """
    # 1) 입력 파일 이름과 확장자 분리
    name1 = os.path.splitext(os.path.basename(video1_path))[0]
    name2 = os.path.splitext(os.path.basename(video2_path))[0]

    # 2) WAV 추출 및 싱크 결과물 경로 정의
    wav1 = os.path.join(out_dir, f"{name1}.wav")
    wav2 = os.path.join(out_dir, f"{name2}.wav")
    synced1 = os.path.join(out_dir, f"{name1}_synced.mp4")
    synced2 = os.path.join(out_dir, f"{name2}_synced.mp4")

    # 3) FFmpeg로 비디오에서 오디오만 WAV로 추출
    for vid_path, wav_path in ((video1_path, wav1), (video2_path, wav2)):
        subprocess.run([
            'ffmpeg', '-y', '-i', vid_path,
            '-vn', '-ac', '1', '-ar', str(sr), wav_path
        ], check=True)

    # 4) librosa로 WAV 로드 후 상호 상관 계산
    y1, _ = librosa.load(wav1, sr=sr)
    y2, _ = librosa.load(wav2, sr=sr)
    # 길이 맞추기: 짧은 쪽을 패딩
    if len(y1) < len(y2):
        y1 = np.pad(y1, (0, len(y2) - len(y1)))
    else:
        y2 = np.pad(y2, (0, len(y1) - len(y2)))
    corr = correlate(y1, y2, mode='full', method='fft')
    lag = np.argmax(corr) - (len(y2) - 1)

    # 5) 프레임 기반 시작 시간 계산 (초 단위)
    if lag > 0:
        start1, start2 = lag / sr, 0.0
    else:
        start1, start2 = 0.0, -lag / sr
    print(f"[Sync] {name1} vs {name2} → lag={lag} samples, "
          f"start1={start1:.3f}s, start2={start2:.3f}s")

    # 6) 비디오 메타정보(프레임 수, FPS) 가져오기 함수 정의
    def get_info(path):
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frames, fps

    f1, fps1 = get_info(video1_path)
    f2, fps2 = get_info(video2_path)

    # 7) 두 영상의 공통 길이 계산
    rem1 = (f1 - start1 * fps1) / fps1
    rem2 = (f2 - start2 * fps2) / fps2
    duration = min(rem1, rem2)

    # 8) 시작 시간에 맞춰 비디오/오디오 컷 & 재결합
    cuts = []
    for name, vid_path, ss in ((name1, video1_path, start1), (name2, video2_path, start2)):
        vid_cut = os.path.join(out_dir, f"{name}_cut.mp4")
        aud_cut = os.path.join(out_dir, f"{name}_cut.wav")
        cuts.append((vid_cut, aud_cut))

        # 비디오 컷
        subprocess.run([
            'ffmpeg', '-y', '-i', vid_path,
            '-ss', f'{ss:.6f}', '-t', f'{duration:.6f}',
            '-r', str(fps1), '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '18', '-an', vid_cut
        ], check=True)

        # 오디오 컷
        subprocess.run([
            'ffmpeg', '-y', '-i', vid_path,
            '-ss', f'{ss:.6f}', '-t', f'{duration:.6f}',
            '-ac', '1', '-ar', str(sr), '-vn', aud_cut
        ], check=True)

    # 9) 컷된 영상 + 오디오 재결합
    for (vid_cut, aud_cut), final in zip(cuts, (synced1, synced2)):
        subprocess.run([
            'ffmpeg', '-y', '-i', vid_cut, '-i', aud_cut,
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', final
        ], check=True)

    print(f"✅ Merged: {synced1}, {synced2}")
    return synced1, synced2


# ─── 스크립트 실행용 진입점: import 시 실행되지 않음 ─────────────────
def main():
    """
    data_dir 경로에서 *_1.mp4/_2.mp4 쌍을 찾아
    sync_pair 함수를 일괄 실행합니다.
    """
    # tqdm은 스크립트 실행 시에만 필요
    from tqdm import tqdm

    # 데이터 디렉토리의 모든 mp4 파일 목록 생성
    files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith('.mp4'))
    pairs = []
    for f in files:
        if f.endswith('_1.mp4'):
            mate = f[:-6] + '_2.mp4'
            if mate in files:
                pairs.append((f, mate))

    print(f"▶ 총 {len(pairs)} 쌍 발견됨\n")
    # tqdm을 사용하여 진행바 표시
    for v1, v2 in tqdm(pairs, desc='Processing pairs'):
        sync_pair(
            os.path.join(DATA_DIR, v1),
            os.path.join(DATA_DIR, v2),
            DATA_DIR
        )


if __name__ == '__main__':
    main()


# import os
# import subprocess
# import numpy as np
# import librosa
# import cv2
# from scipy.signal import correlate  # FFT-based correlation for speed
# from tqdm import tqdm

# # ─── 설정 ───────────────────────────────────────────────
# data_dir = './data'
# sr       = 22050   # 오디오 샘플링 레이트
# out_dir  = os.path.join(data_dir)
# # 출력 디렉토리 생성
# os.makedirs(out_dir, exist_ok=True)

# # ─── 1) 데이터 파일에서 *_1.mp4 / *_2.mp4 페어 검색 ────
# files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith('.mp4'))
# pairs = []
# for f in files:
#     if f.endswith('_1.mp4'):
#         mate = f[:-6] + '_2.mp4'
#         if mate in files:
#             pairs.append((f, mate))

# print(f'▶ 총 {len(pairs)} 쌍 발견됨\n')

# # ─── 동기화 처리 함수 ────────────────────────────────────
# def sync_pair(video1, video2):
#     name1, name2 = os.path.splitext(video1)[0], os.path.splitext(video2)[0]
#     wav1 = os.path.join(data_dir, name1 + '.wav')
#     wav2 = os.path.join(data_dir, name2 + '.wav')
#     final1 = os.path.join(out_dir, name1 + '_synced.mp4')
#     final2 = os.path.join(out_dir, name2 + '_synced.mp4')

#     # 1) WAV 추출
#     for vid, wav in ((video1, wav1), (video2, wav2)):
#         subprocess.run([
#             'ffmpeg', '-y', '-i', os.path.join(data_dir, vid),
#             '-vn', '-ac', '1', '-ar', str(sr), wav
#         ], check=True)

#     # 2) 크로스-상관(cross-correlation)으로 오프셋 계산
#     y1, _ = librosa.load(wav1, sr=sr)
#     y2, _ = librosa.load(wav2, sr=sr)
#     # 길이 맞추기: 짧은 쪽 패딩
#     if len(y1) < len(y2):
#         y1 = np.pad(y1, (0, len(y2) - len(y1)))
#     else:
#         y2 = np.pad(y2, (0, len(y1) - len(y2)))
#     # FFT 기반 상호상관 (빠른 수행)
#     corr = correlate(y1, y2, mode='full', method='fft')
#     lag = np.argmax(corr) - (len(y2) - 1)
#     if lag > 0:
#         start1, start2 = lag / sr, 0.0
#     else:
#         start1, start2 = 0.0, -lag / sr
#     print(f'[Sync] {name1}, {name2} → lag={lag} samples, start1={start1:.3f}s, start2={start2:.3f}s')

#     # 3) 영상 정보 (프레임수, FPS)
#     def get_info(path):
#         cap = cv2.VideoCapture(os.path.join(data_dir, path))
#         frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         cap.release()
#         return frames, fps

#     f1, fps1 = get_info(video1)
#     f2, fps2 = get_info(video2)

#     # 4) 동기화된 길이 계산
#     rem1 = (f1 - start1 * fps1) / fps1
#     rem2 = (f2 - start2 * fps2) / fps2
#     duration = min(rem1, rem2)

#     # 임시 파일 경로
#     cuts = []
#     for name, vid, ss in ((name1, video1, start1), (name2, video2, start2)):
#         vid_cut = os.path.join(out_dir, name + '_cut.mp4')
#         aud_cut = os.path.join(out_dir, name + '_cut.wav')
#         cuts.append((vid_cut, aud_cut, vid, ss))
#         # 비디오만 컷
#         subprocess.run([
#             'ffmpeg', '-y', '-i', os.path.join(data_dir, vid),
#             '-ss', f'{ss:.6f}', '-t', f'{duration:.6f}',
#             '-r', str(fps1), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
#             '-an', vid_cut
#         ], check=True)
#         # 오디오만 컷
#         subprocess.run([
#             'ffmpeg', '-y', '-i', os.path.join(data_dir, vid),
#             '-ss', f'{ss:.6f}', '-t', f'{duration:.6f}',
#             '-ac', '1', '-ar', str(sr), '-vn', aud_cut
#         ], check=True)

#     # 5) 비디오+오디오 재결합
#     for vid_cut, aud_cut, final in ((cuts[0][0], cuts[0][1], final1), (cuts[1][0], cuts[1][1], final2)):
#         subprocess.run([
#             'ffmpeg', '-y', '-i', vid_cut, '-i', aud_cut,
#             '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', final
#         ], check=True)
#     print(f'✅ Merged: {final1}, {final2}\n')

# # ─── 모든 쌍 동기화 (진행바 표시) ─────────────────────────
# for video1, video2 in tqdm(pairs, desc='Processing pairs'):
#     sync_pair(video1, video2)

