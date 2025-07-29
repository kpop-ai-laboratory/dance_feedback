# flask-server/pipeline/extract_keypoints/sound_sync.py

import os
import subprocess
import numpy as np
import librosa
import cv2
import time
from scipy.signal import correlate  # FFT-based correlation for speed

def sync_pair(video1_path: str, video2_path: str, out_dir: str, sr: int = 22050):
    """
    두 비디오 파일을 오디오 크로스-상관으로 싱크한 뒤, 똑같은 길이로 잘라
    *_synced.mp4 두 개를 out_dir에 저장하고 그 경로를 반환합니다.

    Args:
      video1_path: 첫 번째 비디오의 전체 경로
      video2_path: 두 번째 비디오의 전체 경로
      out_dir: 결과물을 저장할 디렉토리
      sr: 오디오 샘플링 레이트
    Returns:
      (synced1_path, synced2_path)
    """

    # 1) 타이머 시작
    start_time = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # 1) 베이스 네임 추출
    name1 = os.path.splitext(os.path.basename(video1_path))[0]
    name2 = os.path.splitext(os.path.basename(video2_path))[0]

    # 2) 임시 WAV & 결과 비디오 경로
    wav1 = os.path.join(out_dir, f"{name1}.wav")
    wav2 = os.path.join(out_dir, f"{name2}.wav")
    synced1 = os.path.join(out_dir, f"{name1}_synced.mp4")
    synced2 = os.path.join(out_dir, f"{name2}_synced.mp4")

    # 3) 오디오만 추출
    for vid_path, wav_path in ((video1_path, wav1), (video2_path, wav2)):
        subprocess.run([
            "ffmpeg", "-y", "-i", vid_path,
            "-vn", "-ac", "1", "-ar", str(sr), wav_path
        ], check=True)

    # 4) 크로스-상관으로 지연(lag) 계산
    y1, _ = librosa.load(wav1, sr=sr)
    y2, _ = librosa.load(wav2, sr=sr)
    if len(y1) < len(y2):
        y1 = np.pad(y1, (0, len(y2) - len(y1)))
    else:
        y2 = np.pad(y2, (0, len(y1) - len(y2)))
    corr = correlate(y1, y2, mode="full", method="fft")
    lag = np.argmax(corr) - (len(y2) - 1)
    if lag > 0:
        start1, start2 = lag / sr, 0.0
    else:
        start1, start2 = 0.0, -lag / sr

    print(f"[Sync] {name1} vs {name2} → lag={lag} samples, start1={start1:.3f}s, start2={start2:.3f}s")

    # 5) 비디오 정보(프레임 수, FPS) 가져오기
    def get_info(path):
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frames, fps

    f1, fps1 = get_info(video1_path)
    f2, fps2 = get_info(video2_path)

    # 6) 동기화된 공통 길이 계산
    rem1 = (f1 - start1 * fps1) / fps1
    rem2 = (f2 - start2 * fps2) / fps2
    duration = min(rem1, rem2)

    # 7) 각 비디오·오디오 컷 & 재결합
    cuts = []
    for name, vid_path, ss in ((name1, video1_path, start1), (name2, video2_path, start2)):
        vid_cut = os.path.join(out_dir, f"{name}_cut.mp4")
        aud_cut = os.path.join(out_dir, f"{name}_cut.wav")
        cuts.append((vid_cut, aud_cut))

        # 비디오만 컷
        subprocess.run([
            "ffmpeg", "-y", "-i", vid_path,
            "-ss", f"{ss:.6f}", "-t", f"{duration:.6f}",
            "-r", str(fps1), "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18",
            "-an", vid_cut
        ], check=True)

        # 오디오만 컷
        subprocess.run([
            "ffmpeg", "-y", "-i", vid_path,
            "-ss", f"{ss:.6f}", "-t", f"{duration:.6f}",
            "-ac", "1", "-ar", str(sr),
            "-vn", aud_cut
        ], check=True)

    # 8) 컷된 비디오 + 오디오 재결합
    subprocess.run([
        "ffmpeg", "-y", "-i", cuts[0][0], "-i", cuts[0][1],
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", synced1
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", cuts[1][0], "-i", cuts[1][1],
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", synced2
    ], check=True)

     # 9) 완료 로그 & 실행 시간
    elapsed = time.time() - start_time
    print(f"[Merged] {synced1} , {synced2}")
    print(f"[Sync Complete] 총 소요 시간: {elapsed:.2f} 초")
    
    return synced1, synced2
