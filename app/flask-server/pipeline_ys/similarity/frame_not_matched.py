# 테스트용 코드 by 준혁


import os
import subprocess
import numpy as np
import librosa
import cv2
from scipy.signal import correlate  # FFT-based correlation for speed
from tqdm import tqdm

# ─── 설정 ───────────────────────────────────────────────
data_dir = r'D:\python-yslee\workspace\dance_feedback\data_files'
sr       = 22050   # 오디오 샘플링 레이트
out_dir  = os.path.join(data_dir)
# 출력 디렉토리 생성
os.makedirs(out_dir, exist_ok=True)

# ─── 1) 데이터 파일에서 *_1.mp4 / *_2.mp4 페어 검색 ────
files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith('.mp4'))
pairs = []
for f in files:
    if f.endswith('_1.mp4'):
        mate = f[:-6] + '_2.mp4'
        if mate in files:
            pairs.append((f, mate))

print(f'▶ 총 {len(pairs)} 쌍 발견됨\n')

# ─── 동기화 처리 함수 ────────────────────────────────────
def sync_pair(video1, video2):
    name1, name2 = os.path.splitext(video1)[0], os.path.splitext(video2)[0]
    wav1 = os.path.join(data_dir, name1 + '.wav')
    wav2 = os.path.join(data_dir, name2 + '.wav')
    final1 = os.path.join(out_dir, name1 + '_synced.mp4')
    final2 = os.path.join(out_dir, name2 + '_synced.mp4')

    # 1) WAV 추출
    for vid, wav in ((video1, wav1), (video2, wav2)):
        subprocess.run([
            'ffmpeg', '-y', '-i', os.path.join(data_dir, vid),
            '-vn', '-ac', '1', '-ar', str(sr), wav
        ], check=True)

    # 2) 크로스-상관(cross-correlation)으로 오프셋 계산
    y1, _ = librosa.load(wav1, sr=sr)
    y2, _ = librosa.load(wav2, sr=sr)
    # 길이 맞추기: 짧은 쪽 패딩
    if len(y1) < len(y2):
        y1 = np.pad(y1, (0, len(y2) - len(y1)))
    else:
        y2 = np.pad(y2, (0, len(y1) - len(y2)))
    # FFT 기반 상호상관 (빠른 수행)
    corr = correlate(y1, y2, mode='full', method='fft')
    lag = np.argmax(corr) - (len(y2) - 1)
    if lag > 0:
        start1, start2 = lag / sr, 0.0
    else:
        start1, start2 = 0.0, -lag / sr
    print(f'[Sync] {name1}, {name2} → lag={lag} samples, start1={start1:.3f}s, start2={start2:.3f}s')

    # 3) 영상 정보 (프레임수, FPS)
    def get_info(path):
        cap = cv2.VideoCapture(os.path.join(data_dir, path))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frames, fps

    f1, fps1 = get_info(video1)
    f2, fps2 = get_info(video2)

    # 4) 동기화된 길이 계산
    rem1 = (f1 - start1 * fps1) / fps1
    rem2 = (f2 - start2 * fps2) / fps2
    duration = min(rem1, rem2)

    # 임시 파일 경로
    cuts = []
    for name, vid, ss in ((name1, video1, start1), (name2, video2, start2)):
        vid_cut = os.path.join(out_dir, name + '_cut.mp4')
        aud_cut = os.path.join(out_dir, name + '_cut.wav')
        cuts.append((vid_cut, aud_cut, vid, ss))
        # 비디오만 컷
        subprocess.run([
            'ffmpeg', '-y', '-i', os.path.join(data_dir, vid),
            '-ss', f'{ss:.6f}', '-t', f'{duration:.6f}',
            '-r', str(fps1), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
            '-an', vid_cut
        ], check=True)
        # 오디오만 컷
        subprocess.run([
            'ffmpeg', '-y', '-i', os.path.join(data_dir, vid),
            '-ss', f'{ss:.6f}', '-t', f'{duration:.6f}',
            '-ac', '1', '-ar', str(sr), '-vn', aud_cut
        ], check=True)

    def get_frame_count(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

    # 두 컷 영상의 프레임 수 가져오기
    frames_cut1 = get_frame_count(cuts[0][0])
    frames_cut2 = get_frame_count(cuts[1][0])

    # 두 영상의 최소 프레임 수로 맞추기
    min_frames = min(frames_cut1, frames_cut2)

    # 두 컷 영상의 프레임 수가 같지 않다면, 적은 쪽 기준으로 다시 자름 (추가할 부분)
    if frames_cut1 != frames_cut2:
        print(f'⚠️ Frame mismatch: {frames_cut1} vs {frames_cut2}. Adjusting frames...')

        adjusted_duration = min_frames / fps1  # 최소 프레임 수를 기준으로 재조정

        for vid_cut, aud_cut, vid_orig, ss in cuts:
            # 비디오만 다시 컷 (프레임 수 조정)
            subprocess.run([
                'ffmpeg', '-y', '-i', os.path.join(data_dir, vid_orig),
                '-ss', f'{ss:.6f}', '-t', f'{adjusted_duration:.6f}',
                '-r', str(fps1), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
                '-an', vid_cut
            ], check=True)

            # 오디오도 다시 컷 (프레임 수 조정)
            subprocess.run([
                'ffmpeg', '-y', '-i', os.path.join(data_dir, vid_orig),
                '-ss', f'{ss:.6f}', '-t', f'{adjusted_duration:.6f}',
                '-ac', '1', '-ar', str(sr), '-vn', aud_cut
            ], check=True)

        print(f'✅ Frames aligned to {min_frames} frames.\n')

    # 5) 비디오+오디오 재결합
    for vid_cut, aud_cut, final in ((cuts[0][0], cuts[0][1], final1), (cuts[1][0], cuts[1][1], final2)):
        subprocess.run([
            'ffmpeg', '-y', '-i', vid_cut, '-i', aud_cut,
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', final
        ], check=True)
    print(f'✅ Merged: {final1}, {final2}\n')

# ─── 모든 쌍 동기화 (진행바 표시) ─────────────────────────
for video1, video2 in tqdm(pairs, desc='Processing pairs'):
    sync_pair(video1, video2)