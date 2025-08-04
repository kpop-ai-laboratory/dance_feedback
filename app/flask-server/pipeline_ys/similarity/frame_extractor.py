# 테스트용 by 준혁

import subprocess
import os

# 1) 처리할 동영상 파일 목록
video_paths = [
    r'D:\python-yslee\workspace\dance_feedback\data_files\out-corr-files\attention_1_synced.mp4',
    r'D:\python-yslee\workspace\dance_feedback\data_files\out-corr-files\attention_2_synced.mp4'
]

for video_path in video_paths:
    base = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join('.', f'{base}_frames')
    os.makedirs(frames_dir, exist_ok=True)

    # 2) ffmpeg 로 일괄 추출
    #    PNG나 JPG 원하시는 포맷으로 확장자만 바꾸세요.
    subprocess.run([
        'ffmpeg', '-y',
        '-i', video_path,
        os.path.join(frames_dir, 'frame_%05d.png')
    ], check=True)

    print(f'✅ FFmpeg: "{video_path}" → frames in "{frames_dir}"')