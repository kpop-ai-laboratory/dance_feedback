import os
from yt_dlp import YoutubeDL

VIDEO_DIR    = r'C:\Users\human\Desktop\real_kpop\data'
PLAYLIST_URL = 'https://www.youtube.com/watch?v=vMAn2KVbcCM'

os.makedirs(VIDEO_DIR, exist_ok=True)

ytdl_opts = {
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
    'outtmpl': os.path.join(VIDEO_DIR, '%(id)s.%(ext)s'),
    'ignoreerrors': True,
    'quiet': False,
    'noplaylist': True,
    'postprocessors': [
        {
            'key': 'FFmpegMerger',   # ← preferredformat 은 제거
        }
    ],
}

# 영상/음성 따로
# ytdl_opts = {
#     'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
#     'merge_output_format': 'mp4',      # ← 이 옵션만 남긴다
#     'ffmpeg_location': r'C:\tools\ffmpeg\bin',  # 이미 설치·등록된 경우 생략 가능
#     'outtmpl': os.path.join(VIDEO_DIR, '%(id)s.%(ext)s'),
#     'ignoreerrors': True,
#     'quiet': False,
#     'noplaylist': True,
#     # ▶ postprocessors 블록은 제거!
# }

print("📥 다운로드 시작...")
with YoutubeDL(ytdl_opts) as ydl:
    ydl.download([PLAYLIST_URL])
print("✅ 다운로드 완료!")
#설정: 출력 폴더(VIDEO_DIR), URL

#옵션: MP4 + M4A 자동 합치기, 플레이리스트 옵션
