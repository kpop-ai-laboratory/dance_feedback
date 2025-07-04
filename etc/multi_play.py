from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.compositing.CompositeVideoClip import clips_array

# 1) 비디오 & 오디오 로드
clip1 = VideoFileClip(r"C:\Users\human\Desktop\kpop_project\data-files\result1\nextlevel_1_synced.mp4")
clip2 = VideoFileClip(r"C:\Users\human\Desktop\kpop_project\data-files\result2\nextlevel_2_synced.mp4")
audio = AudioFileClip(r"C:\Users\human\Desktop\kpop_project\data-files\video\nextlevel_1.wav")

# 2) 세로(stacked)로 합치기
final = clips_array([[clip1],
                     [clip2]]).with_audio(audio)

# 3) 결과물 저장
final.write_videofile(
    "dance_comparison_with_audio.mp4",
    codec="libx264",
    audio_codec="aac",
    fps=clip1.fps, 
    remove_temp=True
)

# GPU 가속

# final.write_videofile(
#     "dance_comparison_with_audio.mp4",
#     codec="h264_nvenc",             # NVENC 인코더
#     audio_codec="aac",
#     fps=clip1.fps,
#     preset="fast",                  # NVENC 프리셋 (옵션: slow, medium, fast, p1..p7)
#     threads=0,                      # ffmpeg에 자동 스레드 할당
#     ffmpeg_params=[
#         "-hwaccel", "cuda",             # CUDA 디코드 가속
#         "-hwaccel_device", "0",         # 사용할 GPU 인덱스
#         "-hwaccel_output_format", "cuda" 
#     ],
#     remove_temp=True
# )
