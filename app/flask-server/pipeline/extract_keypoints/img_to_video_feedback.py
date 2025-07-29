# # flask-server/pipeline/extract_keypoints/img_to_video_feedback.py

# import os
# import json
# import subprocess
# import cv2
# import numpy as np
# from tqdm import tqdm
# from PIL import Image, ImageDraw, ImageFont

# def render_feedback_video(
#     feedback_json: str,
#     teacher_frames: str,
#     student_frames: str,
#     out_video_path: str,
#     fps: int = 30,
#     font_path: str = r"C:\Windows\Fonts\malgun.ttf",
#     font_size: int = 24
# ):
#     """
#     feedback_json: {frame_idx: [메시지, ...], ...}
#     teacher_frames/student_frames: 두 영상의 프레임 이미지 폴더
#     out_video_path: 최종 비디오(.mp4) 경로
#     """
#     # # 1) 피드백 불러오기
#     # with open(feedback_json, 'r', encoding='utf-8') as f:
#     #     raw = json.load(f)

#     # # 2) 프레임 수 결정
#     total = min(len(os.listdir(teacher_frames)), len(os.listdir(student_frames)))
#     print("img to video :",teacher_frames)
#     print("img to video :",student_frames)

#     # 3) 캔버스용 임시 폴더
#     canvas_dir = os.path.join(os.path.dirname(out_video_path), "canvas_frames")
#     os.makedirs(canvas_dir, exist_ok=True)

#     # 4) 한글 폰트
#     font = ImageFont.truetype(font_path, font_size)

#     # 5) 프레임별 렌더링
#     for i in tqdm(range(total), desc="Rendering feedback frames"):
#         msgs = raw.get(str(i), [])
#         t_img = cv2.imread(os.path.join(teacher_frames, f"frame_{i:06d}.jpg"))
#         s_img = cv2.imread(os.path.join(student_frames, f"frame_{i:06d}.jpg"))
#         if t_img is None or s_img is None:
#             continue
#         h, w = t_img.shape[:2]
#         s_img = cv2.resize(s_img, (w, h))
#         canvas = np.zeros((h*2, w, 3), dtype=np.uint8)
#         canvas[:h] = t_img
#         canvas[h:] = s_img

#         if msgs:
#             pil = Image.fromarray(canvas)
#             draw = ImageDraw.Draw(pil)
#             padding_x, padding_y = 15, 15
#             line_spacing = 6

#             # 텍스트 박스 크기 계산
#             sizes = [draw.textbbox((0,0), m, font=font) for m in msgs]
#             widths = [s[2]-s[0] for s in sizes]
#             heights= [s[3]-s[1] for s in sizes]
#             box_w = max(widths)+padding_x*2.5
#             box_h = sum(heights)+padding_y*2.5 + line_spacing*(len(msgs)-1)

#             x0 = (w - box_w)//2
#             y0 = int(h*0.6)
#             draw.rectangle([x0, y0, x0+box_w, y0+box_h], fill=(255,255,255))

#             y = y0+padding_y
#             for m,hgt in zip(msgs, heights):
#                 draw.text((x0+padding_x, y), m, font=font, fill=(0,0,0))
#                 y += hgt + line_spacing

#             canvas = np.array(pil)

#         out_path = os.path.join(canvas_dir, f"frame_{i:06d}.jpg")
#         cv2.imwrite(out_path, canvas)

#     # 6) ffmpeg로 비디오 생성
#     subprocess.run([
#         "ffmpeg", "-y",
#         "-framerate", str(fps),
#         "-i", os.path.join(canvas_dir, "frame_%06d.jpg"),
#         "-c:v", "libx264", "-pix_fmt", "yuv420p",
#         out_video_path
#     ], check=True)

#     return out_video_path


import os
import subprocess
import cv2
import numpy as np
from tqdm import tqdm

def render_feedback_video(
    feedback_json: str,         # (사용 안 함)
    teacher_frames: str,        # teacher 프레임이 있는 폴더
    student_frames: str,        # student 프레임이 있는 폴더
    out_video_path: str,
    fps: int = 30,
):
    """
    순수하게 teacher_frames 와 student_frames 를 세로로 쌓아서 비디오로 만듭니다.
    feedback_json 은 더 이상 사용되지 않습니다.
    """
    # 1) 폴더들에 들어있는 이미지 파일 개수로 총 프레임 수 결정
    t_list = sorted(os.listdir(teacher_frames))
    s_list = sorted(os.listdir(student_frames))
    total = min(len(t_list), len(s_list))

    # 2) 캔버스용 임시 폴더 생성
    canvas_dir = os.path.join(os.path.dirname(out_video_path), "canvas_frames")
    os.makedirs(canvas_dir, exist_ok=True)

    # 3) 프레임별 합치기
    for i in tqdm(range(total), desc="Merging frames"):
        t_path = os.path.join(teacher_frames,    t_list[i])
        s_path = os.path.join(student_frames,    s_list[i])
        t_img = cv2.imread(t_path)
        s_img = cv2.imread(s_path)
        if t_img is None or s_img is None:
            continue

        # 두 이미지 크기 맞추기 (너비/높이 같게)
        h, w = t_img.shape[:2]
        s_img = cv2.resize(s_img, (w, h))

        # 세로로 쌓기
        canvas = np.vstack([t_img, s_img])

        out_path = os.path.join(canvas_dir, f"frame_{i:06d}.jpg")
        cv2.imwrite(out_path, canvas)

    # 4) ffmpeg 로 비디오 생성
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(canvas_dir, "frame_%06d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_video_path
    ], check=True)

    return out_video_path