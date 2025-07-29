
# yolo_and_medaipipe_pose 돌리고 나온 이미지를 토대로 영상으로 변환

import os, cv2

IMAGE_DIR    = r".../frames"
OUTPUT_VIDEO = r".../stitched_video.mp4"

# 1) 파일 목록 & 해상도
img_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg'))
h, w = cv2.imread(os.path.join(IMAGE_DIR, img_files[0])).shape[:2]

# 2) VideoWriter 설정 (fps = 29.97)
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    29.97,
    (w, h)
)

# 3) 순차적 이미지 프레임 쓰기
for fname in img_files:
    frame = cv2.imread(os.path.join(IMAGE_DIR, fname))
    writer.write(frame)

writer.release()
cv2.destroyAllWindows()
print("✔ Video saved:", OUTPUT_VIDEO)

