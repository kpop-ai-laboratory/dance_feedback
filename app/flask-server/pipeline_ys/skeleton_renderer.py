# import cv2
# import numpy as np
# import os
# from pipeline_ys.similarity.constants import JOINT_NAMES, JOINT_PAIRS

# def draw_skeleton(canvas, keypoints, color=(0, 255, 0), radius=4, thickness=2):
#     """
#     🔹 2D 캔버스 위에 키포인트 기반 스켈레톤 그리기 함수
#     """
#     for i, point in enumerate(keypoints):
#         x, y = point[:2]  # (x, y, ...)에서 앞 두 개만 사용
#         if x > 0 and y > 0:
#             cv2.circle(canvas, (int(x), int(y)), radius, color, -1)

#     for i, j in JOINT_PAIRS:
#         xi, yi = keypoints[i][:2]
#         xj, yj = keypoints[j][:2]
#         if xi > 0 and yi > 0 and xj > 0 and yj > 0:
#             cv2.line(canvas, (int(xi), int(yi)), (int(xj), int(yj)), color, thickness)

# def render_skeleton_video_pair(
#     keypoints_ref, keypoints_usr,
#     output_path,
#     image_size=(480, 854),  # 🔸 여기만 수정: 480p (16:9)
#     fps=30
# ):
#     """
#     🔹 스켈레톤 페어 영상 생성 (좌: 기준자, 우: 연습생)
#     keypoints_ref, keypoints_usr: 각 프레임당 [33, 4] 형태 리스트
#     """
#     H, W = image_size
#     canvas_size = (H, W * 2, 3)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (W * 2, H))

#     for i in range(min(len(keypoints_ref), len(keypoints_usr))):
#         canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # 흰 배경

#         draw_skeleton(canvas[:, :W], keypoints_ref[i], color=(255, 0, 0))   # 왼쪽: 기준자
#         draw_skeleton(canvas[:, W:], keypoints_usr[i], color=(0, 0, 255))   # 오른쪽: 연습생

#         out.write(canvas)

#     out.release()

#     print(f"[✔] 병렬 스켈레톤 렌더링 완료 → {output_path}")


import cv2
import numpy as np
import os
from pipeline_ys.similarity.constants import JOINT_NAMES, JOINT_PAIRS

def draw_skeleton(canvas, keypoints, color=(0, 255, 0), radius=4, thickness=2):
    """
    🔹 2D 캔버스 위에 키포인트 기반 스켈레톤 그리기
    keypoints: 정규화된 (x_rel,y_rel,z,vis) 리스트
    canvas: numpy array (H, W, 3)
    """
    H, W = canvas.shape[:2]

    # 관절 점 찍기
    for point in keypoints:
        x_rel, y_rel = point[:2]
        # 픽셀 좌표로 변환
        x = int(x_rel * W)
        y = int(y_rel * H)
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(canvas, (x, y), radius, color, -1)

    # 관절 연결 선 그리기
    for i, j in JOINT_PAIRS:
        x1_rel, y1_rel = keypoints[i][:2]
        x2_rel, y2_rel = keypoints[j][:2]
        x1 = int(x1_rel * W)
        y1 = int(y1_rel * H)
        x2 = int(x2_rel * W)
        y2 = int(y2_rel * H)
        if (0 <= x1 < W and 0 <= y1 < H and
            0 <= x2 < W and 0 <= y2 < H):
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

def render_skeleton_video_pair(
    keypoints_ref, keypoints_usr,
    output_path,
    image_size=(480, 854),  # 480p 16:9
    fps=30
):
    """
    🔹 스켈레톤 페어 영상 생성 (좌: 기준자, 우: 연습생)
    keypoints_ref, keypoints_usr: T x 33 x 4 정규화된 배열
    """
    H, W = image_size
    canvas_size = (H, W * 2, 3)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W * 2, H))

    for i in range(min(len(keypoints_ref), len(keypoints_usr))):
        # 흰 배경
        canvas = np.ones(canvas_size, dtype=np.uint8) * 255

        # 왼쪽: 기준자, 오른쪽: 연습생
        draw_skeleton(canvas[:, :W], keypoints_ref[i], color=(255, 0, 0))
        draw_skeleton(canvas[:, W:], keypoints_usr[i], color=(0, 0, 255))

        out.write(canvas)

    out.release()
    print(f"[✔] 병렬 스켈레톤 렌더링 완료 → {output_path}")
