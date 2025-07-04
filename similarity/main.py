# main.py

import json
import numpy as np

from data_utils import load_mediapipe_json, interpolate_missing, smooth_keypoints, normalize_keypoints
from angle_utils import calc_interior_angles_2d, angle_diff
from similarity_utils import compute_frame_similarities, aggregate_per_second, identify_misaligned_joints
from feedback_utils import generate_frame_feedback

def main(
    ref_json: str = "D:\kpop\similarity\data\results1\keypoints.json",
    user_json: str = "D:\kpop\similarity\data\results3\keypoints3.json",
    fps: int = 30,
    angle_report_thresh: float = 10.0,
    proc_thresh: float = 0.1
):
    # 1) Load 원본 픽셀 좌표 보관
    kp_ref_raw, vis_ref   = load_mediapipe_json(ref_json)
    kp_user_raw, vis_user = load_mediapipe_json(user_json)

    # 🔍 Debug: 원본 픽셀 좌표에서 Frame 2 각도 계산
    frame = 2
    angles_ref_raw = calc_interior_angles_2d(kp_ref_raw)
    angles_user_raw = calc_interior_angles_2d(kp_user_raw)
    raw_ref_deg = np.degrees(angles_ref_raw[frame])
    raw_usr_deg = np.degrees(angles_user_raw[frame])
    raw_diff_deg = np.abs([angle_diff(r, u) * 180/np.pi 
                           for r, u in zip(angles_ref_raw[frame], angles_user_raw[frame])])
    print(f"[원본 픽셀] Frame {frame} ref angles: {raw_ref_deg}")
    print(f"[원본 픽셀] Frame {frame} usr angles: {raw_usr_deg}")
    print(f"[원본 픽셀] Frame {frame} diffs: {raw_diff_deg}")

    # 2) Preprocess (정규화된 좌표)
    kp_ref  = normalize_keypoints(smooth_keypoints(interpolate_missing(kp_ref_raw, vis_ref)))
    kp_user = normalize_keypoints(smooth_keypoints(interpolate_missing(kp_user_raw, vis_user)))

    # 3) Compute frame-level similarities
    res = compute_frame_similarities(kp_ref, kp_user, angle_weight=0.6)
    final_scores = res['final']

    # 4) Aggregate per-second
    sec_scores = aggregate_per_second(final_scores, fps)

    # 5) Identify misaligned frames/joints
    dynamic_thresh = np.percentile(res['angle_diffs'].flatten(), 95)
    print(f"Using dynamic angle threshold: {np.degrees(dynamic_thresh):.1f}° (95th percentile)")
    bad_frames, bad_joints = identify_misaligned_joints(
        res['angle_diffs'], res['proc_dists'],
        angle_thresh=dynamic_thresh,
        proc_thresh=proc_thresh
    )
    if not bad_frames:
        print("어긋난 프레임이 없습니다. (현재 임계치로는 모두 정상)")
    else:
        print(f"어긋난 프레임: {bad_frames}")

    # 6) Generate feedback using 원본 픽셀 각도
    angle_thresh_rad = np.deg2rad(angle_report_thresh)
    feedback = {}
    for t in bad_frames:
        msgs = generate_frame_feedback(
            kp_ref_raw[t],   # 원본 픽셀 좌표
            kp_user_raw[t],
            angle_thresh=angle_thresh_rad
        )
        feedback[t] = msgs

    with open('feedback.json', 'w', encoding='utf-8') as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    print("Feedback이 feedback.json에 저장되었습니다.")

    # # 7) 추가 출력: 프레임별 & 초별 유사도
    # print("\n== 프레임별 최종 유사도 ==")
    # for i, s in enumerate(final_scores):
    #     print(f"Frame {i:3d}: {s:.3f}")
    # print("\n== 초별 평균 유사도 ==")
    # for sec, s in enumerate(sec_scores):
    #     print(f"Second {sec:2d}: {s:.3f}")

if __name__ == '__main__':
    main()
