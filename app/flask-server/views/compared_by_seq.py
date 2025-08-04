# flask-server/views/compared_by_seq.py

import os
import uuid
import json
import subprocess
from flask import Blueprint, current_app, request, jsonify
from concurrent.futures import ThreadPoolExecutor

from pipeline_ys.extract_keypoints.sound_sync import sync_pair
from pipeline_ys.extract_keypoints.yolo_and_mediapipe_pose import extract_keypoints
from pipeline_ys.similarity.data_utils import (
    load_mediapipe_json,
    interpolate_missing,
    smooth_keypoints,
    normalize_keypoints
)
from pipeline_ys.similarity.similarity_utils import compute_frame_similarities
from pipeline_ys.similarity.sequence_analyzer import analyze_sequence
from pipeline_ys.similarity.feedback_generator_extended import generate_sequence_feedback

compared_by_seq_bp = Blueprint(
    'compared_by_seq',
    __name__,
    url_prefix='/compared_by_seq'
)

@compared_by_seq_bp.route('/', methods=['POST'])
def compare_and_render_feedback():
    # A) 업로드 검증
    dancer = request.files.get('dancer')
    trainee = request.files.get('trainee')
    if not dancer or not trainee:
        return jsonify(error="댄서와 연습생 영상을 모두 업로드해주세요"), 400

    # B) 작업 디렉토리 준비
    base_dir = current_app.config['DATA_DIR']
    job_id   = uuid.uuid4().hex
    work_dir = os.path.join(base_dir, job_id)
    os.makedirs(work_dir, exist_ok=True)

    # C) 원본 저장
    path_d = os.path.join(work_dir, 'dancer.mp4')
    path_t = os.path.join(work_dir, 'trainee.mp4')
    dancer.save(path_d)
    trainee.save(path_t)

    # D) 오디오/비디오 싱크
    synced_d, synced_t = sync_pair(path_d, path_t, work_dir)

    # E) 480p 리사이즈
    resized_d = os.path.join(work_dir, 'dancer_480p.mp4')
    resized_t = os.path.join(work_dir, 'trainee_480p.mp4')
    def _resize(inp, out):
        subprocess.run([
            'ffmpeg', '-y', '-i', inp,
            '-vf', 'scale=-2:480', '-r', '30',
            '-c:v', 'libx264', '-preset', 'veryfast',
            out
        ], check=True)
    with ThreadPoolExecutor(2) as pool:
        pool.submit(_resize, synced_d, resized_d)
        pool.submit(_resize, synced_t, resized_t)

    # F) 키포인트 추출
    d_kp_dir = os.path.join(work_dir, 'dancer_kp')
    t_kp_dir = os.path.join(work_dir, 'trainee_kp')
    os.makedirs(d_kp_dir, exist_ok=True)
    os.makedirs(t_kp_dir, exist_ok=True)
    def _extract(video, out_dir):
        _, json_fp, _ = extract_keypoints(video, out_dir)
        return json_fp
    json_d = _extract(resized_d, d_kp_dir)
    json_t = _extract(resized_t, t_kp_dir)

    # G) 전처리 & 정규화
    raw_d, vis_d = load_mediapipe_json(json_d)
    raw_t, vis_t = load_mediapipe_json(json_t)
    n = min(len(raw_d), len(raw_t))
    norm_d = normalize_keypoints(smooth_keypoints(interpolate_missing(raw_d[:n], vis_d[:n])))
    norm_t = normalize_keypoints(smooth_keypoints(interpolate_missing(raw_t[:n], vis_t[:n])))

    # H) 시퀀스 분석 & LLM 피드백 생성
    fps = 30
    sequences = analyze_sequence(norm_d, norm_t)
    sequence_feedbacks = []
    for seq in sequences:
        s, e = seq['start_time'], seq['end_time']
        sf, ef = int(s*fps), int(e*fps)
        fb = generate_sequence_feedback(seq, norm_d[sf:ef], norm_t[sf:ef]).strip()
        sequence_feedbacks.append({
            "start_time": round(s,2),
            "end_time": round(e,2),
            "feedback": fb
        })

    # 🔸 JSON 파일로 저장
    with open(os.path.join(work_dir, 'sequence_feedback.json'), 'w', encoding='utf-8') as f:
        json.dump(sequence_feedbacks, f, ensure_ascii=False, indent=2)

    # I) 좌우 병렬 영상 합성
    combined_mp4 = os.path.join(work_dir, 'skeleton_combined.mp4')
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(d_kp_dir, 'frames', 'frame_%06d.jpg'),
        '-framerate', str(fps),
        '-i', os.path.join(t_kp_dir, 'frames', 'frame_%06d.jpg'),
        '-filter_complex', '[0:v][1:v]hstack=inputs=2',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        combined_mp4
    ], check=True)

    # J) 오디오 추출
    audio_aac = os.path.join(work_dir, 'audio.aac')
    subprocess.run([
        'ffmpeg', '-y',
        '-i', path_d,
        '-vn', '-acodec', 'copy',
        audio_aac
    ], check=True)

    # K) 영상 + 오디오 병합
    final_mp4 = os.path.join(work_dir, 'rendered.mp4')
    subprocess.run([
        'ffmpeg', '-y',
        '-i', combined_mp4,
        '-i', audio_aac,
        '-c:v', 'copy', '-c:a', 'aac', '-shortest',
        final_mp4
    ], check=True)

    # L) 결과 반환
    return jsonify({
        "final_video": f"{job_id}/rendered.mp4",
        "sequence_feedbacks": sequence_feedbacks
    }), 200


# # flask-server/views/compared_by_seq.py

# import os
# import uuid
# import subprocess
# from flask import Blueprint, current_app, request, jsonify
# from concurrent.futures import ThreadPoolExecutor

# from pipeline_ys.extract_keypoints.sound_sync import sync_pair
# from pipeline_ys.extract_keypoints.yolo_and_mediapipe_pose import extract_keypoints
# from pipeline_ys.similarity.data_utils import (
#     load_mediapipe_json,
#     interpolate_missing,
#     smooth_keypoints,
#     normalize_keypoints
# )
# from pipeline_ys.similarity.similarity_utils import compute_frame_similarities
# from pipeline_ys.similarity.sequence_analyzer import analyze_sequence
# from pipeline_ys.similarity.feedback_generator_extended import generate_sequence_feedback

# compared_by_seq_bp = Blueprint(
#     'compared_by_seq',
#     __name__,
#     url_prefix='/compared_by_seq'
# )

# @compared_by_seq_bp.route('/', methods=['POST'])
# def compare_and_render_feedback():
#     # A) 업로드 검증
#     dancer = request.files.get('dancer')
#     trainee = request.files.get('trainee')
#     if not dancer or not trainee:
#         return jsonify(error="댄서와 연습생 영상을 모두 업로드해주세요"), 400

#     # B) 작업 디렉토리 준비
#     base_dir = current_app.config['DATA_DIR']            # e.g. static/data
#     job_id   = uuid.uuid4().hex
#     work_dir = os.path.join(base_dir, job_id)
#     os.makedirs(work_dir, exist_ok=True)

#     # C) 원본 저장
#     path_d = os.path.join(work_dir, 'dancer.mp4')
#     path_t = os.path.join(work_dir, 'trainee.mp4')
#     dancer.save(path_d)
#     trainee.save(path_t)

#     # D) 오디오/비디오 싱크
#     synced_d, synced_t = sync_pair(path_d, path_t, work_dir)

#     # E) 480p 리사이즈（병렬）
#     resized_d = os.path.join(work_dir, 'dancer_480p.mp4')
#     resized_t = os.path.join(work_dir, 'trainee_480p.mp4')
#     def _resize(inp, out):
#         subprocess.run([
#             'ffmpeg', '-y', '-i', inp,
#             '-vf', 'scale=-2:480', '-r', '30',
#             '-c:v', 'libx264', '-preset', 'veryfast',
#             out
#         ], check=True)
#     with ThreadPoolExecutor(2) as pool:
#         pool.submit(_resize, synced_d, resized_d)
#         pool.submit(_resize, synced_t, resized_t)

#     # F) 키포인트 추출 → JSON + frames/ 폴더 생성
#     d_kp_dir = os.path.join(work_dir, 'dancer_kp')
#     t_kp_dir = os.path.join(work_dir, 'trainee_kp')
#     os.makedirs(d_kp_dir, exist_ok=True)
#     os.makedirs(t_kp_dir, exist_ok=True)
#     def _extract(video, out_dir):
#         _, json_fp, _ = extract_keypoints(video, out_dir)
#         return json_fp
#     json_d = _extract(resized_d, d_kp_dir)
#     json_t = _extract(resized_t, t_kp_dir)

#     # G) 전처리 & 정규화
#     raw_d, vis_d = load_mediapipe_json(json_d)
#     raw_t, vis_t = load_mediapipe_json(json_t)
#     n = min(len(raw_d), len(raw_t))
#     norm_d = normalize_keypoints(smooth_keypoints(interpolate_missing(raw_d[:n], vis_d[:n])))
#     norm_t = normalize_keypoints(smooth_keypoints(interpolate_missing(raw_t[:n], vis_t[:n])))

#     # H) 시퀀스 분석 & LLM 피드백 리스트 생성
#     fps = 30
#     sequences = analyze_sequence(norm_d, norm_t)
#     sequence_feedbacks = []
#     for seq in sequences:
#         s, e = seq['start_time'], seq['end_time']
#         sf, ef = int(s*fps), int(e*fps)
#         fb = generate_sequence_feedback(seq, norm_d[sf:ef], norm_t[sf:ef]).strip()
#         sequence_feedbacks.append({
#             "start_time": round(s,2),
#             "end_time": round(e,2),
#             "feedback": fb
#         })

#     # I) 좌우 병렬(hstack) 영상 합성
#     combined_mp4 = os.path.join(work_dir, 'skeleton_combined.mp4')
#     subprocess.run([
#         'ffmpeg', '-y',
#         '-framerate', str(fps),
#         '-i', os.path.join(d_kp_dir, 'frames', 'frame_%06d.jpg'),
#         '-framerate', str(fps),
#         '-i', os.path.join(t_kp_dir, 'frames', 'frame_%06d.jpg'),
#         '-filter_complex', '[0:v][1:v]hstack=inputs=2',
#         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
#         '-pix_fmt', 'yuv420p',
#         combined_mp4
#     ], check=True)

#     # J) 댄서 원본에서 오디오만 추출
#     audio_aac = os.path.join(work_dir, 'audio.aac')
#     subprocess.run([
#         'ffmpeg', '-y',
#         '-i', path_d,
#         '-vn', '-acodec', 'copy',
#         audio_aac
#     ], check=True)

#     # K) 영상＋오디오 병합 → 최종 rendered.mp4
#     final_mp4 = os.path.join(work_dir, 'rendered.mp4')
#     subprocess.run([
#         'ffmpeg', '-y',
#         '-i', combined_mp4,
#         '-i', audio_aac,
#         '-c:v', 'copy', '-c:a', 'aac', '-shortest',
#         final_mp4
#     ], check=True)

#     # L) 결과 반환
#     return jsonify({
#         "final_video": f"{job_id}/rendered.mp4",
#         "sequence_feedbacks": sequence_feedbacks
#     }), 200

