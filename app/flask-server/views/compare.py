# flask-server/views/compare_view.py

import os
import uuid
import time
import subprocess
from flask import Blueprint, current_app, request, jsonify, send_from_directory

from pipeline.extract_keypoints.sound_sync              import sync_pair
from pipeline.extract_keypoints.yolo_and_mediapipe_pose import extract_keypoints
from pipeline.similarity.main                          import compute_feedback
from pipeline.extract_keypoints.img_to_video_feedback   import render_feedback_video

compare_bp = Blueprint('compare', __name__, url_prefix='/compare')


@compare_bp.route('/', methods=['POST'])
def compare_videos():
    # 1) 업로드 확인
    dancer  = request.files.get('dancer')
    trainee = request.files.get('trainee')
    if not dancer or not trainee:
        return jsonify(error="댄서/연습생 영상을 모두 업로드하세요"), 400

    # 2) 작업 디렉토리 생성
    base   = current_app.config['DATA_DIR']
    job_id = uuid.uuid4().hex
    work   = os.path.join(base, job_id)
    os.makedirs(work, exist_ok=True)

    # 3) 원본 저장
    dancer_path  = os.path.join(work, 'dancer.mp4')
    trainee_path = os.path.join(work, 'trainee.mp4')
    dancer.save(dancer_path)
    trainee.save(trainee_path)

    durations = {}

    # 4) 싱크
    start = time.time()
    try:
        synced_dancer, synced_trainee = sync_pair(dancer_path, trainee_path, work)
    except Exception as e:
        return jsonify(error=f"싱크 실패: {e}"), 500
    durations['sync'] = time.time() - start

    # FPS 읽기 
    import cv2
    cap = cv2.VideoCapture(synced_dancer)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(round(fps)) if fps and fps > 0 else 30
    cap.release()

    # 5) 키포인트 추출 (댄서)
    d_kp = os.path.join(work, 'dancer_kp')
    os.makedirs(d_kp, exist_ok=True)
    start = time.time()
    #_, ref_json, ref_frames = extract_keypoints(synced_dancer, d_kp)
    _, ref_json, ref_frames_dir = extract_keypoints(synced_dancer, d_kp)
    durations['extract_dancer'] = time.time() - start

    # 6) 키포인트 추출 (연습생)
    t_kp = os.path.join(work, 'trainee_kp')
    os.makedirs(t_kp, exist_ok=True)
    start = time.time()
    #_, usr_json, usr_frames = extract_keypoints(synced_trainee, t_kp)
    _, usr_json, usr_frames_dir = extract_keypoints(synced_trainee, t_kp)
    durations['extract_trainee'] = time.time() - start

   # n = min(len(ref_frames), len(usr_frames))
   # ref_frames = ref_frames[:n]
   # usr_frames = usr_frames[:n]

    # 7) 피드백 계산
    start = time.time()
    feedback_json, scores_json = compute_feedback(ref_json, usr_json, fps=fps)
    durations['feedback'] = time.time() - start

    # 8) 최종 비디오 렌더링
    final_video = os.path.join(work, 'final_feedback.mp4')
    start = time.time()
    render_feedback_video(
        feedback_json,
        teacher_frames = ref_frames_dir,
        student_frames = usr_frames_dir,
        out_video_path=final_video,
        fps=fps
    )
    durations['rendering'] = time.time() - start

    # 9) 오디오 머지 (댄서 영상 오디오 사용)
    merged_video = os.path.join(work, 'final_feedback_with_audio.mp4')
    start = time.time()
    subprocess.run([
        "ffmpeg", "-y",
        "-i", final_video,
        "-i", synced_dancer,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        merged_video
    ], check=True)
    durations['audio_merge'] = time.time() - start
    final_video = merged_video


    rel = job_id

    # 10) response에 feedback_json도 포함
    response = {
        'final_video': f"{rel}/{os.path.basename(final_video)}",
        'feedback_json': f"{rel}/dancer_kp/{os.path.basename(feedback_json)}",
        'scores_json': f"{rel}//dancer_kp/{os.path.basename(scores_json)}",
        'durations': durations
    }
    print(response)
    return jsonify(response), 200
