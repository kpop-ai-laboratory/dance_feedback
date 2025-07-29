# flask-server/views/compare_view.py

import os
import uuid
from flask import Blueprint, current_app, request, jsonify
from pipeline.extract_keypoints.sound_sync import sync_pair
from pipeline.extract_keypoints.yolo_and_mediapipe_pose import extract_keypoints

compare_bp = Blueprint('compare', __name__, url_prefix='/compare')

@compare_bp.route('/', methods=['POST'])
def compare_videos():
    # 1) 파일 가져오기
    dancer  = request.files.get('dancer')
    trainee = request.files.get('trainee')
    print('dancer' , dancer)
    print('trainee' , trainee)
    if not dancer or not trainee:
        return jsonify(error="댄서/연습생 영상을 모두 업로드하세요"), 400

    # 2) DATA_DIR 아래 job 폴더 생성
    base     = current_app.config['DATA_DIR']
    job_id   = uuid.uuid4().hex
    work_dir = os.path.join(base, job_id)
    os.makedirs(work_dir, exist_ok=True)

    print("base",base)
    print("job_id",job_id)
    print("work_dir",work_dir)


    # 3) 업로드된 파일 저장
    dancer_path  = os.path.join(work_dir, 'dancer.mp4')
    trainee_path = os.path.join(work_dir, 'trainee.mp4')
    print("dancer_path",dancer_path)
    print("trainee_path",trainee_path)

    #dancer.save(dancer_path)
    # trainee.save(trainee_path)
    dancer.save(dancer_path); trainee.save(trainee_path)
    synced1, synced2 = sync_pair(dancer_path, trainee_path, work_dir)

    # 3) keypoint 추출
    ref_dir  = os.path.join(work_dir, 'kp_ref')
    user_dir = os.path.join(work_dir, 'kp_user')
    kp1 = extract_keypoints(synced1, ref_dir)
    kp2 = extract_keypoints(synced2, user_dir)


    # 4) 싱크 실행
    try:
        synced1, synced2 = sync_pair(
            video1_path=dancer_path,
            video2_path=trainee_path,
            out_dir=work_dir
        )
    except Exception as e:
        return jsonify(error=f"싱크 중 오류: {e}"), 500

    # 5) 결과 리턴
    return jsonify({
        "original": {
            "dancer":  dancer.filename,
            "trainee": trainee.filename
        },
        "synced": {
            "dancer":  os.path.basename(synced1),
            "trainee": os.path.basename(synced2)
        }
    }), 200
