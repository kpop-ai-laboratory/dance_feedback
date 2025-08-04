# flask-server/views/youtube_download.py

import os
import uuid
import glob
from flask import Blueprint, request, jsonify, current_app,send_from_directory
from yt_dlp import YoutubeDL

youtube_bp = Blueprint('youtube_download', __name__)

@youtube_bp.route('/youtube-download', methods=['POST'])
def download_youtube():
    data = request.get_json() or {}
    url  = data.get('url')
    if not url:
        return jsonify({'error': 'URL이 입력되지 않았습니다.'}), 400

    # 🔧 저장 디렉토리: static/data/youtube 안에 저장되도록 수정
    base_dir = os.path.join(current_app.config['DATA_DIR'], 'youtube')
    uid      = uuid.uuid4().hex
    out_dir  = os.path.join(base_dir, uid)
    os.makedirs(out_dir, exist_ok=True)

    # 🎞️ yt_dlp 다운로드 옵션
    ytdl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(out_dir, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'merge_output_format': 'mp4',
        'postprocessors': [
            {
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4'
            }
        ]
    }


    try:
        with YoutubeDL(ytdl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # 📁 저장된 mp4 파일 경로 가져오기
        mp4_files = glob.glob(os.path.join(out_dir, '*.mp4'))
        if not mp4_files:
            raise FileNotFoundError(f"No mp4 file found in {out_dir}")
        saved_path = mp4_files[0]  # 가장 먼저 찾은 파일

        rel_path = os.path.relpath(saved_path, start=os.path.join(current_app.config['DATA_DIR']))
        url_path = '/media/' + rel_path.replace('\\', '/')

        print(f"🔗 url_path: {url_path}")

        return jsonify({'path': saved_path, 'url': url_path}), 200

    except Exception as e:
        current_app.logger.exception("YouTube 다운로드 실패")
        return jsonify({'error': str(e)}), 500


# 🎯 추가: 다운로드 파일을 실제로 브라우저에 서빙하는 라우터
@youtube_bp.route('/media/<path:filename>')
def serve_youtube_file(filename):
    """저장된 유튜브 영상 파일을 브라우저에서 다운로드 가능하게 서빙"""
    # 실제 경로: static/data 폴더 기준
    data_dir = current_app.config['DATA_DIR']
    return send_from_directory(data_dir, filename, as_attachment=True)