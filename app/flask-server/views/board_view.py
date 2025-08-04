# flask-server/views/board_view.py
import os
from flask import Blueprint, request, jsonify, current_app, session
from werkzeug.utils import secure_filename
from db.boad_util import (
    get_posts_by_page,
    insert_post,
    get_post_by_id,
    increase_view_count,
    update_post,
    delete_post
)

board_bp = Blueprint('board', __name__, url_prefix='/board')

# 업로드 가능한 확장자
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@board_bp.route('/upload_media', methods=['POST'])
def upload_media():
    if 'file' not in request.files:
        return jsonify({'error':'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error':'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    save_dir = os.path.join(current_app.root_path, 'static', 'uploads')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    file.save(filepath)

    url = f'/static/uploads/{filename}'
    return jsonify({'url': url})

@board_bp.route('/posts', methods=['GET'])
def list_posts():
    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    posts, total = get_posts_by_page(page, per_page)

    return jsonify({'posts': posts, 'total': total})

@board_bp.route('/posts/<int:post_id>', methods=['GET'])
def get_post(post_id):
    post = get_post_by_id(post_id)
    print("게시글 한개 조회", post)
    if not post:
        return jsonify({'error':'Not found'}), 404
    increase_view_count(post_id)
    return jsonify(post)

@board_bp.route('/posts', methods=['POST'])
def create_post():
    if 'loginuser' not in session:
        return jsonify({'error':'Unauthorized'}), 401
    data    = request.get_json() or {}
    title   = (data.get('title')   or '').strip()
    content = (data.get('content') or '').strip()
    user_id = session['loginuser']['id']
    if not title or not content:
        return jsonify({'error':'제목과 내용을 모두 입력해주세요.'}), 400
    ok = insert_post(user_id, title, content)
    return (jsonify({'success':True}), 201) if ok else (jsonify({'error':'DB 오류'}), 500)

@board_bp.route('/posts/<int:post_id>', methods=['PUT'])
def edit_post(post_id):
    if 'loginuser' not in session:
        return jsonify({'error':'Unauthorized'}), 401
    post = get_post_by_id(post_id)
    if not post:
        return jsonify({'error':'Not found'}), 404
    if post['user_id'] != session['loginuser']['id']:
        return jsonify({'error':'Forbidden'}), 403
    data    = request.get_json() or {}
    title   = (data.get('title')   or '').strip()
    content = (data.get('content') or '').strip()
    if not title or not content:
        return jsonify({'error':'제목과 내용을 모두 입력해주세요.'}), 400
    ok = update_post(post_id, title, content)
    return (jsonify({'success':True})) if ok else (jsonify({'error':'DB 오류'}), 500)

@board_bp.route('/posts/<int:post_id>', methods=['DELETE'])
def remove_post(post_id):
    if 'loginuser' not in session:
        return jsonify({'error':'Unauthorized'}), 401
    post = get_post_by_id(post_id)
    if not post:
        return jsonify({'error':'Not found'}), 404
    if post['user_id'] != session['loginuser']['id']:
        return jsonify({'error':'Forbidden'}), 403
    ok = delete_post(post_id)
    return (jsonify({'success':True})) if ok else (jsonify({'error':'DB 오류'}), 500)
