# app/use_auth.py
from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from db.user_util import select_user_by_email, insert_user

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    print("회원가입")
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    username = data.get('username')
    role = data.get('role', 'user')

    # 이미 존재하는 이메일 체크
    if select_user_by_email(email):
        return jsonify(success=False, message='이미 등록된 이메일입니다.'), 409

    # 비밀번호 해시
    hashed = generate_password_hash(password)
    if insert_user(email, hashed, username, role):
        return jsonify(success=True), 201
    return jsonify(success=False, message='회원가입 실패'), 500

# @auth_bp.route('/login', methods=['POST'])
# def login():
#     print("로그인")
#     data = request.get_json()
#     print(data)
#     email = data.get('userEmail')
#     password = data.get('password')

#     # 사용자 조회 및 비밀번호 검증
#     user = select_user_by_email(email)
#     if user and check_password_hash(user[2], password):
#         session['user'] = user[1]  # 이메일 저장
#         return jsonify(success=True, username=user[3])
#     return jsonify(success=False, message='로그인 실패'), 401

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email    = data.get('userEmail')
    password = data.get('password')

    user = select_user_by_email(email)
    if user and check_password_hash(user[2], password):
        # session['user'] = user[1]
        # ↓ 이렇게 바꿔줍니다
        session['loginuser'] = {
          'id':       user[0],  # id
          'email':    user[1],
          'username': user[3],
          'usertype': user[4]
        }
        return jsonify(success=True, user=session['loginuser'])
    return jsonify(success=False, message='로그인 실패'), 401

@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.pop('loginuser', None)
    return jsonify(success=True)

@auth_bp.route('/me', methods=['GET'])
def me():
    loginuser = session.get('loginuser')
    if loginuser:
        return jsonify(authenticated=True, user=loginuser)
    return jsonify({'authenticated': False}), 401