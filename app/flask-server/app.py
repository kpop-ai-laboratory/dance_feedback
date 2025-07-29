# flask-server/app.py
from flask import Flask
from config import DATA_DIR
from views.compare import compare_bp
from views.youtube_download import youtube_bp
from views.auth_view import auth_bp

from flask import send_from_directory


app = Flask(__name__, static_folder='static')
app.config['DATA_DIR'] = DATA_DIR
app.config['SECRET_KEY'] = 'humanda5-secret-key' # 세션(session) 등을 사용하기 필요한 설정


app.register_blueprint(compare_bp)
app.register_blueprint(youtube_bp)
app.register_blueprint(auth_bp)

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(app.config['DATA_DIR'], filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
