# flask-server/config.py 에 업로드/출력 디렉터리를 설정해 놓습니다.
import os

# static/data 아래에 결과물이 저장/서빙됩니다.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'static', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
