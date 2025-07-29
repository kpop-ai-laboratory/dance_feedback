import pymysql
from datetime import date

#  공통 DB 연결 함수
def get_connection():
    return pymysql.connect(
        host="localhost",
        database="masdb",
        user="nielpj",
        password="nielpj",
        charset="utf8"  
    )

#  회원 등록 함수
def insert_user(email, passwd, username):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        sql = """
        INSERT INTO users (email, passwd, username, usertype, regdate)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (email, passwd, username, 0, date.today()))
        conn.commit()
        return True  # 성공 시 True 반환

    except pymysql.err.IntegrityError:
        print("이미 등록된 이메일입니다.")
        return False  # 중복 이메일
    except Exception as e:
        print("사용자 등록 중 오류 발생:", type(e).__name__, str(e))
        return False
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# 이메일로 회원 정보 조회 (로그인 시 사용)
def select_user_by_email(email):
    row = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        sql = """
        SELECT id, email, passwd, username, usertype, regdate
        FROM users
        WHERE email = %s AND deleted = 0
        """
        cursor.execute(sql, (email,))
        row = cursor.fetchone()

    except Exception as e:
        print("사용자 조회 중 오류 발생:", type(e).__name__, str(e))
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

    return row
