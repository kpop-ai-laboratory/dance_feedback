import pymysql
from datetime import date

def get_connection():
    return pymysql.connect(
        host="192.168.0.88",
        database="masdb",
        user="nielpj",
        password="nielpj",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

# ──────────────────────────────────────────────────────────
# 1) 페이지 단위로 게시글 목록 조회 (deleted = 0 조건 추가)
# ──────────────────────────────────────────────────────────
def get_posts_by_page(page, per_page=10):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            offset = (page - 1) * per_page

            # 총 게시글 수 (deleted = 0 인 것만)
            cursor.execute("SELECT COUNT(*) as total FROM board WHERE deleted = 0")
            total = cursor.fetchone()['total']

            # 실제 게시글 목록도 deleted = 0 조건 포함
            cursor.execute("""
                SELECT id, user_id, title, content, view_count, created_at, updated_at
                FROM board
                WHERE deleted = 0
                ORDER BY id DESC
                LIMIT %s OFFSET %s
            """, (per_page, offset))
            posts = cursor.fetchall()

        return posts, total
    finally:
        conn.close()

# ──────────────────────────────────────────────────────────
# 2) 게시글 삽입
# ──────────────────────────────────────────────────────────
def insert_post(user_id, title, content):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO board (user_id, title, content)
            VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (user_id, title, content))
        conn.commit()
        return True
    except Exception as e:
        print("글쓰기 실패:", e)
        return False
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# ──────────────────────────────────────────────────────────
# 3) 게시글 ID로 조회
# ──────────────────────────────────────────────────────────
def get_post_by_id(post_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = """
            SELECT id, user_id, title, content, view_count, created_at, updated_at
            FROM board
            WHERE id = %s AND deleted = 0
        """
        cursor.execute(sql, (post_id,))
        return cursor.fetchone()
    except Exception as e:
        print("게시글 조회 오류:", e)
        return None
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# ──────────────────────────────────────────────────────────
# 4) 조회수 증가
# ──────────────────────────────────────────────────────────
def increase_view_count(post_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE board SET view_count = view_count + 1 WHERE id = %s", (post_id,))
        conn.commit()
    except Exception as e:
        print("조회수 증가 오류:", e)
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# ──────────────────────────────────────────────────────────
# 5) 게시글 수정
# ──────────────────────────────────────────────────────────
def update_post(post_id, title, content):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        sql = """
            UPDATE board
            SET title = %s, content = %s, updated_at = NOW()
            WHERE id = %s
        """
        cursor.execute(sql, (title, content, post_id))
        conn.commit()
        return True
    except Exception as e:
        print("게시글 수정 오류:", e)
        return False
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# ──────────────────────────────────────────────────────────
# 6) 게시글 삭제 (soft delete)
# ──────────────────────────────────────────────────────────
def delete_post(post_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE board SET deleted = 1 WHERE id = %s", (post_id,))
        conn.commit()
        return True
    except Exception as e:
        print("게시글 삭제 오류:", e)
        return False
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()
