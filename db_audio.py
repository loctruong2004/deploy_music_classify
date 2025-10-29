# db_tracks_mssql.py
import os, re, pyodbc
from typing import List, Dict, Optional, Tuple

# ====== KHAI BÁO KẾT NỐI MSSQL ======
SERVER   = r"localhost\SQLEXPRESS"   # ví dụ: '172.11.12.52' hoặc 'localhost\\SQLEXPRESS'
DATABASE = "mussic"                   # tên DB của bạn
USERNAME = "loctruong"
PASSWORD = "11012004"
PORT     = "1433"

CONN_STR = (
    f"DRIVER={{SQL Server}};"
    f"SERVER={SERVER},{PORT};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};PWD={PASSWORD}"
)

# ====== HELPERS ======
def correct_drive_path(path: str) -> str:
    """
    Chuyển 'C/Users/...' -> 'C:/Users/...' để Python mở được file.
    Giữ nguyên nếu đã đúng. Chuẩn hoá slash cho OS hiện tại.
    """
    if not path:
        return ""
    m = re.match(r"^([A-Za-z])[\\/](.+)", path)
    if m:
        fixed = f"{m.group(1).upper()}:/{m.group(2)}"
        return os.path.normpath(fixed)
    return os.path.normpath(path)

def _connect():
    # Mẹo: thêm autocommit=True để read-only mượt
    return pyodbc.connect(CONN_STR, autocommit=True)

# ====== API CHÍNH ======
def fetch_all_tracks(limit_top: Optional[int] = 100) -> List[Dict]:
    """
    Lấy danh sách bài hát: id, title, artist, genre, filepath, duration_sec
    """
    top = f"TOP {int(limit_top)} " if limit_top else ""
    sql = f"""
    SELECT {top} id, title, artist, genre, filepath, duration_sec
    FROM tracks
    ORDER BY genre, title, id
    """
    rows = []
    try:
        with _connect() as cn, cn.cursor() as cur:
            cur.execute(sql)
            for r in cur.fetchall():
                rows.append({
                    "id": int(r.id),
                    "title": r.title,
                    "artist": r.artist,
                    "genre": r.genre,
                    "filepath": correct_drive_path(r.filepath),
                    "duration_sec": int(r.duration_sec) if r.duration_sec is not None else None
                })
    except Exception as e:
        print("❌ fetch_all_tracks error:", e)
    return rows

def fetch_genres() -> List[str]:
    """
    Lấy danh sách thể loại duy nhất (genre)
    """
    sql = "SELECT DISTINCT genre FROM tracks ORDER BY genre"
    gs = []
    try:
        with _connect() as cn, cn.cursor() as cur:
            cur.execute(sql)
            gs = [row.genre for row in cur.fetchall() if row.genre]
    except Exception as e:
        print("❌ fetch_genres error:", e)
    return gs

def fetch_tracks_by_genre(genre: Optional[str], limit_top: Optional[int] = 200) -> List[Dict]:
    """
    Lọc theo thể loại (None hoặc 'Tất cả' -> không lọc)
    """
    if not genre or genre == "Tất cả":
        return fetch_all_tracks(limit_top)

    top = f"TOP {int(limit_top)} " if limit_top else ""
    sql = f"""
    SELECT {top} id, title, artist, genre, filepath, duration_sec
    FROM tracks
    WHERE genre = ?
    ORDER BY title, id
    """
    rows = []
    try:
        with _connect() as cn, cn.cursor() as cur:
            cur.execute(sql, (genre,))
            for r in cur.fetchall():
                rows.append({
                    "id": int(r.id),
                    "title": r.title,
                    "artist": r.artist,
                    "genre": r.genre,
                    "filepath": correct_drive_path(r.filepath),
                    "duration_sec": int(r.duration_sec) if r.duration_sec is not None else None
                })
    except Exception as e:
        print("❌ fetch_tracks_by_genre error:", e)
    return rows
