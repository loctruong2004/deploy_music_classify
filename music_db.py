# music_db.py
import os, sqlite3
from contextlib import contextmanager
from typing import List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MUSIC_DB_PATH = os.getenv("MUSIC_DB_PATH", os.path.join(BASE_DIR, "music.db"))
print("[music_db] Using DB at:", MUSIC_DB_PATH)

@contextmanager
def get_conn():
    conn = sqlite3.connect(MUSIC_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_music_db(seed_demo: bool = True):
    with get_conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            artist TEXT,
            genre TEXT NOT NULL,
            filepath TEXT NOT NULL,
            duration_sec INTEGER
        );
        """)
        # optional seed
        if seed_demo:
            cur = c.execute("SELECT COUNT(1) AS n FROM tracks;")
            n = cur.fetchone()["n"]
            if n == 0:
                demo = [
                    ("Chiều Tím", "Ca sĩ A", "bolero", os.path.join(BASE_DIR, "demo_audio", "bolero_01.mp3"), 210),
                    ("Dạ Cổ Hoài Lang", "Nghệ sĩ B", "cailuong", os.path.join(BASE_DIR, "demo_audio", "cailuong_01.mp3"), 480),
                    ("Trống Quân", "Nghệ sĩ C", "danca", os.path.join(BASE_DIR, "demo_audio", "danca_01.mp3"), 240),
                    ("Đi Cày", "Nghệ sĩ D", "nhacdo", os.path.join(BASE_DIR, "demo_audio", "nhacdo_01.mp3"), 200),
                    ("Hát Xẩm Demo", "Nghệ sĩ E", "cheo", os.path.join(BASE_DIR, "demo_audio", "cheo_01.mp3"), 230),
                ]
                c.executemany("INSERT INTO tracks (title, artist, genre, filepath, duration_sec) VALUES (?, ?, ?, ?, ?);", demo)

def list_genres() -> List[str]:
    with get_conn() as c:
        rows = c.execute("SELECT DISTINCT genre FROM tracks ORDER BY genre;").fetchall()
    genres = [r["genre"] for r in rows]
    return genres

def list_tracks_by_genre(genre: Optional[str]) -> List[Tuple[int, str, str, str, str, int]]:
    """
    Trả về list tuple: (id, title, artist, genre, filepath, duration_sec)
    genre=None hoặc 'Tất cả' => không lọc
    """
    with get_conn() as c:
        if not genre or genre == "Tất cả":
            rows = c.execute("SELECT * FROM tracks ORDER BY genre, title;").fetchall()
        else:
            rows = c.execute("SELECT * FROM tracks WHERE genre = ? ORDER BY title;", (genre,)).fetchall()
    out = []
    for r in rows:
        out.append((r["id"], r["title"], r["artist"], r["genre"], r["filepath"], r["duration_sec"]))
    return out

def search_tracks(query: str, genre: Optional[str]) -> List[Tuple[int, str, str, str, str, int]]:
    q = f"%{(query or '').strip()}%"
    with get_conn() as c:
        if genre and genre != "Tất cả":
            rows = c.execute("""
                SELECT * FROM tracks
                WHERE genre = ? AND (title LIKE ? OR artist LIKE ?)
                ORDER BY title;
            """, (genre, q, q)).fetchall()
        else:
            rows = c.execute("""
                SELECT * FROM tracks
                WHERE (title LIKE ? OR artist LIKE ?)
                ORDER BY genre, title;
            """, (q, q)).fetchall()
    return [(r["id"], r["title"], r["artist"], r["genre"], r["filepath"], r["duration_sec"]) for r in rows]
