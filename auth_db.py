# auth_db.py (bản bỏ hash)
import os, sqlite3, time, hmac
from contextlib import contextmanager
from typing import Optional, Tuple
import jwt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("AUTH_DB_PATH", os.path.join(BASE_DIR, "auth_users.db"))
print("[auth_db] Using DB at:", DB_PATH)

JWT_SECRET = os.getenv("JWT_SECRET", "change_this_secret")
JWT_ALG = "HS256"
JWT_EXPIRE_SECONDS = 60 * 60 * 12

def _norm_user(u: str) -> str:
    return (u or "").strip().lower()
def _norm_pwd(p: str) -> str:
    return (p or "").strip()

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        """)

def create_user(username: str, password: str) -> Tuple[bool, str]:
    username = _norm_user(username)
    password = _norm_pwd(password)
    if not username or not password:
        return False, "Username / password không được rỗng."
    if len(password) < 3:
        return False, "Password tối thiểu 3 ký tự."
    try:
        with get_conn() as c:
            c.execute(
                "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                (username, password, int(time.time()))
            )
        print(f"[auth_db] Created user: {username} / {password}")
        return True, f"Tạo tài khoản thành công cho '{username}'."
    except sqlite3.IntegrityError:
        return False, "Username đã tồn tại."

def verify_user(username: str, password: str) -> Tuple[bool, str]:
    username = _norm_user(username)
    password = _norm_pwd(password)
    if not username or not password:
        return False, "not_found"
    with get_conn() as c:
        row = c.execute(
            "SELECT password FROM users WHERE username = ?",
            (username,)
        ).fetchone()
    if not row:
        print(f"[auth_db] verify_user: {username} -> not_found")
        return False, "not_found"
    stored_pw = row[0]
    if stored_pw == password:
        print(f"[auth_db] verify_user: {username} -> OK")
        return True, "ok"
    else:
        print(f"[auth_db] verify_user: {username} -> wrong_password (expect {stored_pw}, got {password})")
        return False, "wrong_password"

def gen_jwt(username: str) -> str:
    now = int(time.time())
    payload = {"sub": username, "iat": now, "exp": now + JWT_EXPIRE_SECONDS}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def verify_jwt(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None

def safe_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)

def list_users():
    with get_conn() as c:
        return [r[0] for r in c.execute("SELECT username FROM users").fetchall()]
