# run_server.py
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr

from login_page import login_app
from main_app import main_app
from player_page import player_app

from auth_db import verify_user, gen_jwt, JWT_EXPIRE_SECONDS  # dùng để set cookie

app = FastAPI(title="Music Classifier with Auth & Player")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ===== NEW: API auth đặt/huỷ cookie =====
@app.post("/auth/login")
async def api_login(req: Request):
    data = await req.json()
    u = (data.get("username") or "").strip().lower()
    p = (data.get("password") or "")
    ok, _ = verify_user(u, p)
    if not ok:
        return JSONResponse({"ok": False, "msg": "Invalid credentials"}, status_code=401)

    token = gen_jwt(u)
    resp = JSONResponse({"ok": True})
    # Ở production hãy đặt secure=True + samesite="Strict" và dùng HTTPS
    resp.set_cookie(
        key="access_token",
        value=token,
        max_age=JWT_EXPIRE_SECONDS,
        httponly=True,
        secure=False,         # True nếu dùng HTTPS
        samesite="Lax",       # 'Strict' nếu muốn chặt chẽ hơn
        path="/"
    )
    return resp

@app.post("/auth/logout")
async def api_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("access_token", path="/")
    return resp
# ========================================

# Mount các trang
app = gr.mount_gradio_app(app, login_app,  path="/login")
app = gr.mount_gradio_app(app, main_app,   path="/app")
app = gr.mount_gradio_app(app, player_app, path="/player")

if __name__ == "__main__":
    uvicorn.run("run_server:app", host="0.0.0.0", port=8001, reload=True)
