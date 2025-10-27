# main_app.py
import gradio as gr
from http.cookies import SimpleCookie
from auth_db import verify_jwt
from app_ui import build_app_ui
from ui_navbar import add_navbar

def _read_token_from_cookie(request: gr.Request):
    raw = request.headers.get("cookie", "") if request and request.headers else ""
    c = SimpleCookie(); 
    try: c.load(raw)
    except: return None
    return c.get("access_token").value if "access_token" in c else None

def ui():
    with gr.Blocks(title="🎧 Music Genre Classifier (Protected)") as demo:
        add_navbar(active="app")
        gate_msg = gr.Markdown("⛔ Chưa đăng nhập. Vui lòng vào **/login**.")
        app_group = gr.Group(visible=False)

        with app_group:
            build_app_ui()

        def on_load(request: gr.Request):
            token = _read_token_from_cookie(request)
            user = verify_jwt(token) if token else None
            if not user:
                return gr.update(visible=True), gr.update(visible=False), "⛔ Chưa đăng nhập. Vui lòng vào **/login**."
            return gr.update(visible=False), gr.update(visible=True), f"👋 Xin chào **{user}**"

        demo.load(on_load, None, [gate_msg, app_group, gate_msg])
    return demo

main_app = ui()
