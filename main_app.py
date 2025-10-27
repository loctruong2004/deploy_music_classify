# main_app.py
import gradio as gr
from auth_db import verify_jwt
from app_ui import build_app_ui  # giao diện & gradio_predict từ app_ui.py

def ui():
    with gr.Blocks(title="🎧 Music Genre Classifier (Protected)") as demo:
        gate_msg = gr.Markdown("⛔ Chưa đăng nhập. Vui lòng vào **/login**.")
        app_group = gr.Group(visible=False)

        # Tải UI app thật (của bạn) vào app_group
        with app_group:
            app = build_app_ui()

        def on_load(request: gr.Request):
            token = request.query_params.get("token") if request and request.query_params else None
            user = verify_jwt(token) if token else None
            if not user:
                return gr.update(visible=True), gr.update(visible=False), "⛔ Chưa đăng nhập. Vui lòng vào **/login**."
            return gr.update(visible=False), gr.update(visible=True), f"👋 Xin chào **{user}**"

        demo.load(on_load, None, [gate_msg, app_group, gate_msg])

    return demo

main_app = ui()
