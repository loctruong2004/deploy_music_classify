# login_page.py
import gradio as gr
from auth_db import init_db, create_user
from ui_navbar import add_navbar

init_db()

def do_register(u, p, p2):
    if (p or "") != (p2 or ""):
        return "❌ Mật khẩu nhập lại không khớp."
    u = (u or "").strip().lower()
    ok, msg = create_user(u, p or "")
    return ("✅ " + msg) if ok else ("❌ " + msg)

def ui():
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #1f1f1f, #2c2c2c);
        color: white;
        font-family: 'Inter', sans-serif;
    }
    .gr-tab {
        background-color: transparent !important;
    }
    .gr-group {
        border: 1px solid #444;
        border-radius: 14px;
        background-color: #2b2b2b;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    input, textarea {
        background-color: #1e1e1e !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 8px !important;
    }
    button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 0 !important;
    }
    #login-msg, #reg-msg {
        text-align: center;
        font-size: 15px;
        font-weight: 500;
        margin-top: 10px;
    }
    h2 {
        text-align: center;
        color: #f0f0f0;
    }
    """

    with gr.Blocks(css=custom_css, title="🔐 Login & Register") as demo:
        # add_navbar(active="login")

        with gr.Row():
            gr.Markdown("## 🔐 Đăng nhập / Đăng ký", elem_id="header")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("")
            with gr.Column(scale=2):
                with gr.Tab("Đăng nhập"):
                    with gr.Group():
                        u1 = gr.Textbox(label="👤 Username", placeholder="ví dụ: loc")
                        p1 = gr.Textbox(label="🔑 Password", type="password")
                        login_btn = gr.Button("Đăng nhập", variant="primary")
                        login_msg = gr.Markdown("", elem_id="login-msg")

                        login_btn.click(
                            fn=None, inputs=[u1, p1], outputs=[],
                            js=r"""
                            (u, p) => {
                              fetch('/auth/login', {
                                method: 'POST',
                                headers: {'Content-Type':'application/json'},
                                body: JSON.stringify({username: (u||'').trim().toLowerCase(), password: p||''})
                              }).then(async r => {
                                if (r.ok) {
                                  try { window.top.location.href = '/app'; }
                                  catch(e){ window.location.href = '/app'; }
                                } else {
                                  const el = document.querySelector('#login-msg');
                                  if (el) el.innerHTML = "❌ Sai username hoặc password.";
                                }
                              }).catch(() => {
                                const el = document.querySelector('#login-msg');
                                if (el) el.innerHTML = "❌ Lỗi kết nối server.";
                              });
                            }
                            """
                        )

                with gr.Tab("Đăng ký"):
                    with gr.Group():
                        u2 = gr.Textbox(label="👤 Username mới")
                        p2 = gr.Textbox(label="🔑 Password", type="password")
                        p3 = gr.Textbox(label="🔁 Nhập lại password", type="password")
                        reg_btn = gr.Button("Tạo tài khoản", variant="secondary")
                        reg_msg = gr.Markdown("", elem_id="reg-msg")
                        reg_btn.click(do_register, [u2, p2, p3], [reg_msg], show_progress=False)

            with gr.Column(scale=1):
                gr.Markdown("")

    return demo

login_app = ui()
