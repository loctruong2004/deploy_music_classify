# login_page.py
import gradio as gr
from auth_db import init_db, create_user
from ui_navbar import add_navbar

init_db()

def do_register(u, p, p2):
    if (p or "") != (p2 or ""): return "❌ Mật khẩu nhập lại không khớp."
    u = (u or "").strip().lower()
    ok, msg = create_user(u, p or "")
    return ("✅ " + msg) if ok else ("❌ " + msg)

def ui():
    with gr.Blocks(title="🔐 Login & Register") as demo:
        gr.Markdown("## 🔐 Đăng nhập / Đăng ký")

        with gr.Tab("Đăng nhập"):
            u1 = gr.Textbox(label="Username", placeholder="ví dụ: loc")
            p1 = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Đăng nhập", variant="primary")
            login_msg = gr.Markdown("", elem_id="login-msg")

            # Gọi fetch tới /auth/login để set cookie HttpOnly rồi redirect
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
            u2 = gr.Textbox(label="Username mới")
            p2 = gr.Textbox(label="Password", type="password")
            p3 = gr.Textbox(label="Nhập lại password", type="password")
            reg_btn = gr.Button("Tạo tài khoản")
            reg_msg = gr.Markdown("")
            reg_btn.click(do_register, [u2, p2, p3], [reg_msg], show_progress=False)
    return demo

login_app = ui()
