# login_page.py
import gradio as gr
from auth_db import init_db, create_user, verify_user, gen_jwt, DB_PATH

init_db()

def do_register(u, p, p2):
    if (p or "") != (p2 or ""):
        return "❌ Mật khẩu nhập lại không khớp."
    u = (u or "").strip().lower()
    ok, msg = create_user(u, p or "")
    return ("✅ " + msg) if ok else ("❌ " + msg)

def do_login_return_token_and_msg(u, p):
    """
    Trả về (token_or_empty, ui_message, fallback_html)
    - token_or_empty: chuỗi token nếu ok, "" nếu fail
    - ui_message: message hiển thị ngay dưới nút
    - fallback_html: nếu ok -> trả thẻ meta refresh để phòng JS bị chặn; nếu fail -> ""
    """
    u = (u or "").strip().lower()
    p = (p or "")
    ok, reason = verify_user(u, p)
    if ok:
        token = str(gen_jwt(u))
        msg = "✅ Đăng nhập thành công. Đang chuyển trang…"
        html = f'<meta http-equiv="refresh" content="0; url=/app?token={token}">'
        return token, msg, html
    else:
        if reason == "not_found":
            msg = "❌ Không tìm thấy username."
        elif reason == "wrong_password":
            msg = "❌ Mật khẩu không đúng."
        else:
            msg = "❌ Sai username hoặc password."
        return "", msg, ""

def ui():
    with gr.Blocks(title="🔐 Login & Register") as demo:
        gr.Markdown("## 🔐 Đăng nhập / Đăng ký")

        # ===== Tab Đăng nhập =====
        with gr.Tab("Đăng nhập"):
            u1 = gr.Textbox(label="Username", placeholder="ví dụ: loc")
            p1 = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Đăng nhập", variant="primary")
            login_msg = gr.Markdown("", elem_id="login-msg")
            token_state = gr.State("")
            redirect_html = gr.HTML("")  # fallback khi JS bị chặn

            # 1) click -> server trả (token, message, meta_refresh)
            login_evt = login_btn.click(
                fn=do_login_return_token_and_msg,
                inputs=[u1, p1],
                outputs=[token_state, login_msg, redirect_html],
                show_progress=False
            )

            # 2) nếu có token -> JS redirect (ổn định hơn meta refresh)
            login_evt.then(
                fn=None,
                inputs=[token_state],
                outputs=[],
                js="""
                (token) => {
                    if (token) {
                        const url = `/app?token=${token}`;
                        try { window.top.location.href = url; }
                        catch(e) { window.location.href = url; }
                    }
                }
                """
            )

            gr.Markdown("> Mẹo: Username tự chuyển về **chữ thường** khi đăng ký/đăng nhập. DB: `" + DB_PATH.replace("\\","/") + "`")

        # ===== Tab Đăng ký =====
        with gr.Tab("Đăng ký"):
            u2 = gr.Textbox(label="Username mới", placeholder="chỉ chữ & số, không dấu cách")
            p2 = gr.Textbox(label="Password", type="password")
            p3 = gr.Textbox(label="Nhập lại password", type="password")
            reg_btn = gr.Button("Tạo tài khoản")
            reg_msg = gr.Markdown("")
            reg_btn.click(do_register, [u2, p2, p3], [reg_msg], show_progress=False)

    return demo

login_app = ui()
