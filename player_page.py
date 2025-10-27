# player_page.py
import os
import gradio as gr
from http.cookies import SimpleCookie

from ui_navbar import add_navbar
# Nếu bạn dùng đăng nhập bằng cookie:
from auth_db import verify_jwt

from db_tracks_mssql import fetch_genres, fetch_tracks_by_genre, fetch_all_tracks

def _read_token_from_cookie(request: gr.Request):
    raw = request.headers.get("cookie", "") if request and hasattr(request, "headers") and request.headers else ""
    c = SimpleCookie()
    try: c.load(raw)
    except: return None
    return c.get("access_token").value if "access_token" in c else None

def _rows_to_playlist(rows):
    """
    rows: [{id,title,artist,genre,filepath,duration_sec}, ...]
    -> names: "id • title • artist", path: filepath
    chỉ giữ file tồn tại thật sự
    """
    out = []
    for r in rows:
        p = r.get("filepath") or ""
        if p and os.path.exists(p):
            display = f"{r['id']} • {r['title'] or os.path.basename(p)}" + (f" • {r['artist']}" if r.get("artist") else "")
            out.append((display, p))
    return out

# ----- Events -----
def load_initial():
    # genres + all tracks
    genres = fetch_genres()
    genres = ["Tất cả"] + genres if genres else ["Tất cả"]
    rows = fetch_all_tracks(100)
    playlist = _rows_to_playlist(rows)
    names = [n for (n, _) in playlist]
    first_name, first_src = (playlist[0] if playlist else (None, None))
    return gr.update(choices=genres, value=genres[0]), gr.update(choices=names, value=first_name), first_src, f"✅ Nạp {len(playlist)} bài từ DB."

def on_change_genre(genre):
    rows = fetch_tracks_by_genre(genre, 200)
    playlist = _rows_to_playlist(rows)
    names = [n for (n, _) in playlist]
    first_name, first_src = (playlist[0] if playlist else (None, None))
    return gr.update(choices=names, value=first_name), first_src

def on_refresh(genre):
    return on_change_genre(genre)

def on_select_song(name, genre):
    rows = fetch_tracks_by_genre(genre, 200)
    playlist = _rows_to_playlist(rows)
    for n, p in playlist:
        if n == name:
            return p
    return None

def next_prev(current_name, genre, direction="next"):
    rows = fetch_tracks_by_genre(genre, 200)
    playlist = _rows_to_playlist(rows)
    if not playlist:
        return None, None
    names = [n for (n, _) in playlist]
    paths = [p for (_, p) in playlist]
    try:
        idx = names.index(current_name) if current_name in names else -1
    except ValueError:
        idx = -1
    if idx < 0:
        idx = 0
    else:
        idx = (idx + 1) % len(names) if direction == "next" else (idx - 1) % len(names)
    return names[idx], paths[idx]

def build_player_ui():
    with gr.Blocks(title="🎵 Music Player (SQL Server)") as demo:
        add_navbar(active="player")

        # Gate bằng cookie (bỏ nếu không dùng auth)
        gate_msg = gr.Markdown("⛔ Chưa đăng nhập. Vui lòng vào **/login**.")
        app_group = gr.Group(visible=False)

        with app_group:
            gr.Markdown("## 🎵 Trình phát nhạc (đọc từ SQL Server → phát theo đường dẫn)")

            with gr.Row():
                genre_dd = gr.Dropdown(choices=[], label="Thể loại", interactive=True, scale=1)
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=0)
                status_md = gr.Markdown("")

            with gr.Row():
                song_dd = gr.Dropdown(choices=[], label="Danh sách bài (từ DB)", interactive=True, scale=2)
                prev_btn = gr.Button("⏮️ Trước", scale=1)
                next_btn = gr.Button("⏭️ Sau", scale=1)

            now_playing = gr.Textbox(label="Đang chọn", interactive=False)
            player = gr.Audio(label="Trình phát", autoplay=True, interactive=False)

            # Load ban đầu
            demo.load(
                fn=load_initial,
                inputs=None,
                outputs=[genre_dd, song_dd, player, status_md]
            ).then(
                lambda name: name, [song_dd], [now_playing], show_progress=False
            )

            # Đổi thể loại
            genre_dd.change(
                fn=on_change_genre,
                inputs=[genre_dd],
                outputs=[song_dd, player]
            ).then(
                lambda name: name, [song_dd], [now_playing], show_progress=False
            )

            # Refresh danh sách theo thể loại hiện tại
            refresh_btn.click(
                fn=on_refresh,
                inputs=[genre_dd],
                outputs=[song_dd, player]
            ).then(
                lambda name: name, [song_dd], [now_playing], show_progress=False
            )

            # Chọn bài
            song_dd.change(
                fn=on_select_song,
                inputs=[song_dd, genre_dd],
                outputs=[player]
            ).then(
                lambda name: name, [song_dd], [now_playing], show_progress=False
            )

            # Next/Prev
            next_btn.click(
                fn=lambda cur, g: next_prev(cur, g, "next"),
                inputs=[now_playing, genre_dd],
                outputs=[song_dd, player]
            ).then(
                lambda name: name, [song_dd], [now_playing], show_progress=False
            )

            prev_btn.click(
                fn=lambda cur, g: next_prev(cur, g, "prev"),
                inputs=[now_playing, genre_dd],
                outputs=[song_dd, player]
            ).then(
                lambda name: name, [song_dd], [now_playing], show_progress=False
            )

            gr.Markdown("> Chỉ **đọc DB** để lấy `filepath` và phát nhạc. Đảm bảo đường dẫn tồn tại trên máy server.")

        # Auth gate
        def on_load(request: gr.Request):
            token = _read_token_from_cookie(request)
            user = verify_jwt(token) if token else None
            if not user:
                return gr.update(visible=True), gr.update(visible=False), "⛔ Chưa đăng nhập. Vui lòng vào **/login**."
            return gr.update(visible=False), gr.update(visible=True), f"🎧 Xin chào **{user}**"

        demo.load(on_load, None, [gate_msg, app_group, gate_msg])

    return demo

player_app = build_player_ui()
