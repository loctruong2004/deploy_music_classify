# player_page.py
import os
import gradio as gr
from http.cookies import SimpleCookie
from typing import List, Tuple

from auth_db import verify_jwt
from ui_navbar import add_navbar
from music_db import init_music_db, list_genres, list_tracks_by_genre, search_tracks

# Khởi tạo DB nhạc (có seed demo nếu thiếu)
init_music_db(seed_demo=True)

def _read_token_from_cookie(request: gr.Request):
    raw = request.headers.get("cookie", "") if request and hasattr(request, "headers") and request.headers else ""
    c = SimpleCookie()
    try:
        c.load(raw)
    except Exception:
        return None
    return c.get("access_token").value if "access_token" in c else None

def _filter_existing_files(rows: List[Tuple[int, str, str, str, str, int]]):
    """
    Lọc những track có file tồn tại thật sự trên đĩa,
    tránh trường hợp DB có mà file thiếu.
    """
    out = []
    for r in rows:
        _, title, artist, genre, path, duration = r
        if path and os.path.exists(path):
            out.append(r)
    return out

def _to_table(rows: List[Tuple[int, str, str, str, str, int]]):
    """
    Convert rows -> dữ liệu bảng hiển thị
    """
    data = []
    for (tid, title, artist, genre, path, dur) in rows:
        mins = (dur or 0) // 60
        secs = (dur or 0) % 60
        data.append([tid, title, artist or "", genre, f"{mins:02d}:{secs:02d}", path])
    return data

def load_genres_and_tracks():
    genres = list_genres()
    genres = ["Tất cả"] + genres if genres else ["Tất cả"]
    rows = _filter_existing_files(list_tracks_by_genre(None))
    table = _to_table(rows)
    # default lựa chọn track đầu tiên nếu có
    first_src = rows[0][4] if rows else None
    first_title = rows[0][1] if rows else None
    return gr.update(choices=genres, value=genres[0]), table, first_title, first_src

def on_change_genre(genre):
    rows = _filter_existing_files(list_tracks_by_genre(genre))
    table = _to_table(rows)
    first_src = rows[0][4] if rows else None
    first_title = rows[0][1] if rows else None
    return table, first_title, first_src

def on_search(q, cur_genre):
    rows = _filter_existing_files(search_tracks(q, cur_genre))
    table = _to_table(rows)
    first_src = rows[0][4] if rows else None
    first_title = rows[0][1] if rows else None
    return table, first_title, first_src

def on_select_row(evt: gr.SelectData, current_table):
    """
    Khi người dùng click 1 dòng trong Dataframe, evt.index trả về (row_idx, col_idx)
    Ta lấy path ở cột cuối cùng (index -1).
    """
    if not current_table:
        return None, None
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx is None or row_idx >= len(current_table):
        return None, None
    row = current_table[row_idx]
    title = row[1]       # cột Title
    path  = row[-1]      # cột Filepath (ẩn trong UI)
    return title, path

def next_prev(current_title, current_table, direction="next"):
    if not current_table:
        return None, None
    titles = [r[1] for r in current_table]
    paths  = [r[-1] for r in current_table]
    if not titles:
        return None, None
    try:
        idx = titles.index(current_title) if current_title in titles else -1
    except ValueError:
        idx = -1
    if idx < 0:
        idx = 0
    else:
        if direction == "next":
            idx = (idx + 1) % len(titles)
        else:
            idx = (idx - 1) % len(titles)
    return titles[idx], paths[idx]

def build_player_ui():
    with gr.Blocks(title="🎵 Music Player") as demo:
        # Navbar
        add_navbar(active="player")

        # Cổng xác thực
        gate_msg = gr.Markdown("⛔ Chưa đăng nhập. Vui lòng vào **/login**.")
        app_group = gr.Group(visible=False)

        with app_group:
            gr.Markdown("## 🎵 Trình phát nhạc theo thể loại (từ CSDL)")

            # State giữ bảng hiện tại (để xử lý next/prev & click)
            table_state = gr.State(value=[])

            with gr.Row():
                genre_dd = gr.Dropdown(choices=[], label="Thể loại", interactive=True, scale=1)
                search_box = gr.Textbox(label="Tìm bài/ca sĩ", placeholder="Nhập tên bài hoặc ca sĩ…", scale=2)
                search_btn = gr.Button("🔎 Tìm", variant="secondary", scale=0)

            # Bảng danh sách bài hát (ẩn cột filepath bằng cách để cuối, user không cần quan tâm)
            tracks_table = gr.Dataframe(
                headers=["ID", "Title", "Artist", "Genre", "Duration", "Filepath"],
                datatype=["number", "str", "str", "str", "str", "str"],
                row_count=(0, "dynamic"),
                col_count=(6, "fixed"),
                wrap=True,
                interactive=False,
                label="Danh sách bài hát (click một dòng để phát)"
            )

            with gr.Row():
                prev_btn = gr.Button("⏮️ Trước", variant="secondary")
                play_title = gr.Textbox(label="Đang phát", interactive=False)
                next_btn = gr.Button("⏭️ Sau", variant="secondary")

            player = gr.Audio(label="Trình phát", autoplay=True, interactive=False)

            # --- Nạp genres + tracks ban đầu ---
            demo.load(
                fn=load_genres_and_tracks,
                inputs=None,
                outputs=[genre_dd, tracks_table, play_title, player]
            )

            # Đồng bộ state bảng khi nạp/đổi thể loại/tìm kiếm
            def _sync_state(df):
                return df
            tracks_table.change(_sync_state, [tracks_table], [table_state], show_progress=False)

            # Đổi thể loại
            genre_dd.change(
                fn=on_change_genre,
                inputs=[genre_dd],
                outputs=[tracks_table, play_title, player]
            ).then(_sync_state, [tracks_table], [table_state], show_progress=False)

            # Tìm kiếm
            search_btn.click(
                fn=on_search,
                inputs=[search_box, genre_dd],
                outputs=[tracks_table, play_title, player]
            ).then(_sync_state, [tracks_table], [table_state], show_progress=False)

            # Chọn 1 dòng để phát
            tracks_table.select(
                fn=on_select_row,
                inputs=[tracks_table],
                outputs=[play_title, player]
            )

            # Next / Prev
            next_btn.click(
                fn=lambda t, df: next_prev(t, df, "next"),
                inputs=[play_title, table_state],
                outputs=[play_title, player]
            )
            prev_btn.click(
                fn=lambda t, df: next_prev(t, df, "prev"),
                inputs=[play_title, table_state],
                outputs=[play_title, player]
            )

            gr.Markdown("> Mẹo: Click một dòng trong bảng để phát. Có thể lọc theo **Thể loại** hoặc dùng ô **Tìm kiếm**.")

        # Kiểm tra cookie để mở khóa trang
        def on_load(request: gr.Request):
            token = _read_token_from_cookie(request)
            user = verify_jwt(token) if token else None
            if not user:
                return gr.update(visible=True), gr.update(visible=False), "⛔ Chưa đăng nhập. Vui lòng vào **/login**."
            return gr.update(visible=False), gr.update(visible=True), f"🎧 Xin chào **{user}**"

        demo.load(on_load, None, [gate_msg, app_group, gate_msg])

    return demo

player_app = build_player_ui()
