# player_standalone.py
import os
import gradio as gr
from pathlib import Path

# =========================
# CẤU HÌNH
# =========================
# Thư mục gốc chứa nhạc, mỗi subfolder là 1 "Thể loại" (genre)
MUSIC_ROOT = r"C:\demo_audio"      # <--- ĐỔI CHO PHÙ HỢP
SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

# =========================
# CHỈNH SỬA GIAO DIỆN — KHÔNG PHỤ THUỘC BÊN NGOÀI
# =========================

# Dạng row: {id, title, artist, genre, filepath, duration_sec(=None)}
_index_rows = []
_genres = []

def _build_index():
    """
    Quét MUSIC_ROOT -> tạo index rows + danh sách genres.
    Genre = tên thư mục con ngay dưới MUSIC_ROOT (hoặc sâu hơn: lấy thư mục chứa file).
    """
    global _index_rows, _genres
    _index_rows = []
    seen_genres = set()
    _id = 1

    root = Path(MUSIC_ROOT)
    if not root.exists():
        print(f"[WARN] MUSIC_ROOT không tồn tại: {MUSIC_ROOT}")
        _genres = []
        return

    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in SUPPORTED_EXTS:
                continue
            if not p.exists():
                continue

            # genre = tên thư mục chứa file (parent)
            genre = p.parent.name
            title = p.stem
            artist = ""  # không dùng DB nên để trống
            row = {
                "id": _id,
                "title": title,
                "artist": artist,
                "genre": genre,
                "filepath": str(p),
                "duration_sec": None,
            }
            _index_rows.append(row)
            seen_genres.add(genre)
            _id += 1

    _genres = sorted(seen_genres)

def fetch_genres():
    return list(_genres)

def fetch_all_tracks(limit=100):
    return _index_rows[:limit]

def fetch_tracks_by_genre(genre, limit=200):
    if not _index_rows:
        return []
    if not genre or genre == "Tất cả":
        return _index_rows[:limit]
    out = [r for r in _index_rows if r.get("genre") == genre]
    return out[:limit]

def _rows_to_playlist(rows):
    """
    rows: [{id,title,artist,genre,filepath,duration_sec}, ...]
    -> [(display_name, filepath), ...]
    Chỉ giữ file còn tồn tại.
    """
    out = []
    for r in rows:
        p = r.get("filepath") or ""
        if p and os.path.exists(p):
            display = f"{r['id']} • {r['title'] or os.path.basename(p)}"
            if r.get("artist"):
                display += f" • {r['artist']}"
            out.append((display, p))
    return out

# ----- Events -----
def load_initial():
    genres = fetch_genres()
    genres = ["Tất cả"] + genres if genres else ["Tất cả"]
    rows = fetch_all_tracks(1000)
    playlist = _rows_to_playlist(rows)
    names = [n for (n, _) in playlist]
    first_name, first_src = (playlist[0] if playlist else (None, None))
    status = f"✅ Nạp {len(playlist)} bài từ thư mục." if playlist else "⚠️ Không tìm thấy bài nào. Kiểm tra MUSIC_ROOT và đuôi file."
    return (
        gr.update(choices=genres, value=genres[0]),
        gr.update(choices=names, value=first_name),
        first_src,
        status,
    )

def on_change_genre(genre):
    rows = fetch_tracks_by_genre(genre, 1000)
    playlist = _rows_to_playlist(rows)
    names = [n for (n, _) in playlist]
    first_name, first_src = (playlist[0] if playlist else (None, None))
    return gr.update(choices=names, value=first_name), first_src

def on_refresh(genre):
    # Cho phép refresh lại index từ đĩa (có thể bạn vừa thêm file)
    _build_index()
    return on_change_genre(genre)

def on_select_song(name, genre):
    rows = fetch_tracks_by_genre(genre, 1000)
    playlist = _rows_to_playlist(rows)
    for n, p in playlist:
        if n == name:
            return p
    return None

def next_prev(current_name, genre, direction="next"):
    rows = fetch_tracks_by_genre(genre, 1000)
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
    with gr.Blocks(title="🎵 Music Player — Standalone", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🎵 Trình phát nhạc (đọc từ thư mục cục bộ) — *Standalone demo*")
        with gr.Row():
            genre_dd   = gr.Dropdown(choices=[], label="Thể loại", interactive=True, scale=1)
            refresh_btn= gr.Button("🔄 Refresh", variant="secondary", scale=0)
            status_md  = gr.Markdown("")

        with gr.Row(equal_height=True):
            song_dd = gr.Dropdown(choices=[], label="Danh sách bài", interactive=True, scale=3)
            with gr.Column(scale=1):
                prev_btn = gr.Button("⏮️ Trước")
                next_btn = gr.Button("⏭️ Sau")

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

        # Refresh (quét lại thư mục + reload theo genre hiện tại)
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

        gr.Markdown(
            "> 📂 **Mẹo:** Sắp xếp nhạc như `MUSIC_ROOT/bolero/*.mp3`, `MUSIC_ROOT/cailuong/*.mp3`… "
            "để dropdown *Thể loại* hiển thị theo tên folder."
        )

    return demo

if __name__ == "__main__":
    _build_index()
    app = build_player_ui()
    # Mở LAN nếu muốn test trên máy khác: server_name="0.0.0.0"
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
