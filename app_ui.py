# app_ui.py
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
import torch.nn.functional as F
import torch
from torchvision import models, transforms
import os
import librosa as lb
import soundfile as sf
import numpy as np
import datetime
import tempfile
import re
import random

# === NEW: for YouTube download ===
from yt_dlp import YoutubeDL

class_names =  ['bolero', 'cailuong', 'cheo', 'danca', 'nhacdo']
img_height,img_width=224,224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    if model_name == "Model 5 class: bolero, cailuong, cheo, danca, nhacdo":
        classes = ['bolero', 'cailuong', 'cheo', 'danca', 'nhacdo']
        model_path = "./efficientnet_b0_5_lager.pth"
        num_classes = 5
    elif model_name == "Model 7 class: bolero, cai luong, cheo, dan ca, nhac do, thieu nhi,other":
        classes = ['bolero', 'cailuong', 'cheo', 'danca', 'nhacdo', 'other', 'thieunhi']
        model_path = "./efficientnet_b0_7_final.pth"
        num_classes = 7
    elif model_name == "Model 8 class: bolero, cai luong, chauvan, cheo, dan ca, pop ,remix, rap":
        classes = ['Pop', 'bolero', 'cailuong', 'chauvan', 'cheo', 'danca', 'rap', 'remix']
        model_path = "./train_with_8_class.pth"
        num_classes = 8
    else:
        raise ValueError("Model không hợp lệ")

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_fft(samples, n_fft=2048, hop_length=512):
    for index, item in samples.items():
        D = np.abs(lb.stft(item["sampling"], n_fft=n_fft, hop_length=hop_length))
        samples[index]["stft"] = D
    return samples

def get_mel_spectrogram(samples, sr=22050):
    for index, item in samples.items():
        S = lb.feature.melspectrogram(y=item["sampling"], sr=sr)
        S_db = lb.amplitude_to_db(S, ref=np.max)
        samples[index]["mel-spec-db"] = S_db
    return samples

def save_mel_spec(samples, root):
    image_paths = []
    for index, item in samples.items():
        S_db = item["mel-spec-db"]
        os.makedirs(root, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(item["dir"]))[0]
        out_path = os.path.join(root, file_name + ".png")
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        S_rgb = cm.viridis(S_db_norm)[:, :, :3]
        S_rgb = (S_rgb * 255).astype(np.uint8)
        im = Image.fromarray(S_rgb).resize((224, 224))
        im.save(out_path)
        image_paths.append(out_path)
    return image_paths
def _split_signal_random(
    y: np.ndarray,
    sr: int,
    segment_duration: float = 30.0,
    segments_per_file: int | None = None,
    min_segments: int = 1,
    max_segments: int = 12,
    overlap_ratio: float = 0.5,
    avoid_silence: bool = True,
    top_db: int = 35,
    seed: int | None = None,
    pad_to_full: bool = True
):
    """
    Trả về danh sách tuple: (seg_audio_np, start_sample, end_sample, start_sec, end_sec)
    theo đúng logic random/overlap/né im lặng.
    """
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    seg_len = int(segment_duration * sr)
    if len(y) < max(seg_len // 3, 1):
        return []

    # 1) Vùng ứng viên
    candidate_intervals = [(0, len(y))]
    if avoid_silence:
        intervals = lb.effects.split(y, top_db=top_db)
        candidate_intervals = [(int(s), int(e)) for s, e in intervals if (e - s) >= max(seg_len // 3, 1)]
        if not candidate_intervals:
            candidate_intervals = [(0, len(y))]

    # 2) Suy số đoạn auto
    usable_len = sum(e - s for s, e in candidate_intervals)
    if segments_per_file is None:
        if usable_len < seg_len:
            n_auto = 1
        else:
            step = max(int(seg_len * (1 - max(min(overlap_ratio, 0.95), 0.0))), 1)
            n_auto = int(np.floor((usable_len - seg_len) / step)) + 1
        n_target = int(np.clip(n_auto, min_segments, max_segments))
    else:
        n_target = max(0, segments_per_file)

    if n_target == 0:
        return []

    # 3) Chọn start times ngẫu nhiên
    starts = []
    attempts, max_attempts = 0, n_target * 80
    step = max(int(seg_len * (1 - max(min(overlap_ratio, 0.95), 0.0))), 1)

    start_grid = []
    for s, e in candidate_intervals:
        if e - s < seg_len:
            if pad_to_full and (e - s) > 0:
                start_grid.append(s)
            continue
        pos = s
        while pos <= e - seg_len:
            start_grid.append(pos)
            pos += step

    if not start_grid:
        mid_start = max(0, (len(y) - seg_len) // 2)
        start_grid = [mid_start]

    while len(starts) < n_target and attempts < max_attempts:
        attempts += 1
        st = random.choice(start_grid)
        ok = True
        if overlap_ratio <= 0.01:
            for s0 in starts:
                if not (st + seg_len <= s0 or s0 + seg_len <= st):
                    ok = False; break
        if ok:
            starts.append(st)

    if not starts:
        return []

    # 4) Lấy segment
    segments = []
    for st in sorted(starts)[:n_target]:
        ed = st + seg_len
        seg = y[st:ed]
        if len(seg) < seg_len and pad_to_full:
            seg = np.pad(seg, (0, seg_len - len(seg)), mode="constant")
        elif len(seg) < seg_len and not pad_to_full:
            continue
        start_sec = st / sr
        end_sec = min(ed, len(y)) / sr
        segments.append((seg, st, ed, start_sec, end_sec))

    return segments
def gradio_predict(audio_file, selected_model):
    print(f"\n👉 Nhận file: {audio_file}")
    model, class_names_local = load_model(selected_model)
    results = []
    try:
        y, sr = lb.load(audio_file, sr=None, mono=True)
    except Exception as e:
        return f"❌ Lỗi khi load file: {e}"

    # ====== Cấu hình cắt theo logic mới ======
    SEGMENT_DURATION = 30.0
    MIN_SEGMENTS     = 1
    MAX_SEGMENTS     = 12
    OVERLAP_RATIO    = 0   #  10% chồng lấn
    AVOID_SILENCE    = True
    TOP_DB           = 35
    PAD_TO_FULL      = True
    SEED             = None

    segs = _split_signal_random(
        y=y, sr=sr,
        segment_duration=SEGMENT_DURATION,
        segments_per_file=None,         # auto theo độ dài
        min_segments=MIN_SEGMENTS,
        max_segments=MAX_SEGMENTS,
        overlap_ratio=OVERLAP_RATIO,
        avoid_silence=AVOID_SILENCE,
        top_db=TOP_DB,
        seed=SEED,
        pad_to_full=PAD_TO_FULL
    )

    if not segs:
        return "⚠️ File quá ngắn hoặc không trích xuất được đoạn hợp lệ."

    base_filename = os.path.splitext(os.path.basename(audio_file))[0]
    folder_path = os.path.dirname(audio_file)
    output_folder = os.path.join(folder_path, "predict")
    os.makedirs(output_folder, exist_ok=True)

    # Ghi các đoạn đã cắt và tạo mel-spec
    samples = {}
    for i, (seg, st, ed, t0, t1) in enumerate(segs, 1):
        new_filename = f"{base_filename}_part{i}.wav"
        new_path = os.path.join(output_folder, new_filename)
        sf.write(new_path, seg, sr)
        samples[i] = {"dir": new_path, "sampling": seg, "t0": t0, "t1": t1}

    samples = get_fft(samples)
    samples = get_mel_spectrogram(samples, sr)
    mel_root = os.path.join(output_folder, "mel-images")
    os.makedirs(mel_root, exist_ok=True)
    list_test = save_mel_spec(samples, mel_root)

    # Suy đoán từng ảnh mel
    for idx, path in enumerate(list_test, 1):
        image_pil = Image.open(path).convert("RGB").resize((224, 224))
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            output_class = class_names_local[pred.item()]
            percentage = conf.item() * 100

            # Dùng time từ dict samples tương ứng
            t0 = samples[idx]["t0"]
            t1 = samples[idx]["t1"]
            start_time = str(datetime.timedelta(seconds=int(t0)))
            end_time   = str(datetime.timedelta(seconds=int(t1)))
            results.append(f"[{start_time} → {end_time}] → {output_class} ({percentage:.2f}%)")

    return "\n".join(results)


# === NEW: helper để tải YouTube và trích xuất WAV ===
_SANITIZE = re.compile(r'[^a-zA-Z0-9_\-\.]+')

def _safe_name(s: str) -> str:
    return _SANITIZE.sub('_', s).strip('_')
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

def _yt_opts_common(out_dir: str, use_cookies: bool = False):
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title).80s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "concurrent_fragment_downloads": 4,
        "http_headers": {
            # UA “thật” để hạn chế bị 403
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.youtube.com/",
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "prefer_ffmpeg": True,
        "nocheckcertificate": True,    # tránh cert lỗi lặt vặt
        "geo_bypass": True,
        "forceipv4": True,             # tránh IPv6 gây 403 ở một số mạng
    }
    if use_cookies:
        # Tự lấy cookie từ Chrome (Windows). Nếu dùng Edge/Firefox, đổi lại tuple.
        # Yêu cầu đang đăng nhập YT trên Chrome ở máy này.
        opts["cookiesfrombrowser"] = ("chrome", )
    return opts

def download_youtube_audio(url: str, out_dir: str) -> str:
    """
    Tải bestaudio từ YouTube -> WAV. Thử 2 bước:
    1) Không cookie
    2) Nếu thất bại (403/DownloadError) -> dùng cookies từ Chrome
    Trả về đường dẫn file WAV.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _try_download(use_cookies: bool):
        ydl_opts = _yt_opts_common(out_dir, use_cookies=use_cookies)
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "audio")
            wav_guess = _safe_name(title) + ".wav"
            candidate = os.path.join(out_dir, wav_guess)
            if os.path.exists(candidate):
                return candidate
            # fallback: lấy file .wav mới nhất
            wavs = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith(".wav")]
            if not wavs:
                raise RuntimeError("Không tìm thấy WAV sau khi trích xuất. Kiểm tra ffmpeg/yt_dlp.")
            return max(wavs, key=os.path.getmtime)

    # Thử không cookie
    try:
        return _try_download(use_cookies=False)
    except DownloadError as e:
        msg = str(e).lower()
        # Nếu có dấu hiệu 403/forbidden -> thử lại với cookies
        if "403" in msg or "forbidden" in msg:
            try:
                return _try_download(use_cookies=True)
            except Exception as e2:
                raise RuntimeError(f"YT 403. Thử lại với cookiesfrombrowser thất bại: {e2}") from e
        raise
def predict_from_ui(audio_path, model_name, input_mode, yt_url):
    try:
        if input_mode == "YouTube URL":
            if not yt_url or not yt_url.strip():
                return "⚠️ Vui lòng nhập YouTube URL."
            tmp_dir = os.path.join(os.getcwd(), "downloads_youtube")
            wav_path = download_youtube_audio(yt_url.strip(), tmp_dir)
            return gradio_predict(wav_path, model_name)
        else:
            if not audio_path:
                return "⚠️ Vui lòng chọn file âm thanh (WAV/MP3)."
            return gradio_predict(audio_path, model_name)
    except Exception as e:
        hint = (
            "\n\n🛠️ Gợi ý khắc phục:\n"
            "• `pip install -U yt-dlp`\n"
            "• Cài `ffmpeg` và thêm vào PATH\n"
            "• Nếu vẫn 403: đăng nhập YouTube trên Chrome cùng máy, rồi chạy lại (app sẽ tự lấy cookies)\n"
            "• Hoặc export `cookies.txt` và thay bằng `cookiefile` trong yt_dlp options"
        )
        return f"❌ Lỗi xử lý: {e}{hint}"
# === GỢI Ý NHẠC TƯƠNG TỰ ===
def suggest_similar_songs(result_text):
    if not result_text or "→" not in result_text:
        return "⚠️ Vui lòng chạy dự đoán trước khi gợi ý nhạc tương tự."

    # Lấy dòng cuối cùng và trích thể loại dự đoán
    last_line = result_text.strip().splitlines()[-1]
    match = re.search(r"→\s*(\w+)", last_line)
    if not match:
        return "⚠️ Không nhận diện được thể loại để gợi ý."
    genre = match.group(1).lower()

    suggestions = {
        "bolero": ["Duyên Phận - Như Quỳnh", "Ai Cho Tôi Tình Yêu - Lệ Quyên",
                    "Sầu Tím Thiệp Hồng - Quang Lê", "Nỗi Buồn Hoa Phượng - Hương Lan",
                    "Cô Hàng Xóm - Trường Vũ"],
        "cailuong": ["Tình Anh Bán Chiếu - Minh Cảnh", "Lan Và Điệp - Phượng Liên",
                      "Bông Hồng Cài Áo - Thanh Kim Huệ", "Trách Ai Vô Tình - Thanh Sang",
                      "Giọt Lệ Đài Trang - Lệ Thủy"],
        "cheo": ["Quan Âm Thị Kính", "Lưu Bình Dương Lễ", "Trương Viên", "Tấm Cám", "Chị Dậu"],
        "danca": ["Trống Cơm", "Lý Cây Đa", "Cò Lả", "Bèo Dạt Mây Trôi", "Lý Ngựa Ô"],
        "nhacdo": ["Tiến Quân Ca", "Bác Vẫn Cùng Chúng Cháu Hành Quân", "Như Có Bác Hồ Trong Ngày Vui Đại Thắng",
                   "Chiều Trên Bản Thượng", "Cô Gái Mở Đường"],
        "pop": ["Em Của Ngày Hôm Qua - Sơn Tùng M-TP", "Nơi Này Có Anh - Sơn Tùng M-TP",
                "Phía Sau Một Cô Gái - Soobin Hoàng Sơn", "Tháng Tư Là Lời Nói Dối Của Em - Hà Anh Tuấn",
                "Hơn Cả Yêu - Đức Phúc"],
        "remix": ["Đếm Sao Remix - Hồ Quang Hiếu", "Hạ Còn Vương Nắng Remix", 
                  "Chờ Ngày Mưa Tan Remix", "Bay Trên Đường Bay Remix", "Ngày Ấy Bạn Và Tôi Remix"],
        "thieunhi": ["Bé Bé Bồng Bông", "Con Cò Bé Bé", "Một Con Vịt", "Bắc Kim Thang", "Em Đi Giữa Biển Vàng"],
        "other": ["Gánh Hàng Rong", "Trở Về Dòng Sông Tuổi Thơ", "Nỗi Nhớ Mùa Đông", "Chiếc Lá Cuối Cùng", "Tình Khúc Vàng"]
    }

    top_songs = suggestions.get(genre, [])
    if not top_songs:
        return f"🎵 Chưa có gợi ý cho thể loại `{genre}`."

    out = f"### 🎶 Gợi ý nhạc tương tự ({genre.capitalize()})\n"
    out += "\n".join([f"- {s}" for s in top_songs])
    return out


def build_app_ui():
    with gr.Blocks(title="🔊 Dự đoán thể loại nhạc từ file âm thanh") as app:
        gr.Markdown("## 🎧 Phân tích & dự đoán 30s/đoạn")

        with gr.Row():
            input_mode = gr.Radio(
                choices=["Tải file", "YouTube URL"],
                value="Tải file",
                label="Nguồn dữ liệu"
            )

        with gr.Row():
            audio_in = gr.Audio(type="filepath", label="🎵 Chọn file âm thanh (WAV, MP3)", visible=True)
            yt_url_in = gr.Textbox(label="🔗 YouTube URL", placeholder="https://www.youtube.com/watch?v=...", visible=False)

        with gr.Row():
            model_in = gr.Dropdown(
                choices=[
                    "Model 8 class: bolero, cai luong, chauvan, cheo, dan ca, pop ,remix, thieu nhi",
                    "Model 5 class: bolero, cailuong, cheo, danca, nhacdo",
                    "Model 7 class: bolero, cai luong, cheo, dan ca, nhac do, thieu nhi,other",
                ],
                value="Model 8 class: bolero, cai luong, chauvan, cheo, dan ca, pop ,remix, thieu nhi",
                label="🧠 Chọn mô hình"
            )

        run_btn = gr.Button("Phân tích", variant="primary")
        result_out = gr.Textbox(lines=15, label="Kết quả dự đoán theo từng đoạn 30s")

        suggest_btn = gr.Button("🎶 Gợi ý nhạc tương tự")
        suggest_out = gr.Markdown(label="Gợi ý bài hát tương tự")

        # Toggle input
        def _toggle(mode):
            return (
                gr.update(visible=(mode == "Tải file")),
                gr.update(visible=(mode == "YouTube URL"))
            )

        input_mode.change(_toggle, [input_mode], [audio_in, yt_url_in])

        run_btn.click(
            predict_from_ui,
            inputs=[audio_in, model_in, input_mode, yt_url_in],
            outputs=[result_out]
        )

        suggest_btn.click(suggest_similar_songs, inputs=[result_out], outputs=[suggest_out])

    return app
