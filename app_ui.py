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
    elif model_name == "Model 7 class: bolero, cai luong, cheo, dan ca, pop ,remix, thieu nhi":
        classes = ['bolero', 'cailuong', 'cheo', 'danca', 'pop', 'remix', 'thieunhi']
        model_path = "./train_with_7_class_03.pth"
        num_classes = 7
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

def gradio_predict(audio_file, selected_model):
    print(f"\n👉 Nhận file: {audio_file}")
    model, class_names_local = load_model(selected_model)
    results = []
    try:
        y, sr = lb.load(audio_file, sr=None)
    except Exception as e:
        return f"❌ Lỗi khi load file: {e}"

    duration = 30
    segment_samples = duration * sr
    total_samples = len(y)
    num_segments = total_samples // segment_samples
    if num_segments == 0:
        return "⚠️ File quá ngắn (< 30s). Không thể xử lý."

    base_filename = os.path.splitext(os.path.basename(audio_file))[0]
    folder_path = os.path.dirname(audio_file)
    output_folder = os.path.join(folder_path, "predict")
    os.makedirs(output_folder, exist_ok=True)

    samples = {}
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]
        new_filename = f"{base_filename}_part{i+1}.wav"
        new_path = os.path.join(output_folder, new_filename)
        sf.write(new_path, segment, sr)
        samples[i] = {"dir": new_path, "sampling": segment}

    samples = get_fft(samples)
    samples = get_mel_spectrogram(samples, sr)
    mel_root = os.path.join(output_folder, "mel-images")
    os.makedirs(mel_root, exist_ok=True)
    list_test = save_mel_spec(samples, mel_root)

    for idx, path in enumerate(list_test):
        image_pil = Image.open(path).convert("RGB").resize((224, 224))
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = torch.softmax(models_output(model, image_tensor), dim=1) if False else None  # placeholder

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            output_class = class_names_local[pred.item()]
            percentage = conf.item() * 100
            start_time = str(datetime.timedelta(seconds=idx * duration))
            end_time = str(datetime.timedelta(seconds=(idx + 1) * duration))
            results.append(f"[{start_time} → {end_time}] → {output_class} ({percentage:.2f}%)")

    return "\n".join(results)

def build_app_ui():
    with gr.Blocks(title="🔊 Dự đoán thể loại nhạc từ file âm thanh") as app:
        gr.Markdown("## 🎧 Phân tích & dự đoán 30s/đoạn")
        with gr.Row():
            audio_in = gr.Audio(type="filepath", label="🎵 Chọn file âm thanh (WAV, MP3)")
            model_in = gr.Dropdown(
                choices=[
                    "Model 5 class: bolero, cailuong, cheo, danca, nhacdo",
                    "Model 7 class: bolero, cai luong, cheo, dan ca, pop ,remix, thieu nhi"
                ],
                label="🧠 Chọn mô hình"
            )
        run_btn = gr.Button("Phân tích", variant="primary")
        result_out = gr.Textbox(lines=15, label="Kết quả dự đoán theo từng đoạn 30s")

        run_btn.click(gradio_predict, [audio_in, model_in], [result_out])
    return app
