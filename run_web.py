import gradio as gr
import matplotlib.pyplot as plt
# import cv2
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
# class_names = ['CachMang', 'TruTinh', 'bolero', 'cailuong', 'cheo', 'danca', 'nhacxuan']
# class_names =  ['bolero', 'cailuong', 'cheo', 'danca', 'nhacdo', 'pop']
class_names =  ['bolero', 'cailuong', 'cheo', 'danca', 'nhacdo']
img_height,img_width=224,224
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.efficientnet_b0(pretrained=False)
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)  # 5 class
# model.load_state_dict(torch.load("/efficientnet_b0_5_lager.pth", map_location=device))
# model.to(device)
# model.eval()
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

# Transform ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# Hàm xử lý STFT
def get_fft(samples, n_fft=2048, hop_length=512):
    for index, item in samples.items():
        D = np.abs(lb.stft(item["sampling"], n_fft=n_fft, hop_length=hop_length))
        samples[index]["stft"] = D
    return samples

# Hàm xử lý mel spectrogram
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

        # Normalize để đưa về 0–255
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())  # 0–1
        S_rgb = cm.viridis(S_db_norm)[:, :, :3]  # RGB, bỏ A
        S_rgb = (S_rgb * 255).astype(np.uint8)

        # Resize về 224x224 rồi lưu
        im = Image.fromarray(S_rgb).resize((224, 224))
        im.save(out_path)

        image_paths.append(out_path)

    return image_paths

# Hàm chính: predict và sinh ảnh spectrogram
def predict(file_path, duration=30):
    try:
        y, sr = lb.load(file_path, sr=None)
    except Exception as e:
        print(f" Lỗi khi load: {file_path} ({e})")
        return

    segment_samples = duration * sr
    total_samples = len(y)
    num_segments = total_samples // segment_samples
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    if num_segments == 0:
        return

    folder_path = os.path.dirname(file_path)

    # Folder lưu wav đã cắt
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
        samples[i] = {
            "dir": new_path,
            "sampling": segment
        }

    samples = get_fft(samples)
    samples = get_mel_spectrogram(samples, sr)

    mel_root = os.path.join(output_folder, "mel-images")
    os.makedirs(mel_root, exist_ok=True)

    list_test = save_mel_spec(samples, mel_root)

    # === DỰ ĐOÁN TỪ ẢNH MEL ===
    print("\n🔍 Đang dự đoán")
    for path in list_test:
        image_pil = Image.open(path).convert("RGB") 
        image_pil = image_pil.resize((224, 224))
        # Apply transforms và chuyển sang tensor
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)  # chuyển logits thành xác suất
            conf, pred = torch.max(probs, 1)   # lấy class có xác suất cao nhất
        
            output_class = class_names[pred.item()]
            percentage = conf.item() * 100
            print(f"Ảnh {os.path.basename(path)} → Dự đoán: {output_class} ({percentage:.2f}%)")

def gradio_predict(audio_file, selected_model):
    print(f"\n👉 Nhận file: {audio_file}")
    model, class_names = load_model(selected_model)
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
        samples[i] = {
            "dir": new_path,
            "sampling": segment
        }

    samples = get_fft(samples)
    samples = get_mel_spectrogram(samples, sr)
    mel_root = os.path.join(output_folder, "mel-images")
    os.makedirs(mel_root, exist_ok=True)
    list_test = save_mel_spec(samples, mel_root)

    for idx, path in enumerate(list_test):
        image_pil = Image.open(path).convert("RGB")
        image_pil = image_pil.resize((224, 224))
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            output_class = class_names[pred.item()]
            percentage = conf.item() * 100

            start_time = str(datetime.timedelta(seconds=idx * duration))
            end_time = str(datetime.timedelta(seconds=(idx + 1) * duration))

            results.append(f"[{start_time} → {end_time}] → {output_class} ({percentage:.2f}%)")

    return "\n".join(results)


# Gradio UI
ui = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Audio(type="filepath", label="🎵 Chọn file âm thanh (WAV, MP3)"),
        gr.Dropdown(choices=["Model 5 class: bolero, cailuong, cheo, danca, nhacdo", "Model 7 class: bolero, cai luong, cheo, dan ca, pop ,remix, thieu nhi"] , label="🧠 Chọn mô hình")
    ],
    outputs="text",
    title="🔊 Dự đoán thể loại nhạc từ file âm thanh",
    description="Upload file âm thanh >30s. Chọn mô hình và hệ thống sẽ phân tích, dự đoán từng đoạn 30s."
)

ui.launch()

