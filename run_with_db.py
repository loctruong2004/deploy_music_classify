from get_db import get_db
# from get_db import update_prediction_to_db
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import librosa as lb
import numpy as np
import matplotlib.cm as cm
import os
import gc
import pyodbc
import re

# Các class tương ứng với mô hình
class_names = ['bolero', 'cailuong', 'cheo', 'danca', 'nhacdo']
img_height, img_width = 224, 224

# Lấy danh sách đường dẫn từ CSDL


# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Tải mô hình
def load_model(model_name):
    if model_name == "6_class":
        classes = ['bolero', 'cailuong', 'cheo', 'danca', 'other', 'thieunhi']
        model_path = "./efficientnet_b0_6_final_smoothing.pth"
        num_classes = 6
    elif model_name == "7_class":
        classes = ['bolero', 'cailuong', 'cheo', 'danca', 'pop', 'remix', 'thieunhi']
        model_path = "./train_with_7_class_03.pth"
        num_classes = 7
    else:
        raise ValueError("Model không hợp lệ")

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, classes

# Chuyển ảnh sang tensor
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dự đoán từ file .wav (KHÔNG ghi file)
def predict_from_existing_wav(file_path, model, class_names, duration=30):
    try:
        y, sr = lb.load(file_path, sr=None)
    except Exception as e:
        print(f"Lỗi khi load: {file_path} ({e})")
        return

    segment_samples = duration * sr
    total_samples = len(y)
    num_segments = total_samples // segment_samples

    if num_segments == 0:
        print(f"File quá ngắn (<{duration}s): {file_path}")
        return

    print(f"\n File: {file_path}")
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]

        # Mel spectrogram
        S = lb.feature.melspectrogram(y=segment, sr=sr)
        S_db = lb.amplitude_to_db(S, ref=np.max)
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        S_rgb = cm.viridis(S_db_norm)[:, :, :3]
        S_rgb = (S_rgb * 255).astype(np.uint8)

        # Tạo ảnh RGB resize về 224x224
        image = Image.fromarray(S_rgb).resize((img_width, img_height)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            output_class = class_names[pred.item()]
            percentage = conf.item() * 100
            print(f" Đoạn {i+1}: {output_class} ({percentage:.2f}%)")
from collections import Counter

from collections import Counter, defaultdict
from get_db import save_segment_prediction_to_db
def predict_from_existing_wav_final(file_path, model, class_names, id, duration=30):
    try:
        y, sr = lb.load(file_path, sr=None)
    except Exception as e:
        print(f"Lỗi khi load: {file_path} ({e})")
        return

    segment_samples = duration * sr
    total_samples = len(y)
    num_segments = total_samples // segment_samples

    if num_segments == 0:
        print(f"File quá ngắn (<{duration}s): {file_path}")
        return

    print(f"\nFile: {file_path}")

    count_classes = Counter()
    sum_confidence = defaultdict(float)

    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]

        S = lb.feature.melspectrogram(y=segment, sr=sr)
        S_db = lb.amplitude_to_db(S, ref=np.max)
        S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        S_rgb = cm.viridis(S_db_norm)[:, :, :3]
        S_rgb = (S_rgb * 255).astype(np.uint8)

        image = Image.fromarray(S_rgb).resize((img_width, img_height)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

            pred_class = class_names[pred.item()]
            confidence = conf.item()

            count_classes[pred_class] += 1
            sum_confidence[pred_class] += confidence

            # 🟢 LƯU DỮ LIỆU SEGMENT VÀO DB
            segment_time = start // sr  # giây bắt đầu
            save_segment_prediction_to_db(
                song_id=id,
                segment_time=segment_time,
                predicted_class=pred_class,
                confidence=confidence
            )


from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
def chunks(iterable, size=5):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])
# === MAIN ===  
if __name__ == '__main__':
    list_path = get_db("")
    print("Danh sách file lấy từ DB:")
    model_name = "7_class"
    model, class_names = load_model(model_name)

    def process_item(item):
        id = item['id']
        path = item['path']
        print(f"Đang xử lý id: {id}, path: {path}")
        predict_from_existing_wav_final(path, model, class_names, id)

    # Duyệt từng batch 5 file
    batch_list = chunks(list_path, size=5)
    for batch_num, batch in enumerate(batch_list):
        print(f"\n===> Đang xử lý batch {batch_num + 1} <===")
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_item, item) for item in batch]
            for future in as_completed(futures):
                future.result()  # Nếu muốn kiểm tra lỗi
