import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from ultralytics import YOLO

# ───────────── 모델 정의 ─────────────
class ConvLSTMBackbone(nn.Module):
    def __init__(self, input_size, conv_filters, lstm_hidden, dropout_rate, bidirectional):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(conv_filters, lstm_hidden, batch_first=True, bidirectional=bidirectional)
        self.out_dim = lstm_hidden * (2 if bidirectional else 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return out[:, -1, :]

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feat):
        return self.fc(feat)

class AggressionDetectionModel(nn.Module):
    def __init__(self, input_size, conv_filters, lstm_hidden, dropout_rate, bidirectional, num_classes):
        super().__init__()
        self.backbone = ConvLSTMBackbone(input_size, conv_filters, lstm_hidden, dropout_rate, bidirectional)
        self.classifier = ClassificationHead(self.backbone.out_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

# ───────────── 설정 ─────────────
INPUT_DIR = "dataset"
OUTPUT_CSV = "predictions.csv"
MODEL_PATH = "model_final.pth"
MAX_SEQ_LEN = 300
KEYPOINT_DIM = 34  # 17 keypoints x (x, y) = 34
conv_filters = 128
lstm_hidden = 64
dropout_rate = 0.4
bidirectional = False
SEED = 63065
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────── 모델 로딩 ─────────────
def load_model():
    model = AggressionDetectionModel(
        input_size=KEYPOINT_DIM,
        conv_filters=conv_filters,
        lstm_hidden=lstm_hidden,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        num_classes=2
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

# ───────────── YOLO Pose 추론 ─────────────
yolo_pose = YOLO('yolov8n-pose.pt')

def extract_pose_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    poses = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_pose.predict(frame, verbose=False)
        if not results or not results[0].keypoints:
            continue
        kps = results[0].keypoints  # (N, 17, 2)
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            continue

        # 모든 사람의 keypoints 평균 (N, 17, 2) → (17, 2) → flatten(34)
        all_persons = kps.xy[0]  # (N, 17, 2)
        if all_persons.shape[0] > 0:
            avg_person = torch.mean(all_persons, dim=0).reshape(-1)  # (34,)
            if avg_person.shape[0] == KEYPOINT_DIM:
                poses.append(avg_person.cpu().numpy())

    cap.release()
    return np.array(poses, dtype=np.float32)


# ───────────── 예측 함수 ─────────────
def predict_pose_sequence(model, sequence):
    sequence = sequence[:MAX_SEQ_LEN]
    padded = np.zeros((MAX_SEQ_LEN, KEYPOINT_DIM), dtype=np.float32)
    padded[:sequence.shape[0]] = sequence
    tensor = torch.tensor(padded[None, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

# ───────────── 전체 실행 ─────────────
def collect_videos(root_dir):
    video_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                video_files.append(os.path.join(root, f))
    return video_files

if __name__ == "__main__":
    model = load_model()
    video_paths = collect_videos(INPUT_DIR)
    results = []

    for vid_path in tqdm(video_paths, desc="🎥 영상 처리 중"):
        try:
            poses = extract_pose_sequence(vid_path)
            if poses.shape[0] == 0:
                pred = -1  # 너무 적은 프레임 → 무효
            else:
                pred = predict_pose_sequence(model, poses)
        except Exception as e:
            pred = "ERROR"
        results.append((vid_path, pred))

    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("video_path,prediction\n")
        for path, pred in results:
            f.write(f"{path},{pred}\n")

    print(f"\n✅ 예측 완료: 결과 저장됨 → {OUTPUT_CSV}")
