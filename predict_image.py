import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp

# ---------------------------
# 1) Pose 추출 설정
# ---------------------------
mp_pose = mp.solutions.pose.Pose(static_image_mode=True)

def extract_pose_sequence_from_frame(frame, input_size): #BGR 을 RGB로 수정후 mediapipe로 랜드마크 배열을 kp에 저장
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(img_rgb)
    if not results.pose_landmarks:
        return None
    kp = []
    for lm in results.pose_landmarks.landmark:
        kp.extend([lm.x, lm.y])
    kp = np.array(kp, dtype=np.float32)
    if kp.size < input_size:#강제로 인풋길이를 맞춤(정상적인경우 인풋길이 차이없음)
        kp = np.pad(kp, (0, input_size - kp.size))
    else:
        kp = kp[:input_size]
    return kp

# ---------------------------
# 2) 데이터 로드(사전정의된경로)
# ---------------------------
X = np.load("npy_merged/ucf_pose_features.npy")  # (N, seq_len, input_size)
y = np.load("npy_merged/ucf_labels.npy")         # (N,)
seq_len, input_size = X.shape[1], X.shape[2]

# ---------------------------
# 3) PyTorch Dataset & Model
# ---------------------------
class PoseDataset(Dataset):#X Y 를 텐서로 래핑, 길이 , 인덱싱 인터페이스
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ConvLSTMModel(nn.Module):#Conv+LSTM 정의
    def __init__(self, input_size, conv_filters, lstm_hidden,
                 dropout_rate, bidirectional):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(conv_filters)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(conv_filters)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        self.lstm  = nn.LSTM(conv_filters, lstm_hidden,
                             batch_first=True, bidirectional=bidirectional)
        self.drop_lstm = nn.Dropout(dropout_rate)
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc    = nn.Linear(out_dim, 2)
    def forward(self, x):#입력차원변환,합성곱 처리, 차원변환, 은닉벡터,FC
        x = x.transpose(1,2)
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = x.transpose(1,2)
        out, _ = self.lstm(x)
        out = self.drop_lstm(out[:,-1,:])
        return self.fc(out)

# ---------------------------
# 4) 학습/평가 함수
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):#같은결과를 받을 수 있게 seed를 고정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


BEST_PARAMS = dict(#사전에 구한 최적 파라미터
    conv_filters=256,
    lstm_hidden=64,
    dropout_rate=0.1,
    bidirectional=True,
    optimizer="sgd",
    seed=75006,
    epochs=160
)

def train_model(X, y, params):#학습및 분할
    set_seed(params['seed'])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, random_state=params['seed'])#시드설정
    train_loader = DataLoader(
        PoseDataset(X_tr,y_tr), batch_size=64,
        shuffle=True, worker_init_fn=lambda i: np.random.seed(params['seed']+i)
    )
    model = ConvLSTMModel(#파라미터 설정
        input_size, params['conv_filters'],
        params['lstm_hidden'], params['dropout_rate'],
        params['bidirectional']
    ).to(device)
    if params['optimizer']=="adam":
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
    elif params['optimizer']=="rmsprop":
        optim = torch.optim.RMSprop(model.parameters(), lr=0.001)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, params['epochs']+1):#에폭만큼 반복
        total_loss=0
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out,yb)
            loss.backward()
            optim.step()
            total_loss += loss.item()*xb.size(0)
        # 매 10 에폭마다 검증 정확도
        if epoch % 10 == 0:
            model.eval()
            correct=total=0
            for xb,yb in DataLoader(PoseDataset(X_te,y_te), batch_size=64):
                xb,yb=xb.to(device),yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred==yb).sum().item()
                total += yb.size(0)
            print(f"Epoch {epoch}/{params['epochs']}: Val Acc {100*correct/total:.2f}%")
            model.train()
    return model#학습된 모델 반환                              

# ---------------------------
# 5) 이미지 예측
# ---------------------------
def predict_images(model, folder="testcase"):
    print("\n-- Image Predictions --")
    for img in glob.glob(os.path.join(folder,"*.jpg")):
        frame = cv2.imread(img)
        kp = extract_pose_sequence_from_frame(frame, input_size)#프레임을 통한 포즈추출
        if kp is None:#포즈추출실패시
            print(os.path.basename(img),"→ 판별 실패")
            continue
        seq = np.tile(kp,(seq_len,1))
        tensor = torch.tensor(seq,dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(tensor).argmax(dim=1).item()#예측
        print(os.path.basename(img),"→", "공격적" if pred==1 else "비공격적")

# ---------------------------
# 6) 동영상 예측
# ---------------------------
def predict_video(model, video_path, seq_len=500, step=30):
    """
    model: 학습된 모델
    video_path: 동영상 파일 경로
    seq_len: 윈도우 길이 (프레임 수)
    step: 윈도우 이동 간격
    """
    try:
        print(f"-- Processing video: {video_path} --")
        cap = cv2.VideoCapture(video_path)#프레임마다 진행로그 
        if not cap.isOpened():#동영상이 안열렸으면
            print(f"Error: Cannot open video file {video_path}")
            return []
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Read {frame_count} frames...")
        cap.release()
        print(f"Total frames read: {frame_count}")

        results = []

        
        if len(frames) < seq_len:#너무 짧은경우
            pad_count = seq_len - len(frames)
            print(f"Video shorter than seq_len, padding {pad_count} frames")
            frames = [frames[0]] * pad_count + frames

        
        for start in range(0, len(frames) - seq_len + 1, step):
            try:#각윈도우마다 포즈 특징 추출
                seq_feats = []
                for f_idx, f in enumerate(frames[start:start + seq_len]):
                    kp = extract_pose_sequence_from_frame(f, input_size)
                    if kp is None:
                        kp = np.zeros((input_size,), dtype=np.float32)
                    seq_feats.append(kp)
                seq_arr = np.stack(seq_feats, axis=0)
                tensor = torch.tensor(seq_arr, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(tensor).argmax(dim=1).item()
                results.append((start, pred))
            except Exception as e_frame:
                print(f"Error at window start {start}: {e_frame}")
        print(f"Prediction windows: {len(results)}")
        return results
    except Exception as e:
        print(f"Exception in predict_video: {e}")
        return []

# ---------------------------
# 7) 실행
# ---------------------------
if __name__=="__main__":
    model = train_model(X, y, BEST_PARAMS)
    predict_images(model)
    
    video_results = predict_video(model, "testcase/video.mp4")
    print("-- Video Predictions --", video_results)
