import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 시드 고정 함수
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ✅ 사용자 정의 Dataset
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ✅ Conv1D + LSTM 모델 구조
class ConvLSTMModel(nn.Module):
    def __init__(self, input_size=30, seq_len=500, num_classes=2,
                 conv_filters=64, lstm_hidden=64, dropout_rate=0.3, bidirectional=False):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_filters)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(conv_filters, lstm_hidden, batch_first=True, bidirectional=bidirectional)
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.dropout_lstm = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)                   # (B, D, T)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = x.transpose(1, 2)                   # (B, T, F)
        out, _ = self.lstm(x)
        out = self.dropout_lstm(out[:, -1, :])
        return self.fc(out)


# ✅ 학습 및 평가 함수
def train_and_evaluate(X, y, num_classes=2,
                       optimizer_type="adam",
                       loss_type="crossentropy",
                       use_bidirectional=False,
                       dropout_rate=0.3,
                       conv_filters=64,
                       lstm_hidden=64,
                       seed=42):  # <- 시드 추가

    set_seed(seed)  # 시드 고정

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)

    train_loader = DataLoader(PoseDataset(X_train, y_train), batch_size=64, shuffle=True,
                              pin_memory=True, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    test_loader = DataLoader(PoseDataset(X_test, y_test), batch_size=64, pin_memory=True)

    model = ConvLSTMModel(
        input_size=X.shape[2],
        dropout_rate=dropout_rate,
        bidirectional=use_bidirectional,
        conv_filters=conv_filters,
        lstm_hidden=lstm_hidden,
        num_classes=num_classes
    ).to(device)

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    criterion = nn.CrossEntropyLoss() if loss_type == "crossentropy" else nn.NLLLoss()

    model.train()
    for _ in range(40):
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            out = model(xb)
            if loss_type == "nll":
                out = F.log_softmax(out, dim=1)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = torch.argmax(out, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    acc = 100 * correct / total
    print(f"✅ 테스트 정확도: {acc:.2f}%")
    return acc