import os
import numpy as np
from tqdm import tqdm
from collections import Counter

# ✅ 현재 스크립트 기준으로 상대경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "processed")
SAVE_DIR = os.path.join(BASE_DIR, "npy_merged")
os.makedirs(SAVE_DIR, exist_ok=True)

X_all = []
y_all = []
shape_counter = Counter()
invalid_files = []

print("📦 npy 파일 통합 시작")

def safe_load(path):
    try:
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            return data
    except Exception as e:
        print(f"❌ 파일 로딩 실패: {path}, 이유: {e}")
    return None

def pad_sequence(seq, target_len):
    if seq.shape[0] == target_len:
        return seq
    elif seq.shape[0] < target_len:
        pad_len = target_len - seq.shape[0]
        pad = np.zeros((pad_len, seq.shape[1]))
        return np.vstack([seq, pad])
    else:
        return seq[:target_len]

TARGET_SEQ_LEN = 30  # 고정 프레임 수 (ex. 30프레임으로 맞추기)

for folder_name in tqdm(os.listdir(INPUT_DIR), desc="폴더별 처리"):
    folder_path = os.path.join(INPUT_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        if not file_name.endswith("_features.npy"):
            continue

        feature_path = os.path.join(folder_path, file_name)
        label_path = feature_path.replace("_features.npy", "_label.npy")

        if not os.path.exists(label_path):
            continue

        X = safe_load(feature_path)
        y = safe_load(label_path)

        if X is None or y is None:
            continue

        # ✅ 폴더명이 NonFight면 라벨을 무조건 0으로 덮어쓰기
        if folder_name.lower() == "nonfight":
            y = 0
            np.save(label_path, np.array(y))

        if len(X.shape) == 2:
            X = pad_sequence(X, TARGET_SEQ_LEN)
            shape_counter[X.shape] += 1
            if X.shape[0] == TARGET_SEQ_LEN and X.shape[1] != 34:
                X_all.append(X)
                y_all.append(int(y))
            else:
                if X.shape[1] == 34:
                    print(f"❌ 스킵됨 (shape 30x34): {feature_path}")
                invalid_files.append((feature_path, X.shape))
        else:
            invalid_files.append((feature_path, X.shape))

if len(X_all) > 0:
    try:
        X_all = np.stack(X_all)
        y_all = np.array(y_all)

        np.save(os.path.join(SAVE_DIR, "X_features.npy"), X_all)
        np.save(os.path.join(SAVE_DIR, "y_labels.npy"), y_all)

        print(f"✅ 통합 완료: 총 {X_all.shape[0]}개 샘플")
    except Exception as e:
        print(f"❌ 통합 실패: {e}")
else:
    print("⚠️ 통합할 데이터가 없습니다.")

# ✅ 부적절한 shape 출력
if invalid_files:
    print("\n⚠️ 스킵된 파일 목록 (shape 불일치):")
    for path, shape in invalid_files:
        print(f" - {path}, shape: {shape}")

# ✅ shape 분포 출력
if shape_counter:
    print("\n📊 shape 분포:")
    for shape, count in shape_counter.items():
        print(f"  {shape}: {count}개")
