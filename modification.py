import os
import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline

# ✅ 증강 함수들
def jitter(seq, sigma=0.02):
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise

def scaling(seq, sigma=0.1):
    factor = np.random.normal(1.0, sigma)
    return seq * factor

def rotation(seq, angle_sigma=10):
    angle = np.deg2rad(np.random.normal(0, angle_sigma))
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    seq = seq.reshape(seq.shape[0], -1, 2)
    rotated = np.dot(seq, rotation_matrix)
    return rotated.reshape(seq.shape[0], -1)

def time_warp(seq, sigma=0.2):
    time = np.arange(seq.shape[0])
    warp = np.random.normal(loc=1.0, scale=sigma, size=seq.shape[0])
    warp = np.cumsum(warp)
    warp = (warp - warp.min()) / (warp.max() - warp.min()) * (seq.shape[0] - 1)
    warp, unique_indices = np.unique(warp, return_index=True)
    seq = seq[unique_indices]
    if len(warp) < 2:
        return seq
    cs = CubicSpline(warp, seq, axis=0)
    return cs(time)

def frame_drop(seq, drop_rate=0.1):
    keep = np.random.rand(seq.shape[0]) > drop_rate
    kept_seq = seq[keep]
    if kept_seq.shape[0] < seq.shape[0]:
        pad = np.zeros((seq.shape[0] - kept_seq.shape[0], seq.shape[1]))
        return np.vstack((kept_seq, pad))
    return kept_seq

def shuffle(seq, portion=0.2):
    seq = seq.copy()
    num_frames = int(seq.shape[0] * portion)
    indices = np.random.choice(seq.shape[0], num_frames, replace=False)
    np.random.shuffle(seq[indices])
    return seq

def flip(seq):
    seq = seq.copy()
    seq[:, ::2] = 1.0 - seq[:, ::2]
    return seq

def affine(seq, shift=0.05):
    seq = seq.copy()
    dx = np.random.uniform(-shift, shift)
    dy = np.random.uniform(-shift, shift)
    seq[:, ::2] += dx
    seq[:, 1::2] += dy
    return seq

def gaussian_noise(seq, sigma=0.02):
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise

def crop(seq, crop_portion=0.1):
    length = seq.shape[0]
    crop_size = int(length * crop_portion)
    return seq[crop_size: length-crop_size] if length - 2*crop_size > 1 else seq

def stretch(seq, stretch_factor_range=(0.8, 1.2)):
    factor = np.random.uniform(*stretch_factor_range)
    indices = np.arange(0, len(seq), factor)
    indices = indices[indices < len(seq)].astype(int)
    return seq[indices]

def reverse(seq):
    return seq[::-1]

def noise_masking(seq, portion=0.1):
    seq = seq.copy()
    num_points = int(seq.shape[1] * portion)
    points = np.random.choice(seq.shape[1], num_points, replace=False)
    seq[:, points] = np.random.rand(seq.shape[0], len(points))
    return seq

def random_erasing(seq, portion=0.1):
    seq = seq.copy()
    num_points = int(seq.shape[1] * portion)
    points = np.random.choice(seq.shape[1], num_points, replace=False)
    seq[:, points] = 0
    return seq

def random_shift(seq, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift)
    seq = np.roll(seq, shift, axis=0)
    return seq

def random_scale(seq, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return seq * scale

def speed_perturbation(seq, speed_range=(0.9, 1.1)):
    factor = np.random.uniform(*speed_range)
    indices = np.arange(0, len(seq), factor)
    indices = np.clip(indices, 0, len(seq)-1).astype(int)
    return seq[indices]

def temporal_crop(seq, crop_size=20):
    if seq.shape[0] <= crop_size:
        return seq
    start = np.random.randint(0, seq.shape[0] - crop_size)
    return seq[start:start+crop_size]

def joint_dropout(seq, num_joints=5):
    seq = seq.copy()
    joint_indices = np.random.choice(seq.shape[1] // 2, num_joints, replace=False)
    for j in joint_indices:
        seq[:, 2*j] = 0.0
        seq[:, 2*j + 1] = 0.0
    return seq

def random_time_shift(seq, max_shift=5):
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift > 0:
        pad = np.zeros((shift, seq.shape[1]))
        return np.vstack([pad, seq[:-shift]])
    elif shift < 0:
        pad = np.zeros((-shift, seq.shape[1]))
        return np.vstack([seq[-shift:], pad])
    else:
        return seq

def magnitude_warping(seq, sigma=0.2, knot=4):
    time_steps = np.arange(seq.shape[0])
    anchor_pts = np.linspace(0, seq.shape[0]-1, knot)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot, seq.shape[1]))
    warp_curve = CubicSpline(anchor_pts, random_warps, axis=0)(time_steps)
    return seq * warp_curve

augmentations = [
    jitter, scaling, rotation, time_warp, frame_drop, shuffle, flip, affine, gaussian_noise,
    crop, stretch, reverse, noise_masking, random_erasing, random_shift, random_scale,
    speed_perturbation, temporal_crop, joint_dropout, random_time_shift, magnitude_warping
]

# ✅ 파일 불러오기
input_dir = r"C:\\Users\\User\\Desktop\\MinOfficialProject\\pose_detection_project_rwfonly\\npy_merged"
X = np.load(os.path.join(input_dir, "ucf_pose_features.npy"))
y = np.load(os.path.join(input_dir, "ucf_labels.npy"))

print(f"🔍 원본 데이터 수: {len(X)} (공격성 1: {(y==1).sum()}, 공격성 0: {(y==0).sum()})")

# ✅ 데이터 분리
X_attack = X[y == 1]
y_attack = y[y == 1]
X_nonattack = X[y == 0]
y_nonattack = y[y == 0]

# ✅ 증강 함수
def augment_data(X, label):
    X_aug = []
    for seq in tqdm(X, desc=f"📈 데이터 증강 중 (라벨 {label})"):
        X_aug.append(seq)
        for aug_func in augmentations:
            aug_seq = aug_func(seq)
            if aug_seq.shape[0] != seq.shape[0]:
                pad_len = seq.shape[0] - aug_seq.shape[0]
                if pad_len > 0:
                    pad = np.zeros((pad_len, seq.shape[1]))
                    aug_seq = np.vstack([aug_seq, pad])
                else:
                    aug_seq = aug_seq[:seq.shape[0]]
            X_aug.append(aug_seq)
    return np.array(X_aug), np.full(len(X_aug), label)

# ✅ 공격성만 증강
X_attack_aug, y_attack_aug = augment_data(X_attack, 1)

# ✅ 비공격성은 그대로 유지
X_nonattack_aug = X_nonattack
y_nonattack_aug = y_nonattack

print(f"📊 증강된 공격성 데이터 수: {len(X_attack_aug)}")
print(f"📊 비공격성 데이터 수: {len(X_nonattack_aug)}")

# ✅ 최종 병합
X_final = np.vstack([X_attack_aug, X_nonattack_aug])
y_final = np.hstack([y_attack_aug, y_nonattack_aug])

print(f"✅ 최종 데이터 수: {len(X_final)} (공격성 1: {(y_final==1).sum()}, 공격성 0: {(y_final==0).sum()})")

# ✅ 저장
save_dir = "npy_final"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "ucf_pose_features_aug.npy"), X_final)
np.save(os.path.join(save_dir, "ucf_labels_aug.npy"), y_final)

print(f"🎯 저장 완료: {save_dir}/ucf_pose_features_aug.npy, {save_dir}/ucf_labels_aug.npy")
