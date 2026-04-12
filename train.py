import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# NOTE:
# This script provides a simplified training example for
# repository validation and reproducibility purposes.
# It does NOT reproduce the full industrial-scale training
# procedure described in the paper.
# =========================================================


def gaussian_smooth_1d(x: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Simple 1D Gaussian smoothing without external dependencies.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    radius = kernel_size // 2
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    padded = np.pad(x, (radius, radius), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def compute_gradient_feature(x: np.ndarray) -> np.ndarray:
    """
    Compute first-order gradient after Gaussian smoothing.
    """
    x_smooth = gaussian_smooth_1d(x, kernel_size=5, sigma=1.0)
    grad = np.gradient(x_smooth)
    return grad.astype(np.float32)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """
    Normalize depth to [0, 1].
    """
    d_min = depth.min()
    d_max = depth.max()
    if np.isclose(d_min, d_max):
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """
    Z-score normalization.
    """
    mean = np.mean(x)
    std = np.std(x)
    if np.isclose(std, 0):
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build 9-channel features:
    [GR, AC, DEN, RLLD, dGR, dAC, dDEN, dRLLD, Norm_Depth]
    """
    required_cols = ["DEPTH", "GR", "AC", "DEN", "RLLD"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    depth = df["DEPTH"].to_numpy(dtype=np.float32)
    gr = df["GR"].to_numpy(dtype=np.float32)
    ac = df["AC"].to_numpy(dtype=np.float32)
    den = df["DEN"].to_numpy(dtype=np.float32)
    rlld = df["RLLD"].to_numpy(dtype=np.float32)

    d_gr = compute_gradient_feature(gr)
    d_ac = compute_gradient_feature(ac)
    d_den = compute_gradient_feature(den)
    d_rlld = compute_gradient_feature(rlld)
    norm_depth = normalize_depth(depth)

    gr_n = zscore_normalize(gr)
    ac_n = zscore_normalize(ac)
    den_n = zscore_normalize(den)
    rlld_n = zscore_normalize(rlld)

    d_gr_n = zscore_normalize(d_gr)
    d_ac_n = zscore_normalize(d_ac)
    d_den_n = zscore_normalize(d_den)
    d_rlld_n = zscore_normalize(d_rlld)

    features = np.stack(
        [
            gr_n, ac_n, den_n, rlld_n,
            d_gr_n, d_ac_n, d_den_n, d_rlld_n,
            norm_depth
        ],
        axis=1
    ).astype(np.float32)

    return features


def encode_labels(df: pd.DataFrame):
    """
    Encode string labels from the LAYER column into integer indices.
    """
    if "LAYER" not in df.columns:
        raise ValueError("Missing required column: LAYER")

    labels_str = df["LAYER"].astype(str).to_numpy()
    unique_labels = sorted(np.unique(labels_str).tolist())
    label_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    labels = np.array([label_to_idx[x] for x in labels_str], dtype=np.int64)
    return labels, idx_to_label


class StratDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(features).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DemoClassifier(nn.Module):
    """
    A lightweight demo classifier for repository validation.
    This is NOT the full DA-TransResUNet.
    """

    def __init__(self, in_channels: int = 9, num_classes: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def split_train_val(features: np.ndarray, labels: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    """
    Random split for demo purposes.
    """
    np.random.seed(seed)
    indices = np.arange(len(features))
    np.random.shuffle(indices)

    n_val = max(1, int(len(features) * val_ratio))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train = features[train_idx]
    y_train = labels[train_idx]
    x_val = features[val_idx]
    y_val = labels[val_idx]

    return x_train, y_train, x_val, y_val


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def main():
    print("=" * 70)
    print("DA-TransResUNet Demo Training Script")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "sample_data.csv"
    ckpt_dir = repo_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Sample data not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded sample data from: {data_path}")
    print(f"[INFO] Data shape: {df.shape}")

    # Build features and labels
    features = build_features(df)
    labels, idx_to_label = encode_labels(df)

    num_classes = len(idx_to_label)
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] Feature shape: {features.shape}")

    # Train/validation split
    x_train, y_train, x_val, y_val = split_train_val(features, labels, val_ratio=0.2, seed=42)
    print(f"[INFO] Train size: {len(x_train)}")
    print(f"[INFO] Val size: {len(x_val)}")

    train_dataset = StratDataset(x_train, y_train)
    val_dataset = StratDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Model
    torch.manual_seed(42)
    model = DemoClassifier(in_channels=9, num_classes=num_classes).to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    best_val_acc = -1.0
    best_ckpt_path = ckpt_dir / "demo_model.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=100)

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            train_correct += (pred == y).sum().item()
            train_samples += x.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(train_correct / train_samples):.4f}"
            )

        train_loss = train_loss_sum / train_samples
        train_acc = train_correct / train_samples

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"[INFO] Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "label_mapping": idx_to_label,
                },
                best_ckpt_path
            )
            print(f"[INFO] Best model saved to: {best_ckpt_path}")

    print("=" * 70)
    print("[SUCCESS] Demo training completed successfully.")
    print(f"[INFO] Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()