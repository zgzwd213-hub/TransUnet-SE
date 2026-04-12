import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# NOTE:
# This script provides a simplified and runnable demonstration
# of the workflow described in the paper.
# It does NOT reproduce the full DA-TransResUNet model training
# because the full industrial dataset and trained weights are
# not publicly available.
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
    This follows the idea described in the paper.
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
    Simple z-score normalization for demo inference.
    """
    mean = np.mean(x)
    std = np.std(x)
    if np.isclose(std, 0):
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mean) / std).astype(np.float32)


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build the 9-channel input features used in the paper:
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

    # Normalize raw and gradient features for a stable demo
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
    Encode string labels in the LAYER column into integer indices.
    """
    if "LAYER" not in df.columns:
        return None, None

    labels_str = df["LAYER"].astype(str).to_numpy()
    unique_labels = sorted(np.unique(labels_str).tolist())
    label_to_idx = {name: idx for idx, name in enumerate(unique_labels)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    labels = np.array([label_to_idx[x] for x in labels_str], dtype=np.int64)
    return labels, idx_to_label


class DemoClassifier(nn.Module):
    """
    A lightweight demo classifier for repository validation.
    This is NOT the full DA-TransResUNet model.
    It is only used to demonstrate the inference workflow.
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


def main() -> None:
    print("=" * 70)
    print("DA-TransResUNet Quick Test")
    print("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "sample_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Sample data not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    print(f"[INFO] Sample data loaded from: {data_path}")
    print(f"[INFO] Data shape: {df.shape}")
    print(f"[INFO] Available columns: {list(df.columns)}")

    # Build features
    features = build_features(df)
    print(f"[INFO] Feature matrix shape: {features.shape}")
    print("[INFO] Input channels = [GR, AC, DEN, RLLD, dGR, dAC, dDEN, dRLLD, Norm_Depth]")

    # Encode labels
    labels, idx_to_label = encode_labels(df)
    if labels is None:
        raise ValueError("The sample data must contain the 'LAYER' column.")

    num_classes = len(idx_to_label)
    print(f"[INFO] Number of stratigraphic units in sample data: {num_classes}")
    print(f"[INFO] Stratigraphic labels: {[idx_to_label[i] for i in range(num_classes)]}")

    # Convert to tensor
    x = torch.from_numpy(features)

    # Build demo model
    torch.manual_seed(42)
    model = DemoClassifier(in_channels=9, num_classes=num_classes)
    model.eval()

    # Demo inference
    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()

    pred_labels = [idx_to_label[i] for i in pred_idx]
    true_labels = [idx_to_label[i] for i in labels]

    # Demo accuracy
    acc = float((pred_idx == labels).mean())

    print(f"[INFO] Prediction shape: {pred_idx.shape}")
    print(f"[INFO] Demo accuracy: {acc:.4f}")

    print("\n[INFO] First 10 predictions vs. labels:")
    for i in range(min(10, len(pred_labels))):
        print(
            f"  Depth={df['DEPTH'].iloc[i]:.2f} m | "
            f"Pred={pred_labels[i]} | True={true_labels[i]}"
        )

    print("\n[SUCCESS] Quick test completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()