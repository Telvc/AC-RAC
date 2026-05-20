from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rac.data import save_prediction_cache

from .prepare_data import SIDE_COLS, frame_to_arrays, load_movielens_frames


class MovieLensClassifDataset(Dataset):
    def __init__(self, df):
        self.user_ids, self.item_ids, self.side_info, self.y = frame_to_arrays(df)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.user_ids[idx], dtype=torch.long),
            torch.tensor(self.item_ids[idx], dtype=torch.long),
            torch.tensor(self.side_info[idx], dtype=torch.float32),
        ), torch.tensor(self.y[idx], dtype=torch.long)


class DeepRecommenderClassifier(nn.Module):
    def __init__(self, num_users: int, num_items: int, side_in_dim: int = len(SIDE_COLS), num_classes: int = 5):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 32)
        self.item_emb = nn.Embedding(num_items, 32)
        self.side_mlp = nn.Sequential(
            nn.Linear(side_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, user_ids, item_ids, side_info):
        combined = torch.cat([self.user_emb(user_ids), self.item_emb(item_ids), self.side_mlp(side_info)], dim=1)
        return self.final_mlp(combined)


def _train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    for (user_ids, item_ids, side_info), y in loader:
        user_ids, item_ids, side_info, y = user_ids.to(device), item_ids.to(device), side_info.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(user_ids, item_ids, side_info), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / max(total, 1)


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for (user_ids, item_ids, side_info), y in loader:
            user_ids, item_ids, side_info = user_ids.to(device), item_ids.to(device), side_info.to(device)
            probs = torch.softmax(model(user_ids, item_ids, side_info), dim=1)
            probs_list.append(probs.cpu().numpy())
            labels_list.append(y.numpy())
    return np.concatenate(probs_list, axis=0), np.concatenate(labels_list, axis=0)


def cache_movielens_predictions(
    zip_path: str = "ml-100k.zip",
    output_dir: str = "artifacts/cached_predictions",
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, calib_df, test_df, user2idx, item2idx = load_movielens_frames(zip_path, ".", seed=seed)
    train_loader = DataLoader(MovieLensClassifDataset(train_df), batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(MovieLensClassifDataset(calib_df), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MovieLensClassifDataset(test_df), batch_size=batch_size, shuffle=False)
    model = DeepRecommenderClassifier(len(user2idx), len(item2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        _train_one_epoch(model, train_loader, optimizer, device)
    cal_probs, cal_labels = _predict(model, calib_loader, device)
    test_probs, test_labels = _predict(model, test_loader, device)
    save_prediction_cache(output_dir, cal_probs, cal_labels, test_probs, test_labels, prefix="movielens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default="ml-100k.zip")
    parser.add_argument("--output-dir", default="artifacts/cached_predictions")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cache_movielens_predictions(args.zip_path, args.output_dir, args.epochs, args.batch_size, args.seed)
