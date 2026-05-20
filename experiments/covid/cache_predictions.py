from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rac.data import save_prediction_cache


CLASS_TO_IDX = {"Normal": 0, "Viral Pneumonia": 1, "COVID": 2, "Lung_Opacity": 3}


class ChestXRayDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def _collect_records(dataset_dir: Path, image_subdir: str = "auto") -> list[tuple[Path, int]]:
    records = []
    for class_name, label in CLASS_TO_IDX.items():
        if image_subdir == "auto":
            candidates = [dataset_dir / class_name / "masks", dataset_dir / class_name / "images"]
            folder = next((p for p in candidates if p.exists()), candidates[0])
        else:
            folder = dataset_dir / class_name / image_subdir
        records.extend((path, label) for path in folder.glob("*.png"))
    if not records:
        raise FileNotFoundError(f"No COVID PNG files found under {dataset_dir}")
    return records


def _build_model(checkpoint: Path, device: torch.device):
    model = models.inception_v3(weights=None, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 4)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _predict(model, loader, device):
    probs_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)
            probs_list.append(probs.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(probs_list, axis=0), np.concatenate(labels_list, axis=0)


def cache_covid_predictions(
    dataset_dir: str | Path,
    checkpoint: str | Path = "my_inception_model.pth",
    output_dir: str | Path = "artifacts/cached_predictions",
    image_subdir: str = "auto",
    seed: int = 42,
    batch_size: int = 32,
) -> None:
    dataset_dir = Path(dataset_dir)
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    records = _collect_records(dataset_dir, image_subdir=image_subdir)
    random.shuffle(records)
    x = [item[0] for item in records]
    y = [item[1] for item in records]
    _, x_rem, _, y_rem = train_test_split(x, y, test_size=0.30, random_state=seed, stratify=y)
    x_cal, x_test, y_cal, y_test = train_test_split(x_rem, y_rem, test_size=1 - 0.3333, random_state=seed, stratify=y_rem)

    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    workers = 0 if torch.cuda.is_available() else 0
    cal_loader = DataLoader(ChestXRayDataset(x_cal, y_cal, transform), batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(ChestXRayDataset(x_test, y_test, transform), batch_size=batch_size, shuffle=False, num_workers=workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(checkpoint, device)
    cal_probs, cal_labels = _predict(model, cal_loader, device)
    test_probs, test_labels = _predict(model, test_loader, device)
    save_prediction_cache(output_dir, cal_probs, cal_labels, test_probs, test_labels, prefix="covid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache COVID checkpoint probabilities.")
    parser.add_argument("--dataset-dir", required=True, help="Extracted COVID-19_Radiography_Dataset directory.")
    parser.add_argument("--checkpoint", default="my_inception_model.pth")
    parser.add_argument("--output-dir", default="artifacts/cached_predictions")
    parser.add_argument("--image-subdir", default="auto", choices=["auto", "images", "masks"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cache_covid_predictions(
        args.dataset_dir,
        args.checkpoint,
        args.output_dir,
        args.image_subdir,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
