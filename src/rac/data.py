from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


COVID_UTILITY_MATRIX = np.array(
    [
        [10, 2, 2, 4],
        [0, 10, 3, 7],
        [0, 3, 10, 8],
        [1, 4, 4, 10],
    ],
    dtype=float,
)

COVID_ACTION_NAMES = {
    0: "No Action",
    1: "Antibiotics",
    2: "Quarantine",
    3: "Additional Testing",
}

COVID_LABEL_NAMES = {
    0: "Normal",
    1: "Pneumonia",
    2: "COVID-19",
    3: "Lung Opacity",
}

MOVIELENS_UTILITY_MATRIX = np.array(
    [
        [0, -2],
        [0, -1],
        [0, 0],
        [0, 1],
        [0, 2],
    ],
    dtype=float,
)

MOVIELENS_ACTION_NAMES = {
    0: "No-Recommend",
    1: "Recommend",
}

MOVIELENS_LABEL_NAMES = {
    0: "Rating 1",
    1: "Rating 2",
    2: "Rating 3",
    3: "Rating 4",
    4: "Rating 5",
}


def make_utility_fn(matrix: np.ndarray) -> Callable[[int, int], float]:
    mat = np.asarray(matrix, dtype=float)

    def utility_fn(action: int, true_label: int) -> float:
        return float(mat[int(true_label), int(action)])

    utility_fn.matrix = mat  # type: ignore[attr-defined]
    return utility_fn


def load_prediction_cache(directory: str | Path, prefix: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(directory)
    stem = f"{prefix}_" if prefix else ""
    cal_probs = np.load(root / f"{stem}cal_probs.npy")
    cal_labels = np.load(root / f"{stem}cal_labels.npy")
    test_probs = np.load(root / f"{stem}test_probs.npy")
    test_labels = np.load(root / f"{stem}test_labels.npy")
    return cal_probs, cal_labels, test_probs, test_labels


def save_prediction_cache(
    directory: str | Path,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    prefix: str = "",
) -> None:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_" if prefix else ""
    np.save(root / f"{stem}cal_probs.npy", cal_probs)
    np.save(root / f"{stem}cal_labels.npy", cal_labels)
    np.save(root / f"{stem}test_probs.npy", test_probs)
    np.save(root / f"{stem}test_labels.npy", test_labels)
