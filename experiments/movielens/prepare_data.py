from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def extract_movielens_100k(zip_path: str | Path = "ml-100k.zip", extract_to: str | Path = ".") -> Path:
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    folder = extract_to / "ml-100k"
    if not folder.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    return folder


def load_ratings(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"], engine="python")


def load_users(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"], engine="python")


def load_items(path: str | Path) -> pd.DataFrame:
    item_cols = [
        "item_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]
    return pd.read_csv(path, sep="|", names=item_cols, encoding="latin-1", engine="python")


def split_data(df: pd.DataFrame, train_frac: float = 0.8, calib_frac: float = 0.1, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    train_end = int(train_frac * n)
    calib_end = int((train_frac + calib_frac) * n)
    return df_shuffled.iloc[:train_end], df_shuffled.iloc[train_end:calib_end], df_shuffled.iloc[calib_end:]


def _extract_year(date_str) -> int:
    if pd.isna(date_str):
        return 0
    parts = str(date_str).split("-")
    if len(parts) != 3:
        return 0
    try:
        return int(parts[2])
    except ValueError:
        return 0


def preprocess_data_classification(merged_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    df = merged_df.copy()
    df["rating_class"] = df["rating"] - 1
    user2idx = {u: i for i, u in enumerate(df["user_id"].unique())}
    item2idx = {item: i for i, item in enumerate(df["item_id"].unique())}
    df["user_id_idx"] = df["user_id"].map(user2idx)
    df["item_id_idx"] = df["item_id"].map(item2idx)
    df["gender"] = df["gender"].map({"M": 0, "F": 1}).fillna(0).astype(int)
    df["occupation"] = df["occupation"].map({o: i for i, o in enumerate(df["occupation"].unique())}).fillna(0).astype(int)
    df["zip_code"] = df["zip_code"].map({z: i for i, z in enumerate(df["zip_code"].unique())}).fillna(0).astype(int)
    df["release_year"] = df["release_date"].apply(_extract_year)
    return df.drop(columns=["movie_title", "release_date", "video_release_date", "imdb_url"]), user2idx, item2idx


def load_movielens_frames(
    zip_path: str | Path = "ml-100k.zip",
    data_dir: str | Path = ".",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    folder = extract_movielens_100k(zip_path, data_dir)
    ratings = load_ratings(folder / "u.data")
    users = load_users(folder / "u.user")
    items = load_items(folder / "u.item")
    merged = ratings.merge(users, on="user_id", how="left").merge(items, on="item_id", how="left")
    df, user2idx, item2idx = preprocess_data_classification(merged)
    train_df, calib_df, test_df = split_data(df, 0.8, 0.1, seed=seed)
    return train_df, calib_df, test_df, user2idx, item2idx


SIDE_COLS = [
    "age",
    "gender",
    "occupation",
    "zip_code",
    "timestamp",
    "release_year",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def frame_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    work = df.copy()
    for col in SIDE_COLS:
        if col not in work:
            work[col] = 0
    return (
        work["user_id_idx"].to_numpy(np.int64),
        work["item_id_idx"].to_numpy(np.int64),
        work[SIDE_COLS].to_numpy(np.float32),
        work["rating_class"].to_numpy(np.int64),
    )
