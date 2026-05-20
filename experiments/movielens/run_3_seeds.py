from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rac.conformal import select_action_max_min  # noqa: E402
from rac.data import MOVIELENS_UTILITY_MATRIX, load_prediction_cache, make_utility_fn, save_prediction_cache  # noqa: E402
from rac.metrics import (  # noqa: E402
    METHOD_COLORS,
    METHOD_MARKERS,
    METHOD_ORDER,
    run_acrac,
    run_best_response,
    run_rac,
    run_score_one_minus_true,
    run_score_tail,
)

from .prepare_data import load_movielens_frames
from .train_model import DeepRecommenderClassifier, MovieLensClassifDataset, _predict, _train_one_epoch


FIGURE1_METHODS = ["AC-RAC", "RAC", "Score-1", "Score-2"]
CRITICAL_METHODS = ["Best Response", "RAC", "AC-RAC"]
PAPER_ALPHA_LIST = [0.01, 0.02, 0.03, 0.05, 0.10]
PAPER_TUNED_AC_EPSILON = 0.045
PAPER_TUNED_AC_MAX_ITER = 100
PAPER_TUNED_AC_ETA0 = 2.0


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    if ":" in value:
        start, stop = value.split(":", 1)
        return list(range(int(start), int(stop)))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def movielens_utility_matrix(tie_break_epsilon: float = 0.0) -> np.ndarray:
    """Return the MovieLens utility matrix used for reruns.

    The visible paper table has zero utility for No-Recommend. Appendix D.2
    mentions a small epsilon tie break; callers can opt into that perturbation
    for policy selection. Reported utilities default to the visible table.
    """
    matrix = MOVIELENS_UTILITY_MATRIX.copy()
    if tie_break_epsilon:
        ratings = np.arange(1, matrix.shape[0] + 1)
        matrix[:, 0] = tie_break_epsilon * (3 - ratings)
    return matrix


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_predict_for_seed(
    zip_path: str | Path,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    set_reproducible_seed(seed)
    train_df, calib_df, test_df, user2idx, item2idx = load_movielens_frames(zip_path, ".", seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        MovieLensClassifDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    calib_loader = DataLoader(MovieLensClassifDataset(calib_df), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(MovieLensClassifDataset(test_df), batch_size=batch_size, shuffle=False, num_workers=0)

    model = DeepRecommenderClassifier(len(user2idx), len(item2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        _train_one_epoch(model, train_loader, optimizer, device)
    cal_probs, cal_labels = _predict(model, calib_loader, device)
    test_probs, test_labels = _predict(model, test_loader, device)
    return cal_probs, cal_labels, test_probs, test_labels


def movielens_seed_cache_dir(cache_root: str | Path, seed: int) -> Path:
    return Path(cache_root) / f"seed_{seed}"


def has_movielens_seed_cache(cache_root: str | Path, seed: int) -> bool:
    root = movielens_seed_cache_dir(cache_root, seed)
    return all((root / name).exists() for name in ["cal_probs.npy", "cal_labels.npy", "test_probs.npy", "test_labels.npy"])


def get_or_create_predictions_for_seed(
    zip_path: str | Path,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed_cache_root: str | Path | None = "artifacts/cached_predictions/movielens_seeded",
    refresh_seed_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if seed_cache_root is not None and not refresh_seed_cache and has_movielens_seed_cache(seed_cache_root, seed):
        print(f"[MovieLens] seed={seed}: loading cached predictions from {movielens_seed_cache_dir(seed_cache_root, seed)}")
        return load_prediction_cache(movielens_seed_cache_dir(seed_cache_root, seed))

    cal_probs, cal_labels, test_probs, test_labels = train_and_predict_for_seed(zip_path, seed, epochs, batch_size, device)
    if seed_cache_root is not None:
        cache_dir = movielens_seed_cache_dir(seed_cache_root, seed)
        save_prediction_cache(cache_dir, cal_probs, cal_labels, test_probs, test_labels)
        print(f"[MovieLens] seed={seed}: saved predictions to {cache_dir}")
    return cal_probs, cal_labels, test_probs, test_labels


def recommendation_error_rates(acts: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rates = {}
    for label, name in [(0, "rating1_recommend_rate"), (1, "rating2_recommend_rate")]:
        mask = labels == label
        rates[name] = float(np.mean(acts[mask] == 1)) if np.any(mask) else np.nan
    return rates


def report_utility_metrics(result: dict, labels: np.ndarray, actions: list[int], utility_fn) -> dict[str, float]:
    acts = result["acts"]
    sets_list = result["sets"]
    realized_utils = [float(utility_fn(int(a), int(y))) for a, y in zip(acts, labels)]
    maxmin_values = []
    for c_set, action in zip(sets_list, acts):
        _, min_util = select_action_max_min(set(c_set), [int(action)], utility_fn)
        maxmin_values.append(float(min_util))
    return {
        "avg_utility": float(np.mean(realized_utils)) if realized_utils else np.nan,
        "avg_maxmin": float(np.mean(maxmin_values)) if maxmin_values else np.nan,
    }


def rows_for_seed(
    seed: int,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alphas: list[float],
    selection_tie_break_epsilon: float,
    ac_selection_tie_break_epsilon: float | None,
    report_tie_break_epsilon: float,
    ac_max_iter: int,
    ac_eta0: float,
    ac_batch_size: int,
    eval_calib_size: int | None,
    eval_test_size: int | None,
) -> list[dict]:
    if eval_calib_size is not None:
        cal_probs = cal_probs[:eval_calib_size]
        cal_labels = cal_labels[:eval_calib_size]
    if eval_test_size is not None:
        test_probs = test_probs[:eval_test_size]
        test_labels = test_labels[:eval_test_size]

    utility_fn = make_utility_fn(movielens_utility_matrix(selection_tie_break_epsilon))
    ac_utility_fn = make_utility_fn(
        movielens_utility_matrix(
            selection_tie_break_epsilon if ac_selection_tie_break_epsilon is None else ac_selection_tie_break_epsilon
        )
    )
    report_utility_fn = make_utility_fn(movielens_utility_matrix(report_tie_break_epsilon))
    actions = [0, 1]
    rows = []

    best_response = run_best_response(test_probs, test_labels, actions, report_utility_fn)
    best_response_base = {
        "method": "Best Response",
        "coverage_overall": np.nan,
        "miscoverage": np.nan,
        "miscov_a0": np.nan,
        "miscov_a1": np.nan,
        "avg_utility": best_response["avg_utility"],
        "avg_maxmin": np.nan,
        "set_size_mean": np.nan,
        "fdr_overall": np.nan,
        "action0_frequency": best_response["action_frequencies"].get(0, 0.0),
        "action1_frequency": best_response["action_frequencies"].get(1, 0.0),
        **recommendation_error_rates(best_response["acts"], test_labels),
    }

    for alpha in alphas:
        ac_result = run_acrac(
            cal_probs,
            cal_labels,
            test_probs,
            test_labels,
            alpha,
            actions,
            ac_utility_fn,
            max_iter=ac_max_iter,
            eta0=ac_eta0,
            batch_size=ac_batch_size,
            progress_label=f"MovieLens seed={seed} alpha={alpha}",
            progress_every=max(1, len(test_labels) // 10),
        )
        results = {
            "AC-RAC": ac_result,
            "RAC": run_rac(cal_probs, cal_labels, test_probs, test_labels, alpha, actions, utility_fn),
            "Score-1": run_score_tail(cal_probs, cal_labels, test_probs, test_labels, alpha, actions, utility_fn),
            "Score-2": run_score_one_minus_true(cal_probs, cal_labels, test_probs, test_labels, alpha, actions, utility_fn),
        }
        for method, res in results.items():
            cov_a0 = res["coverage_by_action"].get(0, np.nan)
            cov_a1 = res["coverage_by_action"].get(1, np.nan)
            report_metrics = report_utility_metrics(res, test_labels, actions, report_utility_fn)
            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "method": method,
                    "coverage_overall": res.get("coverage_overall", np.nan),
                    "miscoverage": 1.0 - res["coverage_overall"],
                    "miscov_a0": 1.0 - cov_a0 if not np.isnan(cov_a0) else np.nan,
                    "miscov_a1": 1.0 - cov_a1 if not np.isnan(cov_a1) else np.nan,
                    "avg_utility": report_metrics["avg_utility"],
                    "avg_maxmin": res.get("avg_maxmin", np.nan),
                    "report_avg_maxmin": report_metrics["avg_maxmin"],
                    "decision_avg_utility": res["avg_utility"],
                    "set_size_mean": res.get("set_size_mean", np.nan),
                    "fdr_overall": res.get("fdr_overall", np.nan),
                    "action0_frequency": res["action_frequencies"].get(0, 0.0),
                    "action1_frequency": res["action_frequencies"].get(1, 0.0),
                    **recommendation_error_rates(res["acts"], test_labels),
                }
            )
        rows.append({"seed": seed, "alpha": alpha, **best_response_base})
    return rows


def aggregate_seed_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    keys = ["alpha", "method"]
    metric_cols = [
        col
        for col in per_seed.columns
        if col not in {"seed", "method"} and col not in keys and pd.api.types.is_numeric_dtype(per_seed[col])
    ]
    grouped = per_seed.groupby(keys, dropna=False)[metric_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    count = grouped.count().add_suffix("_count")
    out = pd.concat([mean, std, count], axis=1).reset_index()
    for col in metric_cols:
        count_col = f"{col}_count"
        stderr = out[f"{col}_std"] / np.sqrt(out[count_col].replace(0, np.nan))
        out[f"{col}_stderr"] = stderr.fillna(0.0)
    return out


def _series(agg: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    return agg[agg["method"] == method].sort_values("alpha")[["alpha", f"{metric}_mean", f"{metric}_stderr"]]


def _errorbar(ax, df: pd.DataFrame, metric: str, method: str) -> None:
    ax.errorbar(
        df["alpha"],
        df[f"{metric}_mean"],
        yerr=df[f"{metric}_stderr"].fillna(0.0),
        marker=METHOD_MARKERS.get(method, "o"),
        color=METHOD_COLORS.get(method, "black"),
        capsize=2,
        label=method,
    )


def plot_figure3_from_aggregate(agg: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    alpha_list = sorted(agg["alpha"].unique())
    for action, ax in zip([0, 1], axes[0]):
        metric = f"miscov_a{action}"
        for method in METHOD_ORDER:
            sub = _series(agg, method, metric)
            _errorbar(ax, sub, metric, method)
        ax.plot(alpha_list, alpha_list, "k--", alpha=0.45, label="Nominal")
        ax.set_title(f"({chr(97 + action)}) Action {action}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Miscoverage")
    utility_ax = axes[1, 0]
    for method in ["Best Response"] + METHOD_ORDER:
        sub = _series(agg, method, "avg_utility")
        if len(sub):
            _errorbar(utility_ax, sub, "avg_utility", method)
    utility_ax.set_title("(c) Average Realized Utility")
    utility_ax.set_xlabel("alpha")
    utility_ax.set_ylabel("Avg. realized utility")
    axes[1, 1].axis("off")
    handles, labels = [], []
    for legend_ax in [axes[0, 0], utility_ax]:
        for handle, label in zip(*legend_ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_figure1_bottom_from_aggregate(
    agg: pd.DataFrame,
    output_path: str | Path,
    alpha_focus: float = 0.05,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300)
    focus = agg[np.isclose(agg["alpha"], alpha_focus)]
    x = np.arange(2)
    width = 0.18
    for j, method in enumerate(FIGURE1_METHODS):
        row = focus[focus["method"] == method].iloc[0]
        vals = [row["miscov_a0_mean"], row["miscov_a1_mean"]]
        errs = [row["miscov_a0_stderr"], row["miscov_a1_stderr"]]
        axes[0].bar(x + j * width, vals, width, yerr=errs, capsize=2, label=method, color=METHOD_COLORS[method])
    axes[0].axhline(alpha_focus, ls="--", color="gray")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(["No-Recommend", "Recommend"], fontsize=8)
    axes[0].set_ylabel("Miscoverage")
    axes[0].set_title("MovieLens Miscoverage")

    for method in FIGURE1_METHODS:
        sub = _series(agg, method, "avg_maxmin")
        _errorbar(axes[1], sub, "avg_maxmin", method)
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Avg. realized max-min")
    axes[1].set_title("MovieLens Max-Min Utility")

    labels = ["True Rating 1\nRecommend", "True Rating 2\nRecommend"]
    for j, method in enumerate(CRITICAL_METHODS):
        row = focus[focus["method"] == method].iloc[0]
        vals = [100 * row["rating1_recommend_rate_mean"], 100 * row["rating2_recommend_rate_mean"]]
        errs = [100 * row["rating1_recommend_rate_stderr"], 100 * row["rating2_recommend_rate_stderr"]]
        axes[2].bar(x + j * 0.25, vals, 0.25, yerr=errs, capsize=2, label=method, color=METHOD_COLORS.get(method, "black"))
    axes[2].set_xticks(x + 0.25)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("Bad decisions (%)")
    axes[2].set_title("MovieLens Critical Decisions")
    for ax in axes:
        ax.legend(fontsize=7)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_movielens_seed_sweep(
    zip_path: str | Path = "ml-100k.zip",
    seeds: list[int] | None = None,
    alphas: list[float] | None = None,
    epochs: int = 20,
    batch_size: int = 256,
    output_dir: str | Path = "artifacts/results/movielens_3seeds",
    figures_dir: str | Path = "artifacts/figures/movielens_3seeds",
    selection_tie_break_epsilon: float = 0.0,
    ac_selection_tie_break_epsilon: float | None = PAPER_TUNED_AC_EPSILON,
    report_tie_break_epsilon: float = 0.0,
    ac_max_iter: int = PAPER_TUNED_AC_MAX_ITER,
    ac_eta0: float = PAPER_TUNED_AC_ETA0,
    ac_batch_size: int = 32,
    jobs: int = 1,
    eval_calib_size: int | None = None,
    eval_test_size: int | None = None,
    seed_cache_root: str | Path | None = "artifacts/cached_predictions/movielens_seeded",
    refresh_seed_cache: bool = False,
    make_figures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = seeds if seeds is not None else [0, 1, 2]
    alphas = alphas if alphas is not None else PAPER_ALPHA_LIST
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if make_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    for seed in seeds:
        print(f"[MovieLens] seed={seed}: preparing predictions on {device}")
        cal_probs, cal_labels, test_probs, test_labels = get_or_create_predictions_for_seed(
            zip_path,
            seed,
            epochs,
            batch_size,
            device,
            seed_cache_root=seed_cache_root,
            refresh_seed_cache=refresh_seed_cache,
        )
        print(f"[MovieLens] seed={seed}: evaluating {len(alphas)} alpha values")
        if jobs > 1 and len(alphas) > 1:
            worker_count = min(int(jobs), len(alphas))
            print(f"[MovieLens] seed={seed}: parallel alpha workers={worker_count}")
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        rows_for_seed,
                        seed,
                        cal_probs,
                        cal_labels,
                        test_probs,
                        test_labels,
                        [alpha],
                        selection_tie_break_epsilon,
                        ac_selection_tie_break_epsilon,
                        report_tie_break_epsilon,
                        ac_max_iter,
                        ac_eta0,
                        ac_batch_size,
                        eval_calib_size,
                        eval_test_size,
                    )
                    for alpha in alphas
                ]
                for future in as_completed(futures):
                    all_rows.extend(future.result())
        else:
            all_rows.extend(
                rows_for_seed(
                    seed,
                    cal_probs,
                    cal_labels,
                    test_probs,
                    test_labels,
                    alphas,
                    selection_tie_break_epsilon,
                    ac_selection_tie_break_epsilon,
                    report_tie_break_epsilon,
                    ac_max_iter,
                    ac_eta0,
                    ac_batch_size,
                    eval_calib_size,
                    eval_test_size,
                )
            )

    per_seed = pd.DataFrame(all_rows)
    per_seed = per_seed.sort_values(["seed", "alpha", "method"]).reset_index(drop=True)
    aggregate = aggregate_seed_results(per_seed)
    per_seed.to_csv(output_dir / "movielens_per_seed.csv", index=False)
    aggregate.to_csv(output_dir / "movielens_aggregate.csv", index=False)
    if make_figures:
        plot_figure1_bottom_from_aggregate(aggregate, figures_dir / "figure1_movielens_bottom.pdf")
        plot_figure3_from_aggregate(aggregate, figures_dir / "figure3_movielens_appendix.pdf")
    return per_seed, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun MovieLens over multiple seeds and aggregate Figure 1/3 data.")
    parser.add_argument("--zip-path", default="ml-100k.zip")
    parser.add_argument("--seeds", default="0,1,2", help="Comma list such as 0,1,2 or half-open range such as 0:3.")
    parser.add_argument("--alphas", default=",".join(str(a) for a in PAPER_ALPHA_LIST))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="artifacts/results/movielens_3seeds_paper_protocol")
    parser.add_argument("--figures-dir", default="artifacts/figures/movielens_3seeds_paper_protocol")
    parser.add_argument("--selection-tie-break-epsilon", type=float, default=0.0)
    parser.add_argument("--ac-selection-tie-break-epsilon", type=float, default=PAPER_TUNED_AC_EPSILON)
    parser.add_argument("--report-tie-break-epsilon", type=float, default=0.0)
    parser.add_argument("--tie-break-epsilon", type=float, dest="selection_tie_break_epsilon", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--ac-max-iter", type=int, default=PAPER_TUNED_AC_MAX_ITER)
    parser.add_argument("--ac-eta0", type=float, default=PAPER_TUNED_AC_ETA0)
    parser.add_argument("--ac-batch-size", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel alpha workers per seed.")
    parser.add_argument("--eval-calib-size", type=int, default=None, help="Optional prefix size for faster exact AC-RAC checks.")
    parser.add_argument("--eval-test-size", type=int, default=None, help="Optional prefix size for faster exact AC-RAC checks.")
    parser.add_argument("--seed-cache-root", default="artifacts/cached_predictions/movielens_seeded")
    parser.add_argument("--refresh-seed-cache", action="store_true")
    parser.add_argument("--no-seed-cache", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    run_movielens_seed_sweep(
        zip_path=args.zip_path,
        seeds=parse_seed_list(args.seeds),
        alphas=parse_float_list(args.alphas),
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        selection_tie_break_epsilon=args.selection_tie_break_epsilon,
        ac_selection_tie_break_epsilon=args.ac_selection_tie_break_epsilon,
        report_tie_break_epsilon=args.report_tie_break_epsilon,
        ac_max_iter=args.ac_max_iter,
        ac_eta0=args.ac_eta0,
        ac_batch_size=args.ac_batch_size,
        jobs=args.jobs,
        eval_calib_size=args.eval_calib_size,
        eval_test_size=args.eval_test_size,
        seed_cache_root=None if args.no_seed_cache else args.seed_cache_root,
        refresh_seed_cache=args.refresh_seed_cache,
        make_figures=not args.no_figures,
    )


if __name__ == "__main__":
    main()
