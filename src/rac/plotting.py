from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import COVID_ACTION_NAMES, MOVIELENS_ACTION_NAMES
from .metrics import METHOD_COLORS, METHOD_MARKERS, METHOD_ORDER


def _save(fig, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_covid_diagnostic_set_size(exp1_df: pd.DataFrame, output_path: str | Path) -> None:
    alpha_list = sorted(exp1_df["alpha"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    for method in METHOD_ORDER:
        sub = exp1_df[exp1_df["method"] == method].sort_values("alpha")
        axes[0].plot(sub["alpha"], sub["set_size_mean"], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method)
        axes[1].plot(sub["alpha"], sub["miscoverage"], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method)
    axes[0].axhline(1, ls=":", color="gray", alpha=0.5)
    axes[0].axhline(4, ls=":", color="gray", alpha=0.3)
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("Average Prediction Set Size")
    axes[0].set_title("(a) Prediction Set Size")
    axes[0].legend(fontsize=8)
    axes[1].plot(alpha_list, alpha_list, "k--", alpha=0.5, label="nominal alpha")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("Realized Miscoverage")
    axes[1].set_title("(b) Marginal Miscoverage")
    axes[1].legend(fontsize=8)
    _save(fig, output_path)


def plot_covid_scaling(exp2_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    specs = [
        ("set_size_mean", lambda x: x, "(a) Avg Set Size", "Average Set Size"),
        ("coverage_overall", lambda x: 1.0 - x, "(b) Overall Miscoverage", "Miscoverage"),
        ("num_actions_used", lambda x: x, "(c) Number of Distinct Actions Used", "Distinct Actions Used"),
        ("avg_utility", lambda x: x, "(d) Average Utility", "Average Utility"),
    ]
    action_sizes = sorted(exp2_df["|A|"].unique())
    for ax, (key, transform, title, ylabel) in zip(axes.flat, specs):
        for method in METHOD_ORDER:
            sub = exp2_df[exp2_df["method"] == method].sort_values("|A|")
            ax.plot(sub["|A|"], sub[key].apply(transform), marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method)
        if key == "coverage_overall":
            ax.axhline(0.05, ls="--", color="gray", alpha=0.6)
        if key == "num_actions_used":
            ax.plot(action_sizes, action_sizes, ls=":", color="gray", alpha=0.5)
        ax.set_xlabel("|A|")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(action_sizes)
        ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_covid_rare_action(exp3_df: pd.DataFrame, output_path: str | Path) -> None:
    scenarios = list(exp3_df["scenario"].drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    specs = [
        ("target_cov", "(a) Target Action Coverage", "Coverage"),
        ("avg_utility", "(b) Average Utility", "Average Utility"),
        ("set_size_mean", "(c) Average Set Size", "Average Set Size"),
    ]
    for ax, (key, title, ylabel) in zip(axes, specs):
        for method in METHOD_ORDER:
            sub = exp3_df[exp3_df["method"] == method]
            ax.plot(range(len(sub)), sub[key].values, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method)
        if key == "target_cov":
            ax.axhline(0.95, ls="--", color="gray", alpha=0.6)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios)
        ax.set_xlabel("Target action prevalence")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    _save(fig, output_path)


def plot_covid_fdr_frequency(exp4_fdr_df: pd.DataFrame, exp4_freq_df: pd.DataFrame, output_path: str | Path) -> None:
    actions = sorted(exp4_freq_df["action"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    methods_freq = METHOD_ORDER + ["Best Response"]
    width = 0.15
    x_pos = np.arange(len(actions))
    for j, method in enumerate(methods_freq):
        sub = exp4_freq_df[exp4_freq_df["method"] == method]
        freqs = [float(sub[sub["action"] == a]["frequency"].iloc[0]) if len(sub[sub["action"] == a]) else 0.0 for a in actions]
        axes[0].bar(x_pos + j * width, freqs, width, label=method, color=METHOD_COLORS.get(method, "#9467bd"))
    axes[0].set_xticks(x_pos + width * (len(methods_freq) - 1) / 2)
    axes[0].set_xticklabels([COVID_ACTION_NAMES.get(a, str(a)) for a in actions], fontsize=8)
    axes[0].set_ylabel("Selection Frequency")
    axes[0].set_title("(a) Action Selection Frequency")
    axes[0].legend(fontsize=7)

    bars = axes[1].bar(exp4_fdr_df["method"], exp4_fdr_df["fdr_overall"], color=[METHOD_COLORS[m] for m in exp4_fdr_df["method"]])
    axes[1].set_ylabel("False Discovery Rate")
    axes[1].set_title("(b) Overall FDR")
    axes[1].tick_params(axis="x", labelrotation=15)
    for bar, val in zip(bars, exp4_fdr_df["fdr_overall"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", fontsize=8)

    width2 = 0.18
    for j, method in enumerate(METHOD_ORDER):
        row = exp4_fdr_df[exp4_fdr_df["method"] == method].iloc[0]
        vals = [row.get(f"fdr_a{a}", np.nan) for a in actions]
        axes[2].bar(x_pos + j * width2, vals, width2, label=method, color=METHOD_COLORS[method])
    axes[2].set_xticks(x_pos + width2 * (len(METHOD_ORDER) - 1) / 2)
    axes[2].set_xticklabels([COVID_ACTION_NAMES.get(a, str(a)) for a in actions], fontsize=8)
    axes[2].set_ylabel("FDR")
    axes[2].set_title("(c) Per-Action FDR")
    axes[2].legend(fontsize=8)
    _save(fig, output_path)


def plot_appendix_figure2(
    exp1_df: pd.DataFrame,
    output_path: str | Path,
    best_response_utility: float | None = None,
) -> None:
    fig = plt.figure(figsize=(12, 8), dpi=300)
    axes = [fig.add_subplot(2, 4, i + 1) for i in range(4)]
    utility_ax = fig.add_subplot(2, 2, 3)
    alpha_list = sorted(exp1_df["alpha"].unique())
    for action in range(4):
        ax = axes[action]
        for method in METHOD_ORDER:
            sub = exp1_df[exp1_df["method"] == method].sort_values("alpha")
            cov_col = f"cov_a{action}"
            if cov_col in sub:
                ax.plot(sub["alpha"], 1.0 - sub[cov_col], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method)
        ax.plot(alpha_list, alpha_list, "k--", alpha=0.45, label="Nominal" if action == 0 else None)
        ax.set_title(f"({chr(97 + action)}) Action {action}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Miscoverage")
    for method in METHOD_ORDER:
        sub = exp1_df[exp1_df["method"] == method].sort_values("alpha")
        utility_ax.plot(sub["alpha"], sub["avg_utility"], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method)
    if best_response_utility is not None:
        utility_ax.plot(
            alpha_list,
            [best_response_utility] * len(alpha_list),
            marker="o",
            color=METHOD_COLORS["Best Response"],
            label="Best Response",
        )
    utility_ax.set_title("(e) Average Realized Utility")
    utility_ax.set_xlabel("alpha")
    utility_ax.set_ylabel("Avg. realized utility")
    handles, labels = [], []
    for ax in [axes[0], utility_ax]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    _save(fig, output_path)


def plot_appendix_figure3(movie_df: pd.DataFrame, output_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    alpha_list = sorted(movie_df["alpha"].unique())
    for action, ax in zip([0, 1], axes[0]):
        for method in METHOD_ORDER:
            sub = movie_df[movie_df["method"] == method].sort_values("alpha")
            col = f"miscov_a{action}"
            if col in sub:
                ax.plot(sub["alpha"], sub[col], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method)
        ax.plot(alpha_list, alpha_list, "k--", alpha=0.45, label="Nominal")
        ax.set_title(f"({chr(97 + action)}) Action {action}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Miscoverage")
    ax = axes[1, 0]
    for method in ["Best Response"] + METHOD_ORDER:
        sub = movie_df[movie_df["method"] == method].sort_values("alpha")
        if len(sub):
            ax.plot(sub["alpha"], sub["avg_utility"], marker=METHOD_MARKERS.get(method, "o"), color=METHOD_COLORS.get(method, "black"), label=method)
    ax.set_title("(c) Average Realized Utility")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Avg. realized utility")
    axes[1, 1].axis("off")
    handles, labels = [], []
    for legend_ax in [axes[0, 0], ax]:
        for handle, label in zip(*legend_ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    _save(fig, output_path)


def plot_main_figure1(covid_df: pd.DataFrame, movie_df: pd.DataFrame, output_path: str | Path) -> None:
    plot_main_figure1_computed(covid_df, movie_df, output_path, {})


def _row_metric(row: pd.Series, key: str, default=np.nan) -> float:
    if key in row:
        return row[key]
    mean_key = f"{key}_mean"
    if mean_key in row:
        return row[mean_key]
    return default


def _row_stderr(row: pd.Series, key: str) -> float:
    stderr_key = f"{key}_stderr"
    return float(row[stderr_key]) if stderr_key in row and not pd.isna(row[stderr_key]) else 0.0


def plot_main_figure1_computed(
    covid_df: pd.DataFrame,
    movie_df: pd.DataFrame,
    output_path: str | Path,
    covid_best_response_critical: dict[str, float],
    alpha_focus: float = 0.05,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    covid_focus = covid_df[np.isclose(covid_df["alpha"], alpha_focus)]
    covid_actions = [0, 1, 3]
    x = np.arange(len(covid_actions))
    width = 0.18
    for j, method in enumerate(METHOD_ORDER):
        row = covid_focus[covid_focus["method"] == method].iloc[0]
        vals = [1.0 - row.get(f"cov_a{a}", np.nan) for a in covid_actions]
        axes[0, 0].bar(x + j * width, vals, width, label=method, color=METHOD_COLORS[method])
    axes[0, 0].axhline(alpha_focus, ls="--", color="gray")
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels([f"Action {a}" for a in covid_actions], fontsize=8)
    axes[0, 0].set_ylabel("Miscoverage")
    axes[0, 0].set_title("COVID Miscoverage")
    for method in METHOD_ORDER:
        sub = covid_df[covid_df["method"] == method].sort_values("alpha")
        axes[0, 1].plot(sub["alpha"], sub["avg_maxmin"], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], label=method)
    axes[0, 1].set_xlabel("alpha")
    axes[0, 1].set_ylabel("Avg. realized max-min")
    axes[0, 1].set_title("COVID Max-Min Utility")

    labels = ["Pneumonia\nNo action", "COVID-19\nNo action", "Lung Opacity\nNo action"]
    critical_keys = ["pneumonia_no_action_rate", "covid_no_action_rate", "lung_opacity_no_action_rate"]
    crit_methods = ["Best Response", "RAC", "AC-RAC"]
    xx = np.arange(3)
    for j, method in enumerate(crit_methods):
        if method == "Best Response":
            vals = [100 * covid_best_response_critical.get(key, np.nan) for key in critical_keys]
        else:
            row = covid_focus[covid_focus["method"] == method].iloc[0]
            vals = [100 * row.get(key, np.nan) for key in critical_keys]
        axes[0, 2].bar(xx + j * 0.25, vals, 0.25, label=method, color=METHOD_COLORS.get(method, "black"))
    axes[0, 2].set_xticks(xx + 0.25)
    axes[0, 2].set_xticklabels(labels, fontsize=8)
    axes[0, 2].set_ylabel("Bad decisions (%)")
    axes[0, 2].set_title("COVID Critical Decisions")

    movie_focus = movie_df[np.isclose(movie_df["alpha"], alpha_focus)]
    x2 = np.arange(2)
    for j, method in enumerate(METHOD_ORDER):
        row = movie_focus[movie_focus["method"] == method].iloc[0]
        vals = [_row_metric(row, f"miscov_a{a}") for a in range(2)]
        errs = [_row_stderr(row, f"miscov_a{a}") for a in range(2)]
        axes[1, 0].bar(x2 + j * width, vals, width, yerr=errs, capsize=2, label=method, color=METHOD_COLORS[method])
    axes[1, 0].axhline(alpha_focus, ls="--", color="gray")
    axes[1, 0].set_xticks(x2 + width * 1.5)
    axes[1, 0].set_xticklabels([MOVIELENS_ACTION_NAMES[a] for a in range(2)], fontsize=8)
    axes[1, 0].set_ylabel("Miscoverage")
    axes[1, 0].set_title("MovieLens Miscoverage")

    for method in METHOD_ORDER:
        sub = movie_df[movie_df["method"] == method].sort_values("alpha")
        y = sub["avg_maxmin_mean"] if "avg_maxmin_mean" in sub else sub["avg_maxmin"]
        yerr = sub["avg_maxmin_stderr"] if "avg_maxmin_stderr" in sub else None
        axes[1, 1].errorbar(
            sub["alpha"],
            y,
            yerr=yerr,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            capsize=2,
            label=method,
        )
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_ylabel("Avg. realized max-min")
    axes[1, 1].set_title("MovieLens Max-Min Utility")

    movie_keys = ["rating1_recommend_rate", "rating2_recommend_rate"]
    for j, method in enumerate(crit_methods):
        row = movie_focus[movie_focus["method"] == method].iloc[0]
        vals = [100 * _row_metric(row, key) for key in movie_keys]
        errs = [100 * _row_stderr(row, key) for key in movie_keys]
        axes[1, 2].bar(x2 + j * 0.25, vals, 0.25, yerr=errs, capsize=2, label=method, color=METHOD_COLORS.get(method, "black"))
    axes[1, 2].set_xticks(x2 + 0.25)
    axes[1, 2].set_xticklabels(["True Rating 1\nRecommend", "True Rating 2\nRecommend"], fontsize=8)
    axes[1, 2].set_ylabel("Bad decisions (%)")
    axes[1, 2].set_title("MovieLens Critical Decisions")
    for ax in axes.flat:
        ax.legend(fontsize=7)
    _save(fig, output_path)
