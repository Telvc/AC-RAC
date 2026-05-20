from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rac.data import COVID_ACTION_NAMES, COVID_UTILITY_MATRIX, load_prediction_cache, make_utility_fn
from rac.metrics import run_all_methods, run_best_response


ALPHA_LIST = [0.01, 0.02, 0.03, 0.05, 0.10]
ACTION_SIZES = [4, 6, 8, 10]
RARE_PREVALENCES = ["full", 0.01, 0.005, 0.001]
ALPHA_FOCUS = 0.05

FULL_UTILITY_MATRIX = np.array(
    [
        [10, 2, 2, 4, 8, 7, 8, 2, 2, 2],
        [0, 10, 3, 7, 8, 2, 2, 8, 8, 2],
        [0, 3, 10, 8, 2, 8, 2, 8, 2, 9],
        [1, 4, 4, 10, 3, 3, 9, 3, 9, 9],
    ],
    dtype=float,
)


def get_expanded_utility_matrix(num_actions: int) -> np.ndarray:
    return FULL_UTILITY_MATRIX[:, :num_actions]


def load_cached_covid_csvs(source_dir: str | Path = "data/cached/covid") -> dict[str, pd.DataFrame]:
    root = Path(source_dir)
    return {
        "exp1": pd.read_csv(root / "exp1_set_size.csv"),
        "exp2": pd.read_csv(root / "exp2_action_scaling.csv"),
        "exp3": pd.read_csv(root / "exp3_rare_action.csv"),
        "exp4_fdr": pd.read_csv(root / "exp4_fdr.csv"),
        "exp4_freq": pd.read_csv(root / "exp4_action_freq.csv"),
    }


def _no_action_rates(acts: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    rates = {}
    for label, name in [
        (1, "pneumonia_no_action_rate"),
        (2, "covid_no_action_rate"),
        (3, "lung_opacity_no_action_rate"),
    ]:
        mask = labels == label
        rates[name] = float(np.mean(acts[mask] == 0)) if np.any(mask) else np.nan
    return rates


def _result_rows(alpha: float, results: dict[str, dict], actions: list[int], true_labels: np.ndarray | None = None) -> list[dict]:
    rows = []
    for method, res in results.items():
        row = {
            "alpha": alpha,
            "method": method,
            "set_size_mean": res["set_size_mean"],
            "coverage_overall": res["coverage_overall"],
            "miscoverage": 1.0 - res["coverage_overall"],
            "avg_utility": res["avg_utility"],
            "avg_maxmin": res["avg_maxmin"],
            "fdr_overall": res["fdr_overall"],
        }
        row.update({f"cov_a{a}": res["coverage_by_action"].get(a, np.nan) for a in actions})
        row.update({f"setsize_a{a}": res["set_size_by_action"].get(a, np.nan) for a in actions})
        if true_labels is not None and "acts" in res:
            row.update(_no_action_rates(res["acts"], true_labels))
        rows.append(row)
    return rows


def compute_exp1(
    cal_p,
    cal_l,
    test_p,
    test_l,
    alphas: list[float] | None = None,
    ac_max_iter: int = 400,
    ac_eta0: float = 25.0,
    ac_batch_size: int = 32,
) -> tuple[pd.DataFrame, dict[float, dict[str, dict]]]:
    uf = make_utility_fn(COVID_UTILITY_MATRIX)
    actions = [0, 1, 2, 3]
    rows, all_results = [], {}
    for alpha in alphas or ALPHA_LIST:
        print(f"[COVID] exp1 alpha={alpha}", flush=True)
        results = run_all_methods(
            cal_p,
            cal_l,
            test_p,
            test_l,
            alpha,
            actions,
            uf,
            ac_kwargs={
                "max_iter": ac_max_iter,
                "eta0": ac_eta0,
                "batch_size": ac_batch_size,
                "progress_label": f"COVID exp1 alpha={alpha}",
                "progress_every": max(1, len(test_l) // 10),
            },
        )
        all_results[alpha] = results
        rows.extend(_result_rows(alpha, results, actions, test_l))
    return pd.DataFrame(rows), all_results


def compute_exp2(
    cal_p,
    cal_l,
    test_p,
    test_l,
    action_sizes: list[int] | None = None,
    ac_max_iter: int = 400,
    ac_eta0: float = 25.0,
    ac_batch_size: int = 32,
) -> pd.DataFrame:
    rows = []
    for num_actions in action_sizes or ACTION_SIZES:
        print(f"[COVID] exp2 |A|={num_actions}", flush=True)
        uf = make_utility_fn(get_expanded_utility_matrix(num_actions))
        actions = list(range(num_actions))
        results = run_all_methods(
            cal_p,
            cal_l,
            test_p,
            test_l,
            ALPHA_FOCUS,
            actions,
            uf,
            ac_kwargs={
                "max_iter": max(ac_max_iter, num_actions * 50),
                "eta0": ac_eta0,
                "batch_size": ac_batch_size,
                "progress_label": f"COVID exp2 |A|={num_actions}",
                "progress_every": max(1, len(test_l) // 10),
            },
        )
        for method, res in results.items():
            cov_vals = [v for v in res["coverage_by_action"].values() if not np.isnan(v)]
            worst_cov = min(cov_vals) if cov_vals else np.nan
            row = {
                "|A|": num_actions,
                "method": method,
                "set_size_mean": res["set_size_mean"],
                "coverage_overall": res["coverage_overall"],
                "worst_case_coverage": worst_cov,
                "worst_miscoverage": 1.0 - worst_cov if not np.isnan(worst_cov) else np.nan,
                "avg_utility": res["avg_utility"],
                "avg_maxmin": res["avg_maxmin"],
                "fdr_overall": res["fdr_overall"],
                "num_actions_used": sum(1 for a in actions if res["action_counts"].get(a, 0) > 0),
                "critical_error_rate": res["critical_error_rate"],
                "miscoverage": 1.0 - res["coverage_overall"],
            }
            row.update({f"cov_a{a}": res["coverage_by_action"].get(a, np.nan) for a in actions})
            rows.append(row)
    return pd.DataFrame(rows)


def _downsample_for_rarity(ref_actions: np.ndarray, target_action: int, target_prevalence: float, seed: int = 42) -> tuple[np.ndarray, float]:
    idx_target = np.where(ref_actions == target_action)[0]
    idx_other = np.where(ref_actions != target_action)[0]
    if len(idx_target) == 0:
        return np.arange(len(ref_actions)), 0.0
    want = int(round(target_prevalence * len(idx_other) / max(1.0 - target_prevalence, 1e-12)))
    want = max(1, min(want, len(idx_target)))
    rng = np.random.default_rng(seed + int(target_prevalence * 1e6))
    keep_target = rng.choice(idx_target, size=want, replace=False)
    keep = np.sort(np.concatenate([idx_other, keep_target]))
    return keep, want / len(keep)


def compute_exp3(
    cal_p,
    cal_l,
    test_p,
    test_l,
    prevalences=RARE_PREVALENCES,
    ac_max_iter: int = 400,
    ac_eta0: float = 25.0,
    ac_batch_size: int = 32,
) -> pd.DataFrame:
    uf = make_utility_fn(COVID_UTILITY_MATRIX)
    actions = [0, 1, 2, 3]
    print("[COVID] exp3 reference actions", flush=True)
    ref_results = run_all_methods(
        cal_p,
        cal_l,
        cal_p,
        cal_l,
        ALPHA_FOCUS,
        actions,
        uf,
        ac_kwargs={
            "max_iter": ac_max_iter,
            "eta0": ac_eta0,
            "batch_size": ac_batch_size,
            "progress_label": "COVID exp3 reference",
            "progress_every": max(1, len(cal_l) // 10),
        },
    )
    ref_actions = ref_results["AC-RAC"]["acts"]
    ref_counts = Counter(int(a) for a in ref_actions)
    target_action = 0
    rows = []
    for prev in prevalences:
        print(f"[COVID] exp3 prevalence={prev}", flush=True)
        if prev == "full":
            cp, cy = cal_p, cal_l
            actual_prev = ref_counts.get(target_action, 0) / len(ref_actions)
            scenario = "full"
        else:
            keep_idx, actual_prev = _downsample_for_rarity(ref_actions, target_action, float(prev))
            cp, cy = cal_p[keep_idx], cal_l[keep_idx]
            scenario = f"{float(prev) * 100:g}%"
        results = run_all_methods(
            cp,
            cy,
            test_p,
            test_l,
            ALPHA_FOCUS,
            actions,
            uf,
            ac_kwargs={
                "max_iter": ac_max_iter,
                "eta0": ac_eta0,
                "batch_size": ac_batch_size,
                "progress_label": f"COVID exp3 prevalence={prev}",
                "progress_every": max(1, len(test_l) // 10),
            },
        )
        for method, res in results.items():
            target_cov = res["coverage_by_action"].get(target_action, np.nan)
            row = {
                "scenario": scenario,
                "target_prevalence": actual_prev,
                "cal_size": len(cp),
                "method": method,
                "target_action": target_action,
                "target_cov": target_cov,
                "target_miscov": 1.0 - target_cov if not np.isnan(target_cov) else np.nan,
                "coverage_overall": res["coverage_overall"],
                "avg_utility": res["avg_utility"],
                "set_size_mean": res["set_size_mean"],
            }
            row.update({f"cov_a{a}": res["coverage_by_action"].get(a, np.nan) for a in actions})
            rows.append(row)
    return pd.DataFrame(rows)


def compute_exp4(test_p, test_l, focus_results: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = [0, 1, 2, 3]
    fdr_rows = []
    for method, res in focus_results.items():
        row = {
            "method": method,
            "fdr_overall": res["fdr_overall"],
            "set_size_mean": res["set_size_mean"],
            "miscoverage": 1.0 - res["coverage_overall"],
            "avg_utility": res["avg_utility"],
        }
        row.update({f"fdr_a{a}": res["fdr_by_action"].get(a, np.nan) for a in actions})
        fdr_rows.append(row)

    freq_rows = []
    n_test = len(test_l)
    for method, res in focus_results.items():
        for a in actions:
            count = res["action_counts"].get(a, 0)
            freq_rows.append({"method": method, "action": a, "action_name": COVID_ACTION_NAMES[a], "count": count, "frequency": count / n_test})
    br = run_best_response(test_p, test_l, actions, make_utility_fn(COVID_UTILITY_MATRIX))
    for a in actions:
        count = br["action_counts"].get(a, 0)
        freq_rows.append({"method": "Best Response", "action": a, "action_name": COVID_ACTION_NAMES[a], "count": count, "frequency": count / n_test})
    return pd.DataFrame(fdr_rows), pd.DataFrame(freq_rows)


def run_covid_diagnostics(
    fast: bool = True,
    output_dir: str | Path = "artifacts/results/covid",
    cache_dir: str | Path = "data/cached/covid",
    cache_prefix: str = "",
    ac_max_iter: int = 400,
    ac_eta0: float = 25.0,
    ac_batch_size: int = 32,
    fast_smoke: bool = False,
    eval_calib_size: int | None = None,
    eval_test_size: int | None = None,
) -> dict[str, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[COVID] diagnostics start fast={fast}", flush=True)
    if fast:
        dfs = load_cached_covid_csvs(cache_dir)
    else:
        cal_p, cal_l, test_p, test_l = load_prediction_cache(cache_dir, prefix=cache_prefix)
        if eval_calib_size is not None:
            cal_p, cal_l = cal_p[:eval_calib_size], cal_l[:eval_calib_size]
        if eval_test_size is not None:
            test_p, test_l = test_p[:eval_test_size], test_l[:eval_test_size]
        alphas = ALPHA_LIST
        action_sizes = ACTION_SIZES
        prevalences = RARE_PREVALENCES
        if fast_smoke:
            if eval_calib_size is None:
                cal_p, cal_l = cal_p[:200], cal_l[:200]
            if eval_test_size is None:
                test_p, test_l = test_p[:100], test_l[:100]
            alphas = [ALPHA_FOCUS]
            action_sizes = [4]
            prevalences = ["full"]
        exp1, results = compute_exp1(
            cal_p,
            cal_l,
            test_p,
            test_l,
            alphas=alphas,
            ac_max_iter=ac_max_iter,
            ac_eta0=ac_eta0,
            ac_batch_size=ac_batch_size,
        )
        exp2 = compute_exp2(
            cal_p,
            cal_l,
            test_p,
            test_l,
            action_sizes=action_sizes,
            ac_max_iter=ac_max_iter,
            ac_eta0=ac_eta0,
            ac_batch_size=ac_batch_size,
        )
        exp3 = compute_exp3(
            cal_p,
            cal_l,
            test_p,
            test_l,
            prevalences=prevalences,
            ac_max_iter=ac_max_iter,
            ac_eta0=ac_eta0,
            ac_batch_size=ac_batch_size,
        )
        exp4_fdr, exp4_freq = compute_exp4(test_p, test_l, results[ALPHA_FOCUS])
        dfs = {"exp1": exp1, "exp2": exp2, "exp3": exp3, "exp4_fdr": exp4_fdr, "exp4_freq": exp4_freq}

    names = {
        "exp1": "exp1_set_size.csv",
        "exp2": "exp2_action_scaling.csv",
        "exp3": "exp3_rare_action.csv",
        "exp4_fdr": "exp4_fdr.csv",
        "exp4_freq": "exp4_action_freq.csv",
    }
    for key, name in names.items():
        dfs[key].to_csv(out / name, index=False)
    print(f"[COVID] diagnostics done -> {out}", flush=True)
    return dfs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use cached CSV inputs.")
    parser.add_argument("--output-dir", default="artifacts/results/covid")
    parser.add_argument("--cache-dir", default="data/cached/covid")
    parser.add_argument("--cache-prefix", default="")
    parser.add_argument("--ac-max-iter", type=int, default=400)
    parser.add_argument("--ac-eta0", type=float, default=25.0)
    parser.add_argument("--ac-batch-size", type=int, default=32)
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--eval-calib-size", type=int, default=None)
    parser.add_argument("--eval-test-size", type=int, default=None)
    args = parser.parse_args()
    run_covid_diagnostics(
        fast=args.fast,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        cache_prefix=args.cache_prefix,
        ac_max_iter=args.ac_max_iter,
        ac_eta0=args.ac_eta0,
        ac_batch_size=args.ac_batch_size,
        fast_smoke=args.fast_smoke,
        eval_calib_size=args.eval_calib_size,
        eval_test_size=args.eval_test_size,
    )
