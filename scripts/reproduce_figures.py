from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.covid.run_diagnostics import _no_action_rates, run_covid_diagnostics  # noqa: E402
from experiments.movielens.run_3_seeds import plot_figure3_from_aggregate  # noqa: E402
from rac.data import COVID_UTILITY_MATRIX, load_prediction_cache, make_utility_fn  # noqa: E402
from rac.metrics import run_best_response  # noqa: E402
from rac.plotting import (  # noqa: E402
    plot_appendix_figure2,
    plot_covid_diagnostic_set_size,
    plot_covid_fdr_frequency,
    plot_covid_rare_action,
    plot_covid_scaling,
    plot_main_figure1_computed,
)


def _covid_best_response(cache_dir: str | Path, cache_prefix: str) -> tuple[float | None, dict[str, float]]:
    try:
        _, _, test_probs, test_labels = load_prediction_cache(cache_dir, prefix=cache_prefix)
    except FileNotFoundError:
        return None, {}
    result = run_best_response(test_probs, test_labels, [0, 1, 2, 3], make_utility_fn(COVID_UTILITY_MATRIX))
    return float(result["avg_utility"]), _no_action_rates(result["acts"], test_labels)


def _load_movielens_aggregate(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"MovieLens aggregate not found at {source}. "
            "Run `python -m experiments.movielens.run_3_seeds ...` first."
        )
    return pd.read_csv(source)


def _load_covid_result_csvs(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(results_dir)
    files = {
        "exp1": "exp1_set_size.csv",
        "exp2": "exp2_action_scaling.csv",
        "exp3": "exp3_rare_action.csv",
        "exp4_fdr": "exp4_fdr.csv",
        "exp4_freq": "exp4_action_freq.csv",
    }
    missing = [name for name in files.values() if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"COVID result CSVs missing from {root}: {missing}")
    return {key: pd.read_csv(root / name) for key, name in files.items()}


def reproduce_figures(
    fast: bool = True,
    output_dir: str | Path = "artifacts/figures",
    covid_results_dir: str | Path = "artifacts/results/covid",
    covid_cache_dir: str | Path = "data/cached/covid",
    covid_prob_cache_dir: str | Path | None = None,
    covid_cache_prefix: str = "",
    covid_prob_cache_prefix: str | None = None,
    movielens_aggregate_path: str | Path = "artifacts/results/movielens_3seeds_paper_protocol/movielens_aggregate.csv",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_covid_diagnostics(
        fast=fast,
        output_dir=covid_results_dir,
        cache_dir=covid_cache_dir,
        cache_prefix=covid_cache_prefix,
    )
    covid = _load_covid_result_csvs(covid_results_dir)
    movie = _load_movielens_aggregate(movielens_aggregate_path)
    covid_br_source = covid_prob_cache_dir if covid_prob_cache_dir is not None else covid_cache_dir
    covid_br_prefix = covid_cache_prefix if covid_prob_cache_prefix is None else covid_prob_cache_prefix
    covid_br_utility, covid_br_critical = _covid_best_response(covid_br_source, covid_br_prefix)
    plot_main_figure1_computed(covid["exp1"], movie, out / "figure1_main.pdf", covid_br_critical)
    plot_appendix_figure2(covid["exp1"], out / "figure2_covid_appendix.pdf", best_response_utility=covid_br_utility)
    plot_figure3_from_aggregate(movie, out / "figure3_movielens_appendix.pdf")
    plot_covid_diagnostic_set_size(covid["exp1"], out / "covid_exp1_set_size_vs_alpha.pdf")
    plot_covid_scaling(covid["exp2"], out / "covid_exp2_action_scaling.pdf")
    plot_covid_rare_action(covid["exp3"], out / "covid_exp3_rare_action.pdf")
    plot_covid_fdr_frequency(covid["exp4_fdr"], covid["exp4_freq"], out / "covid_exp4_fdr_action_freq.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use cached rerun inputs where available.")
    parser.add_argument("--output-dir", default="artifacts/figures")
    parser.add_argument("--covid-results-dir", default="artifacts/results/covid")
    parser.add_argument("--covid-cache-dir", default="data/cached/covid")
    parser.add_argument("--covid-prob-cache-dir", default=None)
    parser.add_argument("--covid-cache-prefix", default="")
    parser.add_argument("--covid-prob-cache-prefix", default=None)
    parser.add_argument("--movielens-aggregate-path", default="artifacts/results/movielens_3seeds_paper_protocol/movielens_aggregate.csv")
    args = parser.parse_args()
    reproduce_figures(
        fast=args.fast,
        output_dir=args.output_dir,
        covid_results_dir=args.covid_results_dir,
        covid_cache_dir=args.covid_cache_dir,
        covid_prob_cache_dir=args.covid_prob_cache_dir,
        covid_cache_prefix=args.covid_cache_prefix,
        covid_prob_cache_prefix=args.covid_prob_cache_prefix,
        movielens_aggregate_path=args.movielens_aggregate_path,
    )
