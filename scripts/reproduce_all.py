from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.reproduce_figures import reproduce_figures  # noqa: E402
from scripts.reproduce_tables import reproduce_tables  # noqa: E402
from experiments.covid.run_diagnostics import run_covid_diagnostics  # noqa: E402
from experiments.movielens.run_3_seeds import run_movielens_seed_sweep  # noqa: E402


FULL_MOVIELENS_ALPHAS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]


def reproduce_all(
    fast: bool = True,
    covid_results_dir: str | Path = "artifacts/results/covid",
    covid_cache_dir: str | Path = "data/cached/covid",
    covid_cache_prefix: str = "",
    movielens_aggregate_path: str | Path = "artifacts/results/movielens_3seeds_paper_protocol/movielens_aggregate.csv",
    movielens_zip_path: str | Path = "ml-100k.zip",
    movielens_seeds: list[int] | None = None,
    movielens_epochs: int = 20,
    movielens_batch_size: int = 256,
    movielens_jobs: int = 1,
    refresh_movielens_seed_cache: bool = False,
) -> None:
    if not fast:
        run_covid_diagnostics(
            fast=False,
            output_dir=covid_results_dir,
            cache_dir=covid_cache_dir,
            cache_prefix=covid_cache_prefix,
            ac_max_iter=400,
            ac_eta0=25.0,
            ac_batch_size=32,
        )
        aggregate_path = Path(movielens_aggregate_path)
        run_movielens_seed_sweep(
            zip_path=movielens_zip_path,
            seeds=movielens_seeds or [0, 1, 2],
            alphas=FULL_MOVIELENS_ALPHAS,
            epochs=movielens_epochs,
            batch_size=movielens_batch_size,
            output_dir=aggregate_path.parent,
            figures_dir="artifacts/figures/movielens_3seeds_paper_protocol",
            selection_tie_break_epsilon=0.0,
            ac_selection_tie_break_epsilon=0.045,
            report_tie_break_epsilon=0.0,
            ac_max_iter=100,
            ac_eta0=2.0,
            ac_batch_size=32,
            jobs=movielens_jobs,
            refresh_seed_cache=refresh_movielens_seed_cache,
            make_figures=False,
        )
    csv_source = covid_results_dir if not fast else covid_cache_dir
    reproduce_tables(fast=True, covid_cache_dir=csv_source, covid_cache_prefix="" if not fast else covid_cache_prefix)
    reproduce_figures(
        fast=True,
        covid_results_dir=covid_results_dir,
        covid_cache_dir=csv_source,
        covid_prob_cache_dir=covid_cache_dir,
        covid_cache_prefix="" if not fast else covid_cache_prefix,
        covid_prob_cache_prefix=covid_cache_prefix,
        movielens_aggregate_path=movielens_aggregate_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--covid-results-dir", default="artifacts/results/covid")
    parser.add_argument("--covid-cache-dir", default="data/cached/covid")
    parser.add_argument("--covid-cache-prefix", default="")
    parser.add_argument("--movielens-aggregate-path", default="artifacts/results/movielens_3seeds_paper_protocol/movielens_aggregate.csv")
    parser.add_argument("--movielens-zip-path", default="ml-100k.zip")
    parser.add_argument("--movielens-seeds", default="0,1,2")
    parser.add_argument("--movielens-epochs", type=int, default=20)
    parser.add_argument("--movielens-batch-size", type=int, default=256)
    parser.add_argument("--movielens-jobs", type=int, default=1)
    parser.add_argument("--refresh-movielens-seed-cache", action="store_true")
    args = parser.parse_args()
    reproduce_all(
        fast=args.fast,
        covid_results_dir=args.covid_results_dir,
        covid_cache_dir=args.covid_cache_dir,
        covid_cache_prefix=args.covid_cache_prefix,
        movielens_aggregate_path=args.movielens_aggregate_path,
        movielens_zip_path=args.movielens_zip_path,
        movielens_seeds=[int(item.strip()) for item in args.movielens_seeds.split(",") if item.strip()],
        movielens_epochs=args.movielens_epochs,
        movielens_batch_size=args.movielens_batch_size,
        movielens_jobs=args.movielens_jobs,
        refresh_movielens_seed_cache=args.refresh_movielens_seed_cache,
    )

