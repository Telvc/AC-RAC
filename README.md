# Conformal Risk-Averse Decision Making with Action-Conditional Guarantees

This repository contains the code and lightweight cached inputs needed to
reproduce the empirical tables and figures for the ICML 2026 paper
`Conformal Risk-Averse Decision Making with Action-Conditional Guarantees`.

## Quick Start

Install dependencies, then run:

```bash
python scripts/reproduce_all.py --fast
```

This fast mode uses uploaded rerun CSVs and cached probability arrays, then
redraws the paper tables and figures under `artifacts/`.

The repository includes the required lightweight inputs:

- COVID probability caches and paper-protocol diagnostic CSVs in
  `data/cached/covid/`.
- MovieLens 100K archive as `ml-100k.zip`.
- MovieLens 3-seed paper-protocol aggregate CSVs in
  `artifacts/results/movielens_3seeds_paper_protocol/`.

## Reproduction Modes

There is one public orchestration entrypoint:

```bash
python scripts/reproduce_all.py --fast
```

Fast mode loads the uploaded rerun CSVs, copies the COVID diagnostics into
`artifacts/results/covid/`, and redraws:

- `artifacts/figures/figure1_main.pdf`
- `artifacts/figures/figure2_covid_appendix.pdf`
- `artifacts/figures/figure3_movielens_appendix.pdf`
- CSV/Markdown/LaTeX tables in `artifacts/tables/`

Full mode first reruns the experiment pipelines and then draws the plots from
the newly generated CSVs:

```bash
python scripts/reproduce_all.py
```

Full mode is substantially slower because MovieLens evaluates AC-RAC over
three seeds and eight alpha values. Use `--movielens-jobs` to parallelize
alpha evaluations per seed if your machine has enough memory.

## Core Code

The reusable implementation lives in `src/rac/`.

- `src/rac/acrac.py`: AC-RAC candidate-label procedure.
- `src/rac/conformal.py`: conformal prediction sets, RAC, Score-1, Score-2,
  and max-min action selection.
- `src/rac/metrics.py`: coverage, action-conditional coverage, utilities,
  FDR, set sizes, action frequencies, and critical-decision rates.
- `src/rac/plotting.py`: shared plotting functions.

## COVID Experiment

The COVID experiment source of truth is:

```bash
python -m experiments.covid.run_diagnostics ^
  --cache-dir data/cached/covid ^
  --output-dir artifacts/results/covid ^
  --ac-max-iter 400 ^
  --ac-eta0 25.0 ^
  --ac-batch-size 32
```

The paper-protocol COVID hyperparameters are:

- `--ac-max-iter 400`
- `--ac-eta0 25.0`
- `--ac-batch-size 32`

The uploaded fast-path CSVs in `data/cached/covid/` were generated with this
configuration. They include the aggregate bad-decision-rate columns used in
Figure 1:

- `pneumonia_no_action_rate`
- `covid_no_action_rate`
- `lung_opacity_no_action_rate`

These rates are aggregate outputs. Per-test selected actions are not stored in
the CSVs; rerun `experiments.covid.run_diagnostics` if you need a different
critical-decision definition.

To regenerate COVID probability caches from raw images, use:

```bash
python -m experiments.covid.cache_predictions ^
  --dataset-dir path/to/COVID-19_Radiography_Dataset ^
  --checkpoint my_inception_model.pth ^
  --output-dir artifacts/cached_predictions ^
  --seed 42
```

The raw Kaggle image dataset and Inception checkpoint are not required for the
uploaded fast reproduction.

## MovieLens Experiment

The MovieLens experiment source of truth is:

```bash
python -m experiments.movielens.run_3_seeds ^
  --zip-path ml-100k.zip ^
  --seeds 0,1,2 ^
  --epochs 20 ^
  --batch-size 256 ^
  --alphas 0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.25 ^
  --selection-tie-break-epsilon 0.0 ^
  --ac-selection-tie-break-epsilon 0.045 ^
  --report-tie-break-epsilon 0.0 ^
  --ac-max-iter 100 ^
  --ac-eta0 2.0 ^
  --ac-batch-size 32
```

The paper-protocol MovieLens hyperparameters are:

- seeds `0,1,2`
- `--epochs 20`
- `--batch-size 256`
- `--selection-tie-break-epsilon 0.0`
- `--ac-selection-tie-break-epsilon 0.045`
- `--report-tie-break-epsilon 0.0`
- `--ac-max-iter 100`
- `--ac-eta0 2.0`
- `--ac-batch-size 32`

This writes:

- `artifacts/results/movielens_3seeds_paper_protocol/movielens_per_seed.csv`
- `artifacts/results/movielens_3seeds_paper_protocol/movielens_aggregate.csv`

The Figure 1 MovieLens row and Figure 3 are drawn from the aggregate CSV.

For a faster smoke check:

```bash
python -m experiments.movielens.run_3_seeds ^
  --zip-path ml-100k.zip ^
  --seeds 0 ^
  --epochs 20 ^
  --alphas 0.05 ^
  --selection-tie-break-epsilon 0.0 ^
  --ac-selection-tie-break-epsilon 0.045 ^
  --report-tie-break-epsilon 0.0 ^
  --ac-max-iter 100 ^
  --ac-eta0 2.0 ^
  --ac-batch-size 32 ^
  --eval-calib-size 1000 ^
  --eval-test-size 300
```

## Plot and Table Generation

To redraw only figures:

```bash
python scripts/reproduce_figures.py --fast
```

To redraw only tables:

```bash
python scripts/reproduce_tables.py --fast
```

The plotting scripts consume CSV outputs; they do not contain embedded paper
numbers. If a required CSV is missing, the scripts fail with an explicit error.

## Artifact Map

Upload-facing artifacts are intentionally limited to:

```text
data/cached/covid/
  cal_probs.npy
  cal_labels.npy
  test_probs.npy
  test_labels.npy
  exp1_set_size.csv
  exp2_action_scaling.csv
  exp3_rare_action.csv
  exp4_fdr.csv
  exp4_action_freq.csv

artifacts/results/covid/
  exp1_set_size.csv
  exp2_action_scaling.csv
  exp3_rare_action.csv
  exp4_fdr.csv
  exp4_action_freq.csv

artifacts/results/movielens_3seeds_paper_protocol/
  movielens_per_seed.csv
  movielens_aggregate.csv

artifacts/figures/
  figure1_main.pdf
  figure2_covid_appendix.pdf
  figure3_movielens_appendix.pdf
  covid_exp1_set_size_vs_alpha.pdf
  covid_exp2_action_scaling.pdf
  covid_exp3_rare_action.pdf
  covid_exp4_fdr_action_freq.pdf

artifacts/tables/
  table*.csv
  table*.md
  table*.tex
```

Other local debug outputs, raw datasets, notebooks, checkpoints, and validation
archives are excluded from the public repository.
