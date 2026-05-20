# Reproduction Status

The public reproduction path has two modes.

## Fast mode

`python scripts/reproduce_all.py --fast` uses uploaded rerun CSVs and cached probability arrays, then regenerates all upload-facing figures and tables under `artifacts/`.

## Standard/full mode

`python scripts/reproduce_all.py` first reruns the COVID and MovieLens pipelines with the paper-protocol hyperparameters, writes fresh CSVs, and then regenerates the same figures and tables from those CSVs.

## Source of truth

- COVID diagnostics: `experiments.covid.run_diagnostics` with `ac_max_iter=400`, `ac_eta0=25.0`, `ac_batch_size=32`.
- MovieLens: `experiments.movielens.run_3_seeds` with seeds `0,1,2`, 20 epochs, 8 alpha values, AC-RAC `ac_max_iter=100`, `ac_eta0=2.0`, `ac_batch_size=32`, and AC selection tie break `0.045`.

The plotting and table scripts consume CSV outputs; they do not embed paper-result constants.
