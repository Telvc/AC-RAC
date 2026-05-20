# Upload Manifest

This folder is prepared for a GitHub upload. It is not currently initialized as a Git repository on this machine, so use `git init`/`git add` with `.gitignore`, or upload only the files listed below.

## Include

- `README.md`
- `REPRODUCTION_STATUS.md`
- `requirements.txt`
- `.gitignore` if using GitHub/Git to protect local-only files
- `ml-100k.zip`
- `src/rac/*.py`
- `experiments/__init__.py`
- `experiments/covid/__init__.py`
- `experiments/covid/cache_predictions.py`
- `experiments/covid/prepare_data.py`
- `experiments/covid/run_diagnostics.py`
- `experiments/movielens/__init__.py`
- `experiments/movielens/prepare_data.py`
- `experiments/movielens/train_model.py`
- `experiments/movielens/run_3_seeds.py`
- `scripts/__init__.py`
- `scripts/reproduce_all.py`
- `scripts/reproduce_figures.py`
- `scripts/reproduce_tables.py`
- `tests/*.py`
- `data/cached/covid/*.npy`
- `data/cached/covid/*.csv`
- `artifacts/results/covid/*.csv`
- `artifacts/results/movielens_3seeds_paper_protocol/*.csv`
- `artifacts/figures/*.pdf`
- `artifacts/tables/table*.*`

## Exclude

- `.venv/`
- `.local_archive/`
- `.local_notebooks/`
- `.pytest_cache/`
- `.claude/`
- `COVID-19_Radiography_Dataset/`
- `my_inception_model.pth`
- extracted `ml-100k/`
- root-level PDFs and notebooks
- `Iso-lora/`
- any `__pycache__/` or `*.pyc`
