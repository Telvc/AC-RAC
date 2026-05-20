"""COVID data preparation notes.

The fast reproduction path uses cached probability arrays in ``data/cached/covid``.
For a full rerun, download the COVID-19 Radiography Database from Kaggle:

    kaggle datasets download -d tawsifurrahman/covid19-radiography-database

Then extract it and pass the dataset root to ``cache_predictions.py``.
"""
