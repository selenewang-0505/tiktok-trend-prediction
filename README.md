# README

## Overview
This repository contains two main scripts for building and exploring a TikTok engagement/virality prediction pipeline:

- `tiktokpredict.py` — preprocesses data, trains or loads a model, runs predictions and saves results.
- `final_dashboard.py` — interactive dashboard that visualizes data, model metrics and predictions.

## Requirements
- Python 3.8+
- Common packages (install via `pip install -r requirements.txt`). Typical dependencies:
    - pandas, numpy
    - scikit-learn (or your chosen ML library)
    - joblib (or pickle) for model serialization
    - matplotlib / seaborn or plotly for visualizations
    - streamlit (if dashboard is implemented with Streamlit) or dash/Flask if a different framework is used

Add any project-specific packages to `requirements.txt`.

## Installation
1. Clone the repo or place files in your working directory.
2. Create and activate a virtual environment:
     - python -m venv venv
     - source venv/bin/activate (macOS / Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies:
     - pip install -r requirements.txt

## tiktokpredict.py — what it does
- Loads raw TikTok dataset (CSV/JSON) containing features such as captions, hashtags, video metadata, and engagement labels.
- Runs preprocessing (cleaning, feature engineering, encoding).
- Trains a model or loads a pre-trained model (configurable).
- Produces predictions and evaluation metrics (train/validation scores, confusion matrix, feature importances).
- Saves artifacts: model file (e.g., `model.pkl`), prediction CSV (e.g., `predictions.csv`) and a brief metrics JSON/log.

How to run:
- Example (train + predict):
    - python tiktokpredict.py --input data/tiktok_raw.csv --output results/predictions.csv --model results/model.pkl --train
- Example (use existing model to predict):
    - python tiktokpredict.py --input data/new_videos.csv --model results/model.pkl --output results/predictions.csv

Common flags (adapt per implementation):
- `--input` : path to input dataset
- `--output` : path to save predictions
- `--model` : model file to save/load
- `--train` : flag to train a new model
- `--config` : optional config file for hyperparameters

Outputs produced:
- predictions CSV with original rows plus prediction columns (probabilities, predicted label)
- serialized model file (joblib/pickle)
- metrics/log file (JSON or text)
- optional figures (ROC, feature importance)

## final_dashboard.py — what it does
- Loads the prediction output and/or raw dataset and model metrics.
- Provides interactive visualizations: timeline of engagement, distribution of predicted scores, confusion matrix, feature importance, filter by hashtag/author/date.
- Meant to be run locally for analysis and presentation.

How to run:
- Streamlit example:
    - streamlit run final_dashboard.py -- --data results/predictions.csv
- Python script example (if self-hosted):
    - python final_dashboard.py --data results/predictions.csv
- Open the local address printed by the framework (e.g., http://localhost:8501 for Streamlit).

Inputs required:
- Predictions CSV (output from `tiktokpredict.py`)
- Optional model metrics or artifacts folder

Dashboard outputs:
- Interactive plots and tables
- Export options (CSV or image) depending on implementation
- Summary report / dashboard snapshot export

## Typical workflow
1. Prepare raw data: data/tiktok_raw.csv
2. Train and predict:
     - python tiktokpredict.py --input data/tiktok_raw.csv --output results/predictions.csv --model results/model.pkl --train
3. Explore results:
     - streamlit run final_dashboard.py -- --data results/predictions.csv

## Notes
- Adjust CLI flags and dependency list if the scripts use different libraries or arguments.
- Add a `requirements.txt` listing exact package versions for reproducibility.
- Include sample input data and a small model artifact for quick demos.
