# TikTok Trend Prediction

Time-series forecasting of next week's trending hashtags and music.
<img width="1135" height="669" alt="截屏2025-12-18 15 59 01" src="https://github.com/user-attachments/assets/1cf285ae-bef8-4b83-98b9-3193dd4b201f" />


## Overview

Predicts which TikTok hashtags and music will trend next week using weekly aggregation and growth-based machine learning models.

## Files

- **`tiktokpredict.py`** - Main prediction model
  - Aggregates 10K videos by week
  - Engineers time-series features (lags, growth, acceleration)
  - Trains Random Forest, Gradient Boosting, Logistic Regression
  - Outputs top 15 predictions for next week

- **`final_dashboard.py`** - HTML dashboard generator
  - Runs predictions for each week
  - Generates interactive single-file HTML
  - Displays trend probabilities and growth metrics

- **`data/tiktok_data_10000.csv`** - Dataset (24MB, 14 weeks)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run prediction
python tiktokpredict.py

# Generate dashboard
python final_dashboard.py
```

Open `tiktok_trend_dashboard.html` in browser.

## Key Features

- Filters generic hashtags (#fyp, #viral, #foryou)
- Filters placeholder music titles ("Original Sound")
- Uses growth momentum over absolute volume
- 15 time-series features per item
- Temporal validation (train on earlier weeks, test on later)

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
plotly>=5.0.0
datasets>=2.0.0
```

## Project Structure

```
tiktok-trend-prediction/
├── data/
│   └── tiktok_data_10000.csv
├── tiktokpredict.py
├── final_dashboard.py
├── tiktok_trend_dashboard.html
├── requirements.txt
└── README.md
```

## Course

IEOR 242A - Applications in Data Analysis, UC Berkeley, Fall 2025
