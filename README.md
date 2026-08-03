# Chooser Option Pricing Project

## Project overview

This project builds an automated market-data and feature-engineering pipeline
for a chooser option pricing study.

The current workflow:

```text
Yahoo Finance and FRED
        ↓
download_data.py
        ↓
data/week1_initial_raw_dataset.csv
        ↓
preprocessing.ipynb
        ↓
output/week2_feature_dataset_pipeline.csv
```

## Data included

- JPM stock price and trading volume
- VIX
- 10-year U.S. Treasury rate

## Week 2 features

- Daily return
- Log return
- 20-day annualized rolling volatility
- VIX change
- 20-day VIX/JPM rolling correlation
- Interest-rate change
- Trading-volume change

## Project structure

```text
Chooser-Option-Pricing/
├── .github/
│   └── workflows/
│       └── pipeline.yml
├── data/
│   └── .gitkeep
├── output/
│   └── .gitkeep
├── download_data.py
├── preprocessing.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The CSV files are generated automatically after the first successful workflow run.

## Run locally or in Colab

Install packages:

```bash
pip install -r requirements.txt
```

Download the latest Week 1 data:

```bash
python download_data.py
```

Then open and run:

```text
preprocessing.ipynb
```

## Run with GitHub Actions

1. Open the repository's **Actions** tab.
2. Select **Daily Market Data Pipeline**.
3. Click **Run workflow**.
4. After a successful run, check:
   - `data/week1_initial_raw_dataset.csv`
   - `output/week2_feature_dataset_pipeline.csv`

The workflow is also scheduled for weekdays at 8:00 PM New York time.

## Author

Guanyu Hu

