# End-to-End Demand Forecasting

End-to-end data science project for retail demand forecasting: from raw sales data to engineered features, trained models and evaluated forecasts.

## Dataset

[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only/data) (Kaggle):
**913,000 rows** of daily sales — 10 stores × 50 items × 5 years (2013–2017).

The data is not versioned in this repo. To reproduce, download `train.csv` from Kaggle and place it at `data/raw/train.csv`.

## Project structure

```
├── data/
│   ├── raw/          # original Kaggle data (not versioned)
│   ├── processed/    # feature-engineered datasets (not versioned)
│   └── external/     # auxiliary data sources
├── notebooks/        # numbered analysis notebooks
├── src/
│   ├── data/         # data loading utilities
│   ├── features/     # feature engineering pipeline
│   ├── models/       # model training
│   └── evaluation/   # metrics and backtesting
├── outputs/
│   ├── figures/      # exported charts
│   ├── forecasts/    # model predictions
│   └── metrics/      # evaluation results
└── requirements.txt
```

## Pipeline

| Step | Notebook | Status |
|---|---|---|
| 1. Exploratory data analysis | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | ✅ done |
| 2. Feature engineering | [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | ✅ done |
| 3. Baseline & ML models | `notebooks/03_modeling.ipynb` | planned |
| 4. Evaluation & backtesting | `notebooks/04_evaluation.ipynb` | planned |

## Setup

```bash
pip install -r requirements.txt
```

Then place the Kaggle `train.csv` in `data/raw/` and run the notebooks in order.
