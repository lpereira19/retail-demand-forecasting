"""Data loading utilities."""

import pandas as pd


def load_raw(path):
    """Load the raw Kaggle sales CSV, parse dates and sort by series and date."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)
