"""Leakage-safe feature engineering for per-series demand forecasting.

Each (store, item) pair is an independent time series: every lag and
rolling feature is computed within its own series, and every history-based
feature is shifted by at least the forecast horizon, so a row can be
predicted `horizon` days in advance without seeing future data.
"""

SERIES_KEYS = ["store", "item"]


def add_calendar_features(df):
    """Deterministic date features — always known in advance."""
    out = df.copy()
    d = out["date"].dt
    out["year"] = d.year
    out["month"] = d.month
    out["day"] = d.day
    out["dayofweek"] = d.dayofweek
    out["dayofyear"] = d.dayofyear
    out["weekofyear"] = d.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_month_start"] = d.is_month_start.astype(int)
    out["is_month_end"] = d.is_month_end.astype(int)
    return out


def add_lag_features(df, lags):
    """Past sales of the same series, shifted by each lag in days."""
    out = df.copy()
    series = out.groupby(SERIES_KEYS, observed=True)["sales"]
    for lag in lags:
        out[f"sales_lag_{lag}"] = series.shift(lag)
    return out


def add_rolling_features(df, windows, shift):
    """Rolling mean/std over the window ending `shift` days before each row."""
    out = df.copy()
    out["_base"] = out.groupby(SERIES_KEYS, observed=True)["sales"].shift(shift)
    base = out.groupby(SERIES_KEYS, observed=True)["_base"]
    for window in windows:
        out[f"sales_rmean_{window}"] = base.transform(
            lambda s, w=window: s.rolling(w).mean()
        )
        out[f"sales_rstd_{window}"] = base.transform(
            lambda s, w=window: s.rolling(w).std()
        )
    return out.drop(columns="_base")


def build_features(
    df,
    lags=(7, 14, 28, 91, 182, 365),
    windows=(7, 14, 28),
    horizon=1,
    dropna=True,
):
    """Build the model-ready feature frame from raw (date, store, item, sales).

    `horizon` is the forecast distance in days: all lags must be >= horizon,
    and rolling windows end `horizon` days before the target date.
    """
    too_short = [lag for lag in lags if lag < horizon]
    if too_short:
        raise ValueError(
            f"Lags {too_short} are shorter than the forecast horizon "
            f"({horizon} days) and would leak future data."
        )
    out = df.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)
    out = add_calendar_features(out)
    out = add_lag_features(out, lags)
    out = add_rolling_features(out, windows, shift=horizon)
    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out
