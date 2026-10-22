import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features


def make_raw(days=800, stores=(1,), items=(1,)):
    """Synthetic raw frame; sales = row index within each series."""
    frames = []
    for store in stores:
        for item in items:
            frames.append(
                pd.DataFrame(
                    {
                        "date": pd.date_range("2020-01-01", periods=days),
                        "store": store,
                        "item": item,
                        "sales": np.arange(days),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_lag_matches_past_value():
    df = build_features(make_raw(), lags=(7,), windows=(7,), horizon=1)
    # sales = day index, so a 7-day lag is always sales - 7
    assert (df["sales_lag_7"] == df["sales"] - 7).all()


def test_rolling_window_ends_at_horizon():
    horizon = 91
    df = build_features(make_raw(), lags=(91,), windows=(7,), horizon=horizon)
    # mean of the 7 values ending `horizon` days back: sales - horizon - 3
    expected = df["sales"] - horizon - 3
    assert df["sales_rmean_7"].to_numpy() == pytest.approx(expected.to_numpy())


def test_lag_shorter_than_horizon_raises():
    with pytest.raises(ValueError, match="leak"):
        build_features(make_raw(), lags=(7, 91), windows=(7,), horizon=91)


def test_no_leakage_from_future_sales():
    """Changing sales after date D must not change any feature row
    with date < D + horizon (those rows only see data up to D - 1)."""
    horizon = 28
    raw = make_raw(days=400)
    cutoff = pd.Timestamp("2020-10-01")

    mutated = raw.copy()
    mutated.loc[mutated["date"] >= cutoff, "sales"] = 99999

    kwargs = {"lags": (28, 35), "windows": (7,), "horizon": horizon, "dropna": False}
    original = build_features(raw, **kwargs)
    changed = build_features(mutated, **kwargs)

    safe = original["date"] < cutoff + pd.Timedelta(days=horizon)
    assert safe.sum() > 0
    feature_cols = [c for c in original.columns if c not in ("date", "store", "item", "sales")]
    pd.testing.assert_frame_equal(
        original.loc[safe, feature_cols], changed.loc[safe, feature_cols]
    )


def test_series_do_not_cross_contaminate():
    df = build_features(
        make_raw(days=400, stores=(1, 2)), lags=(7,), windows=(7,), horizon=1, dropna=False
    )
    # First rows of store 2 must have NaN lags, not values from store 1
    store2_head = df[df["store"] == 2].head(7)
    assert store2_head["sales_lag_7"].isna().all()


def test_calendar_features_are_correct():
    df = build_features(make_raw(days=40), lags=(7,), windows=(7,), horizon=1, dropna=False)
    saturday = df[df["date"] == "2020-01-04"].iloc[0]
    assert saturday["dayofweek"] == 5
    assert saturday["is_weekend"] == 1
    first = df[df["date"] == "2020-01-01"].iloc[0]
    assert first["is_month_start"] == 1
