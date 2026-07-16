"""Evaluation metrics for demand forecasts."""

import numpy as np


def smape(y_true, y_pred):
    """Symmetric mean absolute percentage error, in percent (0-200).

    Pairs where both actual and predicted are zero contribute zero error
    instead of dividing by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mae(y_true, y_pred):
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(name, y_true, y_pred):
    """All metrics for one model as a dict (one row of a results table)."""
    return {
        "model": name,
        "SMAPE": smape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }
