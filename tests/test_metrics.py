import numpy as np
import pytest

from src.evaluation.metrics import evaluate, mae, rmse, smape


def test_perfect_prediction_is_zero():
    y = np.array([1.0, 5.0, 100.0])
    assert smape(y, y) == 0
    assert mae(y, y) == 0
    assert rmse(y, y) == 0


def test_smape_is_symmetric_in_arguments():
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([12.0, 15.0, 33.0])
    assert smape(a, b) == pytest.approx(smape(b, a))


def test_smape_known_value():
    # |100-50| / ((100+50)/2) = 2/3 -> 66.67%
    assert smape([100.0], [50.0]) == pytest.approx(200 / 3)


def test_smape_bounded_between_0_and_200():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 100, 1000).astype(float)
    p = rng.integers(0, 100, 1000).astype(float)
    assert 0 <= smape(y, p) <= 200


def test_smape_handles_both_zero_without_nan():
    # A day with zero actual and zero predicted sales is a perfect forecast.
    assert smape([0.0, 10.0], [0.0, 10.0]) == 0


def test_evaluate_returns_all_metrics():
    row = evaluate("model", [1.0, 2.0], [2.0, 2.0])
    assert row["model"] == "model"
    assert row["MAE"] == pytest.approx(0.5)
    assert set(row) == {"model", "SMAPE", "MAE", "RMSE"}
