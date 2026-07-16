"""LightGBM training helpers."""

import lightgbm as lgb

DEFAULT_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


def train_lgbm(
    train_x, train_y, eval_x, eval_y, params=None, eval_metric="l1", early_stopping_rounds=100
):
    """Fit a LightGBM regressor with early stopping on the eval set."""
    model = lgb.LGBMRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(
        train_x,
        train_y,
        eval_set=[(eval_x, eval_y)],
        eval_metric=eval_metric,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return model
