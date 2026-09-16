"""LightGBM point (L2) and quantile models with fixed a-priori hyperparameters.

No data-dependent tuning: the same fixed setting is used for the point model, the quantile models
and every CV+ fold model, so CV+ uses a symmetric algorithm and no tuning information crosses roles.
Inputs: sensor bands (raw Rrs; trees are invariant to monotone transforms) plus the four band-ratio
features of the empirical baselines (src.models.features.ratio_features).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.features import feature_columns, ratio_features

LGBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    deterministic=True, force_col_wise=True, verbose=-1,
)


def lgbm_matrix(df: pd.DataFrame, sensor: str) -> np.ndarray:
    X = df[feature_columns(sensor)].to_numpy(float)
    return np.column_stack([X, ratio_features(df, sensor).to_numpy(float)])


class LGBMPoint:
    name = "lgbm"

    def __init__(self, sensor, seed, n_jobs=4):
        self.sensor, self.seed, self.n_jobs = sensor, seed, n_jobs

    def fit(self, df, y, groups=None):
        self.model_ = lgb.LGBMRegressor(objective="regression", random_state=self.seed,
                                        n_jobs=self.n_jobs, **LGBM_PARAMS)
        self.model_.fit(lgbm_matrix(df, self.sensor), y)
        return self

    def predict(self, df):
        return self.model_.predict(lgbm_matrix(df, self.sensor))

    def predict_matrix(self, X):
        return self.model_.predict(X)


class LGBMQuantile:
    def __init__(self, sensor, seed, q, n_jobs=4):
        self.sensor, self.seed, self.q, self.n_jobs = sensor, seed, q, n_jobs

    def fit(self, df, y, groups=None):
        self.model_ = lgb.LGBMRegressor(objective="quantile", alpha=self.q, random_state=self.seed,
                                        n_jobs=self.n_jobs, **LGBM_PARAMS)
        self.model_.fit(lgbm_matrix(df, self.sensor), y)
        return self

    def predict(self, df):
        return self.model_.predict(lgbm_matrix(df, self.sensor))
