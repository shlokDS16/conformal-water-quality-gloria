"""Trivial and refitted empirical baselines. All targets are log10; every fit sees training rows only.

Common interface: model.fit(df, y, groups) -> self; model.predict(df) -> log10 prediction.
`df` holds the sensor feature columns; `groups` are the inner-CV group labels (wb_group) of the rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.models.features import BAND_ROLES, LOG_FLOOR, feature_columns, log_bands


class ConstantMedian:
    name = "const"

    def fit(self, df, y, groups=None):
        self.m_ = float(np.median(y))
        return self

    def predict(self, df):
        return np.full(len(df), self.m_)


class RidgeLogBands:
    """Ridge on log10 bands; StandardScaler fit on the training rows; alpha by inner GroupKFold(5)."""
    name = "ridge"
    ALPHAS = np.logspace(-3, 3, 13)

    def __init__(self, sensor: str):
        self.cols = feature_columns(sensor)

    def _X(self, df):
        return log_bands(df[self.cols].to_numpy(float))

    def fit(self, df, y, groups=None):
        X = self._X(df)
        best, best_a = np.inf, 1.0
        if groups is not None and len(np.unique(groups)) >= 5:
            cv = GroupKFold(n_splits=5)
            for a in self.ALPHAS:
                err = 0.0
                for tr, va in cv.split(X, y, groups):
                    sc = StandardScaler().fit(X[tr])  # inner scaler: inner-train rows only
                    m = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
                    err += np.sum((m.predict(sc.transform(X[va])) - y[va]) ** 2)
                if err < best:
                    best, best_a = err, a
        self.alpha_ = best_a
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = Ridge(alpha=best_a).fit(self.scaler_.transform(X), y)
        return self

    def predict(self, df):
        return self.model_.predict(self.scaler_.transform(self._X(df)))


class _FallbackMixin:
    """Rows whose band inputs are invalid (non-positive) get the training median; count kept."""

    def _finish(self, pred, valid):
        out = np.where(valid, pred, self.median_)
        self.n_fallback_ = int((~valid).sum())
        return out


class OCMaxBandRatio(_FallbackMixin):
    """OC4-type (OC3-type for MSI): log10 Chl = sum_k a_k log10(MBR)^k, degree 4, refitted by OLS."""
    name = "emp_oc"
    DEGREE = 4

    def __init__(self, sensor):
        r = BAND_ROLES[sensor]
        self.blues = [r[k] for k in ("b443", "b490", "b510") if r[k]]
        self.green = r["g560"]

    def _x(self, df):
        B = df[self.blues].to_numpy(float)
        G = df[self.green].to_numpy(float)
        valid = (B > 0).all(1) & (G > 0)
        x = np.log10(np.maximum(B.max(1), LOG_FLOOR) / np.maximum(G, LOG_FLOOR))
        return x, valid

    def fit(self, df, y, groups=None):
        x, v = self._x(df)
        self.median_ = float(np.median(y))
        self.coef_ = np.polyfit(x[v], y[v], self.DEGREE)
        return self

    def predict(self, df):
        x, v = self._x(df)
        return self._finish(np.polyval(self.coef_, x), v)


class NDCIQuadratic(_FallbackMixin):
    """Red/NIR inland ratio (Mishra and Mishra 2012): log10 Chl = a + b NDCI + c NDCI^2, refitted."""
    name = "emp_ndci"

    def __init__(self, sensor):
        r = BAND_ROLES[sensor]
        self.re, self.red = r["re705"], r["r665"]

    def _x(self, df):
        re, red = df[self.re].to_numpy(float), df[self.red].to_numpy(float)
        valid = (re + red) > 0
        x = (re - red) / np.where(valid, re + red, 1.0)
        return x, valid

    def fit(self, df, y, groups=None):
        x, v = self._x(df)
        self.median_ = float(np.median(y))
        self.coef_ = np.polyfit(x[v], y[v], 2)
        return self

    def predict(self, df):
        x, v = self._x(df)
        return self._finish(np.polyval(self.coef_, x), v)


class NechadTSS(_FallbackMixin):
    """Nechad et al. (2010) form TSS = A rho / (1 - rho / C), rho = pi Rrs(665); A, C refitted in log10 space."""
    name = "emp_nechad"

    def __init__(self, sensor):
        self.band = BAND_ROLES[sensor]["r665"]

    def _rho(self, df):
        rho = np.pi * df[self.band].to_numpy(float)
        return rho, rho > 0

    @staticmethod
    def _f(theta, rho):
        A, C = np.exp(theta)
        return np.log10(A * rho / np.maximum(1.0 - rho / C, 1e-6))

    def fit(self, df, y, groups=None):
        rho, v = self._rho(df)
        self.median_ = float(np.median(y))
        r, t = rho[v], y[v]
        A0 = np.median(10 ** t / r)
        C0 = max(0.2, 2.0 * r.max())
        res = least_squares(lambda th: self._f(th, r) - t, x0=np.log([A0, C0]),
                            bounds=([-np.inf, np.log(1.01 * r.max())], [np.inf, np.inf]))
        self.theta_ = res.x
        return self

    def predict(self, df):
        rho, v = self._rho(df)
        A, C = np.exp(self.theta_)
        v = v & (rho < C)
        return self._finish(self._f(self.theta_, np.where(v, rho, 1e-3)), v)


class LogRatioRegression(_FallbackMixin):
    """log10 y = a + sum_j b_j log10(feature_j); features are band ratios or bands (OLS)."""

    def __init__(self, sensor, terms, name):
        r = BAND_ROLES[sensor]
        self.terms = [(r[a], r[b] if b else None) for a, b in terms]
        self.name = name

    def _X(self, df):
        cols, valid = [], np.ones(len(df), bool)
        for a, b in self.terms:
            va = df[a].to_numpy(float)
            if b is None:
                valid &= va > 0
                cols.append(np.log10(np.maximum(va, LOG_FLOOR)))
            else:
                vb = df[b].to_numpy(float)
                valid &= (va > 0) & (vb > 0)
                cols.append(np.log10(np.maximum(va, LOG_FLOOR) / np.maximum(vb, LOG_FLOOR)))
        X = np.column_stack([np.ones(len(df))] + cols)
        return X, valid

    def fit(self, df, y, groups=None):
        X, v = self._X(df)
        self.median_ = float(np.median(y))
        self.coef_, *_ = np.linalg.lstsq(X[v], y[v], rcond=None)
        return self

    def predict(self, df):
        X, v = self._X(df)
        return self._finish(X @ self.coef_, v)


def empirical_models(target: str, sensor: str) -> list:
    """Empirical baselines per target (prereg s3; dossier 5.3)."""
    if target == "Chla":
        if BAND_ROLES[sensor]["re705"] is None:  # strict band sets: no red-edge band, NDCI not defined
            return [OCMaxBandRatio(sensor)]
        return [OCMaxBandRatio(sensor), NDCIQuadratic(sensor)]
    if target == "TSS":
        return [NechadTSS(sensor)]
    if target == "aCDOM440":  # green/red ratio (Brezonik et al. 2015 family)
        return [LogRatioRegression(sensor, [("g560", "r665")], "emp_ratio")]
    if target == "Secchi_depth":  # blue/green ratio plus red band
        return [LogRatioRegression(sensor, [("b490", "g560"), ("r665", None)], "emp_ratio")]
    raise ValueError(target)
