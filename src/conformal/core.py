"""Interval methods. All arrays are in log10 target space; intervals are [lower, upper].

Finite-sample conformal quantile: with n calibration scores, k = ceil((1 - alpha)(n + 1));
if k > n the interval is infinite (lower = -inf, upper = +inf) and the flag `inf` is True.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

_TOL = 1e-9


def conformal_rank(n: int, alpha: float) -> int:
    """k = ceil((1 - alpha)(n + 1)), robust to float round-off (e.g. 0.9 * 10)."""
    return int(math.ceil((1.0 - alpha) * (n + 1) - _TOL))


def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, bool]:
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = conformal_rank(n, alpha)
    if n == 0 or k > n:
        return math.inf, True
    return float(s[k - 1]), False


def group_subsample(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of one uniformly drawn row per group (Dunn et al. 2022, single subsampling).

    Groups are visited in sorted label order, so the draw depends only on the rng state and the labels.
    """
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    ends = np.r_[starts[1:], len(g_sorted)]
    u = rng.random(len(starts))
    pick = starts + np.floor(u * (ends - starts)).astype(int)
    return order[pick]


def _interval(center_lo, center_hi, q):
    return np.asarray(center_lo, float) - q, np.asarray(center_hi, float) + q


def split_conformal(pred_cal, y_cal, pred_test, alpha, idx=None):
    """Absolute-residual split conformal. idx selects calibration rows (e.g. group subsample)."""
    s = np.abs(np.asarray(y_cal) - np.asarray(pred_cal))
    if idx is not None:
        s = s[idx]
    q, inf = conformal_quantile(s, alpha)
    lo, hi = _interval(pred_test, pred_test, q)
    return lo, hi, {"qhat": q, "inf": inf, "n_cal_units": len(s)}


def normalized_split_conformal(pred_cal, sigma_cal, y_cal, pred_test, sigma_test, alpha, idx=None):
    """Score |y - f(x)| / sigma(x) (Lei et al. 2018, locally weighted)."""
    s = np.abs(np.asarray(y_cal) - np.asarray(pred_cal)) / np.asarray(sigma_cal)
    if idx is not None:
        s = s[idx]
    q, inf = conformal_quantile(s, alpha)
    st = np.asarray(sigma_test, float)
    pt = np.asarray(pred_test, float)
    return pt - q * st, pt + q * st, {"qhat": q, "inf": inf, "n_cal_units": len(s)}


def cqr(qlo_cal, qhi_cal, y_cal, qlo_test, qhi_test, alpha, idx=None):
    """Conformalised quantile regression (Romano et al. 2019): score max(q_lo - y, y - q_hi)."""
    y = np.asarray(y_cal)
    s = np.maximum(np.asarray(qlo_cal) - y, y - np.asarray(qhi_cal))
    if idx is not None:
        s = s[idx]
    q, inf = conformal_quantile(s, alpha)
    lo, hi = _interval(qlo_test, qhi_test, q)
    return lo, hi, {"qhat": q, "inf": inf, "n_cal_units": len(s)}


def gaussian_residual(oof_pred_train, y_train, pred_test, alpha):
    """Homoscedastic Gaussian interval point +- z sigma (prior practice).

    Amendment 2 item 3: sigma = SD of 5-fold group-CV out-of-fold residuals within the training groups,
    so oof_pred_train must be out-of-fold predictions, never in-sample fits.
    """
    sd = float(np.std(np.asarray(y_train) - np.asarray(oof_pred_train), ddof=1))
    z = norm.ppf(1 - alpha / 2)
    pt = np.asarray(pred_test, float)
    return pt - z * sd, pt + z * sd, {"qhat": z * sd, "inf": False, "n_cal_units": len(y_train)}


def quantile_recalibration(mean_fit, sd_fit, y_fit, mean_test, sd_test, alpha):
    """Werther et al. (2025) style: uncertainty_toolbox isotonic quantile recalibrator on Gaussian (mean, SD)."""
    import uncertainty_toolbox as uct

    recal = uct.recalibration.get_quantile_recalibrator(np.asarray(mean_fit, float), np.asarray(sd_fit, float),
                                                        np.asarray(y_fit, float))
    m, s = np.asarray(mean_test, float), np.asarray(sd_test, float)
    lo = recal(m, s, alpha / 2)
    hi = recal(m, s, 1 - alpha / 2)
    return np.asarray(lo, float), np.asarray(hi, float), {"qhat": np.nan, "inf": False, "n_cal_units": len(y_fit)}


def cvplus_bound(n: int, K: int, alpha: float) -> float:
    """Barber et al. (2021) Theorem 4 worst-case bound for CV+ with n exchangeable units and K folds:
    1 - 2 alpha - min{2 (1 - 1/K) / (n/K + 1), (1 - K/n) / (K + 1)}."""
    return 1.0 - 2.0 * alpha - min(2.0 * (1.0 - 1.0 / K) / (n / K + 1.0), (1.0 - K / n) / (K + 1.0))


def cv_plus(mu_oof, y_fit, fold_of_row, mu_fold_test, alpha, idx=None):
    """CV+ (Barber et al. 2021). mu_oof[i] = prediction for row i from the model that did not see its fold;
    mu_fold_test[k, j] = fold-k model prediction for test point j.
    Group CV+ (Amendment 2 item 1): idx = one uniformly drawn row per held-out water body, so the
    n residuals are one per unit and the Theorem 4 bound (cvplus_bound) applies with n = units.
    """
    R = np.abs(np.asarray(y_fit) - np.asarray(mu_oof))
    fold = np.asarray(fold_of_row)
    if idx is not None:
        R, fold = R[idx], fold[idx]
    n = len(R)
    k_hi = conformal_rank(n, alpha)
    k_lo = int(math.floor(alpha * (n + 1) + _TOL))
    M = mu_fold_test[fold]  # (n, m)
    m = M.shape[1]
    if k_hi > n or k_lo < 1:
        return np.full(m, -np.inf), np.full(m, np.inf), {"qhat": math.inf, "inf": True, "n_cal_units": n}
    lower_set = M - R[:, None]
    upper_set = M + R[:, None]
    lo = np.partition(lower_set, k_lo - 1, axis=0)[k_lo - 1]
    hi = np.partition(upper_set, k_hi - 1, axis=0)[k_hi - 1]
    return lo, hi, {"qhat": np.nan, "inf": False, "n_cal_units": n}


# ------------------------------------------------------------------ optional (exploratory) interfaces only
def mondrian_split_conformal(pred_cal, y_cal, class_cal, pred_test, class_test, alpha, min_groups_per_class=None):
    """Mondrian split conformal on a-priori spectral classes (prereg s3 optional). Not implemented."""
    raise NotImplementedError("Optional after the 2026-09-19 go/no-go; interface only.")


def weighted_split_conformal(pred_cal, y_cal, w_cal, pred_test, w_test, alpha):
    """Weighted conformal (Tibshirani et al. 2019) heuristic with ESS diagnostics. Not implemented."""
    raise NotImplementedError("Optional after the 2026-09-19 go/no-go; interface only.")
