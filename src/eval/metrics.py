"""Evaluation metrics and inference helpers (prereg s4-s5).

Aggregation rule (written once; research/EXPERIMENT_PLAN.md s5):
  per split: water-body-averaged coverage = mean over test water bodies of within-body coverage;
  across repeats: mean and Nadeau-Bengio corrected CI/test; within a repeat: cluster bootstrap over
  test water bodies, descriptive only.
All interval inputs are log10; widths are multiplicative, 10^(upper - lower).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

LN10 = math.log(10.0)


# ------------------------------------------------------------------ point accuracy (log10 inputs)
def point_metrics(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = LN10 * (yhat - y)  # ln(yhat / y)
    med = np.median(e)
    return {
        "mdsa": 100.0 * (math.exp(np.median(np.abs(e))) - 1.0),
        "sspb": 100.0 * np.sign(med) * (math.exp(abs(med)) - 1.0),
        "logmae": float(np.mean(np.abs(yhat - y))),   # log10 units (Seegers et al. 2018)
        "logbias": float(np.mean(yhat - y)),
        "logrmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
    }


# ------------------------------------------------------------------ intervals
def winkler_score(y, lo, hi, alpha) -> np.ndarray:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    with np.errstate(invalid="ignore"):
        s = (hi - lo) + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    s = np.where(np.isinf(lo) | np.isinf(hi), np.inf, s)
    return s


def _group_codes(groups):
    codes, uniq = pd.factorize(np.asarray(groups), sort=True)
    return codes, len(uniq)


def _group_mean(values, codes, G):
    s = np.bincount(codes, weights=values, minlength=G)
    c = np.bincount(codes, minlength=G)
    return s / c, c


def interval_metrics(y, lo, hi, groups, alpha, min_group_n: int = 10) -> dict:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    inf_row = np.isinf(lo) | np.isinf(hi)
    cover = ((y >= lo) & (y <= hi)).astype(float)
    codes, G = _group_codes(groups)
    cov_g, n_g = _group_mean(cover, codes, G)
    with np.errstate(over="ignore"):
        width = np.where(inf_row, np.inf, 10.0 ** (hi - lo))
    med_w = np.array([np.median(width[codes == g]) for g in range(G)])
    wink = winkler_score(y, lo, hi, alpha)
    wink_g = np.array([np.mean(wink[codes == g]) for g in range(G)])
    big = n_g >= min_group_n
    with np.errstate(divide="ignore", invalid="ignore"):
        logw = np.log10(med_w)
    return {
        "cov_wb": float(cov_g.mean()),
        "cov_pooled": float(cover.mean()),
        "cov_worst_ge10": float(cov_g[big].min()) if big.any() else np.nan,
        "n_wb_ge10": int(big.sum()),
        "width_wb": float(med_w.mean()),
        "width_wb_log10mean": float(logw.mean()),
        "width_pooled_median": float(np.median(width)),
        "winkler_wb": float(wink_g.mean()),
        "winkler_pooled": float(wink.mean()),
        "inf_rate": float(inf_row.mean()),
    }


def per_group_coverage(y, lo, hi, groups) -> pd.DataFrame:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    cover = ((y >= lo) & (y <= hi)).astype(float)
    df = pd.DataFrame({"g": np.asarray(groups), "c": cover})
    return df.groupby("g", sort=True)["c"].agg(["mean", "size"]).rename(columns={"mean": "coverage", "size": "n"})


# ------------------------------------------------------------------ inference
def nadeau_bengio_ttest(values, mu0: float, n_test: float, n_train: float, alternative: str = "greater",
                        level: float = 0.95) -> dict:
    """Corrected resampled t-test (Nadeau and Bengio 2003) over J repeats.

    var_corr = (1/J + n_test/n_train) * s^2, df = J - 1. Amendment 2 item 6: n_test = test water bodies,
    n_train = training plus calibration water bodies (realised per split, averaged over repeats).
    alternative 'greater': H1 mean > mu0; 'less': H1 mean < mu0.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    J = len(v)
    if J < 2:
        return {"J": J, "mean": float(v.mean()) if J else np.nan, "t": np.nan, "p": np.nan,
                "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    m, s2 = v.mean(), v.var(ddof=1)
    se = math.sqrt((1.0 / J + n_test / n_train) * s2)
    t = (m - mu0) / se if se > 0 else (math.inf if m != mu0 else 0.0) * (1 if m >= mu0 else -1)
    if alternative == "greater":
        p = float(stats.t.sf(t, J - 1))
    elif alternative == "less":
        p = float(stats.t.cdf(t, J - 1))
    else:
        p = float(2 * stats.t.sf(abs(t), J - 1))
    h = stats.t.ppf(0.5 + level / 2, J - 1) * se
    return {"J": J, "mean": float(m), "t": float(t), "p": p, "se": se, "ci_lo": float(m - h), "ci_hi": float(m + h)}


def nadeau_bengio_diff_test(values_a, values_b, n_test_a, n_train_a, n_test_b, n_train_b, delta: float = 0.0,
                            alternative: str = "greater", level: float = 0.95) -> dict:
    """Unpaired Welch-type comparison of two protocols' corrected means (Amendment 2 item 6, H4).

    d = mean_a - mean_b; var = var_NB(a) + var_NB(b) with var_NB = (1/J + n_test/n_train) s^2;
    conservative df = min(J_a, J_b) - 1. H1 ('greater'): d > delta.
    """
    a = np.asarray(values_a, float); a = a[np.isfinite(a)]
    b = np.asarray(values_b, float); b = b[np.isfinite(b)]
    Ja, Jb = len(a), len(b)
    va = (1.0 / Ja + n_test_a / n_train_a) * a.var(ddof=1)
    vb = (1.0 / Jb + n_test_b / n_train_b) * b.var(ddof=1)
    d = a.mean() - b.mean()
    se = math.sqrt(va + vb)
    df = min(Ja, Jb) - 1
    t = (d - delta) / se
    p = float(stats.t.sf(t, df)) if alternative == "greater" else (
        float(stats.t.cdf(t, df)) if alternative == "less" else float(2 * stats.t.sf(abs(t), df)))
    h = stats.t.ppf(0.5 + level / 2, df) * se
    return {"diff": float(d), "t": float(t), "p": p, "se": se, "df": df, "ci_lo": float(d - h), "ci_hi": float(d + h)}


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, (m - r) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cluster_bootstrap_coverage(y, lo, hi, groups, B: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Resample test water bodies with replacement; percentile CI for wb-averaged and pooled coverage."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    cover = ((y >= lo) & (y <= hi)).astype(float)
    codes, G = _group_codes(groups)
    cov_g, n_g = _group_mean(cover, codes, G)
    hits = cov_g * n_g
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, size=(B, G))
    cnt = np.apply_along_axis(np.bincount, 1, draws, minlength=G)  # (B, G) multiplicities
    wb = (cnt * cov_g).sum(1) / G
    pooled = (cnt * hits).sum(1) / (cnt * n_g).sum(1)
    a = (1 - level) / 2
    return {"cov_wb_lo": float(np.quantile(wb, a)), "cov_wb_hi": float(np.quantile(wb, 1 - a)),
            "cov_pooled_lo": float(np.quantile(pooled, a)), "cov_pooled_hi": float(np.quantile(pooled, 1 - a))}


def beta_coverage(n_cal_units: int, alpha: float, level: float = 0.95) -> dict:
    """Distribution of conditional coverage of split conformal given n exchangeable calibration units:
    Beta(k, n + 1 - k), k = ceil((1 - alpha)(n + 1)) (Vovk 2012; Angelopoulos and Bates 2023). k > n: coverage 1.
    """
    from src.conformal.core import conformal_rank

    n = int(n_cal_units)
    k = conformal_rank(n, alpha)
    if k > n:
        return {"n": n, "k": k, "a": np.nan, "b": np.nan, "mean": 1.0, "sd": 0.0, "q_lo": 1.0, "q_hi": 1.0,
                "infinite": True}
    a, b = k, n + 1 - k
    dist = stats.beta(a, b)
    return {"n": n, "k": k, "a": a, "b": b, "mean": float(dist.mean()), "sd": float(dist.std()),
            "q_lo": float(dist.ppf((1 - level) / 2)), "q_hi": float(dist.ppf(0.5 + level / 2)), "infinite": False}
