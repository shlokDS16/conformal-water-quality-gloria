"""Tests for interval methods, metrics and leakage rules (phase 06 build)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.conformal import core as C
from src.eval import metrics as E

ALPHA = 0.10


# ------------------------------------------------------------------ helpers
def _linfit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ coef


def _draw_groups(rng, G):
    """Random-effects population: 20 % large low-noise groups (n=50, sd 0.3), 80 % small noisy groups (n=3, sd 1.0)."""
    big = rng.random(G) < 0.2
    sizes = np.where(big, 50, 3)
    sd = np.where(big, 0.3, 1.0)
    b = rng.normal(0, 0.5, G)
    gid = np.repeat(np.arange(G), sizes)
    x = rng.normal(size=len(gid))
    y = 2.0 * x + b[gid] + rng.normal(size=len(gid)) * sd[gid]
    return x, y, gid


# ------------------------------------------------------------------ coverage on synthetic data
def test_split_conformal_iid_coverage_nominal():
    rng = np.random.default_rng(1)
    reps, n_cal, n_test = 500, 100, 1000
    cov = np.empty(reps)
    for r in range(reps):
        xc, xt = rng.normal(size=n_cal), rng.normal(size=n_test)
        yc, yt = xc + rng.standard_t(5, n_cal), xt + rng.standard_t(5, n_test)
        lo, hi, info = C.split_conformal(xc, yc, xt, ALPHA)
        cov[r] = np.mean((yt >= lo) & (yt <= hi))
    expected = C.conformal_rank(n_cal, ALPHA) / (n_cal + 1)
    se = cov.std(ddof=1) / math.sqrt(reps)
    assert abs(cov.mean() - expected) < 4 * se, (cov.mean(), expected, se)
    assert cov.mean() > 1 - ALPHA - 4 * se


def test_grouped_pooled_undercovers_subsampled_valid():
    rng = np.random.default_rng(2)
    reps, G = 500, 60
    cov_pool, cov_gsub = np.empty(reps), np.empty(reps)
    for r in range(reps):
        xc, yc, gc = _draw_groups(rng, G)
        xt, yt, gt = _draw_groups(rng, G)
        pc, pt = 2.0 * xc, 2.0 * xt  # oracle mean function; residual = random effect + noise
        lo, hi, _ = C.split_conformal(pc, yc, pt, ALPHA)
        cov_pool[r] = E.interval_metrics(yt, lo, hi, gt, ALPHA)["cov_wb"]
        idx = C.group_subsample(gc, rng)
        assert len(idx) == G and len(np.unique(gc[idx])) == G
        lo, hi, info = C.split_conformal(pc, yc, pt, ALPHA, idx)
        assert info["n_cal_units"] == G
        cov_gsub[r] = E.interval_metrics(yt, lo, hi, gt, ALPHA)["cov_wb"]
    se_p = cov_pool.std(ddof=1) / math.sqrt(reps)
    se_g = cov_gsub.std(ddof=1) / math.sqrt(reps)
    assert cov_pool.mean() < 1 - ALPHA - 0.03, cov_pool.mean()                  # clear group-level under-coverage
    assert cov_gsub.mean() >= 1 - ALPHA - 3 * se_g, (cov_gsub.mean(), se_g)     # nominal at water-body level
    assert abs(cov_gsub.mean() - C.conformal_rank(G, ALPHA) / (G + 1)) < 4 * se_g


def test_cvplus_iid_at_least_1_minus_2alpha():
    rng = np.random.default_rng(3)
    reps, n, n_test, K = 200, 200, 500, 10
    cov = np.empty(reps)
    for r in range(reps):
        X, Xt = rng.normal(size=(n, 3)), rng.normal(size=(n_test, 3))
        beta = np.array([1.0, -0.5, 0.2])
        y, yt = X @ beta + rng.normal(size=n), Xt @ beta + rng.normal(size=n_test)
        fold = np.arange(n) % K
        oof, mu = np.empty(n), np.empty((K, n_test))
        for k in range(K):
            f = _linfit(X[fold != k], y[fold != k])
            oof[fold == k] = f(X[fold == k])
            mu[k] = f(Xt)
        lo, hi, _ = C.cv_plus(oof, y, fold, mu, ALPHA)
        cov[r] = np.mean((yt >= lo) & (yt <= hi))
    se = cov.std(ddof=1) / math.sqrt(reps)
    assert cov.mean() >= 1 - 2 * ALPHA - 3 * se
    assert cov.mean() >= C.cvplus_bound(n, K, ALPHA)


def test_group_cvplus_single_subsample_reaches_bound():
    rng = np.random.default_rng(4)
    reps, G, K = 200, 80, 10
    cov = np.empty(reps)
    bounds = np.empty(reps)
    for r in range(reps):
        x, y, g = _draw_groups(rng, G)
        xt, yt, gt = _draw_groups(rng, 60)
        from src.experiment_common import equal_unit_folds
        fold = equal_unit_folds(g, K, rng)  # production fold rule (AUDIT_3 A3-1)
        oof, mu = np.empty(len(y)), np.empty((K, len(yt)))
        for k in range(K):
            f = _linfit(x[fold != k, None], y[fold != k])
            oof[fold == k] = f(x[fold == k, None])
            mu[k] = f(xt[:, None])
        idx = C.group_subsample(g, rng)
        lo, hi, info = C.cv_plus(oof, y, fold, mu, ALPHA, idx)
        assert info["n_cal_units"] == G
        bounds[r] = C.cvplus_bound(info["n_cal_units"], K, ALPHA)
        cov[r] = E.interval_metrics(yt, lo, hi, gt, ALPHA)["cov_wb"]
    se = cov.std(ddof=1) / math.sqrt(reps)
    assert cov.mean() >= bounds.mean() - 3 * se, (cov.mean(), bounds.mean())
    assert cov.mean() >= 1 - 2 * ALPHA - 3 * se


def test_equal_unit_folds_unbalanced_rows_and_group_cvplus_bound():
    """AUDIT_3 A3-1: fold UNIT counts differ by <= 1 even when row counts per water body are very unbalanced
    (one body of 600 rows, a few of 50, many of 1-3), and group CV+ with one row per body covers >= the
    Barber et al. (2021) Theorem 4 bound on synthetic grouped data. n = 83 is not a multiple of K."""
    from src.experiment_common import CVPLUS_FOLD_SEED, equal_unit_folds
    rng = np.random.default_rng(11)
    reps, G, K = 200, 83, 10
    cov = np.empty(reps)
    for r in range(reps):
        sizes = np.where(rng.random(G) < 0.1, 50, rng.integers(1, 4, G))
        sizes[0] = 600
        sd = np.where(sizes >= 50, 0.3, 1.0)
        b = rng.normal(0, 0.5, G)
        g = np.repeat(np.array([f"wb{i:03d}" for i in range(G)]), sizes)
        gi = np.repeat(np.arange(G), sizes)
        x = rng.normal(size=len(g))
        y = 2.0 * x + b[gi] + rng.normal(size=len(g)) * sd[gi]
        fold = equal_unit_folds(g, K, np.random.default_rng(CVPLUS_FOLD_SEED + r))
        units = np.array([len(np.unique(g[fold == k])) for k in range(K)])
        assert units.sum() == G and units.max() - units.min() <= 1, units
        rows = np.bincount(fold, minlength=K)
        if r == 0:
            assert rows.max() - rows.min() > 500  # rows are very unbalanced; units are not
        for gg in np.unique(g):  # each unit lies in exactly one fold
            assert len(np.unique(fold[g == gg])) == 1
        Gt = 60
        bt = rng.normal(0, 0.5, Gt)
        gt = np.repeat(np.arange(Gt), 3)
        xt = rng.normal(size=len(gt))
        yt = 2.0 * xt + bt[gt] + rng.normal(size=len(gt))
        oof, mu = np.empty(len(y)), np.empty((K, len(yt)))
        for k in range(K):
            f = _linfit(x[fold != k, None], y[fold != k])
            oof[fold == k] = f(x[fold == k, None])
            mu[k] = f(xt[:, None])
        idx = C.group_subsample(g, rng)
        lo, hi, info = C.cv_plus(oof, y, fold, mu, ALPHA, idx)
        assert info["n_cal_units"] == G
        cov[r] = E.interval_metrics(yt, lo, hi, gt, ALPHA)["cov_wb"]
    se = cov.std(ddof=1) / math.sqrt(reps)
    assert cov.mean() >= C.cvplus_bound(G, K, ALPHA)
    assert cov.mean() >= 1 - 2 * ALPHA - 3 * se
    # determinism of the fold rule
    g = np.array(list("aabbbcddddeeffgghhiijjkk"))
    f1 = equal_unit_folds(g, 3, np.random.default_rng(4000))
    f2 = equal_unit_folds(g, 3, np.random.default_rng(4000))
    assert (f1 == f2).all()


def test_mdn_member_seeds_collision_free_in_core_matrix():
    """AUDIT_3 A3-5: all (split seed, purpose, gauss-CV fold, member) MDN seeds are distinct."""
    from src import config
    from src.experiment_common import GAUSS_CV_K
    seeds = []
    for s in range(20):                       # largest seed range of any protocol (random, waterbody)
        for m in range(10):                   # n_rounds = 10 members
            seeds.append(config.mdn_member_seed("full", s, m))
            seeds.append(config.mdn_member_seed("r80", s, m))
            for k in range(GAUSS_CV_K):
                seeds.append(config.mdn_member_seed("cv", s, m, fold=k))
    assert len(seeds) == 20 * 10 * (2 + GAUSS_CV_K)
    assert len(set(seeds)) == len(seeds)
    # the full admissible range is collision-free too
    full = [config.mdn_member_seed(p, s, m, fold=k) for s in range(0, 900, 7) for m in range(10)
            for p, k in [("full", 0), ("r80", 0)] + [("cv", k) for k in range(10)]]
    assert len(set(full)) == len(full)
    with pytest.raises(ValueError):
        config.mdn_member_seed("cv", 0, 0, fold=10)
    with pytest.raises(ValueError):
        config.mdn_member_seed("full", 900, 0)
    # the MDN-STREAM member seeds of the full ensemble are unchanged (prereg s9)
    assert config.mdn_member_seed("full", 3, 7) == 1000 + 10 * 3 + 7


def test_mdn_cache_check_raises_on_missing_keys_and_old_rule(tmp_path):
    """AUDIT_3 A3-8: a cache lacking arrays for a requested method, or from another cfg/seed rule, raises."""
    import json
    from src.experiment_common import SEED_RULE, check_mdn_cache
    from src.models.mdn import MDN_DEFAULTS
    base = {k: np.zeros(3) for k in ("train__point", "cal__point", "test__point", "cal__mix_mean", "cal__mix_sd",
                                     "test__mix_mean", "test__mix_sd")}
    base.update({f"{r}__GLORIA_ID": np.array(["a", "b", "c"]) for r in ("train", "cal", "test")})
    meta = {"cfg": MDN_DEFAULTS, "seed_rule": SEED_RULE}
    p = tmp_path / "c.npz"
    np.savez_compressed(p, meta__json=np.array(json.dumps(meta)), **base)
    check_mdn_cache(p, MDN_DEFAULTS, need_cv=False, need_recal=False)
    with pytest.raises(RuntimeError, match="train__oof_point"):
        check_mdn_cache(p, MDN_DEFAULTS, need_cv=True, need_recal=False)
    with pytest.raises(RuntimeError, match="r80hold"):
        check_mdn_cache(p, MDN_DEFAULTS, need_cv=False, need_recal=True)
    with pytest.raises(RuntimeError, match="cfg"):
        check_mdn_cache(p, {**MDN_DEFAULTS, "n_iter": 5}, need_cv=False, need_recal=False)
    np.savez_compressed(p, meta__json=np.array(json.dumps({"cfg": MDN_DEFAULTS})), **base)
    with pytest.raises(RuntimeError, match="seed rule"):
        check_mdn_cache(p, MDN_DEFAULTS, need_cv=False, need_recal=False)


def test_sensitivity_population_and_strict_sensors():
    """AUDIT_3 A3-4: secchi_has_depth population and strict band sets."""
    from src.experiment_common import load_job
    from src.models.features import feature_columns, ratio_features
    d = load_job("Secchi_depth", "hyp", "waterbody", "secchi_has_depth")
    assert (d["Depth"] > 0).all() and d["Depth"].notna().all()
    full = load_job("Secchi_depth", "hyp", "waterbody", "primary")
    assert 0 < len(d) < len(full)
    with pytest.raises(ValueError):
        load_job("Chla", "hyp", "waterbody", "secchi_has_depth")
    assert feature_columns("msi_strict") == ["msi_B1", "msi_B2", "msi_B3", "msi_B4"]
    assert feature_columns("olci_strict") == [f"olci_Oa{i}" for i in range(2, 11)]
    for strict, loose in (("msi_strict", "msi"), ("olci_strict", "olci")):
        ds = load_job("Chla", strict, "waterbody")
        dl = load_job("Chla", loose, "waterbody")
        assert len(ds) >= len(dl)
        assert np.isfinite(ds[feature_columns(strict)].to_numpy()).all()
        rf = ratio_features(ds.head(50), strict)
        assert "x_ndci" not in rf.columns and np.isfinite(rf.to_numpy()).all()


# ------------------------------------------------------------------ quantile index and infinite intervals
def test_conformal_rank_and_infinite_flag():
    assert C.conformal_rank(9, 0.1) == 9          # 0.9 * 10 = 9.000000000000002 in floating point
    q, inf = C.conformal_quantile(np.arange(9), 0.1)
    assert not inf and q == 8
    q, inf = C.conformal_quantile(np.arange(8), 0.1)
    assert inf and math.isinf(q)
    lo, hi, info = C.split_conformal(np.zeros(8), np.arange(8), np.zeros(3), 0.1)
    assert info["inf"] and np.isinf(lo).all() and np.isinf(hi).all()
    m = E.interval_metrics(np.zeros(3), lo, hi, ["a", "a", "b"], 0.1)
    assert m["inf_rate"] == 1.0 and m["cov_wb"] == 1.0 and math.isinf(m["width_wb"])
    # conservative ties: k-th order statistic, no randomisation
    q, _ = C.conformal_quantile(np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2.0]), 0.5)
    assert q == 2.0  # k = ceil(0.5 * 11) = 6 -> 6th smallest


def test_group_subsample_deterministic_and_one_per_group():
    g = np.array(list("abcabcaab"))
    i1 = C.group_subsample(g, np.random.default_rng(2000))
    i2 = C.group_subsample(g, np.random.default_rng(2000))
    assert (i1 == i2).all() and sorted(g[i1]) == ["a", "b", "c"]


def test_cvplus_bound_matches_audit_numbers():
    assert round(C.cvplus_bound(205, 10, 0.10), 3) == 0.716
    assert round(C.cvplus_bound(264, 10, 0.10), 3) == 0.734


# ------------------------------------------------------------------ metrics
def test_point_metrics_factor_two():
    y = np.log10(np.array([1.0, 10.0, 100.0]))
    m = E.point_metrics(y, y + math.log10(2))
    assert m["mdsa"] == pytest.approx(100.0) and m["sspb"] == pytest.approx(100.0)
    m = E.point_metrics(y, y - math.log10(2))
    assert m["mdsa"] == pytest.approx(100.0) and m["sspb"] == pytest.approx(-100.0)
    assert m["logmae"] == pytest.approx(math.log10(2))


def test_interval_metrics_by_hand():
    y = np.array([0.0, 0.0, 0.0, 1.0])
    lo = np.array([-1.0, 0.5, -1.0, 0.0])
    hi = np.array([1.0, 1.0, 1.0, 2.0])
    g = np.array(["a", "a", "a", "b"])
    m = E.interval_metrics(y, lo, hi, g, 0.2)
    assert m["cov_pooled"] == pytest.approx(0.75)
    assert m["cov_wb"] == pytest.approx((2 / 3 + 1) / 2)
    assert m["width_wb"] == pytest.approx((100.0 + 100.0) / 2)   # medians: a: 10^2, b: 10^2
    w = E.winkler_score(y, lo, hi, 0.2)
    assert w[1] == pytest.approx(0.5 + (2 / 0.2) * 0.5)
    assert m["inf_rate"] == 0.0


def test_beta_coverage_mean():
    b = E.beta_coverage(51, 0.10)
    assert b["k"] == 47 and b["mean"] == pytest.approx(47 / 52)
    assert E.beta_coverage(8, 0.10)["infinite"]


def test_nadeau_bengio_and_holm():
    v = np.array([0.90, 0.92, 0.88, 0.91, 0.89])
    r = E.nadeau_bengio_ttest(v, 0.88, n_test=50, n_train=200, alternative="greater")
    se = math.sqrt((1 / 5 + 50 / 200) * v.var(ddof=1))
    assert r["se"] == pytest.approx(se) and r["t"] == pytest.approx((v.mean() - 0.88) / se)
    d = E.nadeau_bengio_diff_test(v + 0.05, v, 50, 200, 60, 250, delta=0.02)
    assert d["df"] == 4 and d["diff"] == pytest.approx(0.05)
    assert list(E.holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])


def test_cluster_bootstrap_brackets_estimate():
    rng = np.random.default_rng(5)
    g = np.repeat(np.arange(30), 5)
    y = rng.normal(size=len(g))
    lo, hi = np.full(len(g), -1.64), np.full(len(g), 1.64)
    est = E.interval_metrics(y, lo, hi, g, 0.1)["cov_wb"]
    ci = E.cluster_bootstrap_coverage(y, lo, hi, g, B=500, seed=1)
    assert ci["cov_wb_lo"] <= est <= ci["cov_wb_hi"]


# ------------------------------------------------------------------ leakage and shared row set
def _small_split(target="Chla", sensor="hyp", protocol="waterbody", seed=0):
    from src.experiment_common import iter_splits, load_job
    df = load_job(target, sensor, protocol)
    s, f, sp = next(iter_splits(df, [seed]))
    return df, sp


def test_load_job_floor_and_shared_rows():
    from src.models.features import LOG_FLOOR, feature_columns
    df, sp = _small_split()
    X = df[feature_columns("hyp")].to_numpy()
    assert np.isfinite(X).all() and X.min() >= LOG_FLOOR
    assert df["nonpos_any"].sum() > 0                          # flag recorded before flooring
    assert np.isfinite(df["y"]).all()


def test_fitted_transforms_see_only_train_rows():
    from src.models.baselines import RidgeLogBands, OCMaxBandRatio
    from src.models.features import feature_columns, log_bands
    from src.models.lgbm import LGBMPoint
    _, sp = _small_split()
    tr, ca, te = (sp[sp["role"] == r].reset_index(drop=True) for r in ("train", "cal", "test"))
    ytr = tr["y"].to_numpy()
    rg = RidgeLogBands("hyp").fit(tr, ytr, tr["wb_group"].to_numpy())
    np.testing.assert_allclose(rg.scaler_.mean_, log_bands(tr[feature_columns("hyp")].to_numpy()).mean(0))
    # models must not change when calibration or test labels change
    te2, ca2 = te.copy(), ca.copy()
    te2["y"] = 0.0; ca2["y"] = 0.0
    for M in (lambda: RidgeLogBands("hyp"), lambda: OCMaxBandRatio("hyp"), lambda: LGBMPoint("hyp", 1000, n_jobs=4)):
        p1 = M().fit(tr, ytr, tr["wb_group"].to_numpy()).predict(te)
        p2 = M().fit(tr, ytr, tr["wb_group"].to_numpy()).predict(te2)
        np.testing.assert_array_equal(p1, p2)
    # conformal quantile depends only on calibration scores
    pc, pt = np.zeros(len(ca)), np.zeros(len(te))
    lo1, hi1, _ = C.split_conformal(pc, ca["y"], pt, ALPHA)
    lo2, hi2, _ = C.split_conformal(pc, ca["y"], pt + 0 * te2["y"].to_numpy(), ALPHA)
    np.testing.assert_array_equal(hi1, hi2)


def test_mdn_tasks_use_train_rows_only(monkeypatch, tmp_path):
    import src.models.mdn as mdn
    from src.experiment_common import build_mdn_cache, recal_holdout_index
    from src.models.features import feature_columns
    _, sp = _small_split()
    tr = sp[sp["role"] == "train"]
    Xtr = tr[feature_columns("hyp")].to_numpy(float)
    seen = []
    real = mdn.fit_members

    def spy(tasks, **kw):
        seen.extend(tasks)
        return real(tasks, **kw)

    monkeypatch.setattr(mdn, "fit_members", spy)
    build_mdn_cache([("t", 0, 0, sp, False)], "hyp", tmp_path, [0.1], device="cpu",
                    cfg={"n_iter": 5, "n_rounds": 2}, with_recal=True, with_cv=True, log=lambda *a: None)
    assert len(seen) == 2 + 2 + 5 * 2
    train_rows = {r.tobytes() for r in Xtr}
    for t in seen:
        assert all(r.tobytes() in train_rows for r in t.X)          # every member sees train rows only
    hold = recal_holdout_index(len(tr), 0)
    held = {r.tobytes() for r in Xtr[hold]} - {r.tobytes() for r in Xtr[~hold]}  # duplicate spectra share train
    for t in seen[2:4]:                                                # recal models exclude the 20 % holdout
        assert not any(r.tobytes() in held for r in t.X)
    m = mdn._prepare(seen[0], {**mdn.MDN_DEFAULTS, "n_iter": 5})
    rng = np.random.default_rng(seen[0].seed)
    bag = rng.permutation(len(Xtr))[: int(0.75 * len(Xtr))]
    np.testing.assert_allclose(m[3][0], np.median(Xtr[bag], 0))       # RobustScaler fit on the member bag


def test_all_models_scored_on_identical_rows(tmp_path):
    """Amendment 2 item 2: one shared row set; every model and method is scored on the same GLORIA_IDs."""
    import src.run_experiment as R
    from src.experiment_common import build_mdn_cache, split_stem
    _, sp = _small_split()
    stem = split_stem("Chla", "hyp", "waterbody", 0, 0)
    build_mdn_cache([(stem, 0, 0, sp, False)], "hyp", tmp_path, [0.1], device="cpu",
                    cfg={"n_iter": 5, "n_rounds": 2}, log=lambda *a: None)
    settings = dict(target="Chla", sensor="hyp", protocol="waterbody", population="primary", noise_mult=0.0,
                    noise_add=0.0, noise_tag="", tag="test", out_dir=str(tmp_path), models=R.ALL_MODELS,
                    methods=R.ALL_METHODS, alphas=[0.1], lgbm_threads=8, boot=0)
    R.process_split(sp, settings)
    pred = pd.read_parquet(tmp_path / f"pred_{stem}.parquet")
    met = pd.read_csv(tmp_path / f"metrics_{stem}.csv")
    test_ids = set(sp.loc[sp["role"] == "test", "GLORIA_ID"])
    sets = pred.groupby(["model", "method"], observed=True)["GLORIA_ID"].agg(lambda s: frozenset(s))
    assert all(s == test_ids for s in sets)
    assert pred.groupby(["model", "method"], observed=True).size().nunique() == 1
    assert set(met["model"]) == {"const", "ridge", "emp_oc", "emp_ndci", "lgbm", "mdn"}
    assert not pred[["point", "y"]].isna().any().any()
    iv = pred[pred["method"] != "point"]
    assert not iv[["lower", "upper"]].isna().any().any()
    for col in ("n_test_wb", "k_cal", "n_rows_nonpositive", "inf_rate", "k_cal_rows", "k_cal_units", "cv_folds",
                "calib_scheme", "cvplus_fold_units", "n_fallback_test"):
        assert col in met.columns
    iv = met[met["method"] != "point"].set_index(["model", "method"])
    assert (iv.loc[("lgbm", "scp_gsub"), "k_cal_units"] == iv.loc[("lgbm", "scp_gsub"), "n_cal_wb"])
    assert (iv.loc[("lgbm", "scp_pool"), "k_cal_rows"] == iv.loc[("lgbm", "scp_pool"), "n_cal"])
    assert iv.loc[("lgbm", "cvplus"), "cv_folds"] == 10 and iv.loc[("lgbm", "gauss"), "cv_folds"] == 5
    fu = [int(u) for u in str(iv.loc[("lgbm", "cvplus"), "cvplus_fold_units"]).split(";")]
    assert len(fu) == 10 and max(fu) - min(fu) <= 1 and sum(fu) == iv.loc[("lgbm", "cvplus"), "k_cal_units"]
