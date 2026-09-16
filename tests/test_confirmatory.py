"""Tests for src/analysis/confirmatory.py on SYNTHETIC metrics files with known answers (no real results)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.analysis import confirmatory as CF

TARGETS = CF.TARGETS
J = 20


def _rows(target, protocol, model, method, values, col, rng, alpha=0.10, extra=None):
    out = []
    for s, v in enumerate(values):
        r = dict(target=target, sensor="hyp", protocol=protocol, population="primary", noise_mult=0.0,
                 noise_add=0.0, seed=s, fold=0, test_unit="", descriptive_only=False, tag="synthetic",
                 model=model, method=method, alpha=alpha,
                 n_train=1200 + s, n_cal=400, n_test=400 + (s % 3),          # rows
                 n_train_wb=230 + (s % 2), n_cal_wb=77, n_test_wb=77 + (s % 2),  # water bodies
                 cov_wb=0.9, cov_pooled=0.9, width_wb_log10mean=0.5)
        r[col] = v
        if extra:
            r.update(extra)
        out.append(r)
    return out


def _scenario(rng, h1_mdn=0.75, h1_gauss=0.80, h2=0.91, h3_diff=-0.05, h4_random=0.95, h4_wb=0.88,
              sd=0.01, width_inf_seed=None):
    """One synthetic metrics table. Values = level + N(0, sd), centred exactly on the level."""
    def vals(level):
        e = rng.normal(0, sd, J)
        return level + e - e.mean()

    rows = []
    for t in TARGETS:
        rows += _rows(t, "waterbody", "mdn", "native", vals(h1_mdn), "cov_wb", rng)
        rows += _rows(t, "waterbody", "lgbm", "gauss", vals(h1_gauss), "cov_wb", rng)
        w_scp = vals(0.6)
        w_cqr = w_scp + vals(h3_diff)
        if width_inf_seed is not None:
            w_cqr[width_inf_seed] = np.inf
        scp = _rows(t, "waterbody", "lgbm", "scp_gsub", vals(h2), "cov_wb", rng)
        cqr = _rows(t, "waterbody", "lgbm", "cqr_gsub", vals(0.9), "cov_wb", rng)
        rows += [dict(r, width_wb_log10mean=w) for r, w in zip(scp, w_scp)]
        rows += [dict(r, width_wb_log10mean=w) for r, w in zip(cqr, w_cqr)]
        rows += _rows(t, "random", "lgbm", "scp_pool", vals(h4_random), "cov_pooled", rng)
        rows += _rows(t, "waterbody", "lgbm", "scp_pool", vals(h4_wb), "cov_wb", rng)
        # distractors that must be ignored: other alpha, other sensor, a point row
        rows += _rows(t, "waterbody", "lgbm", "scp_gsub", vals(0.10), "cov_wb", rng, alpha=0.20)
        rows += [dict(r, sensor="msi") for r in _rows(t, "waterbody", "lgbm", "scp_gsub", vals(0.10), "cov_wb", rng)]
    return pd.DataFrame(rows)


def _write(df, tmp_path):
    for (t, p, s), d in df.groupby(["target", "protocol", "seed"]):
        d.to_csv(tmp_path / f"metrics_{t}_hyp_{p}_s{s:02d}_f0.csv", index=False)
    return CF.load_metrics(tmp_path)


def _nb_p(values, mu0, n_test, n_train, alternative):
    v = np.asarray(values, float)
    se = math.sqrt((1 / len(v) + n_test / n_train) * v.var(ddof=1))
    t = (v.mean() - mu0) / se
    return (stats.t.cdf(t, len(v) - 1) if alternative == "less" else stats.t.sf(t, len(v) - 1)), se


def test_all_supported_known_answers(tmp_path):
    rng = np.random.default_rng(0)
    met = _write(_scenario(rng), tmp_path)
    conf, comps = CF.confirmatory_table(met)
    assert len(conf) == 16 and set(conf["hypothesis"]) == {"H1", "H2", "H3", "H4"}
    assert (conf["decision"] == "reject_H0_supported").all(), conf[["hypothesis", "target", "p_raw", "p_holm"]]
    assert conf["complete"].all()
    # Holm adjustment equals the reference implementation on the 16 raw p-values
    np.testing.assert_allclose(conf["p_holm"], CF.E.holm(conf["p_raw"].to_numpy()))
    # H2 by hand, water-body units averaged over repeats
    sub = met[(met.target == "TSS") & (met.protocol == "waterbody") & (met.model == "lgbm")
              & (met.method == "scp_gsub") & (met.sensor == "hyp") & np.isclose(met.alpha, 0.10)].sort_values("seed")
    p, se = _nb_p(sub.cov_wb, 0.88, sub.n_test_wb.mean(), (sub.n_train_wb + sub.n_cal_wb).mean(), "greater")
    row = conf[(conf.hypothesis == "H2") & (conf.target == "TSS")].iloc[0]
    assert row["p_raw"] == pytest.approx(p, rel=1e-10)
    assert row["estimate"] == pytest.approx(0.91, abs=1e-12)
    h = stats.t.ppf(0.975, J - 1) * se
    assert row["ci_lo"] == pytest.approx(0.91 - h) and row["ci_hi"] == pytest.approx(0.91 + h)
    assert row["df"] == J - 1 and row["J"] == J
    # H3 estimate is the mean paired log-width difference
    h3 = conf[(conf.hypothesis == "H3") & (conf.target == "Chla")].iloc[0]
    assert h3["estimate"] == pytest.approx(-0.05, abs=1e-12)
    # H1 components: two arms, binding arm has the larger p
    c = comps[comps.target == "aCDOM440"]
    assert set(c["arm"]) == {"mdn_native", "lgbm_gauss"}
    h1 = conf[(conf.hypothesis == "H1") & (conf.target == "aCDOM440")].iloc[0]
    assert h1["p_raw"] == pytest.approx(c["p_raw"].max())
    assert h1["binding_arm"] == c.loc[c["p_raw"].idxmax(), "arm"]


def test_h2_fails_when_coverage_below_088(tmp_path):
    rng = np.random.default_rng(1)
    met = _write(_scenario(rng, h2=0.85), tmp_path)
    conf, _ = CF.confirmatory_table(met)
    h2 = conf[conf.hypothesis == "H2"]
    assert (h2["decision"] == "not_rejected").all()
    assert (h2["p_raw"] > 0.5).all() and (h2["estimate"] < 0.88).all()
    # the other hypotheses are unaffected
    assert (conf[conf.hypothesis != "H2"]["decision"] == "reject_H0_supported").all()


def test_h4_holds_with_sample_units_for_random_arm(tmp_path):
    rng = np.random.default_rng(2)
    met = _write(_scenario(rng, h4_random=0.95, h4_wb=0.88, sd=0.02), tmp_path)
    conf, _ = CF.confirmatory_table(met)
    row = conf[(conf.hypothesis == "H4") & (conf.target == "Secchi_depth")].iloc[0]
    assert row["decision"] == "reject_H0_supported"
    assert row["estimate"] == pytest.approx(0.07, abs=1e-12)
    ra = met[(met.target == "Secchi_depth") & (met.protocol == "random") & (met.method == "scp_pool")].sort_values("seed")
    wb = met[(met.target == "Secchi_depth") & (met.protocol == "waterbody") & (met.method == "scp_pool")].sort_values("seed")
    va = (1 / J + ra.n_test.mean() / (ra.n_train + ra.n_cal).mean()) * ra.cov_pooled.var(ddof=1)      # rows
    vb = (1 / J + wb.n_test_wb.mean() / (wb.n_train_wb + wb.n_cal_wb).mean()) * wb.cov_wb.var(ddof=1)  # water bodies
    t = (0.07 - 0.02) / math.sqrt(va + vb)
    assert row["t"] == pytest.approx(t, rel=1e-10)
    assert row["p_raw"] == pytest.approx(stats.t.sf(t, J - 1), rel=1e-10)
    assert row["df"] == J - 1
    # wrong (water-body) units for the random arm would give a different statistic
    va_wrong = (1 / J + ra.n_test_wb.mean() / (ra.n_train_wb + ra.n_cal_wb).mean()) * ra.cov_pooled.var(ddof=1)
    assert abs((0.05 / math.sqrt(va_wrong + vb)) - row["t"]) > 1e-6


def test_h4_not_supported_when_difference_below_delta(tmp_path):
    rng = np.random.default_rng(3)
    met = _write(_scenario(rng, h4_random=0.895, h4_wb=0.885), tmp_path)
    conf, _ = CF.confirmatory_table(met)
    assert (conf[conf.hypothesis == "H4"]["decision"] == "not_rejected").all()


def test_h1_intersection_union_requires_both_arms(tmp_path):
    rng = np.random.default_rng(4)
    met = _write(_scenario(rng, h1_mdn=0.70, h1_gauss=0.92), tmp_path)
    conf, comps = CF.confirmatory_table(met)
    h1 = conf[conf.hypothesis == "H1"]
    assert (h1["decision"] == "not_rejected").all()
    assert (h1["binding_arm"] == "lgbm_gauss").all() and (h1["p_raw"] > 0.5).all()


def test_nonfinite_width_gives_conservative_p(tmp_path):
    rng = np.random.default_rng(5)
    met = _write(_scenario(rng, width_inf_seed=3), tmp_path)
    conf, _ = CF.confirmatory_table(met)
    h3 = conf[conf.hypothesis == "H3"]
    assert (h3["p_raw"] == 1.0).all() and (h3["decision"] == "not_rejected").all()
    assert h3["note"].str.contains("non-finite").all()


def test_missing_cell_and_duplicates(tmp_path):
    rng = np.random.default_rng(6)
    df = _scenario(rng)
    df = df[~((df.target == "Chla") & (df.method == "native"))]
    df_short = df[~((df.target == "TSS") & (df.method == "scp_gsub") & (df.seed >= 15))]
    met = _write(df_short, tmp_path)
    conf, _ = CF.confirmatory_table(met)
    h1 = conf[(conf.hypothesis == "H1") & (conf.target == "Chla")].iloc[0]
    assert h1["p_raw"] == 1.0 and "J = 0" in h1["note"] and not h1["complete"]
    h2 = conf[(conf.hypothesis == "H2") & (conf.target == "TSS")].iloc[0]
    assert h2["J"] == 15 and not h2["complete"]
    assert len(conf) == 16
    dup = pd.concat([met, met[met.seed == 0]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        CF.confirmatory_table(dup)


def test_exploratory_and_budget_summaries(tmp_path):
    rng = np.random.default_rng(7)
    met = _write(_scenario(rng), tmp_path)
    met["mdsa"] = 50.0
    ex = CF.exploratory_cells(met)
    assert (ex["label"] == "exploratory").all() and len(ex) > 0
    # budget: 3 seeds x 4 draws; per-seed means 0.8, 0.9, 1.0; draw 0 equals the seed mean
    rows = []
    for s, base in enumerate([0.8, 0.9, 1.0]):
        for d, off in enumerate([0.0, 0.05, -0.05, 0.0]):
            rows.append(dict(target="Chla", sensor="hyp", population="primary", n_cal_wb_target="10", method="scp_gsub",
                             alpha=0.1, seed=s, draw=d, cov_wb=base + off, k_cal_rows=10, k_cal=10,
                             beta_mean=10 / 11, inf_flag=False, width_wb_log10mean=0.4))
    b = CF.budget_group_summary(pd.DataFrame(rows)).iloc[0]
    assert b["mean_cov"] == pytest.approx(0.9)
    assert b["sd_seed_draw0"] == pytest.approx(0.1) and b["sd_seed_means"] == pytest.approx(0.1)
    assert b["sd_draws_conditional_on_split"] == pytest.approx(np.std([0, 0.05, -0.05, 0], ddof=1))
    assert b["beta_sd"] == pytest.approx(CF.E.beta_coverage(10, 0.1)["sd"]) and b["beta_law_applicable"]
