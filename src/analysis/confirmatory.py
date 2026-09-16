"""Confirmatory analysis of the preregistered hypotheses H1-H4 (research/PREREGISTRATION.md s5 with
Amendment 2 item 6 and the AUDIT_3 A3-2/A3-3 unit convention). Written blind, before any core result was read.

Usage (run only after the core matrix has finished and this file's SHA256 is logged):
  python -m src.analysis.confirmatory --results results/core_v2 --budget results/budget_v2 --out tables

Inputs: per-split metrics files `metrics_*.csv` written by src/run_experiment.py (one row per model, method,
alpha). The confirmatory cell is sensor hyp, population primary, no noise, alpha 0.10, fold 0, not
descriptive-only, protocol waterbody (H1-H4) and random (H4 reference arm), for each of the 4 targets.

Per split the inputs are already water-body-averaged (`cov_wb`, `width_wb_log10mean`) or sample-pooled
(`cov_pooled`); the across-repeat statistic is the Nadeau and Bengio (2003) corrected resampled t-test:
    var = (1/J + n_test/n_train) s^2,   df = J - 1,   one-sided,
with n_test and n_train the means over the J repeats of the realised per-split counts.
Unit convention (Amendment 2 item 6, AUDIT_3 A3-3):
    waterbody protocol: n_test = test water bodies, n_train = training + calibration water bodies;
    random protocol (H4 reference arm only): n_test = test rows, n_train = training + calibration rows.

Tests (16, one Holm family, FWER 0.05):
  H1(t) intersection-union: MDN `native` cov_wb < 0.88 AND LightGBM `gauss` cov_wb < 0.88; each arm a
        one-sided 'less' test against 0.88; p = the larger of the two p-values.
  H2(t) non-inferiority: LightGBM `scp_gsub` cov_wb; H0 mean < 0.88, H1 mean >= 0.88 ('greater').
  H3(t) paired by split: d = width_wb_log10mean(cqr_gsub) - width_wb_log10mean(scp_gsub), LightGBM; 'less' than 0.
  H4(t) Welch-type, unpaired: cov_pooled(random, LightGBM scp_pool) - cov_wb(waterbody, LightGBM scp_pool)
        > delta = 0.02; variance = sum of the two corrected variances; df = min(J_random, J_waterbody) - 1.
Effect sizes: the estimate with a two-sided 95 % interval using the same corrected standard error.
Decision: reject H0 (hypothesis supported) when the Holm-adjusted p <= 0.05.

Conservative rules fixed before any result was seen:
  * a hypothesis whose inputs are missing, have J < 2 or contain a non-finite value (for example an
    infinite width) gets p_raw = 1 and a note; it stays in the Holm family of 16;
  * duplicated (target, protocol, model, method, alpha, seed) rows across input files raise an error;
  * J is reported and `complete` is False when J differs from the preregistered 20 repeats.

Exploratory outputs (labelled exploratory; no tests): tables/exploratory_cells.csv (every cell),
tables/exploratory_budget_group.csv and tables/exploratory_budget_local.csv (calibration budget).
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import metrics as E

ALPHA = 0.10
DELTA = 0.02
COV_TARGET = 0.88  # 1 - alpha - delta, written as a literal to avoid float round-off
FWER = 0.05
TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
J_EXPECTED = {"waterbody": 20, "random": 20}
CELL = {"sensor": "hyp", "population": "primary"}
KEY = ["target", "sensor", "protocol", "population", "noise_mult", "noise_add", "model", "method", "alpha",
       "seed", "fold"]
OUT_COLS = ["hypothesis", "target", "test", "estimate", "ci_lo", "ci_hi", "null_value", "alternative", "t", "df",
            "J", "complete", "unit", "n_test_mean", "n_train_mean", "binding_arm", "p_raw", "p_holm", "decision",
            "note"]


# ------------------------------------------------------------------ loading
def load_metrics(dirs) -> pd.DataFrame:
    files = []
    for d in ([dirs] if isinstance(dirs, (str, Path)) else dirs):
        files += sorted(glob.glob(str(Path(d) / "metrics_*.csv")))
    if not files:
        raise FileNotFoundError(f"no metrics_*.csv in {dirs}")
    met = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    met["population"] = met["population"].fillna("primary")
    for c in ("noise_mult", "noise_add"):
        met[c] = met[c].fillna(0.0).astype(float)
    return met


def cell_values(met: pd.DataFrame, target: str, protocol: str, model: str, method: str,
                alpha: float = ALPHA) -> pd.DataFrame:
    """Rows of the confirmatory cell for one (model, method): one row per seed, sorted by seed."""
    m = met[(met["target"] == target) & (met["protocol"] == protocol) & (met["sensor"] == CELL["sensor"])
            & (met["population"] == CELL["population"]) & (met["noise_mult"] == 0) & (met["noise_add"] == 0)
            & (met["model"] == model) & (met["method"] == method)
            & np.isclose(met["alpha"].astype(float), alpha) & (met["fold"] == 0)
            & (~met["descriptive_only"].astype(bool))]
    dup = m.duplicated(subset=KEY, keep=False)
    if dup.any():
        raise ValueError(f"duplicated split rows for {target}/{protocol}/{model}/{method}: "
                         f"seeds {sorted(m.loc[dup, 'seed'].unique())}")
    return m.sort_values("seed").reset_index(drop=True)


def units(rows: pd.DataFrame, protocol: str) -> tuple[float, float, str]:
    """(mean n_test, mean n_train) under the preregistered unit convention for the protocol."""
    if protocol == "random":
        return float(rows["n_test"].mean()), float((rows["n_train"] + rows["n_cal"]).mean()), "rows"
    return float(rows["n_test_wb"].mean()), float((rows["n_train_wb"] + rows["n_cal_wb"]).mean()), "water_bodies"


# ------------------------------------------------------------------ single tests
def _not_evaluable(note: str, **kw) -> dict:
    out = dict(estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan, t=np.nan, df=np.nan, p_raw=1.0, note=note)
    out.update(kw)
    return out


def nb_one_sample(values, mu0: float, n_test: float, n_train: float, alternative: str) -> dict:
    v = np.asarray(values, float)
    J = len(v)
    if J < 2:
        return _not_evaluable(f"J = {J} < 2: conservative p = 1", J=J)
    if not np.isfinite(v).all():
        return _not_evaluable(f"{int((~np.isfinite(v)).sum())} non-finite split values: conservative p = 1", J=J)
    r = E.nadeau_bengio_ttest(v, mu0, n_test=n_test, n_train=n_train, alternative=alternative)
    return dict(estimate=r["mean"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], t=r["t"], df=J - 1, p_raw=r["p"],
                J=J, se=r["se"], note="")


def run_h1(met, target) -> tuple[dict, list[dict]]:
    arms = [("mdn", "native"), ("lgbm", "gauss")]
    comps = []
    for model, method in arms:
        rows = cell_values(met, target, "waterbody", model, method)
        n_te, n_tr, unit = units(rows, "waterbody") if len(rows) else (np.nan, np.nan, "water_bodies")
        r = nb_one_sample(rows["cov_wb"], COV_TARGET, n_te, n_tr, "less")
        r.update(arm=f"{model}_{method}", unit=unit, n_test_mean=n_te, n_train_mean=n_tr,
                 seeds=";".join(map(str, rows["seed"].tolist())))
        comps.append(r)
    b = max(range(len(comps)), key=lambda i: (comps[i]["p_raw"], i))  # larger p binds (intersection-union)
    out = {k: comps[b].get(k) for k in ("estimate", "ci_lo", "ci_hi", "t", "df", "J", "unit", "n_test_mean",
                                        "n_train_mean")}
    notes = [f"{c['arm']}: {c['note']}" for c in comps if c["note"]]
    out.update(test="IUT max p of NB one-sample 'less' (mdn native, lgbm gauss) on cov_wb", null_value=COV_TARGET,
               alternative="less", binding_arm=comps[b]["arm"], p_raw=max(c["p_raw"] for c in comps),
               note="; ".join(notes),
               complete=all(c.get("J") == J_EXPECTED["waterbody"] for c in comps))
    return out, [dict(hypothesis="H1", target=target, **c) for c in comps]


def run_h2(met, target) -> dict:
    rows = cell_values(met, target, "waterbody", "lgbm", "scp_gsub")
    n_te, n_tr, unit = units(rows, "waterbody") if len(rows) else (np.nan, np.nan, "water_bodies")
    r = nb_one_sample(rows["cov_wb"], COV_TARGET, n_te, n_tr, "greater")
    r.update(test="NB one-sample 'greater' (non-inferiority) on cov_wb, lgbm scp_gsub", null_value=COV_TARGET,
             alternative="greater", unit=unit, n_test_mean=n_te, n_train_mean=n_tr, binding_arm="lgbm_scp_gsub",
             complete=r.get("J") == J_EXPECTED["waterbody"])
    return r


def run_h3(met, target) -> dict:
    a = cell_values(met, target, "waterbody", "lgbm", "cqr_gsub")
    b = cell_values(met, target, "waterbody", "lgbm", "scp_gsub")
    cols = ["seed", "width_wb_log10mean", "n_test_wb", "n_train_wb", "n_cal_wb"]
    p = a[cols].merge(b[cols], on="seed", suffixes=("_cqr", "_scp"), validate="one_to_one")
    note = ""
    if len(p) != max(len(a), len(b)):
        note = f"unpaired seeds dropped (cqr {len(a)}, scp {len(b)}, paired {len(p)})"
    for c in ("n_test_wb", "n_train_wb", "n_cal_wb"):
        if not (p[f"{c}_cqr"] == p[f"{c}_scp"]).all():
            raise ValueError(f"H3 {target}: {c} differs within a split between cqr_gsub and scp_gsub")
    d = p["width_wb_log10mean_cqr"] - p["width_wb_log10mean_scp"]
    n_te = float(p["n_test_wb_scp"].mean()) if len(p) else np.nan
    n_tr = float((p["n_train_wb_scp"] + p["n_cal_wb_scp"]).mean()) if len(p) else np.nan
    with np.errstate(invalid="ignore"):
        r = nb_one_sample(d.to_numpy(float), 0.0, n_te, n_tr, "less")
    r["note"] = "; ".join(x for x in (note, r["note"]) if x)
    r.update(test="NB paired 'less' on log10 width difference cqr_gsub - scp_gsub (lgbm)", null_value=0.0,
             alternative="less", unit="water_bodies", n_test_mean=n_te, n_train_mean=n_tr,
             binding_arm="lgbm_cqr_gsub_minus_scp_gsub", complete=len(p) == J_EXPECTED["waterbody"])
    return r


def run_h4(met, target) -> dict:
    ra = cell_values(met, target, "random", "lgbm", "scp_pool")
    wb = cell_values(met, target, "waterbody", "lgbm", "scp_pool")
    a, b = ra["cov_pooled"].to_numpy(float), wb["cov_wb"].to_numpy(float)
    base = dict(test="Welch-type NB difference 'greater': cov_pooled(random) - cov_wb(waterbody), lgbm scp_pool",
                null_value=DELTA, alternative="greater", unit="random: rows; waterbody: water_bodies",
                binding_arm="lgbm_scp_pool", J=f"{len(a)};{len(b)}",
                complete=(len(a) == J_EXPECTED["random"]) and (len(b) == J_EXPECTED["waterbody"]))
    if min(len(a), len(b)) < 2:
        return {**base, **_not_evaluable(f"J = {len(a)};{len(b)} < 2: conservative p = 1"), "J": base["J"]}
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {**base, **_not_evaluable("non-finite split values: conservative p = 1"), "J": base["J"]}
    nte_a, ntr_a, _ = units(ra, "random")
    nte_b, ntr_b, _ = units(wb, "waterbody")
    r = E.nadeau_bengio_diff_test(a, b, nte_a, ntr_a, nte_b, ntr_b, delta=DELTA, alternative="greater")
    return {**base, "estimate": r["diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "t": r["t"], "df": r["df"],
            "p_raw": r["p"], "se": r["se"], "n_test_mean": f"{nte_a:.6g};{nte_b:.6g}",
            "n_train_mean": f"{ntr_a:.6g};{ntr_b:.6g}", "note": ""}


# ------------------------------------------------------------------ family
def confirmatory_table(met: pd.DataFrame, targets=TARGETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, comps = [], []
    for t in targets:
        h1, c = run_h1(met, t)
        comps += c
        for name, r in (("H1", h1), ("H2", run_h2(met, t)), ("H3", run_h3(met, t)), ("H4", run_h4(met, t))):
            rows.append({"hypothesis": name, "target": t, **r})
    out = pd.DataFrame(rows)
    if len(out) != 4 * len(targets):
        raise AssertionError("family size")
    out["p_holm"] = E.holm(out["p_raw"].to_numpy(float))
    out["decision"] = np.where(out["p_holm"] <= FWER, "reject_H0_supported", "not_rejected")
    for c in OUT_COLS:
        if c not in out.columns:
            out[c] = np.nan
    out = out[OUT_COLS + [c for c in out.columns if c not in OUT_COLS]]
    return out, pd.DataFrame(comps)


# ------------------------------------------------------------------ exploratory summaries (no tests)
INTERVAL_COLS = ["cov_wb", "cov_pooled", "width_wb", "width_wb_log10mean", "winkler_wb", "inf_rate",
                 "cov_worst_ge10"]
POINT_COLS = ["mdsa", "sspb", "logmae", "logbias", "logrmse"]


def exploratory_cells(met: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and median over splits for every cell; point accuracy only from method == 'point' rows
    (AUDIT_3 A3-7). Multi-fold protocols (contributor, region) have dependent folds within a seed: the
    SD is descriptive only. Label: exploratory."""
    keys = ["target", "sensor", "protocol", "population", "noise_mult", "noise_add", "model", "method", "alpha"]
    m = met[~met["descriptive_only"].astype(bool) | (met["method"] == "point")].copy()
    m["alpha"] = m["alpha"].fillna(-1.0)
    iv = m[m["method"] != "point"]
    pt = m[m["method"] == "point"]
    parts = []
    for d, cols, is_interval in ((iv, INTERVAL_COLS + ["beta_mean", "cvplus_bound", "k_cal_units", "k_cal_rows"], True),
                                 (pt, POINT_COLS, False)):
        cols = [c for c in cols if c in d.columns]
        if not len(d):
            continue
        dd = d.replace([np.inf, -np.inf], np.nan) if cols else d
        g = dd.groupby(keys, dropna=False)
        agg = g[cols].agg(["mean", "std", "median"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg["n_splits"] = g.size()
        agg["n_seeds"] = g["seed"].nunique()
        if "inf_flag" in d.columns and is_interval:
            agg["n_splits_infinite"] = d.groupby(keys, dropna=False)["inf_flag"].apply(
                lambda s: int(s.fillna(False).astype(bool).sum()))
        parts.append(agg.reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["alpha"] = out["alpha"].replace(-1.0, np.nan)
    out.insert(0, "label", "exploratory")
    return out


def budget_group_summary(bud: pd.DataFrame) -> pd.DataFrame:
    """Calibration budget vs the Beta law (AUDIT_3 A3-11). Draws within a split are dependent, so:
    mean_cov = mean over seeds of the per-seed mean over draws (compared with beta_mean);
    sd_seed_draw0 = SD across seeds of draw 0 only (independent splits; compared with beta_sd);
    sd_seed_means = SD across seeds of per-seed means (smaller than beta_sd by construction);
    sd_draws_conditional_on_split = mean within-split SD across draws (conditional, NOT a Beta comparison)."""
    keys = ["target", "sensor", "population", "n_cal_wb_target", "method", "alpha"]
    b = bud.copy()
    b["population"] = b["population"].fillna("primary")
    b["n_cal_wb_target"] = b["n_cal_wb_target"].astype(str)
    b["beta_sd"] = [E.beta_coverage(int(k), float(a))["sd"] for k, a in zip(b["k_cal_rows"] if "k_cal_rows" in b
                                                                          else b["k_cal"], b["alpha"])]
    rows = []
    for key, d in b.groupby(keys, sort=True):
        per_seed = d.groupby("seed")
        seed_mean = per_seed["cov_wb"].mean()
        draw0 = d[d["draw"] == 0].set_index("seed")["cov_wb"]
        within = per_seed["cov_wb"].std(ddof=1)
        rows.append(dict(zip(keys, key), label="exploratory", n_seeds=int(d["seed"].nunique()),
                         draws_per_seed=float(per_seed.size().mean()),
                         mean_cov=float(seed_mean.mean()), beta_mean=float(per_seed["beta_mean"].mean().mean()),
                         sd_seed_draw0=float(draw0.std(ddof=1)) if len(draw0) > 1 else np.nan,
                         sd_seed_means=float(seed_mean.std(ddof=1)) if len(seed_mean) > 1 else np.nan,
                         beta_sd=float(per_seed["beta_sd"].mean().mean()),
                         beta_law_applicable=str(key[4]).endswith("gsub"),
                         sd_draws_conditional_on_split=float(within.mean()) if within.notna().any() else np.nan,
                         inf_rate_mean=float(d["inf_flag"].astype(bool).mean()),
                         mean_width_wb_log10=float(d["width_wb_log10mean"].replace([np.inf, -np.inf], np.nan).mean())))
    return pd.DataFrame(rows)


def budget_local_summary(loc: pd.DataFrame) -> pd.DataFrame:
    """Local budget: per seed, mean coverage over evaluated water bodies; then mean and SD over seeds."""
    keys = ["target", "order", "eval_set", "k", "score", "variant", "alpha"]
    per_seed = loc.groupby(keys + ["seed"]).agg(cov=("coverage", "mean"), n_wb=("wb_group", "nunique"),
                                                inf_rate=("inf_flag", "mean")).reset_index()
    out = per_seed.groupby(keys).agg(n_seeds=("seed", "nunique"), mean_cov=("cov", "mean"), sd_cov=("cov", "std"),
                                     mean_n_wb=("n_wb", "mean"), inf_rate=("inf_rate", "mean")).reset_index()
    out.insert(0, "label", "exploratory")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", action="append", required=True, help="results dir(s) with metrics_*.csv")
    ap.add_argument("--budget", action="append", default=[], help="budget results dir(s)")
    ap.add_argument("--out", default="tables")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    met = load_metrics(args.results)
    conf, comps = confirmatory_table(met)
    conf.to_csv(out / "confirmatory.csv", index=False)
    comps.to_csv(out / "confirmatory_components.csv", index=False)
    exploratory_cells(met).to_csv(out / "exploratory_cells.csv", index=False)
    for mode, fn in (("group", budget_group_summary), ("local", budget_local_summary)):
        files = [f for d in args.budget for f in sorted(glob.glob(str(Path(d) / f"budget_{mode}_*.csv")))]
        if files:
            fn(pd.concat([pd.read_csv(f) for f in files], ignore_index=True)).to_csv(
                out / f"exploratory_budget_{mode}.csv", index=False)
    print(f"[confirmatory] wrote {out / 'confirmatory.csv'} ({len(conf)} tests)")


if __name__ == "__main__":
    main()
