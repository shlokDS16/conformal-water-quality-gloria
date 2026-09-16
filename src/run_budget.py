"""Calibration budget runner (prereg s6; Amendment 2 conventions).

Group-level (mode group): LightGBM split conformal (group subsampled, plus pooled for contrast) and CQR
(group subsampled and pooled), waterbody protocol. For each split and n_cal in {10,15,20,30,40,all},
R draws of calibration water bodies (rng keyed by (2000+s, n_cal, r)); within a draw, one row per
water body (single subsampling). Metrics: water-body-averaged coverage, width, infinite-interval rate,
Beta law for the realised k.

Local time-forward (mode local; Chl-a and TSS): the k earliest-dated samples (k in {0,1,2,5,10,20}) of
each test water body join calibration; evaluation on that body's later samples only. Variants:
  global   : group-subsampled global calibration scores only (reference, k ignored)
  augment  : global group-subsampled scores + the k local scores
  local    : Mondrian by test water body, the k local scores only (infinite if k too small)
  hybrid   : local when k >= ceil(1/alpha) - 1, else augment
Evaluation sets: `common` = bodies with >= 20 + min_eval dated samples, evaluated after the 20th sample
for every k (paired across k); `per_k` = bodies with >= k + min_eval samples. `--order random` picks k
random samples instead of the earliest (sensitivity).

Output: results/<tag>/budget_group_<stem>.csv, results/<tag>/budget_local_<order>_<stem>.csv (resumable).

Dependence of draws (AUDIT_3 A3-11). Within one split, the R draws for a given n_cal share the fitted
LightGBM models, the calibration pool and the test set; they are NOT independent replicates. The spread of
coverage across draws is conditional on the split and must not be compared with the Beta law. The
aggregation (src/analysis/confirmatory.py, budget_group_summary) compares (i) the mean over seeds of the
per-seed mean coverage with the Beta mean, and (ii) the SD across the 20 independent split seeds of ONE
draw per seed (draw 0) and of the per-seed means with the Beta SD (the per-seed-mean SD is expected to be
smaller than the Beta SD because averaging over draws removes part of the calibration variability).
Draw-level SD within a split is reported only with the label "conditional on split".

Columns `k_cal_rows` (scores used in the quantile) and `k_cal_units` (distinct calibration water bodies)
make the calibration count unambiguous (AUDIT_3 A3-10); legacy `k_cal` = `k_cal_rows`.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.conformal import core as C
from src.eval import metrics as E
from src.experiment_common import GSUB_SEED, RESULTS, iter_splits, load_job, rng_for, split_stem
from src.models.lgbm import LGBMPoint, LGBMQuantile
from src.run_experiment import parse_seeds

N_CAL_GRID = [10, 15, 20, 30, 40, "all"]
K_GRID = [0, 1, 2, 5, 10, 20]


def _fit_lgbm(tr, sensor, seed, alpha, threads):
    ytr = tr["y"].to_numpy(float)
    pt = LGBMPoint(sensor, config.model_seed(seed), n_jobs=threads).fit(tr, ytr)
    ql = LGBMQuantile(sensor, config.model_seed(seed), alpha / 2, n_jobs=threads).fit(tr, ytr)
    qh = LGBMQuantile(sensor, config.model_seed(seed), 1 - alpha / 2, n_jobs=threads).fit(tr, ytr)
    return pt, ql, qh


def _scores(df, pt, ql, qh):
    y = df["y"].to_numpy(float)
    p, lo, hi = pt.predict(df), ql.predict(df), qh.predict(df)
    return {"p": p, "lo": lo, "hi": hi, "s_abs": np.abs(y - p), "s_cqr": np.maximum(lo - y, y - hi), "y": y}


def _intervals(q_abs, q_cqr, te):
    return {"scp": (te["p"] - q_abs, te["p"] + q_abs), "cqr": (te["lo"] - q_cqr, te["hi"] + q_cqr)}


def group_budget(sp, settings, R):
    seed, fold = int(sp["seed"].iloc[0]), int(sp["fold"].iloc[0])
    alpha = settings["alpha"]
    tr, ca, te = (sp[sp["role"] == r].reset_index(drop=True) for r in ("train", "cal", "test"))
    t0 = time.perf_counter()
    pt, ql, qh = _fit_lgbm(tr, settings["sensor"], seed, alpha, settings["threads"])
    sc, st = _scores(ca, pt, ql, qh), _scores(te, pt, ql, qh)
    gte, gca = te["wb_group"].to_numpy(), ca["wb_group"].to_numpy()
    groups = np.unique(gca)
    rows = []
    for n_cal in N_CAL_GRID:
        n_draw = len(groups) if n_cal == "all" else int(n_cal)
        if n_draw > len(groups):
            continue
        reps = 1 if n_cal == "all" else R
        for r in range(reps):
            rng = rng_for(GSUB_SEED + seed, str(n_cal), r)
            chosen = groups if n_cal == "all" else rng.choice(groups, n_draw, replace=False)
            mask = np.isin(gca, chosen)
            sub = np.flatnonzero(mask)
            idx = sub[C.group_subsample(gca[sub], np.random.default_rng(GSUB_SEED + seed) if n_cal == "all" else rng)]
            for scheme, ids in (("gsub", idx), ("pool", sub)):
                q_abs, inf_a = C.conformal_quantile(sc["s_abs"][ids], alpha)
                q_cqr, inf_c = C.conformal_quantile(sc["s_cqr"][ids], alpha)
                for meth, (lo, hi), inf in ((f"scp_{scheme}", _intervals(q_abs, q_cqr, st)["scp"], inf_a),
                                            (f"cqr_{scheme}", _intervals(q_abs, q_cqr, st)["cqr"], inf_c)):
                    row = dict(settings["base"], seed=seed, fold=fold, n_cal_wb_target=str(n_cal), draw=r,
                               method=meth, alpha=alpha, k_cal=len(ids), k_cal_rows=len(ids),
                               k_cal_units=len(np.unique(gca[ids])),
                               calib_scheme="one_row_per_cal_wb" if scheme == "gsub" else "pooled_cal_rows",
                               n_cal_wb=n_draw,
                               n_test_wb=len(np.unique(gte)), inf_flag=inf)
                    row.update(E.interval_metrics(st["y"], lo, hi, gte, alpha))
                    bc = E.beta_coverage(len(ids), alpha)
                    row.update(beta_mean=bc["mean"], beta_q_lo=bc["q_lo"], beta_q_hi=bc["q_hi"])
                    rows.append(row)
    out = pd.DataFrame(rows)
    out["time_split_s"] = time.perf_counter() - t0
    return out


def local_budget(sp, settings, order, min_eval):
    seed, fold = int(sp["seed"].iloc[0]), int(sp["fold"].iloc[0])
    alpha = settings["alpha"]
    tr, ca, te = (sp[sp["role"] == r].reset_index(drop=True) for r in ("train", "cal", "test"))
    t0 = time.perf_counter()
    pt, ql, qh = _fit_lgbm(tr, settings["sensor"], seed, alpha, settings["threads"])
    sc = _scores(ca, pt, ql, qh)
    gca = ca["wb_group"].to_numpy()
    gidx = C.group_subsample(gca, np.random.default_rng(GSUB_SEED + seed))
    te = te.assign(_t=pd.to_datetime(te["Date_Time_UTC"], errors="coerce"))
    if order == "earliest":
        te = te[te["_t"].notna()]
    st_all = _scores(te, pt, ql, qh)
    te = te.assign(**{f"_{k}": v for k, v in st_all.items()})
    kmax = max(K_GRID)
    k_mondrian = math.ceil(1 / alpha) - 1
    rows = []
    for g, body in te.groupby("wb_group", sort=True):
        if order == "earliest":
            body = body.sort_values(["_t", "GLORIA_ID"])
        else:
            body = body.iloc[rng_for("local_random", GSUB_SEED + seed, str(g)).permutation(len(body))]
        n = len(body)
        for k in K_GRID:
            for eval_set in ("common", "per_k"):
                start = kmax if eval_set == "common" else k
                if n < start + min_eval:
                    continue
                loc, ev = body.iloc[:k], body.iloc[start:]
                ye = ev["_y"].to_numpy()
                for score, glob in (("scp", sc["s_abs"][gidx]), ("cqr", sc["s_cqr"][gidx])):
                    lscore = (loc["_s_abs"] if score == "scp" else loc["_s_cqr"]).to_numpy()
                    variants = {"global": glob, "augment": np.r_[glob, lscore], "local": lscore,
                                "hybrid": lscore if k >= k_mondrian else np.r_[glob, lscore]}
                    for vname, s in variants.items():
                        q, inf = C.conformal_quantile(s, alpha)
                        if score == "scp":
                            lo, hi = ev["_p"].to_numpy() - q, ev["_p"].to_numpy() + q
                        else:
                            lo, hi = ev["_lo"].to_numpy() - q, ev["_hi"].to_numpy() + q
                        cover = float(np.mean((ye >= lo) & (ye <= hi)))
                        rows.append(dict(settings["base"], seed=seed, fold=fold, order=order, eval_set=eval_set,
                                         wb_group=g, k=k, score=score, variant=vname, alpha=alpha,
                                         k_cal=len(s), inf_flag=inf, n_eval=len(ev), coverage=cover,
                                         width_median=float(np.median(10 ** (hi - lo))) if not inf else np.inf))
    out = pd.DataFrame(rows)
    if len(out):
        out["time_split_s"] = time.perf_counter() - t0
    return out


def _work(sp, settings, mode, R, order, min_eval):
    seed, fold = int(sp["seed"].iloc[0]), int(sp["fold"].iloc[0])
    stem = split_stem(settings["target"], settings["sensor"], "waterbody", seed, fold, settings["population"])
    out_dir = Path(settings["out_dir"])
    path = out_dir / (f"budget_group_{stem}.csv" if mode == "group" else f"budget_local_{order}_{stem}.csv")
    if path.exists():
        return str(path), True
    df = group_budget(sp, settings, R) if mode == "group" else local_budget(sp, settings, order, min_eval)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return str(path), False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["group", "local"])
    ap.add_argument("--target", required=True, choices=config.RAW_TARGETS)
    ap.add_argument("--sensor", default="hyp", choices=["hyp", "msi", "olci", "msi_strict", "olci_strict"])
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--draws", type=int, default=20, help="calibration-group draws per split and n_cal")
    ap.add_argument("--order", default="earliest", choices=["earliest", "random"])
    ap.add_argument("--min-eval", type=int, default=5)
    ap.add_argument("--population", default="primary")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args(argv)
    if args.mode == "local" and args.target not in ("Chla", "TSS"):
        raise SystemExit("prereg s6.2: local budget only for Chla and TSS")
    out_dir = RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_job(args.target, args.sensor, "waterbody", args.population)
    splits = [sp for _, _, sp in iter_splits(df, parse_seeds(args.seeds))]
    threads = max(1, 24 // max(1, args.n_jobs))
    settings = dict(target=args.target, sensor=args.sensor, population=args.population, alpha=args.alpha,
                    out_dir=str(out_dir), threads=threads,
                    base=dict(target=args.target, sensor=args.sensor, protocol="waterbody",
                              population=args.population, tag=args.tag))
    t0 = time.perf_counter()
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(_work)(sp, settings, args.mode, args.draws, args.order, args.min_eval) for sp in splits)
    print(f"[budget] {args.mode} {args.target}/{args.sensor}: {len(res)} splits "
          f"({sum(not s for _, s in res)} new) in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
