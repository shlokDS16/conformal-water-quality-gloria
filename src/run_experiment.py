"""Core experiment runner (prereg s2-s5, s7; Amendments 1-2).

Example:
  python -m src.run_experiment --target Chla --sensor hyp --protocol waterbody --seeds 0-1 --tag v0_smoke

Outputs per split (resumable: a split is skipped when its metrics file exists):
  results/<tag>/pred_<stem>.parquet     one row per (test row, model, method, alpha)
  results/<tag>/metrics_<stem>.csv      one row per (model, method, alpha) with every setting in columns
Stage 1 (main process, GPU): MDN ensembles for all pending splits, vectorised, cached in results/<tag>/cache.
Stage 2 (joblib processes): all other fits and every interval method, per split.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.conformal import core as C
from src.eval import metrics as E
from src.experiment_common import (CVPLUS_DROP_SEED, CVPLUS_FOLD_SEED, CVPLUS_K, CVPLUS_PERM_SEED, CVPLUS_SEED, GAUSS_CV_K, GSUB_SEED, POPULATIONS,
                                   RESULTS, build_mdn_cache, check_mdn_cache, equal_unit_folds, group_folds,
                                   iter_splits, load_job, mdn_cache_path, recal_holdout_index, split_stem)
from src.models.baselines import ConstantMedian, RidgeLogBands, empirical_models
from src.models.features import SENSORS
from src.models.lgbm import LGBM_PARAMS, LGBMPoint, LGBMQuantile

ALL_MODELS = ["const", "ridge", "emp", "lgbm", "mdn"]
ALL_METHODS = ["gauss", "native", "recal_train20", "recal_cal", "scp_pool", "scp_gsub",
               "nscp_pool", "nscp_gsub", "cqr_pool", "cqr_gsub", "cvplus"]
CONFORMAL = {"scp_pool", "scp_gsub", "nscp_pool", "nscp_gsub", "cqr_pool", "cqr_gsub"}
CODE_VERSION = "amendment4-2026-09-16"

# Calibration bookkeeping columns (AUDIT_3 A3-10). `k_cal` (legacy) = number of scores entering the
# quantile/recalibration step, whose unit depends on the method; aggregation must use the explicit columns:
#   calib_scheme : how calibration scores were formed
#   k_cal_rows   : number of scores / residuals / rows used by the calibration step
#   k_cal_units  : number of distinct water bodies contributing them
#   cv_folds     : number of CV folds (gauss 5, cvplus 10), NaN otherwise
CALIB_SCHEME = {
    "gauss": "oof_groupkfold_train", "native": "none", "recal_train20": "holdout_rows_train20",
    "recal_cal": "pooled_cal_rows", "scp_pool": "pooled_cal_rows", "nscp_pool": "pooled_cal_rows",
    "cqr_pool": "pooled_cal_rows", "scp_gsub": "one_row_per_cal_wb", "nscp_gsub": "one_row_per_cal_wb",
    "cqr_gsub": "one_row_per_cal_wb", "cvplus": "equal_unit_group_cvplus_one_row_per_wb",
}


def parse_seeds(txt: str | None):
    if txt is None or txt == "all":
        return None
    out = []
    for part in txt.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _point_factories(target, sensor, seed, models, lgbm_threads):
    f = []
    if "const" in models:
        f.append(("const", lambda: ConstantMedian()))
    if "ridge" in models:
        f.append(("ridge", lambda: RidgeLogBands(sensor)))
    if "emp" in models:
        for i, m in enumerate(empirical_models(target, sensor)):
            f.append((m.name, (lambda i=i: empirical_models(target, sensor)[i])))  # fresh instance per fit
    if "lgbm" in models:
        f.append(("lgbm", lambda: LGBMPoint(sensor, config.model_seed(seed), n_jobs=lgbm_threads)))
    return f


def _oof_predictions(factory, df, y, groups, K):
    """Out-of-fold predictions from K group folds within the given rows (Gaussian sigma, Amendment 2)."""
    fold = group_folds(groups, K)
    oof = np.full(len(df), np.nan)
    for k in range(K):
        tr, va = fold != k, fold == k
        m = factory().fit(df[tr], y[tr], groups[tr])
        oof[va] = m.predict(df[va])
    return oof


def process_split(sp: pd.DataFrame, settings: dict) -> dict:
    t_split = time.perf_counter()
    target, sensor, protocol = settings["target"], settings["sensor"], settings["protocol"]
    seed, fold = int(sp["seed"].iloc[0]), int(sp["fold"].iloc[0])
    out_dir = Path(settings["out_dir"])
    stem = split_stem(target, sensor, protocol, seed, fold, settings["population"], settings["noise_tag"])
    mpath = out_dir / f"metrics_{stem}.csv"
    if mpath.exists():
        return {"stem": stem, "skipped": True}
    import lightgbm  # noqa: F401  (import inside worker)

    models, methods, alphas = settings["models"], set(settings["methods"]), settings["alphas"]
    desc = bool(sp["descriptive_only"].iloc[0])
    tr = sp[sp["role"] == "train"].reset_index(drop=True)
    ca = sp[sp["role"] == "cal"].reset_index(drop=True)
    te = sp[sp["role"] == "test"].reset_index(drop=True)
    ytr, yca, yte = (d["y"].to_numpy(float) for d in (tr, ca, te))
    gtr, gca, gte = (d["wb_group"].to_numpy() for d in (tr, ca, te))
    base = {
        "target": target, "sensor": sensor, "protocol": protocol, "population": settings["population"],
        "noise_mult": settings["noise_mult"], "noise_add": settings["noise_add"], "seed": seed, "fold": fold,
        "test_unit": str(sp["test_unit"].iloc[0]), "descriptive_only": desc, "tag": settings["tag"],
        "n_train": len(tr), "n_cal": len(ca), "n_test": len(te),
        "n_train_wb": len(np.unique(gtr)), "n_cal_wb": len(np.unique(gca)), "n_test_wb": len(np.unique(gte)),
        "n_rows_nonpositive": int(sp["nonpos_any"].sum()),
        "n_rows_nonpositive_test": int(te["nonpos_any"].sum()),
        "model_seed": config.model_seed(seed), "code_version": CODE_VERSION,
    }
    preds: dict[str, dict] = {}
    t_model: dict[str, float] = {}
    need_gauss = ("gauss" in methods) and not desc

    # ---- point models other than MDN
    for name, fac in _point_factories(target, sensor, seed, models, settings["lgbm_threads"]):
        t0 = time.perf_counter()
        m = fac().fit(tr, ytr, gtr)
        p = {"train": m.predict(tr), "cal": m.predict(ca), "test": m.predict(te)}
        # empirical baselines: test rows with invalid band inputs that fell back to the training median (A3-8)
        p["n_fallback_test"] = getattr(m, "n_fallback_", np.nan)
        if need_gauss:
            p["oof"] = _oof_predictions(fac, tr, ytr, gtr, GAUSS_CV_K)
        preds[name] = p
        t_model[name] = time.perf_counter() - t0

    # ---- MDN from cache
    if "mdn" in models:
        cpath = mdn_cache_path(out_dir, stem)
        z = np.load(cpath, allow_pickle=False)
        # AUDIT_3 A3-8: raise (never skip) when a requested MDN method lacks cache arrays or the cfg differs
        check_mdn_cache(cpath, settings.get("mdn_cfg"), need_cv=need_gauss,
                        need_recal=("recal_train20" in methods) and not desc, z=z)
        assert (z["test__GLORIA_ID"] == te["GLORIA_ID"].to_numpy().astype(str)).all(), "MDN cache row order mismatch"
        assert (z["cal__GLORIA_ID"] == ca["GLORIA_ID"].to_numpy().astype(str)).all()
        assert (z["train__GLORIA_ID"] == tr["GLORIA_ID"].to_numpy().astype(str)).all()
        p = {"train": z["train__point"], "cal": z["cal__point"], "test": z["test__point"], "z": z,
             "n_fallback_test": np.nan}
        if need_gauss:
            p["oof"] = z["train__oof_point"]
        preds["mdn"] = p
        t_model["mdn"] = json.loads(str(z["meta__json"]))["fit_s_share"]

    # ---- LightGBM quantile models (CQR) and CV+ fold models
    qmods = {}
    if "lgbm" in models and not desc and ({"cqr_pool", "cqr_gsub"} & methods):
        t0 = time.perf_counter()
        for a in alphas:
            for q in (a / 2, 1 - a / 2):
                mq = LGBMQuantile(sensor, config.model_seed(seed), q, n_jobs=settings["lgbm_threads"]).fit(tr, ytr)
                qmods[(a, q)] = {"cal": mq.predict(ca), "test": mq.predict(te)}
        t_model["lgbm_quantiles"] = time.perf_counter() - t0
    cvp = None
    if "lgbm" in models and not desc and "cvplus" in methods:
        t0 = time.perf_counter()
        pool = pd.concat([tr, ca], ignore_index=True)  # CV+ uses all non-test rows (train + cal water bodies)
        # Amendment 4 item 1: drop n mod K randomly chosen water bodies (rng 5000 + s) from the CV+ pool so
        # that every fold holds exactly n/K units and Barber et al. (2021) Theorem 4 applies exactly.
        units = np.unique(pool["wb_group"].to_numpy())
        n_drop = len(units) % CVPLUS_K
        if n_drop:
            dropped = np.random.default_rng(CVPLUS_DROP_SEED + seed).choice(units, n_drop, replace=False)
            pool = pool[~pool["wb_group"].isin(dropped)].reset_index(drop=True)
        ypool, gpool = pool["y"].to_numpy(float), pool["wb_group"].to_numpy()
        # AUDIT_3 A3-1: K folds of water bodies with unit counts differing by at most 1 (rng 4000 + s)
        pf = equal_unit_folds(gpool, CVPLUS_K, np.random.default_rng(CVPLUS_FOLD_SEED + seed))
        oof = np.full(len(pool), np.nan)
        mu_test = np.zeros((CVPLUS_K, len(te)))
        for k in range(CVPLUS_K):
            if settings.get("cvplus_perm", False):
                # Amendment 5: uniformly random row order per fold fit (k counted from 0), so the fitted model is
                # distributionally symmetric in its training data despite LightGBM row subsampling.
                trk = np.random.default_rng(CVPLUS_PERM_SEED + 100 * seed + k).permutation(np.flatnonzero(pf != k))
                mk = LGBMPoint(sensor, config.model_seed(seed), n_jobs=settings["lgbm_threads"]).fit(
                    pool.iloc[trk].reset_index(drop=True), ypool[trk])
            else:
                mk = LGBMPoint(sensor, config.model_seed(seed), n_jobs=settings["lgbm_threads"]).fit(pool[pf != k], ypool[pf != k])
            oof[pf == k] = mk.predict(pool[pf == k])
            mu_test[k] = mk.predict(te)
        idx = C.group_subsample(gpool, np.random.default_rng(CVPLUS_SEED + seed))
        units_per_fold = np.bincount(pf[idx], minlength=CVPLUS_K)
        assert units_per_fold.min() == units_per_fold.max(), units_per_fold  # exactly equal (Amendment 4)
        cvp = {"oof": oof, "y": ypool, "fold": pf, "mu_test": mu_test, "idx": idx, "n_wb": len(idx), "n_drop": int(n_drop),
               "units_per_fold": units_per_fold}
        t_model["lgbm_cvplus"] = time.perf_counter() - t0

    base["time_lgbm_quantiles_s"] = t_model.get("lgbm_quantiles", np.nan)
    base["time_lgbm_cvplus_s"] = t_model.get("lgbm_cvplus", np.nan)
    base["n_cvplus_wb"] = cvp["n_wb"] if cvp is not None else np.nan
    base["cvplus_fold_units"] = ";".join(map(str, cvp["units_per_fold"])) if cvp is not None else ""
    base["n_cvplus_wb_dropped"] = cvp["n_drop"] if cvp is not None else np.nan
    if settings.get("cvplus_perm", False):
        base["cvplus_row_perm"] = True  # column only in cvplus_v3 outputs; core_v2 schema unchanged
    n_r80_rows = n_r80_wb = np.nan
    if "mdn" in models and "recal_train20" in methods and not desc:
        hold = recal_holdout_index(len(tr), seed)
        n_r80_rows, n_r80_wb = int(hold.sum()), len(np.unique(gtr[hold]))

    def calib_columns(method, info):
        if method in ("scp_gsub", "nscp_gsub", "cqr_gsub"):
            rows, units, folds = len(gsub), len(np.unique(gca[gsub])), np.nan
        elif method in ("scp_pool", "nscp_pool", "cqr_pool", "recal_cal"):
            rows, units, folds = len(ca), base["n_cal_wb"], np.nan
        elif method == "gauss":
            rows, units, folds = len(tr), base["n_train_wb"], GAUSS_CV_K
        elif method == "recal_train20":
            rows, units, folds = n_r80_rows, n_r80_wb, np.nan
        elif method == "cvplus":
            rows, units, folds = cvp["n_wb"], cvp["n_wb"], CVPLUS_K
        else:  # native: no calibration step
            rows, units, folds = np.nan, np.nan, np.nan
        if method != "native":
            assert rows == info.get("n_cal_units"), (method, rows, info.get("n_cal_units"))
        return {"calib_scheme": CALIB_SCHEME[method], "k_cal_rows": rows, "k_cal_units": units, "cv_folds": folds}

    # ---- intervals and metrics
    gsub = C.group_subsample(gca, np.random.default_rng(GSUB_SEED + seed)) if len(ca) else np.array([], int)
    pred_rows, met_rows = [], []

    def emit(model, method, alpha, lo, hi, point, info, t_method):
        row = dict(base, model=model, method=method, alpha=alpha, time_model_s=t_model.get(model, np.nan),
                   time_method_s=t_method, n_fallback_test=preds[model]["n_fallback_test"])
        # A3-7: point metrics on every row are those of the full-train point model (for cvplus rows too,
        # not the CV+ fold models); read point accuracy only from method == "point" rows.
        row.update(E.point_metrics(yte, point))
        if lo is not None:
            row.update(E.interval_metrics(yte, lo, hi, gte, alpha))
            row.update(qhat=info.get("qhat", np.nan), inf_flag=bool(info.get("inf", False)),
                       k_cal=int(info.get("n_cal_units", -1)))
            row.update(calib_columns(method, info))
            if settings["boot"] > 0 and not info.get("inf", False):
                row.update(E.cluster_bootstrap_coverage(yte, lo, hi, gte, B=settings["boot"],
                                                        seed=config.model_seed(seed)))
            if method in CONFORMAL:
                bc = E.beta_coverage(info["n_cal_units"], alpha)
                row.update(beta_mean=bc["mean"], beta_q_lo=bc["q_lo"], beta_q_hi=bc["q_hi"])
            if method == "cvplus":
                row.update(cvplus_bound=C.cvplus_bound(info["n_cal_units"], CVPLUS_K, alpha))
        met_rows.append(row)
        pred_rows.append(pd.DataFrame({
            "GLORIA_ID": te["GLORIA_ID"].to_numpy(), "wb_group": gte, "chl_technique": te["chl_technique"].to_numpy(),
            "y": yte.astype(np.float32), "model": model, "method": method, "alpha": np.float32(alpha),
            "point": np.asarray(point, np.float32),
            "lower": (np.full(len(te), np.nan) if lo is None else np.asarray(lo)).astype(np.float32),
            "upper": (np.full(len(te), np.nan) if hi is None else np.asarray(hi)).astype(np.float32),
        }))

    for model, p in preds.items():
        emit(model, "point", np.nan, None, None, p["test"], {}, 0.0)
        if desc:
            continue
        for a in alphas:
            def timed(fn):
                t0 = time.perf_counter()
                lo, hi, info = fn()
                return lo, hi, info, time.perf_counter() - t0

            if "gauss" in methods:  # every model has p["oof"] when gauss is requested (MDN: checked cache)
                emit(model, "gauss", a, *_unpack(timed(lambda: C.gaussian_residual(p["oof"], ytr, p["test"], a)), p["test"]))
            if "scp_pool" in methods:
                emit(model, "scp_pool", a, *_unpack(timed(lambda: C.split_conformal(p["cal"], yca, p["test"], a)), p["test"]))
            if "scp_gsub" in methods:
                emit(model, "scp_gsub", a, *_unpack(timed(lambda: C.split_conformal(p["cal"], yca, p["test"], a, gsub)), p["test"]))
            if model == "mdn":
                z = p["z"]
                lv_lo, lv_hi = f"q{a / 2:.3f}", f"q{1 - a / 2:.3f}"
                if "native" in methods:
                    emit(model, "native", a, z[f"test__{lv_lo}"], z[f"test__{lv_hi}"], p["test"],
                         {"inf": False, "n_cal_units": -1}, 0.0)
                if "nscp_pool" in methods:
                    emit(model, "nscp_pool", a, *_unpack(timed(lambda: C.normalized_split_conformal(
                        z["cal__point"], z["cal__mix_sd"], yca, z["test__point"], z["test__mix_sd"], a)), p["test"]))
                if "nscp_gsub" in methods:
                    emit(model, "nscp_gsub", a, *_unpack(timed(lambda: C.normalized_split_conformal(
                        z["cal__point"], z["cal__mix_sd"], yca, z["test__point"], z["test__mix_sd"], a, gsub)), p["test"]))
                if "recal_train20" in methods:  # cache contents were checked above (raises if missing)
                    emit(model, "recal_train20", a, *_unpack(timed(lambda: C.quantile_recalibration(
                        z["r80hold__mix_mean"], z["r80hold__mix_sd"], z["r80hold__y"],
                        z["r80test__mix_mean"], z["r80test__mix_sd"], a)), p["test"]))
                if "recal_cal" in methods:
                    emit(model, "recal_cal", a, *_unpack(timed(lambda: C.quantile_recalibration(
                        z["cal__mix_mean"], z["cal__mix_sd"], yca, z["test__mix_mean"], z["test__mix_sd"], a)), p["test"]))
            if model == "lgbm":
                if qmods:
                    ql, qh = qmods[(a, a / 2)], qmods[(a, 1 - a / 2)]
                    if "cqr_pool" in methods:
                        emit(model, "cqr_pool", a, *_unpack(timed(lambda: C.cqr(ql["cal"], qh["cal"], yca, ql["test"], qh["test"], a)), p["test"]))
                    if "cqr_gsub" in methods:
                        emit(model, "cqr_gsub", a, *_unpack(timed(lambda: C.cqr(ql["cal"], qh["cal"], yca, ql["test"], qh["test"], a, gsub)), p["test"]))
                if cvp is not None:
                    emit(model, "cvplus", a, *_unpack(timed(lambda: C.cv_plus(
                        cvp["oof"], cvp["y"], cvp["fold"], cvp["mu_test"], a, cvp["idx"])), p["test"]))

    pred = pd.concat(pred_rows, ignore_index=True)
    for c in ("model", "method", "wb_group", "chl_technique"):
        pred[c] = pred[c].astype("category")
    met = pd.DataFrame(met_rows)
    met["time_split_s"] = time.perf_counter() - t_split
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_p, tmp_m = out_dir / f"pred_{stem}.parquet.tmp", out_dir / f"metrics_{stem}.csv.tmp"
    pred.to_parquet(tmp_p, index=False)
    met.to_csv(tmp_m, index=False)
    os.replace(tmp_p, out_dir / f"pred_{stem}.parquet")
    os.replace(tmp_m, mpath)  # metrics file written last = completion marker
    return {"stem": stem, "skipped": False, "time_s": time.perf_counter() - t_split, "n_rows": len(pred)}


def _unpack(res, point):
    lo, hi, info, t = res
    return lo, hi, point, info, t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=config.RAW_TARGETS)
    ap.add_argument("--sensor", default="hyp", choices=SENSORS)
    ap.add_argument("--protocol", required=True,
                    choices=["random", "waterbody", "waterbody_5km", "contributor", "contributor_ds", "region", "region_na"])
    ap.add_argument("--seeds", default="all", help="e.g. 0-19 or 0,3,5 or all")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--alpha", default="0.05,0.10,0.20")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-jobs", type=int, default=6, help="parallel split workers (processes)")
    ap.add_argument("--lgbm-threads", type=int, default=0, help="threads per LightGBM fit (0: 24 // n_jobs)")
    ap.add_argument("--population", default="primary", choices=list(POPULATIONS))
    ap.add_argument("--noise-mult", type=float, default=0.0)
    ap.add_argument("--noise-add", type=float, default=0.0)
    ap.add_argument("--boot", type=int, default=2000, help="cluster bootstrap resamples (0 = off)")
    ap.add_argument("--cvplus-perm", action="store_true",
                    help="Amendment 5: permute each CV+ fold model's training rows (seed 6000 + 100 s + k); tag cvplus_v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mdn-chunk", type=int, default=400)
    ap.add_argument("--mdn-n-iter", type=int, default=None, help="override MDN n_iter (smoke tests only)")
    ap.add_argument("--max-splits", type=int, default=None)
    args = ap.parse_args(argv)

    if os.environ.get("PYTHONHASHSEED") is None:
        print("[warn] PYTHONHASHSEED not set; set it in the launcher (prereg s9). Results do not use hash().")
    models = [m for m in args.models.split(",") if m]
    methods = [m for m in args.methods.split(",") if m]
    alphas = [float(a) for a in args.alpha.split(",")]
    seeds = parse_seeds(args.seeds)
    lgbm_threads = args.lgbm_threads or max(1, 24 // max(1, args.n_jobs))
    out_dir = RESULTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_tag = "" if args.noise_mult == 0 and args.noise_add == 0 else f"_nm{args.noise_mult:g}_na{args.noise_add:g}"

    t0 = time.perf_counter()
    df = load_job(args.target, args.sensor, args.protocol, args.population, args.noise_mult, args.noise_add)
    splits = list(iter_splits(df, seeds))
    if args.max_splits:
        splits = splits[: args.max_splits]
    jobs = []
    for s, f, sp in splits:
        stem = split_stem(args.target, args.sensor, args.protocol, s, f, args.population, noise_tag)
        if (out_dir / f"metrics_{stem}.csv").exists():
            continue
        jobs.append((stem, s, f, sp, bool(sp["descriptive_only"].iloc[0])))
    print(f"[run] {args.target}/{args.sensor}/{args.protocol}/{args.population}{noise_tag}: "
          f"{len(splits)} splits, {len(jobs)} pending; load {time.perf_counter() - t0:.1f}s", flush=True)
    run_info = {"argv": sys.argv, "python": platform.python_version(), "lgbm_params": LGBM_PARAMS,
                "lgbm_threads": lgbm_threads, "code_version": CODE_VERSION}
    if not jobs:
        return
    mdn_timing = {}
    mdn_cfg = None
    if "mdn" in models:
        from src.models.mdn import MDN_DEFAULTS
        cfg = {"n_iter": args.mdn_n_iter} if args.mdn_n_iter else None
        mdn_cfg = {**MDN_DEFAULTS, **(cfg or {})}
        mdn_timing = build_mdn_cache(jobs, args.sensor, out_dir, alphas, device=args.device, chunk=args.mdn_chunk,
                                     cfg=cfg, with_recal="recal_train20" in methods, with_cv="gauss" in methods)
    settings = dict(target=args.target, sensor=args.sensor, protocol=args.protocol, population=args.population,
                    noise_mult=args.noise_mult, noise_add=args.noise_add, noise_tag=noise_tag, tag=args.tag,
                    out_dir=str(out_dir), models=models, methods=methods, alphas=alphas,
                    lgbm_threads=lgbm_threads, boot=args.boot, mdn_cfg=mdn_cfg, cvplus_perm=args.cvplus_perm)
    t1 = time.perf_counter()
    if args.n_jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(process_split)(sp, settings) for _, _, _, sp, _ in jobs)
    else:
        res = [process_split(sp, settings) for _, _, _, sp, _ in jobs]
    cpu_s = time.perf_counter() - t1
    run_info.update(mdn_timing=mdn_timing, cpu_stage_s=cpu_s, n_jobs=args.n_jobs, splits=[r["stem"] for r in res])
    with open(out_dir / f"runinfo_{args.target}_{args.sensor}_{args.protocol}_{args.population}{noise_tag}_{int(time.time())}.json", "w") as fh:
        json.dump(run_info, fh, indent=1, default=str)
    print(f"[run] done: MDN {mdn_timing.get('fit_s', 0):.1f}s fit; CPU stage {cpu_s:.1f}s for {len(res)} splits "
          f"(n_jobs {args.n_jobs}, lgbm threads {lgbm_threads})", flush=True)


if __name__ == "__main__":
    main()
