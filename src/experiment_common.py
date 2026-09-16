"""Shared data loading, split iteration and MDN caching for run_experiment.py and run_budget.py."""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.models.features import STRICT_SENSORS, complete_column, complete_mask, feature_columns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

POPULATIONS = {
    "primary": None,
    "strict_qc": "qc_strict",               # drop all Flagged (prereg s7)
    "chla_hplc": "chla_hplc_or_corrected",  # Chl-a HPLC-or-phaeophytin-corrected subset
    "depth_ge3": "~shallow_lt3m",           # exclude Depth < 3 m
    "exclude_nonpositive": "~nonpos_any",   # exploratory: rows with any band <= 0 removed (Amendment 2)
    "secchi_has_depth": "has_depth",        # Secchi rows with a recorded Depth > 0 (A2.8 sensitivity; AUDIT_3 A3-4)
}
POPULATION_TARGETS = {"chla_hplc": {"Chla"}, "secchi_has_depth": {"Secchi_depth"}}

META_COLS = ["GLORIA_ID", "wb_group", "iid_unit", "Dataset_ID", "Date_Time_UTC", "Depth",
             "qc_strict", "chla_hplc_or_corrected", "shallow_lt3m"]


def rng_for(*keys: int | str) -> np.random.Generator:
    """Deterministic generator from a tuple of keys (independent of PYTHONHASHSEED)."""
    ints = [k if isinstance(k, int) else zlib.crc32(str(k).encode()) for k in keys]
    return np.random.default_rng([int(i) & 0xFFFFFFFF for i in ints])


def load_job(target: str, sensor: str, protocol: str, population: str = "primary",
             noise_mult: float = 0.0, noise_add: float = 0.0) -> pd.DataFrame:
    """Rows of the split file for one protocol joined to GLORIA features and target.

    One shared row set per cell (Amendment 2 item 2): complete features for the sensor and a finite
    log10 target; optional population subset. Order of operations on features: optional Rrs noise
    (fixed per GLORIA row), flag `nonpos_any` (any band <= 0), floor every band at LOG_FLOOR.
    All models then see identical, floored inputs.
    """
    from src.models.features import LOG_FLOOR

    if population not in POPULATIONS:
        raise ValueError(f"unknown population {population}")
    if population in POPULATION_TARGETS and target not in POPULATION_TARGETS[population]:
        raise ValueError(f"population {population} is defined only for {sorted(POPULATION_TARGETS[population])}")
    feats = feature_columns(sensor)
    ycol = f"log10_{target}"
    ccol = complete_column(sensor)
    g = pd.read_parquet(DATA / "gloria.parquet",
                        columns=META_COLS + ["chl_technique", "wb_5km", ycol] + ([ccol] if ccol else []) + feats)
    if protocol == "waterbody_5km":
        # 5 km sensitivity: the water-body unit (subsampling, CV+ and Gaussian folds, metrics) is wb_5km.
        g["wb_group"] = g["wb_5km"]
    g = g.drop(columns="wb_5km")
    # completeness is fixed on the raw spectra, before noise (strict sets: all strict bands present)
    g["_complete"] = complete_mask(g, sensor)
    g["has_depth"] = g["Depth"].notna() & (g["Depth"] > 0)
    g = apply_noise(g, sensor, noise_mult, noise_add)
    X = g[feats].to_numpy(float)
    g["nonpos_any"] = (X <= 0).any(1) & g["_complete"].to_numpy()
    g[feats] = np.where(np.isnan(X), np.nan, np.maximum(X, LOG_FLOOR))
    prefix = "splits5km" if protocol == "waterbody_5km" else "splits"  # 5 km sensitivity (src/data/splits_5km.py)
    s = pd.read_parquet(DATA / f"{prefix}_{target}.parquet")
    s = s[s["protocol"] == protocol].copy()
    s["protocol"] = s["protocol"].astype(str)
    s["test_unit"] = s["test_unit"].astype(str)
    s["role"] = s["role"].astype(str)
    d = s.merge(g, on="GLORIA_ID", how="left", validate="many_to_one")
    keep = d["_complete"].fillna(False).astype(bool) & np.isfinite(d[ycol])
    pop = POPULATIONS[population]
    if pop is not None:
        col = pop.lstrip("~")
        flag = d[col].fillna(False).astype(bool)
        keep &= ~flag if pop.startswith("~") else flag
    d = d[keep].reset_index(drop=True)
    d["chl_technique"] = d["chl_technique"].fillna("missing").astype(str)
    d = d.rename(columns={ycol: "y"})
    d.attrs.update(target=target, sensor=sensor, protocol=protocol, population=population,
                   noise_mult=noise_mult, noise_add=noise_add)
    return d


def apply_noise(g: pd.DataFrame, sensor: str, mult: float, add: float) -> pd.DataFrame:
    """Rrs noise sweep (prereg s7, Amendment 3): Rrs' = Rrs exp(mult N1) + add N2, independent per row and band.

    `mult` is the SD of the log-ratio ln(Rrs'/Rrs), matching the ACIX-Aqua median symmetric accuracy
    epsilon via mult = ln(1 + epsilon) / 0.6745 (research/NOISE_AND_5KM_LOG.md); a linear (1 + mult N1)
    factor would turn about 8 % of values negative at mult = 0.70. One fixed noisy realisation per
    (mult, add), identical across splits. Applied before flooring.
    """
    if mult == 0 and add == 0:
        return g
    cols = feature_columns(sensor)
    rng = rng_for("noise", sensor, int(round(mult * 1e6)), int(round(add * 1e9)))
    X = g[cols].to_numpy(float)
    X = X * np.exp(mult * rng.standard_normal(X.shape)) + add * rng.standard_normal(X.shape)
    out = g.copy()
    out[cols] = X
    return out


def iter_splits(df: pd.DataFrame, seeds: list[int] | None):
    keys = df[["seed", "fold"]].drop_duplicates().sort_values(["seed", "fold"])
    for seed, fold in keys.itertuples(index=False):
        if seeds is not None and int(seed) not in seeds:
            continue
        yield int(seed), int(fold), df[(df["seed"] == seed) & (df["fold"] == fold)]


def split_stem(target, sensor, protocol, seed, fold, population="primary", noise_tag=""):
    pop = "" if population == "primary" else f"_{population}"
    return f"{target}_{sensor}_{protocol}{pop}{noise_tag}_s{seed:02d}_f{fold}"


def recal_holdout_index(n_train: int, seed: int) -> np.ndarray:
    """Boolean mask: random 20 % of training rows held out for recalibration (Werther et al. 2025 style)."""
    rng = rng_for("recal20", config.model_seed(seed))
    idx = rng.permutation(n_train)[: int(round(0.2 * n_train))]
    m = np.zeros(n_train, bool)
    m[idx] = True
    return m


def group_folds(groups: np.ndarray, K: int) -> np.ndarray:
    """Fold id per row from sklearn GroupKFold(K) (deterministic). GroupKFold balances ROW counts across
    folds, not group counts (AUDIT_3 A3-6); used only for the Gaussian-sigma out-of-fold residuals and the
    ridge inner CV, where no finite-sample guarantee is claimed."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    fold = np.full(len(groups), -1, int)
    for k, (_, va) in enumerate(GroupKFold(n_splits=K).split(np.zeros(len(groups)), groups=groups)):
        fold[va] = k
    return fold


def equal_unit_folds(groups: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Fold id per row for group CV+ (AUDIT_3 A3-1 fix): K random partitions of the UNITS (water bodies)
    whose unit counts differ by at most 1, whatever the row counts per unit.

    The sorted unique labels are permuted with `rng`; the unit at permuted rank r goes to fold r mod K,
    so fold sizes are ceil(n/K) or floor(n/K). Every row inherits its unit's fold.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    n = len(uniq)
    if n < K:
        raise ValueError(f"{n} units < K = {K}")
    rank = np.empty(n, int)
    rank[rng.permutation(n)] = np.arange(n)
    unit_fold = rank % K
    return unit_fold[inv.ravel()]


GAUSS_CV_K = 5      # Amendment 2 item 3: sigma from 5-fold group-CV out-of-fold residuals within train
CVPLUS_K = 10       # prereg s3: group-K-fold CV+ with K = 10 on wb_group
GSUB_SEED = 2000    # Amendment 2 item 4: calibration group subsample rng = default_rng(2000 + s)
CVPLUS_SEED = 3000  # Amendment 2 item 1: CV+ held-out water-body subsample rng = default_rng(3000 + s)
CVPLUS_FOLD_SEED = 4000  # AUDIT_3 A3-1: CV+ equal-unit fold assignment rng = default_rng(4000 + s)
CVPLUS_DROP_SEED = 5000  # Amendment 4: CV+ drops n mod K water bodies, rng = default_rng(5000 + s)
CVPLUS_PERM_SEED = 6000  # Amendment 5: CV+ fold-k training rows permuted, rng = default_rng(6000 + 100 s + k)


# ------------------------------------------------------------------ MDN cache (GPU vectorised, across splits)
def mdn_cache_path(out_dir: Path, stem: str) -> Path:
    return out_dir / "cache" / f"mdn_{stem}.npz"


def build_mdn_cache(jobs: list[tuple[str, int, int, pd.DataFrame, bool]], sensor: str, out_dir: Path,
                    alphas, device: str = "cuda", chunk: int = 400, cfg: dict | None = None,
                    with_recal: bool = True, with_cv: bool = True, log=print) -> dict:
    """jobs: (stem, seed, fold, split_df, descriptive_only). Trains all missing MDN ensembles, writes npz caches.

    Cache content: full-train ensemble summaries for train (point), cal and test rows; if with_cv, the
    out-of-fold point prediction for every train row from GAUSS_CV_K group folds (one 10-member ensemble
    per fold); if with_recal, mixture mean/SD for the random 20 % recal holdout and the test rows from the
    ensemble trained on the other 80 % of train. Descriptive-only folds get the full-train ensemble only.
    """
    from src.models.mdn import MDN_DEFAULTS, MemberTask, ensemble_summary, fit_members

    cfg = {**MDN_DEFAULTS, **(cfg or {})}
    feats = feature_columns(sensor)
    todo = []
    for j in jobs:
        p = mdn_cache_path(out_dir, j[0])
        if p.exists():  # AUDIT_3 A3-8: an existing cache must match the requested cfg and contents
            check_mdn_cache(p, cfg, need_cv=with_cv and not j[4], need_recal=with_recal and not j[4])
        else:
            todo.append(j)
    timing = {"n_splits": len(todo), "n_members": 0, "fit_s": 0.0, "predict_s": 0.0}
    if not todo:
        return timing
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    tasks, owners = [], []
    R = cfg["n_rounds"]
    for stem, seed, fold, sp, desc in todo:
        tr = sp[sp["role"] == "train"]
        X, y = tr[feats].to_numpy(float), tr["y"].to_numpy(float)
        for m in range(R):
            tasks.append(MemberTask(X, y, config.mdn_member_seed("full", seed, m)))
            owners.append((stem, "full"))
        if desc:
            continue
        if with_recal:
            hold = recal_holdout_index(len(tr), seed)
            for m in range(R):
                tasks.append(MemberTask(X[~hold], y[~hold], config.mdn_member_seed("r80", seed, m)))
                owners.append((stem, "r80"))
        if with_cv:
            gf = group_folds(tr["wb_group"].to_numpy(), GAUSS_CV_K)
            for k in range(GAUSS_CV_K):
                for m in range(R):
                    tasks.append(MemberTask(X[gf != k], y[gf != k], config.mdn_member_seed("cv", seed, m, fold=k)))
                    owners.append((stem, f"cv{k}"))
    timing["n_members"] = len(tasks)
    t0 = time.perf_counter()
    log(f"[mdn] training {len(tasks)} members for {len(todo)} splits on {device} (chunk {chunk})")
    fitted = fit_members(tasks, device=device, cfg=cfg, chunk=chunk)
    timing["fit_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    by: dict = {}
    for key, f in zip(owners, fitted):
        by.setdefault(key, []).append(f)
    for stem, seed, fold, sp, desc in todo:
        tr, ca, te = (sp[sp["role"] == r] for r in ("train", "cal", "test"))
        full = by[(stem, "full")]
        arrs = {}
        arrs["train__point"] = ensemble_summary(full, tr[feats].to_numpy(float), alphas=(), cfg=cfg,
                                                quantiles=False)["point"]
        for name, part in (("cal", ca), ("test", te)):
            summ = ensemble_summary(full, part[feats].to_numpy(float), alphas=alphas, cfg=cfg)
            for k, v in summ.items():
                arrs[f"{name}__{k}"] = v
        for name, part in (("train", tr), ("cal", ca), ("test", te)):
            arrs[f"{name}__GLORIA_ID"] = part["GLORIA_ID"].to_numpy().astype(str)
        if (stem, "r80") in by:
            hold = recal_holdout_index(len(tr), seed)
            r80 = by[(stem, "r80")]
            for name, Xp in (("r80hold", tr[feats].to_numpy(float)[hold]), ("r80test", te[feats].to_numpy(float))):
                summ = ensemble_summary(r80, Xp, alphas=(), cfg=cfg, quantiles=False)
                arrs[f"{name}__mix_mean"] = summ["mix_mean"]
                arrs[f"{name}__mix_sd"] = summ["mix_sd"]
            arrs["r80hold__y"] = tr["y"].to_numpy(float)[hold]
        if (stem, "cv0") in by:
            gf = group_folds(tr["wb_group"].to_numpy(), GAUSS_CV_K)
            oof = np.full(len(tr), np.nan)
            Xtr = tr[feats].to_numpy(float)
            for k in range(GAUSS_CV_K):
                oof[gf == k] = ensemble_summary(by[(stem, f"cv{k}")], Xtr[gf == k], alphas=(), cfg=cfg,
                                                quantiles=False)["point"]
            arrs["train__oof_point"] = oof
        arrs["meta__json"] = np.array(json.dumps({"cfg": cfg, "n_train": len(tr),
                                                  "fit_s_share": timing["fit_s"] / len(todo),
                                                  "seed_rule": SEED_RULE}))
        final = mdn_cache_path(out_dir, stem)
        tmp = final.with_name(final.name + ".tmp")
        with open(tmp, "wb") as fh:  # file handle: numpy would otherwise append ".npz" to the tmp name
            np.savez_compressed(fh, **arrs)
        os.replace(tmp, final)  # atomic: a crash never leaves a partial cache under the final name
    timing["predict_s"] = time.perf_counter() - t1
    log(f"[mdn] fit {timing['fit_s']:.1f}s, predict+write {timing['predict_s']:.1f}s")
    return timing


SEED_RULE = "mdn_member_seed-v2"  # config.mdn_member_seed (AUDIT_3 A3-5); caches from the old rule are rejected


def check_mdn_cache(path: Path, cfg: dict | None, need_cv: bool, need_recal: bool, z=None) -> None:
    """Raise if an MDN cache lacks the arrays a requested method needs or was built with another cfg/seed rule
    (AUDIT_3 A3-8: never silently skip a method)."""
    own = z is None
    z = np.load(path, allow_pickle=False) if own else z
    try:
        files = set(z.files)
        if "meta__json" not in files:
            raise RuntimeError(f"MDN cache {path} has no meta")
        meta = json.loads(str(z["meta__json"]))
        if meta.get("seed_rule") != SEED_RULE:
            raise RuntimeError(f"MDN cache {path} was built with seed rule {meta.get('seed_rule')}, need {SEED_RULE}")
        if cfg is not None and meta["cfg"] != json.loads(json.dumps(cfg)):
            raise RuntimeError(f"MDN cache {path} cfg {meta['cfg']} differs from requested {cfg}")
        need = {"train__point", "cal__point", "test__point", "cal__mix_mean", "cal__mix_sd",
                "test__mix_mean", "test__mix_sd", "train__GLORIA_ID", "cal__GLORIA_ID", "test__GLORIA_ID"}
        if need_cv:
            need.add("train__oof_point")
        if need_recal:
            need |= {"r80hold__mix_mean", "r80hold__mix_sd", "r80hold__y", "r80test__mix_mean", "r80test__mix_sd"}
        missing = sorted(need - files)
        if missing:
            raise RuntimeError(f"MDN cache {path} lacks {missing}; delete it and rerun")
    finally:
        if own:
            z.close()
