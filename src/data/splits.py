"""Split protocols (dossier 5.5, amended by AUDIT_1) with hard leakage assertions.

Output per target: data/processed/splits_<target>.parquet, long format with columns
GLORIA_ID, protocol, seed, fold, test_unit, descriptive_only, role (train/cal/test).

Population per target: the primary boolean in config.PRIMARY_POPULATION (target > 0,
primary QC, plus label rules: Chla without aLH optical estimates, Secchi without censored
values). Sensitivity subsets of it (strict QC, HPLC-or-corrected Chla, shallow exclusion)
can filter these splits without new leakage.

Protocols (RNG seed = config.split_seed(protocol, seed))
- random (i.i.d. reference, optimistic; seeds 0-19): 60/20/20 by iid_unit (identical spectra,
  same lat/lon/datetime replicates and spectral near-duplicates kept together).
- waterbody (seeds 0-19): GroupShuffleSplit on wb_group, 20% of groups test, then 25% of the
  remaining groups calibration (60/20/20 by groups).
- region (seeds 0-4): leave-one-macro-region-out, region assigned per wb_group by majority;
  train/cal from the other regions by GroupShuffleSplit on wb_group (25% of groups cal).
- region_na (seeds 0-4): leave-one-North-America-sub-region-out, same train/cal rule.
- contributor (seeds 0-9): 5 folds over contrib_group (components of Dataset_ID <-> wb_group),
  balanced by sample count with seeded jitter; contrib_groups with < 10 samples are never test.
  Calibration = disjoint contrib_groups from the non-test pool, added in seeded random order
  until >= 25% of non-test samples (a unit is skipped if it would push calibration above 35%).
- contributor_ds (seeds 0-9, alternative for lead review): 5 folds over Dataset_ID; train/cal rows
  whose wb_group occurs in test are dropped, then cal rows whose wb_group occurs in train.
descriptive_only = True for region folds with < 10 test water bodies, and always for
South America and Africa (AUDIT_1 S3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ..config import (PRIMARY_POPULATION, SEEDS_CONTRIBUTOR, SEEDS_RANDOM, SEEDS_REGION, SEEDS_WATERBODY,
                      split_seed)

N_CONTRIB_FOLDS = 5
MIN_TEST_UNIT = 10
MIN_DESCRIPTIVE_GROUPS = 10
ALWAYS_DESCRIPTIVE = {"South America", "Africa"}
ROLES = ["train", "cal", "test"]


def _gss(groups: np.ndarray, test_size: float, rs: int) -> tuple[np.ndarray, np.ndarray]:
    g = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
    return next(g.split(np.zeros(len(groups)), groups=groups))


def _emit(rows, ids, protocol, seed, fold, unit, descriptive, tr, ca, te):
    for role, ix in zip(ROLES, (tr, ca, te)):
        rows.append(pd.DataFrame({"GLORIA_ID": ids[ix], "protocol": protocol, "seed": seed, "fold": fold,
                                  "test_unit": unit, "descriptive_only": descriptive, "role": role}))


def balanced_folds(units: pd.Series, rs: int, k: int = N_CONTRIB_FOLDS) -> dict:
    """Jittered greedy GroupKFold: larger units first (size x U(0.5, 1.5)), each to the lightest fold."""
    sizes = units.value_counts().sort_index()
    eligible = sizes[sizes >= MIN_TEST_UNIT]
    rng = np.random.default_rng(rs)
    jitter = eligible.to_numpy() * rng.uniform(0.5, 1.5, len(eligible))
    order = eligible.index[np.argsort(-jitter, kind="stable")]
    load = np.zeros(k)
    fold_of = {}
    for u in order:
        f = int(np.argmin(load))
        fold_of[u] = f
        load[f] += sizes[u]
    return fold_of


def build_splits(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    popcol = PRIMARY_POPULATION[target]
    pop = df[df[popcol].fillna(False).astype(bool)].reset_index(drop=True)
    ids = pop["GLORIA_ID"].to_numpy()
    wb = pop["wb_group"].to_numpy()
    iid = pop["iid_unit"].to_numpy()
    n = len(pop)
    all_idx = np.arange(n)
    rows: list[pd.DataFrame] = []
    info: dict = {"target": target, "population": popcol, "n": n, "n_wb_group": int(pop["wb_group"].nunique())}

    for s in SEEDS_RANDOM:
        rs = split_seed("random", s)
        rest, te = _gss(iid, 0.2, rs)
        tr, ca = _gss(iid[rest], 0.25, rs + 1)
        _emit(rows, ids, "random", s, 0, "random", False, rest[tr], rest[ca], te)
    for s in SEEDS_WATERBODY:
        rs = split_seed("waterbody", s)
        rest, te = _gss(wb, 0.2, rs)
        tr, ca = _gss(wb[rest], 0.25, rs + 1)
        _emit(rows, ids, "waterbody", s, 0, "waterbody", False, rest[tr], rest[ca], te)

    info["region_folds"] = {}
    for proto, col in [("region", "region_grp"), ("region_na", "na_subregion_grp")]:
        vals = pop[col].astype("string").fillna("").to_numpy(dtype=str)
        units = sorted(u for u in pd.unique(vals) if u != "")
        for f, u in enumerate(units):
            is_te = vals == u
            te = all_idx[is_te]
            rest = all_idx[~is_te]
            n_groups = int(pop.loc[is_te, "wb_group"].nunique())
            descriptive = bool(n_groups < MIN_DESCRIPTIVE_GROUPS or u in ALWAYS_DESCRIPTIVE)
            info["region_folds"][f"{proto}:{u}"] = {"n_test": int(len(te)), "test_wb_groups": n_groups,
                                                   "test_datasets": int(pop.loc[is_te, "Dataset_ID"].nunique()),
                                                   "descriptive_only": descriptive}
            for s in SEEDS_REGION:
                rs = split_seed(proto, s)
                tr, ca = _gss(wb[rest], 0.25, rs)
                _emit(rows, ids, proto, s, f, u, descriptive, rest[tr], rest[ca], te)

    cg = pop["contrib_group"]
    sizes = cg.value_counts()
    info["contrib_groups"] = int(len(sizes))
    info["contrib_groups_never_test"] = int((sizes < MIN_TEST_UNIT).sum())
    info["contrib_group_largest_share"] = float(sizes.max() / n)
    for s in SEEDS_CONTRIBUTOR:
        rs = split_seed("contributor", s)
        fold_of = balanced_folds(cg, rs)
        rng = np.random.default_rng(rs + 1)
        fcol = cg.map(fold_of).to_numpy()
        for f in range(N_CONTRIB_FOLDS):
            te = all_idx[fcol == f]
            pool = all_idx[fcol != f]
            pool_units = np.array(sorted(pd.unique(cg.to_numpy()[pool])))
            pool_units = pool_units[rng.permutation(len(pool_units))]
            cal_units, acc = [], 0
            for u in pool_units:  # skip units that would push calibration above 35% of the pool
                if acc >= 0.25 * len(pool):
                    break
                if acc + sizes[u] <= 0.35 * len(pool):
                    cal_units.append(u)
                    acc += sizes[u]
            if not cal_units:
                cal_units = [min(pool_units, key=lambda u: sizes[u])]
            is_cal = np.isin(cg.to_numpy()[pool], cal_units)
            _emit(rows, ids, "contributor", s, f, f"fold{f}", False, pool[~is_cal], pool[is_cal], te)

    # Alternative (AUDIT_1 L1 option 2): folds on Dataset_ID; train/cal rows sharing a wb_group with
    # test are dropped, and cal rows sharing a wb_group with train are dropped.
    ds = pop["Dataset_ID"]
    ds_sizes = ds.value_counts()
    dropped = []
    for s in SEEDS_CONTRIBUTOR:
        rs = split_seed("contributor_ds", s)
        fold_of = balanced_folds(ds, rs)
        rng = np.random.default_rng(rs + 1)
        fcol = ds.map(fold_of).to_numpy()
        for f in range(N_CONTRIB_FOLDS):
            te = all_idx[fcol == f]
            pool = all_idx[fcol != f]
            pool_units = np.array(sorted(pd.unique(ds.to_numpy()[pool])))
            pool_units = pool_units[rng.permutation(len(pool_units))]
            cal_units, acc = [], 0
            for u in pool_units:
                if acc >= 0.25 * len(pool):
                    break
                if acc + ds_sizes[u] <= 0.35 * len(pool):
                    cal_units.append(u)
                    acc += ds_sizes[u]
            is_cal = np.isin(ds.to_numpy()[pool], cal_units)
            tr, ca = pool[~is_cal], pool[is_cal]
            n0 = len(tr) + len(ca)
            test_wb = np.unique(wb[te])
            tr = tr[~np.isin(wb[tr], test_wb)]
            ca = ca[~np.isin(wb[ca], test_wb)]
            ca = ca[~np.isin(wb[ca], np.unique(wb[tr]))]
            dropped.append(n0 - len(tr) - len(ca))
            _emit(rows, ids, "contributor_ds", s, f, f"fold{f}", False, tr, ca, te)
    info["contributor_ds_dropped_rows_median"] = float(np.median(dropped))
    info["contributor_ds_dropped_rows_max"] = int(np.max(dropped))

    out = pd.concat(rows, ignore_index=True)
    for c in ["protocol", "test_unit", "role"]:
        out[c] = out[c].astype("category")
    out["seed"] = out["seed"].astype("int16")
    out["fold"] = out["fold"].astype("int16")
    return out, info


def assert_no_leakage(splits: pd.DataFrame, table: pd.DataFrame, target: str | None = None) -> dict:
    """Hard checks; raises AssertionError on any violation.

    Every protocol: a GLORIA_ID at most once per split; spec_hash and iid_unit in one role.
    All protocols except random: wb_group in one role.
    contributor: Dataset_ID and contrib_group in one role.
    region / region_na: the test unit never appears in train or cal, all test rows belong to it.
    """
    cols = ["GLORIA_ID", "wb_group", "spec_hash", "iid_unit", "region_grp", "na_subregion_grp", "Dataset_ID", "contrib_group"]
    m = splits.merge(table[cols], on="GLORIA_ID", how="left", validate="many_to_one")
    assert m["wb_group"].notna().all(), "split row not in table"
    if target is not None:
        pop = set(table.loc[table[PRIMARY_POPULATION[target]].fillna(False).astype(bool), "GLORIA_ID"])
        assert set(m["GLORIA_ID"]) <= pop, "split row outside target population"
    key = ["protocol", "seed", "fold"]
    assert not m.duplicated(key + ["GLORIA_ID"]).any(), "sample twice in one split"
    rules = {"spec_hash": None, "iid_unit": None, "wb_group": "non-random", "Dataset_ID": ["contributor", "contributor_ds"],
             "contrib_group": ["contributor"]}
    checked = {}
    for col, scope in rules.items():
        if scope is None:
            sub = m
        elif scope == "non-random":
            sub = m[m["protocol"] != "random"]
        else:
            sub = m[m["protocol"].isin(scope)]
        roles = sub.groupby(key + [col], observed=True)["role"].nunique()
        assert (roles <= 1).all(), f"leakage on {col}: {roles[roles > 1].head()}"
        checked[col] = int(sub.groupby(key, observed=True).ngroups)
    for proto, col in [("region", "region_grp"), ("region_na", "na_subregion_grp")]:
        sub = m[m["protocol"] == proto]
        te = sub[sub["role"] == "test"]
        assert (te[col].astype(str) == te["test_unit"].astype(str)).all(), f"{proto}: test row outside unit"
        nt = sub[sub["role"] != "test"]
        assert not (nt[col].astype(str) == nt["test_unit"].astype(str)).any(), f"{proto}: test unit in train/cal"
        checked[col] = int(sub.groupby(key, observed=True).ngroups)
    # every split has all three roles non-empty
    nroles = m.groupby(key, observed=True)["role"].nunique()
    assert (nroles == 3).all(), f"split missing a role: {nroles[nroles < 3].head()}"
    return checked


def summarise(splits: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    m = splits.merge(table[["GLORIA_ID", "wb_group"]], on="GLORIA_ID")
    per = (m.groupby(["protocol", "seed", "fold", "role"], observed=True)
           .agg(n=("GLORIA_ID", "size"), groups=("wb_group", "nunique")).reset_index())
    wide = per.pivot_table(index=["protocol", "seed", "fold"], columns="role", values=["n", "groups"], observed=True)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    agg = wide.groupby("protocol", observed=True).agg(
        n_splits=("seed", "size"),
        median_train=("n_train", "median"), median_cal=("n_cal", "median"), median_test=("n_test", "median"),
        min_test=("n_test", "min"), max_test=("n_test", "max"),
        median_test_groups=("groups_test", "median"), median_cal_groups=("groups_cal", "median"),
        min_cal_groups=("groups_cal", "min"))
    return agg.reset_index()


def group_level_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """One row per wb_group in the target population (for water-body-averaged coverage)."""
    pop = df[df[PRIMARY_POPULATION[target]].fillna(False).astype(bool)]
    g = pop.groupby("wb_group").agg(
        n_samples=("GLORIA_ID", "size"), region=("region_grp", "first"), na_subregion=("na_subregion_grp", "first"),
        contrib_group=("contrib_group", "first"), n_datasets=("Dataset_ID", "nunique"),
        lat_centroid=("Latitude", "mean"), lon_centroid=("Longitude", "mean"),
        n_timestamps=("Date_Time_UTC", "nunique")).reset_index()
    assert (pop.groupby("wb_group")["region_grp"].nunique() == 1).all()
    assert (pop.groupby("wb_group")["contrib_group"].nunique() == 1).all()
    return g
