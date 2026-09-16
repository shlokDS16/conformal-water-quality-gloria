"""5 km grouping sensitivity: waterbody_5km split protocol (AUDIT_2 S5, PREREGISTRATION.md s7).

Reuses the waterbody-protocol machinery in splits.py (GroupShuffleSplit helper `_gss`, the row
emitter `_emit`, and the leakage assertions in `assert_no_leakage`) but groups on `wb_5km` (the
5 km spatial-merge key from groups.py) instead of the primary `wb_group` (2 km) key. Output
mirrors splits_<target>.parquet exactly (same columns), written to NEW files so the primary
splits and the in-progress experiment run are untouched:

    data/processed/splits5km_<target>.parquet   protocol = "waterbody_5km", seeds 0-9, fold 0.

Does not modify splits.py, groups.py, config.py, build_all.py or any existing output file.

Usage: python -m src.data.splits_5km
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PRIMARY_POPULATION, RAW_TARGETS, split_seed
from .common import PROCESSED, ensure_dirs
from .splits import ROLES, _emit, _gss, assert_no_leakage

PROTOCOL = "waterbody_5km"
SEEDS_WATERBODY_5KM = list(range(10))  # seeds 0-9 (task spec; primary waterbody uses 0-19)

READ_COLS = ["GLORIA_ID", "wb_group", "wb_5km", "spec_hash", "iid_unit", "region_grp",
             "na_subregion_grp", "Dataset_ID", "contrib_group"] + list(PRIMARY_POPULATION.values())


def build_splits_5km(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """Same 60/20/20 GroupShuffleSplit recipe as splits.build_splits's waterbody loop, on wb_5km."""
    popcol = PRIMARY_POPULATION[target]
    pop = df[df[popcol].fillna(False).astype(bool)].reset_index(drop=True)
    ids = pop["GLORIA_ID"].to_numpy()
    wb5 = pop["wb_5km"].to_numpy()
    n = len(pop)
    rows: list[pd.DataFrame] = []
    info: dict = {"target": target, "population": popcol, "n": n, "n_wb_5km": int(pop["wb_5km"].nunique())}

    for s in SEEDS_WATERBODY_5KM:
        rs = split_seed(PROTOCOL, s)
        rest, te = _gss(wb5, 0.2, rs)
        tr, ca = _gss(wb5[rest], 0.25, rs + 1)
        _emit(rows, ids, PROTOCOL, s, 0, PROTOCOL, False, rest[tr], rest[ca], te)

    out = pd.concat(rows, ignore_index=True)
    for c in ["protocol", "test_unit", "role"]:
        out[c] = out[c].astype("category")
    out["seed"] = out["seed"].astype("int16")
    out["fold"] = out["fold"].astype("int16")
    return out, info


def assert_no_leakage_5km(splits: pd.DataFrame, table: pd.DataFrame, target: str) -> dict:
    """Reuses splits.assert_no_leakage (GLORIA_ID uniqueness, spec_hash, iid_unit, and wb_group as
    a corollary check: every wb_group is a subset of exactly one wb_5km component, so a split that
    does not leak wb_5km cannot leak wb_group either) and adds an explicit check on wb_5km itself,
    the actual grouping key of this protocol.
    """
    checked = assert_no_leakage(splits, table, target)
    m = splits.merge(table[["GLORIA_ID", "wb_5km"]], on="GLORIA_ID", how="left", validate="many_to_one")
    assert m["wb_5km"].notna().all(), "split row not in table (wb_5km)"
    key = ["protocol", "seed", "fold"]
    roles = m.groupby(key + ["wb_5km"], observed=True)["role"].nunique()
    assert (roles <= 1).all(), f"leakage on wb_5km: {roles[roles > 1].head()}"
    checked["wb_5km"] = int(m.groupby(key, observed=True).ngroups)
    return checked


def main() -> None:
    ensure_dirs()
    g = pd.read_parquet(PROCESSED / "gloria.parquet", columns=READ_COLS)
    report: dict = {"protocol": PROTOCOL, "seeds": SEEDS_WATERBODY_5KM, "targets": {}}
    for t in RAW_TARGETS:
        sp, info = build_splits_5km(g, t)
        info["leakage_checked_splits"] = assert_no_leakage_5km(sp, g, t)
        per = (sp.merge(g[["GLORIA_ID", "wb_5km"]], on="GLORIA_ID")
                 .groupby(["seed", "role"], observed=True)
                 .agg(n=("GLORIA_ID", "size"), groups=("wb_5km", "nunique")).reset_index())
        wide = per.pivot(index="seed", columns="role", values=["n", "groups"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        info["per_seed"] = wide.reset_index().to_dict(orient="records")
        info["median_train"] = float(wide["n_train"].median())
        info["median_cal"] = float(wide["n_cal"].median())
        info["median_test"] = float(wide["n_test"].median())
        info["median_test_wb5km_groups"] = float(wide["groups_test"].median())
        info["min_test_wb5km_groups"] = int(wide["groups_test"].min())
        info["rows"] = int(len(sp))
        out_path = PROCESSED / f"splits5km_{t}.parquet"
        sp.to_parquet(out_path, index=False)
        report["targets"][t] = info
        print(f"[splits_5km] {t}: n={info['n']} wb_5km_groups={info['n_wb_5km']} rows={len(sp)} "
              f"median(train/cal/test)={info['median_train']:.0f}/{info['median_cal']:.0f}/{info['median_test']:.0f} "
              f"median_test_groups={info['median_test_wb5km_groups']:.0f} leakage assertions passed -> {out_path.name}")
    print(json.dumps({t: {k: v for k, v in i.items() if k != "per_seed"} for t, i in report["targets"].items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
