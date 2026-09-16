"""Data pipeline tests. Run after `python -m src.data.build_all`:  python -m pytest tests/test_data.py -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import bands, gloria, groups, splits, srf
from src.data.common import PROCESSED, TARGETS


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "gloria.parquet")


@pytest.fixture(scope="module")
def raw():
    meta, rrs, qc = gloria.load_raw()
    df = gloria.build_table(meta, qc)
    rrs = rrs.set_index("GLORIA_ID").loc[df["GLORIA_ID"]].reset_index()
    return df, rrs


@pytest.mark.parametrize("unit", ["S2A", "S2B", "S2C", "S3A", "S3B"])
def test_flat_spectrum_band_equals_constant(unit):
    wl = np.arange(350, 901)
    s = srf.load_msi(unit) if unit.startswith("S2") else srf.load_olci(unit)
    band_list = config.MSI_BANDS if unit.startswith("S2") else config.OLCI_BANDS
    for c in (0.0005, 0.0123, 0.05):
        assert bands.flat_spectrum_check(s, band_list, wl, c) < 1e-12


def test_flat_spectrum_hyperspectral():
    wl = np.arange(350, 901)
    R = np.full((2, len(wl)), 0.02)
    h = bands.hyperspectral(R, wl, config.HYP_CENTRES).to_numpy()
    assert np.allclose(h, 0.02)


def test_band_missing_without_coverage():
    wl = np.arange(350, 901)
    R = np.full((1, len(wl)), 0.01)
    R[0, wl < 400] = np.nan  # typical GLORIA spectrum starting at 400 nm
    s = srf.load_olci("S3A")
    sim = bands.simulate(R, wl, s, ["Oa1", "Oa2"], "")
    assert np.isnan(sim.loc[0, "Oa1"]) and np.isclose(sim.loc[0, "Oa2"], 0.01)


def test_olci_centroids_match_file():
    import netCDF4

    ds = netCDF4.Dataset(srf.SRF_DIR / srf.OLCI_URLS["S3A"][1])
    ref = np.asarray(ds.variables["srf_centre_wavelength"][:], dtype=float)
    ds.close()
    summ = srf.srf_summary(srf.load_olci("S3A"), [f"Oa{i}" for i in range(1, 22)])
    assert np.max(np.abs(summ["centroid_nm"].to_numpy() - ref)) < 0.1


def test_row_count(table):
    assert len(table) == 7572
    assert table["GLORIA_ID"].is_unique


def test_qc_subset_and_populations(table):
    assert (table["qc_strict"] <= table["qc_primary"]).all()
    assert not (table["chla_primary"] & table["chla_optical_estimate"]).any()
    assert not (table["secchi_primary"] & table["secchi_censored"]).any()
    for t, col in config.PRIMARY_POPULATION.items():
        assert table.loc[table[col], f"log10_{t}"].notna().all()


def test_feature_prefixes(table):
    assert len([c for c in table.columns if c.startswith("msi_")]) == len(config.MSI_BANDS)
    assert len([c for c in table.columns if c.startswith("olci_")]) == len(config.OLCI_BANDS)
    assert len([c for c in table.columns if c.startswith("hyp_")]) == len(config.HYP_CENTRES)
    for p in ("hyp", "msi", "olci"):
        cols = [c for c in table.columns if c.startswith(p + "_")]
        assert (table[f"complete_{p}"] == table[cols].notna().all(axis=1)).all()


def test_group_keys_deterministic(raw, table):
    df, rrs = raw
    k1, _ = groups.build_groups(df, rrs, TARGETS)
    perm = np.random.default_rng(3).permutation(len(df))  # labels must not depend on row order
    k2, _ = groups.build_groups(df.iloc[perm], rrs.iloc[perm], TARGETS)
    k2 = k2.set_index("GLORIA_ID").loc[k1["GLORIA_ID"]].reset_index()
    for col in ["wb_group", "wb_5km", "wb_raw", "iid_unit", "contrib_group", "spec_hash"]:
        assert (k1[col].to_numpy() == k2[col].to_numpy()).all(), col
        assert (k1[col].to_numpy() == table[col].to_numpy()).all(), f"{col} differs from saved parquet"


def test_group_nesting(table):
    # identical spectra and replicates never split across water bodies; water bodies never split across
    # contributor groups or macro-regions
    assert (table.groupby("spec_hash")["wb_group"].nunique() == 1).all()
    assert (table.groupby("iid_unit")["wb_group"].nunique() == 1).all()
    assert (table.groupby("wb_group")["contrib_group"].nunique() == 1).all()
    assert (table.groupby("Dataset_ID")["contrib_group"].nunique() == 1).all()
    assert (table.groupby("wb_group")["region_grp"].nunique() == 1).all()


@pytest.mark.parametrize("target", TARGETS)
def test_saved_splits_no_leakage(table, target):
    sp = pd.read_parquet(PROCESSED / f"splits_{target}.parquet")
    checked = splits.assert_no_leakage(sp, table, target)
    assert checked["wb_group"] > 0 and checked["Dataset_ID"] > 0
    assert set(sp["protocol"].unique()) == {"random", "waterbody", "region", "region_na", "contributor", "contributor_ds"}
    n = sp.groupby("protocol", observed=True)["seed"].nunique().to_dict()
    assert n["random"] == 20 and n["waterbody"] == 20 and n["contributor"] == 10 and n["region"] == 5
    reg = sp[sp["protocol"] == "region"]
    assert reg.loc[reg["test_unit"].isin(["South America", "Africa"]), "descriptive_only"].all()


@pytest.mark.parametrize("target", TARGETS)
def test_group_tables(table, target):
    g = pd.read_parquet(PROCESSED / f"groups_{target}.parquet")
    pop = table[table[config.PRIMARY_POPULATION[target]]]
    assert g["wb_group"].is_unique
    assert g["n_samples"].sum() == len(pop)


def test_splits_reproducible(table):
    sp_saved = pd.read_parquet(PROCESSED / "splits_TSS.parquet")
    sp_new, _ = splits.build_splits(table, "TSS")
    a = sp_saved.sort_values(["protocol", "seed", "fold", "GLORIA_ID"]).reset_index(drop=True)
    b = sp_new.sort_values(["protocol", "seed", "fold", "GLORIA_ID"]).reset_index(drop=True)
    assert (a["GLORIA_ID"].to_numpy() == b["GLORIA_ID"].to_numpy()).all()
    assert (a["role"].astype(str).to_numpy() == b["role"].astype(str).to_numpy()).all()
