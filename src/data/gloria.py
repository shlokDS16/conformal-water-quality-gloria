"""Load GLORIA, apply QC rules, build targets and region keys."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import GLORIA_DIR, TARGETS

N_SAMPLES = 7572

PRIMARY_EXCLUDE = ["Suspect", "QWIP_fail", "Baseline_shift", "Noisy_red", "Noisy_blue"]
QC_FLAGS = ["Noisy_red", "Noisy_blue", "Baseline_shift", "Oxygen_signal", "Negative_uv_slope", "QWIP_fail", "Suspect", "Flagged"]

# Country -> macro-region (Decisions 2026-09-16; dossier 3.6). Every Country value in the file is listed.
COUNTRY_TO_REGION = {
    "United States of America (the)": "North America",
    "Canada": "North America",
    "Bahamas (the)": "North America",
    "Honduras": "North America",
    "France": "Europe",  # except French Guiana rows, see SITE_REGION_OVERRIDE
    "Netherlands (the)": "Europe",
    "Switzerland": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Poland": "Europe",
    "Estonia": "Europe",
    "Germany": "Europe",
    "United Kingdom of Great Britain and Northern Ireland (the)": "Europe",
    "Norway": "Europe",
    "Belgium": "Europe",
    "Sweden": "Europe",
    "Lithuania": "Europe",
    "Finland": "Europe",
    "China": "East and South-East Asia",
    "Japan": "East and South-East Asia",
    "Korea (the Republic of)": "East and South-East Asia",
    "Viet Nam": "East and South-East Asia",
    "Malaysia": "East and South-East Asia",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "Brazil": "South America",
    "Uruguay": "South America",
    "Peru": "South America",
    "Argentina": "South America",  # 2 rows at 53.8 S 36.6 W (South Georgia, Southern Ocean)
    "South Africa": "Africa",
}
# French Guiana is coded Country = France but lies in South America (lat about 4 N, lon about 52 W).
SITE_REGION_OVERRIDE = {"Guiana": "South America"}

# North America sub-regions from 10 x 10 degree cells (cell key = floor(lat/10)*10, floor(lon/10)*10).
NA_CELL_TO_SUBREGION = {
    (10, -90): "NA1 Gulf of Mexico and Caribbean",
    (20, -100): "NA1 Gulf of Mexico and Caribbean",
    (20, -90): "NA1 Gulf of Mexico and Caribbean",
    (20, -80): "NA1 Gulf of Mexico and Caribbean",
    (30, -90): "NA2 South-East and Mid-Atlantic",
    (30, -80): "NA2 South-East and Mid-Atlantic",
    (40, -90): "NA3 Great Lakes and North-East",
    (40, -80): "NA3 Great Lakes and North-East",
    (30, -100): "NA4 West and Great Plains",
    (30, -130): "NA4 West and Great Plains",
    (40, -100): "NA4 West and Great Plains",
    (40, -110): "NA4 West and Great Plains",
    (50, -100): "NA4 West and Great Plains",
}
# Null-coordinate North American water bodies whose group has no coordinates.
# MSU warm water aquaculture ponds: Mississippi State University (Starkville, MS, about 33.5 N 88.8 W) -> cell (30, -90).
NA_SITE_OVERRIDE = {"MSU warm water aquaculture ponds": "NA2 South-East and Mid-Atlantic"}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(GLORIA_DIR / "GLORIA_meta_and_lab.csv", low_memory=False)
    rrs = pd.read_csv(GLORIA_DIR / "GLORIA_Rrs.csv")
    qc = pd.read_csv(GLORIA_DIR / "GLORIA_qc_flags.csv")
    n_null = int(qc["GLORIA_ID"].isna().sum())
    qc = qc[qc["GLORIA_ID"].notna()]
    assert n_null == 1, n_null
    assert len(meta) == N_SAMPLES and meta["GLORIA_ID"].is_unique
    assert len(rrs) == N_SAMPLES and rrs["GLORIA_ID"].is_unique
    assert len(qc) == N_SAMPLES and qc["GLORIA_ID"].is_unique
    return meta, rrs, qc


def build_table(meta: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    df = meta.merge(qc, on="GLORIA_ID", how="inner", validate="one_to_one")
    assert len(df) == N_SAMPLES
    for f in QC_FLAGS:
        df[f] = df[f].astype("Int8")
    # A missing flag means the test could not be run (paper Usage Notes): treated as not flagged.
    flagged_primary = np.zeros(len(df), dtype=bool)
    for f in PRIMARY_EXCLUDE:
        flagged_primary |= df[f].fillna(0).to_numpy() == 1
    df["qc_primary"] = ~flagged_primary
    df["qc_strict"] = df["Flagged"].fillna(0).to_numpy() != 1
    df["qc_none"] = True
    assert (df["qc_strict"] <= df["qc_primary"]).all(), "strict keep set must be a subset of primary"
    for t in TARGETS:
        v = pd.to_numeric(df[t], errors="coerce")
        df[t] = v
        df[f"log10_{t}"] = np.where(v > 0, np.log10(v.where(v > 0)), np.nan)
    df["region"] = df["Country"].map(COUNTRY_TO_REGION)
    for site, reg in SITE_REGION_OVERRIDE.items():
        df.loc[df["Site_name"] == site, "region"] = reg
    missing = sorted(set(df.loc[df["region"].isna(), "Country"]))
    assert not missing, f"unmapped countries: {missing}"
    return df


def na_subregion_rowwise(df: pd.DataFrame) -> pd.Series:
    """Row-level NA sub-region from coordinates (NaN outside North America or without coordinates)."""
    na = df["region"] == "North America"
    lat_c = (np.floor(df["Latitude"] / 10) * 10)
    lon_c = (np.floor(df["Longitude"] / 10) * 10)
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    ok = na & df["Latitude"].notna()
    keys = list(zip(lat_c[ok].astype(int), lon_c[ok].astype(int)))
    mapped = [NA_CELL_TO_SUBREGION.get(k) for k in keys]
    unmapped = sorted({k for k, v in zip(keys, mapped) if v is None})
    assert not unmapped, f"unmapped NA cells: {unmapped}"
    out[ok] = mapped
    for site, sub in NA_SITE_OVERRIDE.items():
        out[(df["Site_name"] == site) & na & df["Latitude"].isna()] = sub
    return out


TECHNIQUE_CLASS = {
    "HPLC": "HPLC",
    "Spectrophotometry": "spectrophotometry",
    "Fluorometry": "fluorometry",
    "Fluorescence spectroscopy": "fluorometry",
    "aLH": "optical aLH",
}


def chl_method_map() -> pd.DataFrame:
    """(Dataset_ID, Chl_method) -> quantification technique, from the GLORIA xlsx sheet 'Chla methods'."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = pd.read_excel(GLORIA_DIR / "GLORIA_variables_and_methods.xlsx", sheet_name="Chla methods")
    x = x.rename(columns={"Dataset ID": "Dataset_ID", "Methodology short name": "Chl_method",
                          "Pigment quantification technique": "technique"})
    x = x[["Dataset_ID", "Chl_method", "technique"]].dropna(subset=["Chl_method"])
    unknown = sorted(set(x["technique"].dropna()) - set(TECHNIQUE_CLASS))
    assert not unknown, f"unmapped techniques: {unknown}"
    x["chl_technique"] = x["technique"].map(TECHNIQUE_CLASS)
    return x.drop_duplicates(["Dataset_ID", "Chl_method", "chl_technique"])


def add_label_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mp = chl_method_map()
    by_pair = mp.drop_duplicates(["Dataset_ID", "Chl_method"]).set_index(["Dataset_ID", "Chl_method"])["chl_technique"]
    by_name = mp.groupby("Chl_method")["chl_technique"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else "ambiguous")
    tech = []
    source = []
    for d, mth in zip(df["Dataset_ID"], df["Chl_method"]):
        if pd.isna(mth):
            tech.append("missing"); source.append("method missing")
        elif (d, mth) in by_pair.index:
            tech.append(by_pair[(d, mth)]); source.append("dataset+method")
        elif mth in by_name.index:
            tech.append(by_name[mth]); source.append("method name only")
        else:
            tech.append("unmatched"); source.append("not in sheet")
    df["chl_technique"] = tech
    df["chl_technique_source"] = source
    df.loc[df["Chla"].isna(), ["chl_technique", "chl_technique_source"]] = pd.NA
    chla_pos = df["Chla"] > 0
    df["chla_optical_estimate"] = chla_pos & (df["Chl_method"] == "Eawag Chl - aLH")
    df["chla_uncorrected"] = chla_pos & (pd.to_numeric(df["Phaeophytin_correction"], errors="coerce") == 0)
    df["chla_hplc_or_corrected"] = chla_pos & ((df["chl_technique"] == "HPLC") |
                                               (pd.to_numeric(df["Phaeophytin_correction"], errors="coerce") == 1))
    df["chla_primary"] = chla_pos & df["qc_primary"] & ~df["chla_optical_estimate"]
    df["tss_primary"] = (df["TSS"] > 0) & df["qc_primary"]
    df["acdom_primary"] = (df["aCDOM440"] > 0) & df["qc_primary"]
    depth = pd.to_numeric(df["Depth"], errors="coerce")
    df["Depth"] = depth
    df["secchi_censored"] = (df["Secchi_depth"] > 0) & (df["Secchi_depth"] >= 0.95 * depth)
    df["secchi_primary"] = (df["Secchi_depth"] > 0) & df["qc_primary"] & ~df["secchi_censored"]
    df["shallow_lt3m"] = depth < 3
    sd = df["Secchi_depth"]
    df["secchi_rounded_half_m"] = (sd > 0) & np.isclose(sd * 2, np.round(sd * 2))
    table = (df[df["Chla"].notna()].groupby(["Chl_method", "chl_technique", "chl_technique_source"], dropna=False)
             .size().reset_index(name="n_rows").sort_values("n_rows", ascending=False))
    return df, table


def qc_target_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        rows.append({
            "target": t,
            "non_null": int(df[t].notna().sum()),
            "le_zero": int((df[t] <= 0).sum()),
            "positive_no_filter": int(df[f"log10_{t}"].notna().sum()),
            "positive_primary": int((df[f"log10_{t}"].notna() & df["qc_primary"]).sum()),
            "positive_strict": int((df[f"log10_{t}"].notna() & df["qc_strict"]).sum()),
        })
    return pd.DataFrame(rows)
