"""Single entry point: python -m src.data.build_all

Writes
- data/processed/gloria.parquet (one row per GLORIA_ID)
- data/processed/splits_<target>.parquet and groups_<target>.parquet
- data/interim/wb_groups.csv, wb_overrides.csv, wb_merge_candidates.csv
- data/interim/srf_manifest.csv, srf_bands.csv, band_coverage.csv, irradiance_weighting.csv, chl_methods.csv
- data/interim/build_report.json (all counts quoted in research/DATA_PIPELINE_LOG.md)
"""
from __future__ import annotations

import json
import platform
import time
from importlib import metadata

import numpy as np
import pandas as pd

from .. import config
from . import bands, gloria, groups, splits, srf
from .common import GLORIA_DIR, INTERIM, PROCESSED, TARGETS, ensure_dirs, sha256

ID_COLS = ["GLORIA_ID", "Dataset_ID", "Organization_ID", "Sample_ID", "Site_name", "Country", "Country_code",
           "Latitude", "Longitude", "Date_Time_UTC", "Water_body_type", "Water_type", "Depth", "SeaBASS_ID",
           "LIMNADES_ID", "LIMNADES_UID", "Chl_method", "Phaeophytin_correction", "Chla_plus_phaeo"]
LABEL_COLS = ["chl_technique", "chl_technique_source", "chla_optical_estimate", "chla_uncorrected", "chla_hplc_or_corrected",
              "chla_primary", "tss_primary", "acdom_primary", "secchi_censored", "secchi_primary", "shallow_lt3m",
              "secchi_rounded_half_m"]
KEY_COLS = ["site_norm", "wb_node", "wb_group", "wb_5km", "wb_raw", "iid_unit", "contrib_group", "spec_hash",
            "region", "region_grp", "na_subregion", "na_subregion_grp"]


def _csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(INTERIM / name, index=False, lineterminator="\n", encoding="utf-8")


def versions() -> dict:
    out = {"python": platform.python_version(), "os": platform.platform(), "processor": platform.processor()}
    for p in ["numpy", "pandas", "scikit-learn", "pyarrow", "netCDF4", "openpyxl", "scipy", "pytest"]:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "missing"
    return out


def main() -> None:
    t0 = time.time()
    ensure_dirs()
    report: dict = {"versions": versions()}

    # 1. SRFs
    man = srf.fetch_olci()
    man.append({"file": srf.MSI_XLSX, "unit": "S2A/S2B/S2C", "url": "SentiWiki S2 SRF v5.0",
                "bytes": (srf.SRF_DIR / srf.MSI_XLSX).stat().st_size, "sha256": sha256(srf.SRF_DIR / srf.MSI_XLSX)})
    _csv(pd.DataFrame(man), "srf_manifest.csv")
    srfs = {"S2A": srf.load_msi("S2A"), "S2B": srf.load_msi("S2B"), "S2C": srf.load_msi("S2C"),
            "S3A": srf.load_olci("S3A"), "S3B": srf.load_olci("S3B")}
    summ = []
    for u, s in srfs.items():
        x = srf.srf_summary(s, srf.MSI_ALL_VNIR if u.startswith("S2") else srf.OLCI_ALL)
        x.insert(0, "unit", u)
        summ.append(x)
    srf_tab = pd.concat(summ)
    _csv(srf_tab, "srf_bands.csv")
    report["srf_manifest"] = man

    # 2. GLORIA table, QC, targets, label flags, regions
    meta, rrs, qc = gloria.load_raw()
    df = gloria.build_table(meta, qc)
    df, chl_tab = gloria.add_label_flags(df)
    _csv(chl_tab, "chl_methods.csv")
    rrs = rrs.set_index("GLORIA_ID").loc[df["GLORIA_ID"]].reset_index()
    report["qc_flag_counts"] = {f: int((df[f] == 1).sum()) for f in gloria.QC_FLAGS}
    report["qc_keep_counts"] = {q: int(df[q].sum()) for q in ["qc_primary", "qc_strict", "qc_none"]}
    report["target_counts"] = gloria.qc_target_counts(df).to_dict(orient="records")
    report["label_flags"] = {c: int(df[c].fillna(False).astype(bool).sum()) for c in LABEL_COLS if df[c].dtype == bool}
    report["label_flags"]["chla_optical_estimate_primary_qc"] = int((df["chla_optical_estimate"] & df["qc_primary"]).sum())
    report["label_flags"]["shallow_lt3m_primary_qc"] = int((df["shallow_lt3m"] & df["qc_primary"]).sum())
    report["label_flags"]["secchi_censored_primary_qc"] = int((df["secchi_censored"] & df["qc_primary"]).sum())
    report["label_flags"]["chla_hplc_or_corrected_in_chla_primary"] = int((df["chla_hplc_or_corrected"] & df["chla_primary"]).sum())
    report["chl_technique_counts_chla_primary"] = df.loc[df["chla_primary"], "chl_technique"].value_counts(dropna=False).to_dict()
    report["region_counts_country_map"] = df["region"].value_counts().to_dict()

    # 3. Groups
    keys, ginfo = groups.build_groups(df, rrs, TARGETS)
    df = pd.concat([df.reset_index(drop=True), keys.drop(columns="GLORIA_ID")], axis=1)
    df["region_grp"], ginfo["region_conflict_groups"] = groups.majority_by_group(df["region"], df["wb_group"])
    ginfo["rows_region_ne_region_grp"] = int((df["region"] != df["region_grp"]).sum())
    df["na_subregion"] = gloria.na_subregion_rowwise(df)
    na_rows = df["region_grp"] == "North America"
    sub_grp, ginfo["na_subregion_conflict_groups"] = groups.majority_by_group(df["na_subregion"].where(na_rows), df["wb_group"])
    df["na_subregion_grp"] = sub_grp.where(na_rows)
    assert df.loc[na_rows, "na_subregion_grp"].notna().all(), "NA row without sub-region"
    ginfo["rows_na_subregion_reassigned"] = int((df["na_subregion"].notna() & (df["na_subregion"] != df["na_subregion_grp"])).sum())
    ginfo["region_grp_counts"] = df["region_grp"].value_counts().to_dict()
    ginfo["na_subregion_grp_counts"] = df["na_subregion_grp"].value_counts().to_dict()
    gs = df.groupby("wb_group").size()
    ginfo["wb_group_size"] = {"min": int(gs.min()), "median": float(gs.median()), "max": int(gs.max()),
                              "n_ge_20": int((gs >= 20).sum()), "largest": {k: int(v) for k, v in gs.sort_values(ascending=False).head(6).items()}}
    report["groups"] = ginfo
    _csv(groups.group_table(df, keys), "wb_groups.csv")
    _csv(groups.overrides_table(), "wb_overrides.csv")
    _csv(groups.merge_candidates(df, keys), "wb_merge_candidates.csv")

    # 4. Bands
    wl, R = bands.rrs_matrix(rrs)
    cover = []
    for u in ["S2A", "S3A"]:
        bl = srf.MSI_ALL_VNIR if u == "S2A" else srf.OLCI_ALL
        sim = bands.simulate(R, wl, srfs[u], bl, "")
        for b in bl:
            row = srf_tab[(srf_tab["unit"] == u) & (srf_tab["band"] == b)].iloc[0]
            cover.append({"unit": u, "band": b, "support_lo": row["support_lo"], "support_hi": row["support_hi"],
                          "frac_all": round(float(sim[b].notna().mean()), 4),
                          "frac_primary_qc": round(float(sim.loc[df["qc_primary"].to_numpy(), b].notna().mean()), 4)})
    cover = pd.DataFrame(cover)
    _csv(cover, "band_coverage.csv")
    olci_all = [f"Oa{i}" for i in range(1, 13)]
    hyp_all = sorted(config.HYP_CENTRES + config.HYP_EXCLUDED)
    feats = [bands.hyperspectral(R, wl, config.HYP_CENTRES, "hyp_"),
             bands.simulate(R, wl, srfs["S2A"], config.MSI_BANDS, "msi_"),
             bands.simulate(R, wl, srfs["S3A"], config.OLCI_BANDS, "olci_"),
             bands.hyperspectral(R, wl, config.HYP_EXCLUDED, "xband_hyp_"),
             bands.simulate(R, wl, srfs["S3A"], config.OLCI_EXCLUDED, "xband_olci_"),
             bands.simulate(R, wl, srfs["S2B"], config.MSI_BANDS, "sens_s2b_"),
             bands.simulate(R, wl, srfs["S2C"], config.MSI_BANDS, "sens_s2c_"),
             bands.simulate(R, wl, srfs["S3B"], config.OLCI_BANDS, "sens_s3b_")]
    F = pd.concat(feats, axis=1)
    bval: dict = {"flat_max_abs_err": {u: bands.flat_spectrum_check(s, srf.MSI_ALL_VNIR if u.startswith("S2") else olci_all, wl)
                                       for u, s in srfs.items()}}
    bval["b4_vs_665"] = bands.smooth_b4_check(R, wl, F["msi_B4"].to_numpy(), srfs["S2A"]["B4"])
    comp: dict = {}
    for p in ["hyp", "msi", "olci"]:
        cols = [c for c in F.columns if c.startswith(p + "_")]
        ok = F[cols].notna().all(axis=1)
        F[f"complete_{p}"] = ok.to_numpy()
        comp[p] = {"n_features": len(cols), "n_all": int(ok.sum()), "frac_all": round(float(ok.mean()), 4)}
        for t in TARGETS:
            popm = df[config.PRIMARY_POPULATION[t]].to_numpy()
            comp[p][f"{t}: complete/population"] = f"{int((ok.to_numpy() & popm).sum())}/{int(popm.sum())}"
    alt = {"olci_Oa2-Oa10": [f"olci_Oa{i}" for i in range(2, 11)], "msi_B1-B4": [f"msi_B{i}" for i in range(1, 5)],
           "hyp_400-750 (original Decision)": [f"hyp_{c}" for c in config.HYP_CENTRES] + ["xband_hyp_400", "xband_hyp_750"],
           "olci_Oa1-Oa12 (original Decision)": [f"olci_Oa{i}" for i in range(2, 12)] + ["xband_olci_Oa1", "xband_olci_Oa12"]}
    for name, cols in alt.items():
        ok = F[cols].notna().all(axis=1)
        comp[name] = {"n_all": int(ok.sum()), "frac_all": round(float(ok.mean()), 4)}
        for t in TARGETS:
            popm = df[config.PRIMARY_POPULATION[t]].to_numpy()
            comp[name][f"{t}: complete/population"] = f"{int((ok.to_numpy() & popm).sum())}/{int(popm.sum())}"
    bval["completeness"] = comp
    diffs = {}
    for a, b, bl in [("msi_", "sens_s2b_", config.MSI_BANDS), ("msi_", "sens_s2c_", config.MSI_BANDS), ("olci_", "sens_s3b_", config.OLCI_BANDS)]:
        for x in bl:
            u, v = F[a + x], F[b + x]
            ok = u.notna() & v.notna() & (u.abs() > 1e-4)
            diffs[f"{b}{x}"] = round(float(np.median(np.abs(v[ok] - u[ok]) / np.abs(u[ok]))), 5)
    bval["median_rel_diff_vs_default_unit"] = diffs
    lw = pd.read_csv(GLORIA_DIR / "GLORIA_Lw.csv").set_index("GLORIA_ID").loc[df["GLORIA_ID"]]
    es = pd.read_csv(GLORIA_DIR / "GLORIA_Es.csv").set_index("GLORIA_ID").loc[df["GLORIA_ID"]]
    assert list(lw.columns) == [f"Lw_{w}" for w in wl] and list(es.columns) == [f"Es_{w}" for w in wl]
    irr = pd.concat([bands.irradiance_weighting_error(lw.to_numpy(float), es.to_numpy(float), wl, srfs["S2A"], config.MSI_BANDS).assign(unit="S2A"),
                     bands.irradiance_weighting_error(lw.to_numpy(float), es.to_numpy(float), wl, srfs["S3A"], config.OLCI_BANDS).assign(unit="S3A")])
    _csv(irr, "irradiance_weighting.csv")
    report["bands"] = bval

    # 5. Output table
    keep = ID_COLS + gloria.QC_FLAGS + ["qc_primary", "qc_strict", "qc_none"] + \
        [c for t in TARGETS for c in (t, f"log10_{t}")] + LABEL_COLS + KEY_COLS
    out = pd.concat([df[keep].reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    for c in ["Sample_ID", "SeaBASS_ID", "LIMNADES_ID", "LIMNADES_UID", "Date_Time_UTC", "na_subregion", "na_subregion_grp",
              "Water_type", "Chl_method", "chl_technique", "chl_technique_source", "Organization_ID", "Country_code"]:
        out[c] = out[c].astype("string")
    for c in LABEL_COLS:
        if c not in ("chl_technique", "chl_technique_source"):
            out[c] = out[c].fillna(False).astype(bool)
    assert len(out) == gloria.N_SAMPLES and out["GLORIA_ID"].is_unique
    out.to_parquet(PROCESSED / "gloria.parquet", index=False)

    # 6. Splits and group tables
    report["splits"] = {}
    for t in TARGETS:
        sp, info = splits.build_splits(out, t)
        info["leakage_checked_splits"] = splits.assert_no_leakage(sp, out, t)
        info["summary"] = splits.summarise(sp, out).to_dict(orient="records")
        info["rows"] = int(len(sp))
        sp.to_parquet(PROCESSED / f"splits_{t}.parquet", index=False)
        gt = splits.group_level_table(out, t)
        gt.to_parquet(PROCESSED / f"groups_{t}.parquet", index=False)
        report["splits"][t] = info
        print(f"[splits] {t}: n={info['n']} groups={info['n_wb_group']} rows={len(sp)} leakage assertions passed")

    report["outputs"] = []
    files = [PROCESSED / "gloria.parquet"] + [PROCESSED / f"{k}_{t}.parquet" for t in TARGETS for k in ("splits", "groups")]
    for p in files:
        report["outputs"].append({"path": p.relative_to(PROCESSED.parents[1]).as_posix(), "rows": int(pd.read_parquet(p).shape[0]), "sha256": sha256(p)})
    for name in ["wb_groups.csv", "wb_overrides.csv", "wb_merge_candidates.csv", "srf_manifest.csv", "srf_bands.csv",
                 "band_coverage.csv", "irradiance_weighting.csv", "chl_methods.csv"]:
        p = INTERIM / name
        report["outputs"].append({"path": p.relative_to(PROCESSED.parents[1]).as_posix(), "rows": int(len(pd.read_csv(p))), "sha256": sha256(p)})
    report["elapsed_s"] = round(time.time() - t0, 1)
    with open(INTERIM / "build_report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"[build_all] done in {report['elapsed_s']} s; wb_group={ginfo['n_wb_group']}; report data/interim/build_report.json")


if __name__ == "__main__":
    main()
