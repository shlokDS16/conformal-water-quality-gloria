"""Data tables (phase 08, data-only): Table 1 dataset summary, Table 2 split protocols.

Every number is computed here from files in data/ and written to the ledger
tables/data_summary.csv (key, value, rounding, source_file, filter). The LaTeX tables
(booktabs, no vertical rules, no resizebox) are rendered from the ledger values only.
No file in results/ or logs/ is read.
Run: python -m src.make_tables_data
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TAB = ROOT / "tables"
GLORIA = "data/processed/gloria.parquet"

TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
POP = {"Chla": "chla_primary", "TSS": "tss_primary", "aCDOM440": "acdom_primary", "Secchi_depth": "secchi_primary"}
HEAD = {"Chla": r"Chl-a (mg\,m$^{-3}$)", "TSS": r"TSS (g\,m$^{-3}$)", "aCDOM440": r"$a_\mathrm{CDOM}(440)$ (m$^{-1}$)",
        "Secchi_depth": r"Secchi depth (m)"}
HEAD2 = {"Chla": r"\begin{tabular}[b]{@{}r@{}}Chl-a\\(mg\,m$^{-3}$)\end{tabular}",
         "TSS": r"\begin{tabular}[b]{@{}r@{}}TSS\\(g\,m$^{-3}$)\end{tabular}",
         "aCDOM440": r"\begin{tabular}[b]{@{}r@{}}$a_\mathrm{CDOM}(440)$\\(m$^{-1}$)\end{tabular}",
         "Secchi_depth": r"\begin{tabular}[b]{@{}r@{}}Secchi\\depth (m)\end{tabular}"}
SENSORS = {
    "hyp": [f"hyp_{w}" for w in range(405, 746, 5)],
    "msi": [f"msi_B{i}" for i in range(1, 7)],
    "olci": [f"olci_Oa{i}" for i in range(2, 12)],
}
SENSOR_NAME = {"hyp": "Hyperspectral", "msi": "S2A MSI", "olci": "S3A OLCI"}
PROTOCOLS = [
    ("random", "Sample", "splits"),
    ("waterbody", "wb\\_group", "splits"),
    ("waterbody_5km", "wb\\_5km", "splits5km"),
    ("contributor_ds", "Dataset\\_ID", "splits"),
    ("contributor", "contrib\\_group", "splits"),
    ("region", "Macro-region", "splits"),
    ("region_na", "NA sub-region", "splits"),
]
PROTO_NAME = {"random": "Random", "waterbody": "Water body", "waterbody_5km": "Water body 5 km",
              "contributor_ds": "Contributor", "contributor": "Contributor component", "region": "Region",
              "region_na": "North American sub-region"}


class Ledger:
    def __init__(self):
        self.rows: list[dict] = []
        self.val: dict[str, float] = {}

    def add(self, key, value, rounding, source, filt):
        assert key not in self.val, key
        self.val[key] = value
        self.rows.append({"key": key, "value": value, "rounding": rounding, "source_file": source, "filter": filt})

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["key", "value", "rounding", "source_file", "filter"])
            w.writeheader()
            w.writerows(self.rows)


def sig3(x: float) -> str:
    return f"{x:#.3g}".rstrip(".") if abs(x) < 1000 else f"{x:,.0f}"


def table1(L: Ledger, d: pd.DataFrame) -> str:
    for t in TARGETS:
        pos = d[t] > 0
        L.add(f"t1.{t}.n_positive_noqc", int(pos.sum()), "integer", GLORIA, f"{t} > 0")
        L.add(f"t1.{t}.n_positive_qcprimary", int((pos & d["qc_primary"]).sum()), "integer", GLORIA,
              f"{t} > 0 & qc_primary")
        sub = d[d[POP[t]]]
        f = f"{POP[t]}"
        L.add(f"t1.{t}.n_rows", len(sub), "integer", GLORIA, f)
        L.add(f"t1.{t}.n_wb_group", int(sub["wb_group"].nunique()), "integer", GLORIA, f + "; nunique wb_group")
        L.add(f"t1.{t}.n_wb_5km", int(sub["wb_5km"].nunique()), "integer", GLORIA, f + "; nunique wb_5km")
        L.add(f"t1.{t}.n_datasets", int(sub["Dataset_ID"].nunique()), "integer", GLORIA, f + "; nunique Dataset_ID")
        L.add(f"t1.{t}.n_regions", int(sub["region_grp"].nunique()), "integer", GLORIA, f + "; nunique region_grp")
        q = sub[t].quantile([0.05, 0.5, 0.95]).to_numpy()
        for name, v in zip(["p05", "median", "p95"], q):
            L.add(f"t1.{t}.{name}", float(v), "3 significant figures", GLORIA, f + f"; {t} quantile (linear)")
        for s, cols in SENSORS.items():
            comp = sub[sub[f"complete_{s}"]]
            L.add(f"t1.{t}.complete_{s}", len(comp), "integer", GLORIA, f + f" & complete_{s}")
            nle0 = int((comp[cols] <= 0).any(axis=1).sum())
            L.add(f"t1.{t}.anyband_le0_{s}", nle0, "integer", GLORIA, f + f" & complete_{s}; any {s} band <= 0")
    v = L.val
    c = lambda k: f"{int(v[k]):,}"  # noqa: E731
    lines = [
        r"\begin{table*}[!tp]",
        r"\centering",
        r"\caption{GLORIA samples per target after quality control and label rules. Band counts refer to rows "
        r"with complete features for the sensor.}",
        r"\label{tab:data}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        " & ".join([""] + [HEAD2[t] for t in TARGETS]) + r" \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\textit{Samples}} \\",
        " & ".join(["\\quad Target $>0$, no QC"] + [c(f"t1.{t}.n_positive_noqc") for t in TARGETS]) + r" \\",
        " & ".join(["\\quad Target $>0$, primary QC"] + [c(f"t1.{t}.n_positive_qcprimary") for t in TARGETS]) + r" \\",
        " & ".join(["\\quad Primary population$^{a}$"] + [c(f"t1.{t}.n_rows") for t in TARGETS]) + r" \\",
        r"\multicolumn{5}{@{}l}{\textit{Units in the primary population}} \\",
        " & ".join(["\\quad Water bodies, 2 km"] + [c(f"t1.{t}.n_wb_group") for t in TARGETS]) + r" \\",
        " & ".join(["\\quad Water bodies, 5 km"] + [c(f"t1.{t}.n_wb_5km") for t in TARGETS]) + r" \\",
        " & ".join(["\\quad Datasets"] + [c(f"t1.{t}.n_datasets") for t in TARGETS]) + r" \\",
        " & ".join(["\\quad Macro-regions"] + [c(f"t1.{t}.n_regions") for t in TARGETS]) + r" \\",
        r"\multicolumn{5}{@{}l}{\textit{Target value}} \\",
        " & ".join(["\\quad Median"] + [sig3(v[f"t1.{t}.median"]) for t in TARGETS]) + r" \\",
        " & ".join(["\\quad 5th percentile"] + [sig3(v[f"t1.{t}.p05"]) for t in TARGETS]) + r" \\",
        " & ".join(["\\quad 95th percentile"] + [sig3(v[f"t1.{t}.p95"]) for t in TARGETS]) + r" \\",
        r"\multicolumn{5}{@{}l}{\textit{Rows with complete features}} \\",
    ]
    for s in SENSORS:
        lines.append(" & ".join([f"\\quad {SENSOR_NAME[s]}"] + [c(f"t1.{t}.complete_{s}") for t in TARGETS]) + r" \\")
    lines.append(r"\multicolumn{5}{@{}l}{\textit{Complete rows with a non-positive band}$^{b}$} \\")
    for s in SENSORS:
        lines.append(" & ".join([f"\\quad {SENSOR_NAME[s]}"] + [c(f"t1.{t}.anyband_le0_{s}") for t in TARGETS])
                     + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip\raggedright\footnotesize "
        r"$^{a}$ Primary QC (no Suspect, QWIP\_fail, Baseline\_shift, Noisy\_red or Noisy\_blue flag), target $>0$; "
        r"Chl-a without absorption line height estimates; Secchi depth without rows where it is at least 0.95 of the "
        r"recorded water depth (bottom-limited). $^{b}$ Bands are floored at $10^{-5}$ sr$^{-1}$ before any logarithm or ratio. "
        r"Hyperspectral: 405 to 745 nm at 5 nm; S2A MSI: B1 to B6; S3A OLCI: Oa2 to Oa11. "
        r"S2A: Sentinel-2A; S3A: Sentinel-3A.",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def table2(L: Ledger, d: pd.DataFrame) -> str:
    unit_of = d.set_index("GLORIA_ID")[["wb_group", "wb_5km"]]
    for t in TARGETS:
        for proto, _unit, stem in PROTOCOLS:
            src = f"data/processed/{stem}_{t}.parquet"
            s = pd.read_parquet(DATA / "processed" / f"{stem}_{t}.parquet",
                                filters=[("protocol", "==", proto)])
            s = s.join(unit_of, on="GLORIA_ID")
            wbcol = "wb_5km" if proto == "waterbody_5km" else "wb_group"
            key = ["seed", "fold", "test_unit"]
            for c in ["protocol", "test_unit", "role"]:
                s[c] = s[c].astype(str)
            n_split = s[key].drop_duplicates().shape[0]
            n_seed = s["seed"].nunique()
            n_desc = s.loc[s["descriptive_only"], key].drop_duplicates().shape[0]
            filt = f"protocol == {proto}"
            L.add(f"t2.{t}.{proto}.n_splits", int(n_split), "integer", src, filt + "; distinct (seed, fold, test_unit)")
            L.add(f"t2.{t}.{proto}.n_seeds", int(n_seed), "integer", src, filt + "; nunique seed")
            L.add(f"t2.{t}.{proto}.n_descriptive", int(n_desc), "integer", src, filt + "; descriptive_only splits")
            g = s.groupby(key + ["role"], observed=True).agg(rows=("GLORIA_ID", "size"), wb=(wbcol, "nunique"))
            g = g.reset_index()
            for role in ["train", "cal", "test"]:
                r = g[g["role"] == role]
                assert len(r) == n_split, (t, proto, role)
                L.add(f"t2.{t}.{proto}.{role}.rows_median", float(np.median(r["rows"])), "integer (median, half up)",
                      src, filt + f"; role == {role}; median over splits of row count")
                L.add(f"t2.{t}.{proto}.{role}.wb_median", float(np.median(r["wb"])), "integer (median, half up)",
                      src, filt + f"; role == {role}; median over splits of nunique {wbcol}")
    v = L.val

    def m(x):
        return f"{int(np.floor(x + 0.5)):,}"

    lines = [
        r"\begin{table*}[!tp]",
        r"\centering",
        r"\caption{Realized split sizes. Cells give the median over splits of the number of samples, with the "
        r"number of water bodies in parentheses. Water body 5 km and Contributor component are sensitivity "
        r"analyses.}",
        r"\label{tab:splits}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Protocol & Unit & Splits$^{a}$ & Training & Calibration & Test \\",
    ]
    for t in TARGETS:
        lines.append(r"\midrule")
        name, unit_txt = HEAD[t].rsplit(" (", 1)
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{name}}} ({unit_txt}}} \\\\")
        for proto, unit, _stem in PROTOCOLS:
            base = f"t2.{t}.{proto}"
            nsp, nse, nd = int(v[base + ".n_splits"]), int(v[base + ".n_seeds"]), int(v[base + ".n_descriptive"])
            sp = f"{nsp}" + (f" ({nd})" if nd else "")
            cells = [f"{m(v[f'{base}.{r}.rows_median'])} ({m(v[f'{base}.{r}.wb_median'])})"
                     for r in ["train", "cal", "test"]]
            lines.append(" & ".join([PROTO_NAME[proto], unit, sp] + cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip\raggedright\footnotesize "
        r"$^{a}$ Seeds times folds; in parentheses, region folds with fewer than 10 test water bodies or from "
        r"South America and Africa, which are reported descriptively only. Counts are from the split files before "
        r"the per-sensor completeness filter. Random and Water body: 60/20/20, 20 seeds; Water body 5 km: 10 seeds; "
        r"Contributor: 10 seeds $\times$ 5 folds; Region: 5 seeds per left-out region; North American sub-region: "
        r"sub-regions of North America left out in turn. Sample units keep same-time replicates and duplicate "
        r"spectra together. "
        r"Water bodies are counted as wb\_group, or wb\_5km for Water body 5 km.",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def haversine_max_km(lat: np.ndarray, lon: np.ndarray) -> float:
    """Maximum pairwise great-circle distance (km) between unique coordinates."""
    xy = np.unique(np.column_stack([lat, lon]), axis=0)
    if len(xy) < 2:
        return 0.0
    la, lo = np.radians(xy[:, 0]), np.radians(xy[:, 1])
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    h = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return float(2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1))).max())


def text_and_figure_counts(L: Ledger, d: pd.DataFrame) -> None:
    """Counts quoted in the manuscript text (keys txt.*) and printed in Figs. 1, 2 and 4 (keys fig01.*,
    fig02.*, fig04.*). Filters repeat those of the figure scripts in src/figures/."""
    from src.data.groups import EXTRA_MERGES, MANUAL_MERGES

    L.add("txt.all.n_rows", len(d), "integer", GLORIA, "all rows (Fig. 1 box 1)")
    L.add("txt.all.n_qc_primary", int(d["qc_primary"].astype(bool).sum()), "integer", GLORIA,
          "qc_primary (Fig. 1 box 2)")
    L.add("txt.all.n_wb_group", int(d["wb_group"].nunique()), "integer", GLORIA, "all rows; nunique wb_group")
    L.add("txt.all.n_wb_5km", int(d["wb_5km"].nunique()), "integer", GLORIA, "all rows; nunique wb_5km")
    L.add("txt.n_manual_merge_sets", len(MANUAL_MERGES) + len(EXTRA_MERGES), "integer", "src/data/groups.py",
          "len(MANUAL_MERGES) + len(EXTRA_MERGES); name sets merged by override")
    ext = []
    for _g, sub in d.dropna(subset=["Latitude", "Longitude"]).groupby("wb_group"):
        ext.append(haversine_max_km(sub["Latitude"].to_numpy(float), sub["Longitude"].to_numpy(float)))
    L.add("txt.all.n_wb_group_extent_gt200km", int((np.asarray(ext) > 200).sum()), "integer", GLORIA,
          "all rows with coordinates; wb_group with max pairwise haversine distance > 200 km")

    chla = d[d["chla_primary"]]
    for cls, name in [("fluorometry", "fluorometry"), ("spectrophotometry", "spectrophotometry"), ("HPLC", "hplc"),
                      ("missing", "method_missing")]:
        L.add(f"txt.Chla.primary.n_{name}", int((chla["chl_technique"] == cls).sum()), "integer", GLORIA,
              f"chla_primary & chl_technique == {cls}")
    L.add("txt.Chla.primary.n_uncorrected", int(chla["chla_uncorrected"].astype(bool).sum()), "integer", GLORIA,
          "chla_primary & chla_uncorrected (Phaeophytin_correction == 0)")
    L.add("txt.Chla.n_alh_excluded", int(((d["Chla"] > 0) & d["qc_primary"] & d["chla_optical_estimate"]).sum()),
          "integer", GLORIA, "Chla > 0 & qc_primary & chla_optical_estimate (removed from chla_primary)")
    sec = d[d["secchi_primary"]]
    L.add("txt.Secchi_depth.primary.n_no_depth", int(sec["Depth"].isna().sum()), "integer", GLORIA,
          "secchi_primary & Depth missing")

    prim = d[list(POP.values())].astype(bool).any(axis=1)
    dup = d[d.duplicated("spec_hash", keep=False)]
    conflict_hash = [h for h, s in dup.groupby("spec_hash")
                     if any(s[t].dropna().nunique() > 1 for t in TARGETS)]
    in_conf = d["spec_hash"].isin(conflict_hash)
    L.add("txt.dup_spectrum.n_sets", int(dup["spec_hash"].nunique()), "integer", GLORIA,
          "spec_hash values shared by >= 2 rows")
    L.add("txt.dup_spectrum.n_sets_label_conflict", len(conflict_hash), "integer", GLORIA,
          "identical-spectrum sets whose rows differ in any non-null target value")
    L.add("txt.primary_union.n_dup_spectrum_label_conflict", int((in_conf & prim).sum()), "integer", GLORIA,
          "rows of label-conflict sets in the union of the four primary populations")

    srf = pd.read_csv(DATA / "interim" / "srf_bands.csv")
    used = (((srf["unit"] == "S2A") & srf["band"].isin([f"B{i}" for i in range(1, 7)]))
            | ((srf["unit"] == "S3A") & srf["band"].isin([f"Oa{i}" for i in range(2, 12)])))
    L.add("txt.srf.min_frac_integral_in_support", float(srf.loc[used, "frac_integral_in_support"].min()),
          "percent, 2 decimals, rounded down", "data/interim/srf_bands.csv",
          "S2A B1-B6 and S3A Oa2-Oa11; min frac_integral_in_support")
    irr = pd.read_csv(DATA / "interim" / "irradiance_weighting.csv")
    iu = (((irr["unit"] == "S2A") & irr["band"].isin([f"B{i}" for i in range(1, 7)]))
          | ((irr["unit"] == "S3A") & irr["band"].isin([f"Oa{i}" for i in range(2, 12)])))
    L.add("txt.irrad.max_median_rel_diff", float(irr.loc[iu, "median_rel_diff"].max()),
          "percent, 2 decimals, rounded up", "data/interim/irradiance_weighting.csv",
          "S2A B1-B6 and S3A Oa2-Oa11; max over bands of median relative difference")
    b2 = d["msi_B2"]
    L.add("txt.msi_B2.positive_p05", float(b2[b2 > 0].quantile(0.05)), "3 significant figures (sr^-1)", GLORIA,
          "all rows; msi_B2 > 0; 5th percentile (linear)")

    # Fig. 2: union of the four primary populations, one marker per wb_group
    u = d[prim]
    wb = u.groupby("wb_group").agg(lat=("Latitude", "mean"), region_grp=("region_grp", "first"))
    L.add("fig02.union_primary.n_rows", len(u), "integer", GLORIA, "union of primary populations")
    L.add("fig02.union_primary.n_wb_group", len(wb), "integer", GLORIA, "union of primary populations; nunique wb_group")
    L.add("fig02.union_primary.n_wb_no_coord", int(wb["lat"].isna().sum()), "integer", GLORIA,
          "union of primary populations; wb_group with no coordinates (not plotted)")
    for reg, n in wb.groupby("region_grp").size().items():
        L.add(f"fig02.region.{reg.replace(' ', '_')}.n_wb_group", int(n), "integer", GLORIA,
              f"union of primary populations; region_grp (first row) == {reg}; legend count")

    # Fig. 4: union of primary populations with complete hyperspectral features; display class by argmax
    wl = np.arange(405, 746, 5)
    h = d[prim & d["complete_hyp"]]
    lam = wl[np.argmax(h[[f"hyp_{w}" for w in wl]].to_numpy(float), axis=1)]
    L.add("fig04.union_primary_complete_hyp.n_rows", len(h), "integer", GLORIA,
          "union of primary populations & complete_hyp")
    for name, lo, hi in [("blue_peaked", 0, 500), ("green_peaked", 500, 600), ("red_peaked", 600, 1000)]:
        L.add(f"fig04.class.{name}.n_rows", int(((lam >= lo) & (lam < hi)).sum()), "integer", GLORIA,
              f"union of primary populations & complete_hyp; argmax wavelength in [{lo}, {hi}) nm")


def main() -> None:
    cols = (["GLORIA_ID", "Dataset_ID", "wb_group", "wb_5km", "region_grp", "qc_primary"] + TARGETS
            + list(POP.values()) + [f"complete_{s}" for s in SENSORS] + sum(SENSORS.values(), [])
            + ["Latitude", "Longitude", "Depth", "chl_technique", "chla_uncorrected", "chla_optical_estimate",
               "spec_hash"])
    d = pd.read_parquet(DATA / "processed" / "gloria.parquet", columns=cols)
    L = Ledger()
    t1 = table1(L, d)
    t2 = table2(L, d)
    text_and_figure_counts(L, d)
    TAB.mkdir(exist_ok=True)
    (TAB / "table1_data.tex").write_text(t1, encoding="utf-8", newline="\n")
    (TAB / "table2_splits.tex").write_text(t2, encoding="utf-8", newline="\n")
    L.write(TAB / "data_summary.csv")

    # Cross-check against the pipeline's own report (data/interim/build_report.json).
    rep = json.loads((DATA / "interim" / "build_report.json").read_text(encoding="utf-8"))
    tc = {r["target"]: r for r in rep["target_counts"]}
    lf = rep["label_flags"]
    checks = []
    for t in TARGETS:
        checks.append((f"{t} positive_no_filter", L.val[f"t1.{t}.n_positive_noqc"], tc[t]["positive_no_filter"]))
        checks.append((f"{t} positive_primary", L.val[f"t1.{t}.n_positive_qcprimary"], tc[t]["positive_primary"]))
        checks.append((f"{t} population", L.val[f"t1.{t}.n_rows"], lf[POP[t]]))
        checks.append((f"{t} n_wb_group", L.val[f"t1.{t}.n_wb_group"], rep["splits"][t]["n_wb_group"]))
    checks.append(("qc_primary", L.val["txt.all.n_qc_primary"], rep["qc_keep_counts"]["qc_primary"]))
    checks.append(("all rows", L.val["txt.all.n_rows"], rep["qc_keep_counts"]["qc_none"]))
    ct = rep["chl_technique_counts_chla_primary"]
    for cls, name in [("fluorometry", "fluorometry"), ("spectrophotometry", "spectrophotometry"), ("HPLC", "hplc"),
                      ("missing", "method_missing")]:
        checks.append((f"chl {cls}", L.val[f"txt.Chla.primary.n_{name}"], ct[cls]))
    checks.append(("aLH excluded", L.val["txt.Chla.n_alh_excluded"],
                   rep["label_flags"]["chla_optical_estimate_primary_qc"]))
    # Values recorded earlier in PREREGISTRATION Amendments 1-2 and AUDIT_2 (hand-typed there; checked here)
    checks.append(("wb_group all (Amendment 1)", L.val["txt.all.n_wb_group"], 387))
    checks.append(("wb_5km all (Amendment 1)", L.val["txt.all.n_wb_5km"], 340))
    checks.append(("extent > 200 km (Amendment 2)", L.val["txt.all.n_wb_group_extent_gt200km"], 17))
    checks.append(("Secchi no Depth (Amendment 2)", L.val["txt.Secchi_depth.primary.n_no_depth"], 995))
    checks.append(("dup label conflict rows (Amendment 2)", L.val["txt.primary_union.n_dup_spectrum_label_conflict"], 64))
    checks.append(("dup sets (AUDIT_2)", L.val["txt.dup_spectrum.n_sets"], 40))
    checks.append(("dup conflict sets (AUDIT_2)", L.val["txt.dup_spectrum.n_sets_label_conflict"], 31))
    checks.append(("fig02 region sum", sum(v for k, v in L.val.items() if k.startswith("fig02.region.")),
                   L.val["fig02.union_primary.n_wb_group"]))
    bad = [c for c in checks if c[1] != c[2]]
    for c in checks:
        print("check", c[0], c[1], c[2], "OK" if c[1] == c[2] else "MISMATCH")
    print("ledger rows", len(L.rows), "mismatches", len(bad))


if __name__ == "__main__":
    main()
