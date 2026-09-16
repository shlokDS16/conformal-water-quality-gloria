"""Result ledger: one row per quotable number (phase 08b).

Reads only files under `results/` and `tables/confirmatory*.csv`, and writes a single ledger
`tables/summary.csv` with the columns of `tables/data_summary.csv` plus a completeness flag:

    key, value, rounding, source_file, filter, status

Every number that a result table, a result figure caption or the manuscript text may quote must
have a key here. Nothing in this module interprets a result; it aggregates and records.

Run (real outputs, after the runs finish):
    python -m src.make_summary
Run (preview on partial results, used while the runs are still going):
    python -m src.make_summary --out tables/_preview_summary.csv

Sources and the Amendment 5 rule
--------------------------------
* `results/core_v2`   core cells (all models, all interval methods) -- rows with
  `method == "cvplus"` are DROPPED (Amendment 5 item 2: CV+ in core_v2 used unpermuted fold
  training rows, so Barber et al. (2021) Theorem 4 does not apply; those rows are not reported).
* `results/cvplus_v3` the re-run CV+ cells -- only rows with `method == "cvplus"` are kept
  (its `point` rows duplicate the core_v2 LightGBM point rows).
* `results/sens_v2`   sensitivity cells (populations, strict band sets, 5 km grouping).
* `results/noise_v2`  noise sweep (absent until the sweep runs).
* `results/budget_v2` calibration budget, aggregated by the frozen
  `src.analysis.confirmatory.budget_group_summary` / `budget_local_summary`.
* `tables/confirmatory.csv`, `tables/confirmatory_components.csv` when they exist.
Two assertions enforce the Amendment 5 rule: no CV+ row may survive from a tag other than
`cvplus_v3`, and every surviving CV+ row must carry `cvplus_row_perm == True`.

Aggregation across repeats
--------------------------
A cell is (target, sensor, protocol, population, noise, model, method, alpha). Per cell the
ledger records the mean and the 2.5 / 97.5 percentiles across splits, and, for the protocols
where the plan uses it, the Nadeau and Bengio (2003) corrected interval
(`src.eval.metrics.nadeau_bengio_ttest`) with the preregistered unit convention (water bodies,
except the random protocol which uses rows; Amendment 2 item 6, AUDIT_3 A3-3). Infinite widths
and infinite Winkler scores are excluded from means and percentiles; `inf_rate` and
`n_splits_infinite` record them instead. Descriptive-only folds contribute point rows only.

Completeness
------------
`status` is `complete` when the number of splits found equals the number expected for the cell
and `incomplete` otherwise; `n_expected` is repeated in the `filter` text. Expected counts come
from the split files (`data/processed/splits_<target>.parquet`, `splits5km_<target>.parquet`),
reduced by the run matrix of research/EXPERIMENT_PLAN.md section 8 for the blocks that run fewer
seeds (sensitivities and 5 km: 10 seeds; noise sweep: 5 seeds; MDN on MSI contributor folds:
5 seeds). The script is re-runnable at any time and works on partial results.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import metrics as E

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TAB = ROOT / "tables"

CORE_DIR = "results/core_v2"
CVPLUS_DIR = "results/cvplus_v3"
SENS_DIR = "results/sens_v2"
NOISE_DIR = "results/noise_v2"
BUDGET_DIR = "results/budget_v2"

TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
CELL_KEYS = ["target", "sensor", "protocol", "population", "noise_mult", "noise_add", "model", "method", "alpha"]

# Statistics kept per interval cell: column -> (stat kinds, rounding string).
#   "dist" = mean + 2.5/97.5 percentile across splits; "mean" = mean across splits only.
INTERVAL_STATS = {
    "cov_wb": ("dist", "3 decimals"),
    "cov_pooled": ("dist", "3 decimals"),
    "cov_worst_ge10": ("dist", "3 decimals"),
    "width_wb": ("dist", "3 significant figures"),
    "width_wb_log10mean": ("dist", "4 decimals"),
    "winkler_wb": ("dist", "3 significant figures"),
    "winkler_pooled": ("mean", "3 significant figures"),
    "inf_rate": ("mean", "3 decimals"),
    "cov_wb_lo": ("mean", "3 decimals"),
    "cov_wb_hi": ("mean", "3 decimals"),
    "cov_pooled_lo": ("mean", "3 decimals"),
    "cov_pooled_hi": ("mean", "3 decimals"),
    "beta_mean": ("mean", "3 decimals"),
    "beta_q_lo": ("mean", "3 decimals"),
    "beta_q_hi": ("mean", "3 decimals"),
    "cvplus_bound": ("mean", "3 decimals"),
    "k_cal_units": ("mean", "integer"),
    "k_cal_rows": ("mean", "integer"),
    "n_test_wb": ("mean", "integer"),
}
POINT_STATS = {
    "mdsa": ("dist", "percent, 1 decimal"),
    "sspb": ("dist", "percent, 1 decimal"),
    "logmae": ("dist", "3 decimals"),
    "logbias": ("dist", "3 decimals"),
    "logrmse": ("dist", "3 decimals"),
    "n_test": ("mean", "integer"),
    "n_test_wb": ("mean", "integer"),
    "n_fallback_test": ("mean", "integer"),
}
# Endpoints that also get the Nadeau-Bengio corrected interval across repeats.
NB_COLS = {"cov_wb": "3 decimals", "cov_pooled": "3 decimals", "width_wb_log10mean": "4 decimals"}

BUDGET_GROUP_STATS = {
    "mean_cov": "3 decimals", "sd_seed_draw0": "4 decimals", "sd_seed_means": "4 decimals",
    "beta_mean": "3 decimals", "beta_sd": "4 decimals", "sd_draws_conditional_on_split": "4 decimals",
    "inf_rate_mean": "3 decimals", "mean_width_wb_log10": "4 decimals", "n_seeds": "integer",
    "draws_per_seed": "integer",
}
BUDGET_LOCAL_STATS = {"mean_cov": "3 decimals", "sd_cov": "4 decimals", "mean_n_wb": "1 decimal",
                      "inf_rate": "3 decimals", "n_seeds": "integer"}
CONF_FIELDS = {"estimate": "4 decimals", "ci_lo": "4 decimals", "ci_hi": "4 decimals", "t": "3 significant figures",
               "df": "integer", "p_raw": "3 significant figures", "p_holm": "3 significant figures",
               "n_test_mean": "1 decimal", "n_train_mean": "1 decimal"}
CONF_TEXT = ["decision", "binding_arm", "unit", "J", "complete", "note", "test", "alternative", "null_value"]


# ------------------------------------------------------------------ ledger container
class Ledger:
    """Append-only ledger. Keys are unique by construction; a repeat raises."""

    def __init__(self):
        self.rows: list[dict] = []
        self.seen: set[str] = set()

    def add(self, key, value, rounding, source, filt, status="complete"):
        if key in self.seen:
            raise AssertionError(f"duplicate ledger key: {key}")
        self.seen.add(key)
        self.rows.append({"key": key, "value": value, "rounding": rounding, "source_file": source,
                          "filter": filt, "status": status})

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["key", "value", "rounding", "source_file", "filter", "status"])
            w.writeheader()
            w.writerows(self.rows)


# ------------------------------------------------------------------ loading
def _read_dir(d: str, pattern: str = "metrics_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / d / pattern)))
    if not files:
        return pd.DataFrame()
    out = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out["run_dir"] = d
    return out


def load_metrics(core=CORE_DIR, cvplus=CVPLUS_DIR, extra=(SENS_DIR, NOISE_DIR)) -> pd.DataFrame:
    """Core, sensitivity and noise metrics with Amendment 5 applied to the CV+ rows."""
    parts = []
    for d in [core, *extra]:
        m = _read_dir(d)
        if len(m):
            n_cvp = int((m["method"] == "cvplus").sum())
            m = m[m["method"] != "cvplus"].copy()   # Amendment 5 item 2
            m.attrs = {}
            m["n_cvplus_dropped"] = n_cvp
            parts.append(m)
    cv = _read_dir(cvplus)
    if len(cv):
        cv = cv[cv["method"] == "cvplus"].copy()    # its point rows duplicate core_v2
        parts.append(cv)
    if not parts:
        raise FileNotFoundError("no metrics_*.csv found in any results directory")
    met = pd.concat(parts, ignore_index=True)
    met["population"] = met["population"].fillna("primary")
    for c in ("noise_mult", "noise_add"):
        met[c] = met[c].fillna(0.0).astype(float)
    if "cvplus_row_perm" not in met.columns:
        met["cvplus_row_perm"] = np.nan
    cvp = met[met["method"] == "cvplus"]
    # Amendment 5: CV+ is reported from cvplus_v3 only, and every reported CV+ fit permuted its rows.
    assert set(cvp["run_dir"].unique()) <= {cvplus}, \
        f"CV+ rows from a tag other than {cvplus}: {sorted(set(cvp['run_dir'].unique()))}"
    assert bool(cvp["cvplus_row_perm"].eq(True).all()), \
        "a reported CV+ row does not carry cvplus_row_perm == True"
    dup = met.duplicated(subset=CELL_KEYS + ["seed", "fold", "tag"], keep=False)
    if dup.any():
        raise ValueError(f"duplicated split rows: {met.loc[dup, CELL_KEYS + ['seed', 'fold']].head(5)}")
    return met


# ------------------------------------------------------------------ expected repeats
def split_inventory() -> pd.DataFrame:
    """Non-descriptive and total split counts per (target, protocol) from the split files."""
    rows = []
    for t in TARGETS:
        for stem in ("splits", "splits5km"):
            p = DATA / f"{stem}_{t}.parquet"
            if not p.exists():
                continue
            s = pd.read_parquet(p, columns=["protocol", "seed", "fold", "test_unit", "descriptive_only"])
            s["protocol"] = s["protocol"].astype(str)
            key = ["protocol", "seed", "fold", "test_unit"]
            u = s.drop_duplicates(key + ["descriptive_only"])
            g = u.groupby("protocol").agg(n_all=("seed", "size"),
                                          n_desc=("descriptive_only", lambda x: int(x.astype(bool).sum())),
                                          n_seeds=("seed", "nunique"))
            for proto, r in g.iterrows():
                rows.append({"target": t, "protocol": proto, "n_all": int(r["n_all"]),
                             "n_nondesc": int(r["n_all"] - r["n_desc"]),
                             "n_seeds": int(r["n_seeds"])})
    return pd.DataFrame(rows).drop_duplicates(["target", "protocol"]).set_index(["target", "protocol"])


def expected_splits(inv: pd.DataFrame, target, sensor, protocol, population, model, method, noise) -> int:
    """Expected repeats for a cell (EXPERIMENT_PLAN section 8 run matrix). 0 means 'not planned/unknown'."""
    try:
        row = inv.loc[(target, protocol)]
    except KeyError:
        return 0
    n = int(row["n_all"] if method == "point" else row["n_nondesc"])
    # folds per seed, taken from the split file itself: the contributor protocols have 5 folds over
    # 10 seeds and the region protocol 6 folds over 5 seeds, so a fixed divisor mis-counts them.
    per_seed = max(1, round(n / max(1, int(row.get("n_seeds", 20)))))
    if protocol == "waterbody_5km":
        return n                                   # the 5 km split file already holds 10 seeds
    if population != "primary" or sensor.endswith("_strict"):
        return 10 * per_seed                       # blocks D, D2, D3: seeds 0-9
    if noise:
        return 5 * per_seed                        # block F: seeds 0-4
    if model == "mdn" and sensor != "hyp" and protocol.startswith("contributor"):
        return 5 * per_seed                        # block B1: MDN on MSI contributor folds, seeds 0-4
    return n


# ------------------------------------------------------------------ cell aggregation
def _finite(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").to_numpy(float)
    return v[np.isfinite(v)]


def cell_label(row) -> str:
    a = row["alpha"]
    atag = "point" if (isinstance(a, float) and math.isnan(a)) else f"a{round(float(a) * 1000):03d}"
    noise = "" if (row["noise_mult"] == 0 and row["noise_add"] == 0) else \
        f".nm{row['noise_mult']:g}na{row['noise_add']:g}"
    return (f"cell.{row['target']}.{row['sensor']}.{row['protocol']}.{row['population']}{noise}"
            f".{row['model']}.{row['method']}.{atag}")


def runcell_complete(met: pd.DataFrame, inv: pd.DataFrame) -> dict:
    """Per (target, sensor, protocol, population): are all planned split files present?

    A cell can hold exactly as many rows as its own method expects while the run as a whole is
    still missing split files (for example a region protocol whose descriptive-only folds have
    finished and whose remaining folds have not). The cell status uses both counts."""
    out = {}
    keys = ["target", "sensor", "protocol", "population"]
    for key, g in met.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        n = int(g.drop_duplicates(["seed", "fold"]).shape[0])
        n_exp = expected_splits(inv, row["target"], row["sensor"], row["protocol"], row["population"],
                                "lgbm", "point", False)
        out[key] = bool(n_exp) and n == n_exp
    return out


def aggregate_cells(L: Ledger, met: pd.DataFrame, inv: pd.DataFrame, run_ok: dict | None = None) -> dict:
    """One ledger block per cell: distribution across repeats of every reported endpoint."""
    m = met.copy()
    m["descriptive_only"] = m["descriptive_only"].astype(bool)
    m = m[(~m["descriptive_only"]) | (m["method"] == "point")]
    m["alpha"] = pd.to_numeric(m["alpha"], errors="coerce").fillna(-1.0)
    n_cells = n_incomplete = 0
    for key, d in m.groupby(CELL_KEYS, dropna=False, sort=True):
        row = dict(zip(CELL_KEYS, key))
        row["alpha"] = np.nan if row["alpha"] == -1.0 else row["alpha"]
        is_point = row["method"] == "point"
        base = cell_label(row)
        src = f"{sorted(d['run_dir'].unique())[0]}/metrics_*.csv"
        n_splits = int(len(d))
        n_seeds = int(d["seed"].nunique())
        n_exp = expected_splits(inv, row["target"], row["sensor"], row["protocol"], row["population"],
                                row["model"], row["method"], bool(row["noise_mult"] or row["noise_add"]))
        rk = (row["target"], row["sensor"], row["protocol"], row["population"])
        whole_run_ok = True if run_ok is None else run_ok.get(rk, False)
        status = "complete" if (n_exp and n_splits == n_exp and whole_run_ok) else "incomplete"
        # The cell selection is encoded in the key (target.sensor.protocol.population[.noise].model.method.alpha);
        # the filter records only what the key does not carry.
        filt = f"n_expected={n_exp}" + ("" if is_point else "; non-descriptive splits")
        n_cells += 1
        n_incomplete += status == "incomplete"
        L.add(f"{base}.n_splits", n_splits, "integer", src, filt, status)
        L.add(f"{base}.n_seeds", n_seeds, "integer", src, filt, status)
        L.add(f"{base}.n_expected_splits", n_exp, "integer", "research/EXPERIMENT_PLAN.md s8 + split files",
              filt, status)
        stats = POINT_STATS if is_point else INTERVAL_STATS
        if not is_point and "inf_flag" in d.columns:
            L.add(f"{base}.n_splits_infinite", int(d["inf_flag"].eq(True).sum()), "integer",
                  src, filt + "; splits whose conformal quantile was infinite", status)
        for col, (kind, rounding) in stats.items():
            if col not in d.columns:
                continue
            v = _finite(d[col])
            if not len(v):
                continue
            nf = f"{filt}; mean over {len(v)} finite splits"
            L.add(f"{base}.{col}_mean", float(v.mean()), rounding, src, nf, status)
            if kind == "dist" and len(v) > 1:
                lo, hi = np.percentile(v, [2.5, 97.5])
                pf = f"{filt}; percentile over {len(v)} finite splits"
                L.add(f"{base}.{col}_p025", float(lo), rounding, src, pf, status)
                L.add(f"{base}.{col}_p975", float(hi), rounding, src, pf, status)
                # Median across splits. Added 2026-09-16 for AUDIT_4 finding A4-4: the mean width of
                # a cell whose point model extrapolates without bound (the OC-type polynomial) is not
                # a reportable number, and the audit's fix is the median with the 2.5/97.5 percentiles.
                L.add(f"{base}.{col}_median", float(np.median(v)), rounding, src,
                      f"{filt}; median over {len(v)} finite splits", status)
        # Nadeau-Bengio corrected interval across repeats (plan s5 unit convention); the plan uses it for
        # the single-repeat-per-seed protocols only (random, waterbody and the 5 km waterbody sensitivity).
        if not is_point and row["protocol"] in ("random", "waterbody", "waterbody_5km"):
            unit_rows = row["protocol"] == "random"
            ncols = (["n_test", "n_train", "n_cal"] if unit_rows else ["n_test_wb", "n_train_wb", "n_cal_wb"])
            if all(c in d.columns for c in ncols):
                n_te = float(pd.to_numeric(d[ncols[0]], errors="coerce").mean())
                n_tr = float(pd.to_numeric(d[ncols[1]], errors="coerce").mean()
                             + pd.to_numeric(d[ncols[2]], errors="coerce").mean())
                for col, rounding in NB_COLS.items():
                    if col not in d.columns:
                        continue
                    v = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
                    if len(v) < 2 or not np.isfinite(v).all() or not (n_tr > 0):
                        continue
                    r = E.nadeau_bengio_ttest(v, 0.0, n_te, n_tr, "greater")
                    unit = "rows" if unit_rows else "water bodies"
                    f2 = (f"{filt}; Nadeau-Bengio 95 % interval, J={r['J']}, "
                          f"n_test={n_te:.6g}, n_train={n_tr:.6g} ({unit})")
                    L.add(f"{base}.{col}_nb_lo", float(r["ci_lo"]), rounding, src, f2, status)
                    L.add(f"{base}.{col}_nb_hi", float(r["ci_hi"]), rounding, src, f2, status)
                    L.add(f"{base}.{col}_nb_se", float(r["se"]), rounding, src, f2, status)
    return {"n_cells": n_cells, "n_incomplete": n_incomplete}


# ------------------------------------------------------------------ calibration budget
def budget_blocks(L: Ledger, budget_dir: str = BUDGET_DIR) -> dict:
    """Group and local calibration-budget ledgers, aggregated by the frozen confirmatory module."""
    from src.analysis import confirmatory as CF

    out = {"group_rows": 0, "local_rows": 0}
    gfiles = sorted(glob.glob(str(ROOT / budget_dir / "budget_group_*.csv")))
    if gfiles:
        bud = pd.concat([pd.read_csv(f) for f in gfiles], ignore_index=True)
        g = CF.budget_group_summary(bud)
        out["group_rows"] = len(g)
        seeds_exp = 20
        for _, r in g.iterrows():
            base = (f"budget.group.{r['target']}.{r['sensor']}.{r['population']}.{r['method']}"
                    f".ncal{r['n_cal_wb_target']}.a{round(float(r['alpha']) * 1000):03d}")
            status = "complete" if int(r["n_seeds"]) >= seeds_exp else "incomplete"
            filt = (f"beta_law_applicable={r['beta_law_applicable']}; n_expected_seeds={seeds_exp}; "
                    f"draws within a split are dependent (AUDIT_3 A3-11)")
            for col, rounding in BUDGET_GROUP_STATS.items():
                if col in r and pd.notna(r[col]):
                    L.add(f"{base}.{col}", float(r[col]), rounding, f"{budget_dir}/budget_group_*.csv", filt, status)
    lfiles = sorted(glob.glob(str(ROOT / budget_dir / "budget_local_*.csv")))
    if lfiles:
        loc = pd.concat([pd.read_csv(f) for f in lfiles], ignore_index=True)
        lo = CF.budget_local_summary(loc)
        out["local_rows"] = len(lo)
        for _, r in lo.iterrows():
            base = (f"budget.local.{r['target']}.{r['order']}.{r['eval_set']}.{r['score']}.{r['variant']}"
                    f".k{int(r['k'])}.a{round(float(r['alpha']) * 1000):03d}")
            status = "complete" if int(r["n_seeds"]) >= 20 else "incomplete"
            filt = ("per-seed mean over evaluated water bodies, then over seeds; n_expected_seeds=20")
            for col, rounding in BUDGET_LOCAL_STATS.items():
                if col in r and pd.notna(r[col]):
                    L.add(f"{base}.{col}", float(r[col]), rounding,
                          f"{budget_dir}/budget_local_{r['order']}_*.csv", filt, status)
    return out


# ------------------------------------------------------------------ confirmatory tests
def confirmatory_blocks(L: Ledger, tab: Path = TAB) -> dict:
    out = {"tests": 0, "components": 0}
    p = tab / "confirmatory.csv"
    if p.exists():
        c = pd.read_csv(p)
        out["tests"] = len(c)
        for _, r in c.iterrows():
            base = f"conf.{r['hypothesis']}.{r['target']}"
            status = "complete" if bool(r.get("complete", False)) else "incomplete"
            filt = f"{r['hypothesis']} for {r['target']}; {r.get('test', '')}; Holm family of 16, FWER 0.05"
            for f, rounding in CONF_FIELDS.items():
                if f in r and pd.notna(r[f]) and not isinstance(r[f], str):
                    L.add(f"{base}.{f}", float(r[f]), rounding, "tables/confirmatory.csv", filt, status)
            for f in CONF_TEXT:
                if f in r and pd.notna(r[f]) and str(r[f]) != "":
                    L.add(f"{base}.{f}", str(r[f]), "text", "tables/confirmatory.csv", filt, status)
    p = tab / "confirmatory_components.csv"
    if p.exists():
        c = pd.read_csv(p)
        out["components"] = len(c)
        for _, r in c.iterrows():
            base = f"conf.{r['hypothesis']}.{r['target']}.arm.{r['arm']}"
            status = "complete" if int(r.get("J", 0) or 0) >= 20 else "incomplete"
            filt = f"{r['hypothesis']} component arm {r['arm']} for {r['target']} (intersection-union)"
            for f, rounding in CONF_FIELDS.items():
                if f in r and pd.notna(r[f]) and not isinstance(r[f], str):
                    L.add(f"{base}.{f}", float(r[f]), rounding, "tables/confirmatory_components.csv", filt, status)
    return out


# ------------------------------------------------------------------ run inventory
def inventory_blocks(L: Ledger, met: pd.DataFrame, inv: pd.DataFrame) -> None:
    """Row and split counts per run directory and per (target, sensor, protocol, population)."""
    for d, g in met.groupby("run_dir"):
        L.add(f"run.{Path(d).name}.n_metric_rows", int(len(g)), "integer", f"{d}/metrics_*.csv", f"run_dir={d}")
        L.add(f"run.{Path(d).name}.n_splits", int(g.drop_duplicates(
            ["target", "sensor", "protocol", "population", "noise_mult", "noise_add", "seed", "fold"]).shape[0]),
            "integer", f"{d}/metrics_*.csv", f"run_dir={d}; distinct split files")
    keys = ["target", "sensor", "protocol", "population"]
    for key, g in met.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        n = int(g.drop_duplicates(["seed", "fold"]).shape[0])
        n_exp = expected_splits(inv, row["target"], row["sensor"], row["protocol"], row["population"],
                                "lgbm", "point", False)
        L.add(f"runcell.{row['target']}.{row['sensor']}.{row['protocol']}.{row['population']}.n_splits", n,
              "integer", f"{sorted(g['run_dir'].unique())[0]}/metrics_*.csv",
              f"distinct (seed, fold); n_expected={n_exp}",
              "complete" if (n_exp and n == n_exp) else "incomplete")


# ------------------------------------------------------------------ main
def build(out_path: Path, core=CORE_DIR, cvplus=CVPLUS_DIR, extra=(SENS_DIR, NOISE_DIR),
          budget=BUDGET_DIR, tab=TAB) -> dict:
    met = load_metrics(core, cvplus, extra)
    inv = split_inventory()
    L = Ledger()
    info = aggregate_cells(L, met, inv, runcell_complete(met, inv))
    info.update(budget_blocks(L, budget))
    info.update(confirmatory_blocks(L, tab))
    inventory_blocks(L, met, inv)
    L.write(out_path)
    info["ledger_rows"] = len(L.rows)
    info["n_metric_rows"] = int(len(met))
    info["n_incomplete_rows"] = int(sum(r["status"] == "incomplete" for r in L.rows))
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(TAB / "summary.csv"))
    ap.add_argument("--core", default=CORE_DIR)
    ap.add_argument("--cvplus", default=CVPLUS_DIR)
    ap.add_argument("--extra", action="append", default=None, help="extra metrics dirs (default sens_v2, noise_v2)")
    ap.add_argument("--budget", default=BUDGET_DIR)
    ap.add_argument("--tables", default=str(TAB), help="directory holding confirmatory*.csv")
    args = ap.parse_args(argv)
    extra = tuple(args.extra) if args.extra else (SENS_DIR, NOISE_DIR)
    info = build(Path(args.out), args.core, args.cvplus, extra, args.budget, Path(args.tables))
    print(f"[summary] {args.out}: {info['ledger_rows']} ledger rows from {info['n_metric_rows']} metric rows; "
          f"{info['n_cells']} cells ({info['n_incomplete']} incomplete); "
          f"budget group {info['group_rows']}, local {info['local_rows']}; "
          f"confirmatory {info['tests']} tests, {info['components']} components")


if __name__ == "__main__":
    main()
