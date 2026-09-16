"""Shared loading and encoding for the result figures (phase 08b).

Every result figure reads either `tables/summary.csv` (the ledger) or the result files under
`results/`; no number is typed into a figure script. This module centralises

  * loading the per-split metrics with the Amendment 5 rule applied (CV+ only from
    `results/cvplus_v3`, never from `results/core_v2`) by reusing `src.make_summary.load_metrics`;
  * the colour and second-encoding maps (Okabe-Ito colour plus marker or line style), so the
    result figures match the data figures;
  * the label maps and a small argument parser with `--outdir`, used to render previews into
    `figures/_preview/` while the experiments are still running.

Nothing here writes to `results/`.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import style as S

ALPHA_PRIMARY = 0.10
NOMINAL_LEVELS = [0.20, 0.10, 0.05]          # alpha; nominal coverage 0.80, 0.90, 0.95
TARGETS = S.TARGETS

# ------------------------------------------------------------------ labels
PROTOCOL_LABEL = {"random": "Random", "waterbody": "Water body", "contributor_ds": "Contributor",
                  "contributor": "Contributor (graph)", "region": "Region", "region_na": "Region, N. America",
                  "waterbody_5km": "Water body, 5 km"}
PROTOCOL_ORDER = ["random", "waterbody", "contributor_ds", "region"]
SENSOR_LABEL = {"hyp": "Hyperspectral", "msi": "S2A MSI", "olci": "S3A OLCI",
                "msi_strict": "S2A MSI, strict", "olci_strict": "S3A OLCI, strict"}
SENSOR_ORDER = ["hyp", "msi", "olci"]
MODEL_LABEL = {"const": "Training median", "ridge": "Ridge", "emp_oc": "Empirical OC", "emp_ndci": "Empirical NDCI",
               "emp_nechad": "Empirical Nechad", "emp_ratio": "Empirical ratio", "lgbm": "LightGBM", "mdn": "MDN"}
MODEL_ORDER = ["const", "ridge", "emp_oc", "emp_ndci", "emp_nechad", "emp_ratio", "lgbm", "mdn"]
METHOD_LABEL = {"native": "MDN mixture", "recal_train20": "Recal. (train)", "recal_cal": "Recal. (calib.)",
                "gauss": "Gaussian", "scp_pool": "Split, pooled", "scp_gsub": "Split, subsampled",
                "nscp_pool": "Norm. split, pooled", "nscp_gsub": "Norm. split, subsampled",
                "cqr_pool": "CQR, pooled", "cqr_gsub": "CQR, subsampled", "cvplus": "Group CV+"}

# Interval methods shown in the main result figures, in reporting order, with their point model.
MAIN_METHODS = [("mdn", "native"), ("mdn", "recal_train20"), ("mdn", "recal_cal"), ("mdn", "nscp_gsub"),
                ("lgbm", "gauss"), ("lgbm", "scp_pool"), ("lgbm", "scp_gsub"), ("lgbm", "cqr_pool"),
                ("lgbm", "cqr_gsub"), ("lgbm", "cvplus")]

# ------------------------------------------------------------------ encodings (colour + second channel)
# Okabe-Ito order from style.CATEGORICAL, never cycled; every series also carries a marker and a line style.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<"]
LINESTYLES = ["-", "--", "-.", (0, (1, 1)), (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (4, 2)), (0, (2, 2))]


def encoding(keys, offset: int = 0) -> dict:
    """colour, marker and line style per key, in the given order (fixed, never cycled).

    `offset` starts the cycle further along, so two series families drawn in the same figure
    (for example the group budget methods and the local calibration variants) never share a
    colour and marker pair."""
    return {k: {"color": S.CATEGORICAL[(i + offset) % len(S.CATEGORICAL)],
                "marker": MARKERS[(i + offset) % len(MARKERS)],
                "linestyle": LINESTYLES[(i + offset) % len(LINESTYLES)]} for i, k in enumerate(keys)}


PROTOCOL_ENC = encoding(PROTOCOL_ORDER)
SENSOR_ENC = encoding(SENSOR_ORDER)
METHOD_ENC = encoding([f"{m}:{h}" for m, h in MAIN_METHODS])
MODEL_ENC = encoding(MODEL_ORDER)

MARKER_KW = dict(markersize=4.0, markeredgewidth=0.5, markeredgecolor=S.INK)


# ------------------------------------------------------------------ loading
@lru_cache(maxsize=4)
def metrics(core: str = "results/core_v2", cvplus: str = "results/cvplus_v3",
            extra: tuple = ("results/sens_v2", "results/noise_v2")) -> pd.DataFrame:
    """Per-split metrics rows, CV+ taken from cvplus_v3 only (Amendment 5)."""
    from src.make_summary import load_metrics

    met = load_metrics(core, cvplus, extra)
    met["descriptive_only"] = met["descriptive_only"].astype(bool)
    met["alpha"] = pd.to_numeric(met["alpha"], errors="coerce")
    return met


def select(met: pd.DataFrame, *, target=None, sensor=None, protocol=None, population="primary",
           model=None, method=None, alpha=None, noise=False, drop_descriptive=True) -> pd.DataFrame:
    d = met
    if drop_descriptive:
        d = d[~d["descriptive_only"] | (d["method"] == "point")]
    if not noise:
        d = d[(d["noise_mult"] == 0) & (d["noise_add"] == 0)]
    for col, val in (("target", target), ("sensor", sensor), ("protocol", protocol), ("population", population),
                     ("model", model), ("method", method)):
        if val is None:
            continue
        d = d[d[col].isin(val) if isinstance(val, (list, tuple, set)) else d[col] == val]
    if alpha is not None:
        d = d[np.isclose(d["alpha"].astype(float), float(alpha))]
    return d


def agg(d: pd.DataFrame, cols, by) -> pd.DataFrame:
    """Mean and 2.5/97.5 percentiles across splits, infinite values excluded (same rule as the ledger)."""
    def p025(s):
        return np.nanpercentile(s, 2.5) if s.notna().any() else np.nan

    def p975(s):
        return np.nanpercentile(s, 97.5) if s.notna().any() else np.nan

    dd = d.replace([np.inf, -np.inf], np.nan)
    g = dd.groupby(by, dropna=False, observed=True)
    out = g[cols].agg(["mean", "count", p025, p975])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def budget(mode: str, tag: str = "results/budget_v2") -> pd.DataFrame:
    import glob

    files = sorted(glob.glob(str(S.ROOT / tag / f"budget_{mode}_*.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def predictions(target: str, sensor: str = "hyp", protocol: str = "waterbody", seeds=(0,),
                population: str = "primary", tag: str = "results/core_v2",
                columns=("GLORIA_ID", "wb_group", "y", "model", "method", "alpha", "point", "lower", "upper"),
                methods=None) -> pd.DataFrame:
    """Per-test-row predictions and intervals for the given splits (`pred_*.parquet`).

    One frame with a `seed` column. Missing split files are skipped, so the appendix figures work
    on partial results. CV+ rows must come from results/cvplus_v3 (Amendment 5); pass that tag to
    read them."""
    from src.experiment_common import split_stem

    parts = []
    for seed in seeds:
        stem = split_stem(target, sensor, protocol, int(seed), 0, population)
        f = S.ROOT / tag / f"pred_{stem}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=list(columns))
        if methods is not None:
            d = d[d["method"].isin(methods)]
        d["seed"] = int(seed)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def spectral_class(ids) -> pd.Series:
    """Display class of a spectrum from the wavelength of its maximum over 405 to 745 nm, the rule
    already used by Fig. 4 (blue-peaked < 500 nm, green-peaked 500 to < 600 nm, red-peaked >= 600 nm).
    It is a display rule, not an optical water type, and no model uses it."""
    wl = np.arange(405, 746, 5)
    cols = ["GLORIA_ID"] + [f"hyp_{w}" for w in wl]
    g = S.load_gloria(cols).set_index("GLORIA_ID")
    lam = pd.Series(wl[np.argmax(g.to_numpy(float), axis=1)], index=g.index)
    cls = pd.cut(lam, [0, 500, 600, 10_000], right=False,
                 labels=["Blue-peaked", "Green-peaked", "Red-peaked"])
    return pd.Series(pd.Categorical(cls), index=g.index).reindex(pd.Index(ids))


def log_plain(ax, axis: str = "y", minor: bool = True) -> None:
    """Plain decimal labels on a log axis instead of 10^n scientific notation.

    Mathtext superscripts print at 0.8 of the label size (style.py), so an 8 pt tick label carrying
    a 10^n exponent prints at 6.4 pt and fails the 7 pt rule of CLAUDE.md section 7. Plain labels
    keep every span at the full tick size. Minor ticks are labelled only at mantissa 2 and 5, and
    not at all when `minor` is False (a crowded axis spanning many decades)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    def major(v, _pos):
        return "" if v <= 0 else (f"{v:,.0f}" if v >= 1 else f"{v:g}")

    def minor_fmt(v, _pos):
        if v <= 0:
            return ""
        m = v / 10.0 ** np.floor(np.log10(v))
        return major(v, _pos) if int(round(m)) in (2, 5) else ""

    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_formatter(FuncFormatter(major))
    a.set_minor_formatter(FuncFormatter(minor_fmt) if minor else NullFormatter())


def empty_panel(ax, text: str = "not available yet") -> None:
    """Placeholder used only in preview renders, when a run has not produced its cells yet."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=S.SIZE_TICK, color=S.GREY)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def nominal_line(ax, alpha: float = ALPHA_PRIMARY, delta: float = 0.02, horizontal: bool = True) -> None:
    """Nominal coverage 1 - alpha (solid grey) and the tolerance 1 - alpha - delta (dotted grey)."""
    f = ax.axhline if horizontal else ax.axvline
    f(1 - alpha, color=S.GREY, linewidth=0.9, zorder=0)
    f(1 - alpha - delta, color=S.GREY, linewidth=0.8, linestyle=(0, (1, 2)), zorder=0)


def cli(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--outdir", default=None, help="write the PDF and PNG here instead of figures/")
    return ap.parse_args()


def outdir_of(args) -> Path | None:
    return None if args.outdir is None else Path(args.outdir)
