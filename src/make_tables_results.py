"""Result tables (phase 08b, revised 2026-09-16 for AUDIT_4): rendered only from the ledger.

No number is computed here and no file under `results/` is read: every cell is a lookup of a key
written by `src/make_summary.py`, formatted with the `rounding` string recorded next to it in the
ledger. A key that is absent (a cell that has not finished) prints `n.a.`, so the tables compile
on partial results.

Tables:
  Table 3  tables/table3_point.tex        point accuracy per target, model and sensor (water-body
                                          protocol), including the trivial and empirical baselines
  Table 4  tables/table4_coverage.tex     coverage, width, interval score and worst-water-body
                                          coverage at alpha = 0.10, water-body protocol, hyperspectral
  Table 5  tables/table5_worstgroup.tex   worst-water-body coverage by split protocol and the
                                          protocol by method recommendation matrix
  Table 6  tables/table6_sensitivity.tex  sensitivity analyses (populations, strict band sets,
                                          5 km grouping) against the primary cell
  Table 7  tables/table7_budget_local.tex local calibration budget, coverage with the
                                          infinite-interval rate

Reporting rules adopted from research/AUDIT_4.md (binding):
  A4-4  A width mean that is dominated by unbounded extrapolation is not a reportable number. Any
        cell whose across-split mean width exceeds `WIDTH_MEAN_MEDIAN_RATIO` times its median, and
        every cell of the OC-type polynomial baseline, is printed as the median across splits with
        the 2.5 and 97.5 percentiles, marked with a dagger.
  A4-6  Every local-budget coverage is printed next to its infinite-interval rate, and a row with
        an infinite-interval rate of one is printed as "infinite interval" instead of as a
        coverage of 1.000.
  A4-7  Worst-water-body coverage appears in the main coverage table, not only in a later table.
  A4-8  Every coverage printed for a group-subsampled cell carries the number of splits whose
        conformal quantile was infinite, and the caption states that an infinite interval counts as
        covered. The count is zero in the primary cell at every nominal level.

Style: booktabs, no vertical rules, no shading, no \\resizebox, units in the column headers, notes
below the table body (CLAUDE.md section 7; Guide section 7).

Run:  python -m src.make_tables_results
Preview on partial results:
      python -m src.make_tables_results --summary tables/_preview_summary.csv --out tables/_preview
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"

TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
TARGET_HEAD = {"Chla": r"Chl-a", "TSS": r"TSS", "aCDOM440": r"$a_\mathrm{CDOM}(440)$",
               "Secchi_depth": r"Secchi"}
NA = r"n.a."
ALPHA_TAG = "a100"  # alpha = 0.10, the primary nominal level (PREREGISTRATION section 4)

MODEL_NAME = {"const": "Training median", "ridge": "Ridge, log bands", "emp_oc": "Empirical, OC ratio",
              "emp_ndci": "Empirical, NDCI", "emp_nechad": "Empirical, Nechad", "emp_ratio": "Empirical, band ratio",
              "lgbm": "LightGBM", "mdn": "MDN"}
POINT_MODELS = ["const", "ridge", "emp_oc", "emp_ndci", "emp_nechad", "emp_ratio", "lgbm", "mdn"]
SENSOR_NAME = {"hyp": "Hyperspectral, 405 to 745 nm", "msi": "S2A MSI B1 to B6", "olci": "S3A OLCI Oa2 to Oa11"}
# Sensor blocks of Table 3. With all three sensors and every model the table is 24 body rows; if the
# submission layout [preprint,12pt] reports "Float too large for page", regenerate with
# --table3-sensors hyp and move the remaining sensors to the appendix.
TABLE3_SENSORS = ["hyp", "msi", "olci"]

# (model, method) rows of the interval tables, in reporting order.
METHOD_ROWS = [
    ("mdn", "native", "MDN, mixture quantiles"),
    ("mdn", "recal_train20", "MDN, recal. (train)"),
    ("mdn", "recal_cal", "MDN, recal. (calib.)"),
    ("mdn", "nscp_pool", "MDN, norm. conf. (pooled)"),
    ("mdn", "nscp_gsub", "MDN, norm. conf. (subs.)"),
    ("lgbm", "gauss", "LightGBM, Gaussian"),
    ("lgbm", "scp_pool", "LightGBM, split conf. (pooled)"),
    ("lgbm", "scp_gsub", "LightGBM, split conf. (subs.)"),
    ("lgbm", "cqr_pool", "LightGBM, CQR (pooled)"),
    ("lgbm", "cqr_gsub", "LightGBM, CQR (subs.)"),
    ("lgbm", "cvplus", "LightGBM, group CV+"),
]
PROTOCOL_ROWS = [("random", "Random"), ("waterbody", "Water body"), ("contributor_ds", "Contributor"),
                 ("region", "Region")]
# Sensitivity analyses: label -> (sensor, protocol, population)
SENSITIVITIES = [
    ("Primary cell (reference)", "hyp", "waterbody", "primary"),
    ("All quality flags dropped", "hyp", "waterbody", "strict_qc"),
    ("Chl-a HPLC or corrected", "hyp", "waterbody", "chla_hplc"),
    ("Depth at least 3 m", "hyp", "waterbody", "depth_ge3"),
    ("Secchi rows with a depth", "hyp", "waterbody", "secchi_has_depth"),
    ("Non-positive bands excluded", "hyp", "waterbody", "exclude_nonpositive"),
    ("Water bodies at 5 km", "hyp", "waterbody_5km", "primary"),
    ("Strict MSI, B1 to B4", "msi_strict", "waterbody", "primary"),
    ("Strict OLCI, Oa2 to Oa10", "olci_strict", "waterbody", "primary"),
]
COV_TARGET = 0.88  # 1 - alpha - delta (PREREGISTRATION section 4); used only as a printing rule

# --- AUDIT_4 A4-4: when is a mean width unusable?
# The OC-type polynomial extrapolates without bound outside its training range of the band ratio,
# so single splits carry widths of 1e19 and the across-split mean is meaningless. Any cell whose
# mean exceeds this multiple of its median is printed as a median with percentiles instead.
WIDTH_MEAN_MEDIAN_RATIO = 2.0
EXTRAPOLATING_MODELS = {"emp_oc"}
DAGGER = r"$^{\dagger}$"

# --- AUDIT_4 A4-6: local calibration budget
LOCAL_KS = ["0", "1", "2", "5", "10", "20"]
LOCAL_VARIANTS = [("global", "Global"), ("augment", "Global and local"),
                  ("local", "Local only"), ("hybrid", "Hybrid")]
LOCAL_TARGETS = ["Chla", "TSS"]
LOCAL_ORDER, LOCAL_EVAL = "earliest", "common"
INF_TEXT = r"inf.\ int."


# ------------------------------------------------------------------ ledger access
class Summary:
    """Read-only view of tables/summary.csv: value lookup plus formatting by the recorded rounding."""

    def __init__(self, path: Path):
        d = pd.read_csv(path, low_memory=False)
        for c in ("key", "rounding", "status"):
            if c not in d.columns:
                raise ValueError(f"{path} lacks column {c}")
        self.value = dict(zip(d["key"], d["value"]))
        self.rounding = dict(zip(d["key"], d["rounding"]))
        self.status = dict(zip(d["key"], d["status"]))
        self.missing: list[str] = []
        self.incomplete: set[str] = set()

    def raw(self, key: str):
        if key not in self.value:
            self.missing.append(key)
            return None
        if self.status.get(key) == "incomplete":
            self.incomplete.add(key)
        v = self.value[key]
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    def num(self, key: str):
        """Finite float or None (does not record a miss twice)."""
        v = self.raw(key)
        return v if isinstance(v, float) and math.isfinite(v) else None

    def fmt(self, key: str, rounding: str | None = None) -> str:
        v = self.raw(key)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return NA
        if isinstance(v, str):
            return v
        return _format(v, rounding or self.rounding.get(key, "3 significant figures"))


def _format(v: float, rounding: str) -> str:
    r = rounding.lower()
    if r.startswith("integer"):
        return f"{int(round(v)):,}"
    if r.startswith("percent"):
        nd = int(r.split(",")[1].strip().split()[0]) if "decimal" in r else 0
        return f"{v:,.{nd}f}"
    if "decimal" in r:
        nd = int(r.split()[0])
        return f"{v:.{nd}f}"
    if "significant" in r:
        n = int(r.split()[0])
        if v == 0:
            return "0"
        return f"{v:#.{n}g}".rstrip(".")
    return f"{v:g}"


def _sci(v: float) -> str:
    """Three significant figures below 1,000, a LaTeX power of ten above it."""
    if v < 1e3:
        return f"{v:#.3g}".rstrip(".")
    e = int(math.floor(math.log10(v)))
    return rf"${v / 10 ** e:.1f}\times10^{{{e}}}$"


def cell_key(target, sensor, protocol, population, model, method, atag, stat) -> str:
    return f"cell.{target}.{sensor}.{protocol}.{population}.{model}.{method}.{atag}.{stat}"


# ------------------------------------------------------------------ AUDIT_4 printing rules
def cov_cell(S: Summary, base: str, stat: str = "cov_wb_mean",
             marked: list | None = None) -> str:
    """Coverage, with the number of infinite splits appended when it is not zero (A4-8).

    An infinite conformal quantile makes the interval the whole line, so the split counts as covered
    by construction. Wherever that happened the reader must see how often.

    REVIEW_full_v2 V6: pass `marked` to record whether a superscript was actually emitted, so the
    table note can describe the marker in the indicative only when the body carries one. This is the
    same defect AUDIT_5 N5 raised for the dagger, which `dagger_note` already handles."""
    txt = S.fmt(base + stat)
    if txt == NA:
        return NA
    n_inf = S.num(base + "n_splits_infinite")
    if n_inf:
        txt += rf"$^{{{int(round(n_inf))}\infty}}$"
        if marked is not None:
            marked.append(base)
    return txt


def inf_note(marked: list) -> str:
    """The `n\\infty` sentence, in the indicative only when `cov_cell` emitted a superscript.

    REVIEW_full_v2 V6: four Appendix C tables asserted that a superscript in the body marks infinite
    conformal quantiles, while `n_splits_infinite` is zero for every cell they print, so a reviewer
    hunts the body for a marker that is not there. The conditional wording is Table 6's."""
    if marked:
        return ("A coverage with a superscript $n\\infty$ comes from a cell in which $n$ of the 20 "
                "splits had an infinite conformal quantile; such a split covers the test sample by "
                "construction and is counted as covered, which makes the printed coverage an upper "
                "bound for that cell. ")
    return ("A superscript $n\\infty$ would mark a cell in which $n$ of the 20 splits had an "
            "infinite conformal quantile, which counts as covered; that count is zero here. ")


def width_cell(S: Summary, base: str, model: str, flagged: list | None = None,
               label: str = "") -> str:
    """Median multiplicative width, or the mean where the mean is meaningful (A4-4).

    Returns the across-split mean unless the cell extrapolates without bound, in which case the
    median across splits is returned and marked with a dagger. The 2.5 and 97.5 percentiles of every
    daggered cell are appended to `flagged` so that the caller can print them in the table note,
    which keeps the table body inside the text width of both layouts."""
    mean = S.num(base + "width_wb_mean")
    med = S.num(base + "width_wb_median")
    if mean is None:
        return NA
    if med is None:
        return _sci(mean)
    if model in EXTRAPOLATING_MODELS or mean > WIDTH_MEAN_MEDIAN_RATIO * med:
        lo, hi = S.num(base + "width_wb_p025"), S.num(base + "width_wb_p975")
        if flagged is not None and lo is not None and hi is not None:
            flagged.append(f"{label} {_sci(med)} [{_sci(lo)}, {_sci(hi)}]")
        return _sci(med) + DAGGER
    return _sci(mean)


def beta_range(S: Summary, models_methods, protocol="waterbody", sensor="hyp",
               population="primary", atag=ALPHA_TAG) -> str:
    """Range of the theoretical beta mean j/(k+1) over the conformal rows of a table."""
    v = [S.num(cell_key(t, sensor, protocol, population, mo, me, atag, "beta_mean_mean"))
         for t in TARGETS for mo, me, *_ in models_methods]
    v = [x for x in v if x is not None]
    if not v:
        return NA
    return f"{min(v):.3f} to {max(v):.3f}"


def header(caption: str, label: str, colspec: str, tabcolsep: str = "4pt", star: bool = True,
           size: str = r"\footnotesize") -> list[str]:
    """Table preamble. `\\footnotesize` keeps every result table inside the 390 pt text width of the
    submission layout [preprint,12pt,authoryear] as well as the 522 pt of the final two-column layout."""
    env = "table*" if star else "table"
    return [rf"\begin{{{env}}}[!tp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            size, rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}", rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule"]


def na_note(lines: list[str]) -> str:
    """The `n.a.` sentence, emitted only when the table body actually contains an `n.a.` cell.

    REVIEW_full M5: the old wording ("were not available when this table was generated") told the
    reader that the run was unfinished. Every remaining `n.a.` is a not-applicable cell by design,
    an empirical algorithm or a sensitivity analysis that exists for one target only. Table 4 has no
    `n.a.` cell at all, so it must carry no such sentence."""
    return (NA + ": the analysis does not apply to this target. "
            if any(NA in ln for ln in lines) else "")


def dagger_note(flagged: list, extra: str = "") -> str:
    """The dagger sentence, emitted only when `width_cell` actually daggered a value.

    AUDIT_5 N5: `tables/table6_sensitivity.tex` carried a note about a symbol that never occurs in
    its body, which sends a reviewer hunting for it."""
    if not flagged:
        return ""
    return ("Widths marked " + DAGGER + " are medians across splits rather than means"
            + (extra or "") + ". ")


def footer(note: str, star: bool = True) -> list[str]:
    env = "table*" if star else "table"
    return [r"\bottomrule", r"\end{tabular}",
            r"\par\smallskip\raggedright\footnotesize " + note, rf"\end{{{env}}}"]


# ------------------------------------------------------------------ Table 3: point accuracy
def table3_point(S: Summary, sensors: list[str] | None = None) -> str:
    lines = header(
        "Point accuracy under the water-body protocol. Each cell gives the median symmetric accuracy MdSA with "
        "the symmetric signed percentage bias SSPB in parentheses, both in per cent, as the mean over repeats.",
        "tab:point", "@{}l" + "r" * len(TARGETS) + "@{}", "3pt")
    lines.append(r"Model & " + " & ".join(TARGET_HEAD[t] for t in TARGETS) + r" \\")
    lines.append(r" & " + " & ".join([r"MdSA (SSPB)"] * len(TARGETS)) + r" \\")
    for sensor in (sensors or TABLE3_SENSORS):
        block = []
        for model in POINT_MODELS:
            cells = []
            for t in TARGETS:
                base = cell_key(t, sensor, "waterbody", "primary", model, "point", "point", "")
                mdsa = S.fmt(base + "mdsa_mean", "percent, 0 decimals")
                # REVIEW_full m3: the text quotes SSPB to one decimal, so the table must too;
                # an integer rule prints "-0" for a small negative bias.
                sspb = S.fmt(base + "sspb_mean", "percent, 1 decimal")
                cells.append(NA if mdsa == NA else f"{mdsa} ({sspb})")
            if all(c == NA for c in cells):
                continue
            block.append(" & ".join([rf"\quad {MODEL_NAME[model]}"] + cells) + r" \\")
        if not block:
            continue
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + len(TARGETS)}}}{{@{{}}l}}{{\textit{{{SENSOR_NAME[sensor]}}}}} \\")
        lines += block
    lines += footer(
        # REVIEW_final F3: the note named a supplementary CSV that the submission did not carry.
        # src/make_supplementary.py now emits it and make_upload_package.py ships it, so the note
        # names the file itself.
        "Water-body protocol, primary population. MdSA and SSPB follow \\citet{Morley2018}; log-MAE, log-bias "
        "and log-RMSE for the same cells are in Supplementary Table S1. "
        "Empirical rows appear only for the targets that define them (OC-type and NDCI for Chl-a, Nechad-type "
        "for TSS, band-ratio regressions for $a_\\mathrm{CDOM}(440)$ and Secchi depth). "
        "The refitted OC-type polynomial extrapolates without bound outside its training range of the band "
        "ratio, which is visible in its accuracy here and in its interval widths in Table~\\ref{tab:coverage}. "
        "Point accuracy is read from the point rows of the metrics files, never from interval rows. "
        + na_note(lines))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ Table 4: coverage, width, score, worst body
def table4_coverage(S: Summary) -> str:
    blocks = [
        ("Water-body-averaged coverage", "cov", "nominal 0.90"),
        ("Median multiplicative width", "width", None),
        ("Worst-water-body coverage", "cov_worst_ge10_mean",
         "test water bodies with at least 10 samples"),
    ]
    flagged: list[str] = []
    marked: list[str] = []
    lines = header(
        "Interval performance at $\\alpha = 0.10$, water-body protocol, hyperspectral features. Coverage is "
        "averaged over test water bodies; width is the water-body mean of the within-body median multiplicative "
        "width $10^{u-l}$. Mean over repeats.",
        "tab:coverage", "@{}l" + "r" * len(TARGETS) + "@{}", "5pt",
        size=r"\footnotesize\renewcommand{\arraystretch}{0.64}")
    lines.append("Model and interval method & " + " & ".join(TARGET_HEAD[t] for t in TARGETS) + r" \\")
    for title, stat, note in blocks:
        lines.append(r"\midrule")
        head = rf"\multicolumn{{{1 + len(TARGETS)}}}{{@{{}}l}}{{\textit{{{title}}}"
        head += rf". {note}}} \\" if note else r"} \\"
        lines.append(head)
        for model, method, name in METHOD_ROWS:
            cells = []
            for t in TARGETS:
                base = cell_key(t, "hyp", "waterbody", "primary", model, method, ALPHA_TAG, "")
                if stat == "cov":
                    cells.append(cov_cell(S, base, marked=marked))
                elif stat == "width":
                    cells.append(width_cell(S, base, model, flagged,
                                            f"{name}, {TARGET_HEAD[t]}"))
                else:
                    cells.append(S.fmt(base + stat))
            if all(c == NA for c in cells):
                continue
            lines.append(" & ".join([rf"\quad {name}"] + cells) + r" \\")
    beta = beta_range(S, [(m, me) for m, me, _ in METHOD_ROWS])
    lines += footer(
        "Subsampled calibration draws one sample per calibration water body "
        "(single subsampling, \\citealp{Dunn2023}); pooled calibration uses every calibration row. "
        "Interval scores are in Table~\\ref{tab:score}. "
        "For the conformal rows the theoretical mean coverage of the quantile rule, "
        "$\\lceil (1-\\alpha)(k+1) \\rceil / (k+1)$ for the realized $k$, is " + beta + ". "
        + inf_note(marked)
        + "Widths marked " + DAGGER + " are medians across splits, because their mean is dominated by a minority "
        "of splits with unbounded widths; all others are means. "
        "Worst-water-body coverage is a minimum over roughly 14 to 16 qualifying water bodies per split, and "
        "no method claims conditional validity (Remark~\\ref{rem:scope}).")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ Table 5: worst group by protocol, matrix
WORST_BY_PROTOCOL = [("lgbm", "scp_gsub", "Split conformal, subsampled"),
                     ("lgbm", "cvplus", "Group CV+"),
                     ("mdn", "native", "MDN mixture quantiles")]


def table5_worstgroup(S: Summary) -> str:
    marked: list[str] = []
    lines = header(
        "Conditional coverage across split protocols and the protocol by method recommendation matrix at "
        "$\\alpha = 0.10$, hyperspectral features. Panel (a): mean over repeats of the lowest within-body "
        "coverage among test water bodies with at least 10 samples, with the water-body-averaged coverage of "
        "the same cell in parentheses. Panel (b): \\textsc{yes} when the water-body-averaged coverage reaches "
        "$1-\\alpha-\\delta = 0.88$ for every target.",
        "tab:worst", "@{}l" + "r" * len(TARGETS) + "@{}", "6pt")
    lines.append("Protocol & " + " & ".join(TARGET_HEAD[t] for t in TARGETS) + r" \\")
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{1 + len(TARGETS)}}}{{@{{}}l}}{{\textit{{(a) Worst-water-body coverage "
                 rf"(water-body-averaged coverage)}}}} \\")
    for model, method, name in WORST_BY_PROTOCOL:
        block = []
        for proto, pname in PROTOCOL_ROWS:
            cells = []
            for t in TARGETS:
                base = cell_key(t, "hyp", proto, "primary", model, method, ALPHA_TAG, "")
                worst = S.fmt(base + "cov_worst_ge10_mean")
                cov = cov_cell(S, base, marked=marked)
                cells.append(NA if worst == NA else f"{worst} ({cov})")
            if all(c == NA for c in cells):
                continue
            block.append(" & ".join([rf"\quad {pname}"] + cells) + r" \\")
        if not block:
            continue
        lines.append(rf"\multicolumn{{{1 + len(TARGETS)}}}{{@{{}}l}}{{{name}}} \\")
        lines += block
    lines.append(r"\midrule")
    lines.append(rf"\multicolumn{{{1 + len(TARGETS)}}}{{@{{}}l}}{{\textit{{(b) Water-body-averaged coverage "
                 rf"reaches 0.88 in every target}}}} \\")
    lines.append(r"Interval method & " + " & ".join(p for _, p in PROTOCOL_ROWS) + r" \\")
    matrix_methods = [("lgbm", "gauss", "Gaussian residual"),
                      ("lgbm", "scp_pool", "Split conf. (pooled)"),
                      ("lgbm", "scp_gsub", "Split conf. (subs.)"),
                      ("lgbm", "cqr_gsub", "CQR (subs.)"),
                      ("lgbm", "cvplus", "Group CV+"),
                      ("mdn", "native", "MDN quantiles")]
    for model, method, name in matrix_methods:
        cells = []
        for proto, _ in PROTOCOL_ROWS:
            vals, n_inf = [], 0
            for t in TARGETS:
                base = cell_key(t, "hyp", proto, "primary", model, method, ALPHA_TAG, "")
                v = S.num(base + "cov_wb_mean")
                if v is not None:
                    vals.append(v)
                    n_inf += int(round(S.num(base + "n_splits_infinite") or 0))
            if not vals:
                cells.append(NA)
            else:
                mark = rf"$^{{{n_inf}\infty}}$" if n_inf else ""
                cells.append((r"\textsc{yes}" if min(vals) >= COV_TARGET else r"\textsc{no}") + mark)
        lines.append(" & ".join([rf"\quad {name}"] + cells) + r" \\")
    lines += footer(
        "Panel (a): water bodies with fewer than 10 test samples are excluded from the worst-body statistic. "
        "Marginal validity does not imply per-water-body validity, and distribution-free conditional coverage "
        "is impossible in general \\citep{FoygelBarber2021}. "
        "Panel (b): \\textsc{yes} means the tolerance $1-\\alpha-\\delta = 0.88$ is reached for every "
        "target; it summarizes the released cells and is not a hypothesis test. "
        + inf_note(marked)
        + "Contributor and region protocols carry no coverage guarantee (Section~\\ref{sec:theory}). "
        + na_note(lines))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ Table 6: sensitivity summary
def table6_sensitivity(S: Summary) -> str:
    flagged: list[str] = []
    marked: list[str] = []
    lines = header(
        "Sensitivity analyses at $\\alpha = 0.10$ with LightGBM. Each cell gives the water-body-averaged "
        "coverage with the median multiplicative width in parentheses, as the mean over repeats. The first row "
        "of each block is the primary cell for reference.",
        "tab:sensitivity", "@{}l" + "r" * len(TARGETS) + "@{}", "5pt")
    lines.append("Analysis & " + " & ".join(TARGET_HEAD[t] for t in TARGETS) + r" \\")
    for block, method, bname in (("scp", "scp_gsub", "Split conformal, subsampled"),
                                 ("cqr", "cqr_gsub", "CQR, subsampled")):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + len(TARGETS)}}}{{@{{}}l}}{{\textit{{{bname}}}}} \\")
        for label, sensor, protocol, population in SENSITIVITIES:
            cells = []
            for t in TARGETS:
                base = cell_key(t, sensor, protocol, population, "lgbm", method, ALPHA_TAG, "")
                cov = cov_cell(S, base, marked=marked)
                wid = width_cell(S, base, "lgbm", flagged, f"{label}, {TARGET_HEAD[t]}")
                cells.append(NA if cov == NA else f"{cov} ({wid})")
            if all(c == NA for c in cells):
                continue
            lines.append(" & ".join([rf"\quad {label}"] + cells) + r" \\")
    lines += footer(
        "Populations: drop-all-flagged quality control; the Chl-a HPLC or phaeophytin-corrected subset; rows with "
        "a recorded depth of at least 3 m; Secchi rows with any recorded depth; exclusion of rows with a "
        "non-positive band (exploratory). Strict band sets have no red-edge band, so the NDCI ratio feature is "
        "not used. The 5 km analysis replaces the 2 km water-body unit everywhere, including the metric unit. "
        + dagger_note(flagged, " (see Table~\\ref{tab:coverage})")
        + "Sensitivity cells use seeds 0 to 9, the primary cell seeds 0 to 19. "
        + na_note(lines))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ Table 7: local calibration budget
def local_key(target, score, variant, k, stat) -> str:
    return (f"budget.local.{target}.{LOCAL_ORDER}.{LOCAL_EVAL}.{score}.{variant}.k{k}"
            f".{ALPHA_TAG}.{stat}")


def table7_budget_local(S: Summary) -> str:
    lines = header(
        "Local calibration budget at $\\alpha = 0.10$: water-body-averaged coverage when the $k$ earliest-dated "
        "samples of each test water body are added to calibration, evaluated on that water body's later samples "
        "only. Each cell gives the coverage with the infinite-interval rate in parentheses.",
        "tab:localbudget", "@{}l" + "r" * len(LOCAL_VARIANTS) + "@{}", "6pt")
    lines.append("Local samples & "
                 + " & ".join(v for _, v in LOCAL_VARIANTS) + r" \\")
    for target in LOCAL_TARGETS:
        for score, sname in (("scp", "split conformal"), ("cqr", "CQR")):
            block = []
            for k in LOCAL_KS:
                cells = []
                for variant, _ in LOCAL_VARIANTS:
                    cov = S.num(local_key(target, score, variant, k, "mean_cov"))
                    inf = S.num(local_key(target, score, variant, k, "inf_rate"))
                    if cov is None:
                        cells.append(NA)
                    elif inf is not None and inf >= 1.0:
                        # AUDIT_4 A4-6: a coverage of 1.000 with an infinite-interval rate of one is
                        # not a coverage result; the interval is the whole line.
                        cells.append(INF_TEXT)
                    else:
                        cells.append(f"{cov:.3f} ({inf:.2f})" if inf is not None else f"{cov:.3f}")
                if all(c == NA for c in cells):
                    continue
                block.append(" & ".join([rf"\quad $k = {k}$"] + cells) + r" \\")
            if not block:
                continue
            lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{1 + len(LOCAL_VARIANTS)}}}{{@{{}}l}}{{\textit{{"
                         rf"{TARGET_HEAD[target]}, {sname}}}}} \\")
            lines += block
    n_wb = S.fmt(local_key("Chla", "scp", "global", "0", "mean_n_wb"), "1 decimal")
    lines += footer(
        "Local samples are the earliest-dated samples of the test water body, so no later sample informs its own "
        "interval; the evaluation set is held fixed across $k$ (water bodies with more than 20 dated samples, "
        "on average " + n_wb + " water bodies per split for Chl-a). "
        "\\textit{" + INF_TEXT + "} marks a cell whose interval is infinite in every split, which happens "
        "whenever the number of calibration scores is below $\\lceil 1/\\alpha \\rceil - 1 = 9$: the conformal "
        "rank $\\lceil (1-\\alpha)(k+1) \\rceil$ then exceeds $k$ (Eq.~(\\ref{eq:qhat})). Such a cell has a "
        "coverage of 1.000 by construction and is not a coverage result. "
        "The bracketed number is the infinite-interval rate of the cell. "
        "All rows are exploratory (Section~\\ref{sec:stats}).")
    return "\n".join(lines) + "\n"


# ================================================================== Appendix C tables
# REVIEW_full B4: Section 4.4 quotes coverages at alpha = 0.05 and 0.20 and Section 3.6 promises
# per-sensor results, with no table behind either; the interval (Winkler) score was moved out of the
# main coverage table and pointed at "the supplementary CSV". All three now appear in Appendix C.
# AUDIT_4 A4-8 applies to the alpha = 0.05 block: `cov_cell` appends the number of splits with an
# infinite conformal quantile, and the caption states that such a split counts as covered.
APPENDIX_ALPHAS = [("a050", r"$\alpha = 0.05$"), ("a200", r"$\alpha = 0.20$")]
APPENDIX_SENSORS = ["hyp", "msi", "olci"]


def _cov_width_block(S: Summary, sensor: str, atag: str, caption: str, label: str,
                     note_head: str) -> str:
    """One coverage block and one width block for a single (sensor, alpha) combination.

    A single float holding both nominal levels, or all three spectral configurations, is 200 to
    460 pt taller than the text block, which LaTeX reports as "Float too large for page". Each
    combination therefore gets its own float of 24 body rows."""
    flagged: list[str] = []
    marked: list[str] = []
    lines = header(caption, label, "@{}l" + "r" * len(TARGETS) + "@{}", "6pt")
    lines.append("Model and interval method & " + " & ".join(TARGET_HEAD[t] for t in TARGETS) + r" \\")
    for title, stat in (("Coverage", "cov"), ("Median multiplicative width", "width")):
        body = []
        for model, method, name in METHOD_ROWS:
            cells = []
            for t in TARGETS:
                base = cell_key(t, sensor, "waterbody", "primary", model, method, atag, "")
                cells.append(cov_cell(S, base, marked=marked) if stat == "cov"
                             else width_cell(S, base, model, flagged, f"{name}, {TARGET_HEAD[t]}"))
            if all(c == NA for c in cells):
                continue
            body.append(" & ".join([rf"\quad {name}"] + cells) + r" \\")
        if not body:
            continue
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + len(TARGETS)}}}{{@{{}}l}}{{\textit{{{title}}}}} \\")
        lines += body
    note = (note_head
            + inf_note(marked)
            + dagger_note(flagged, " because their mean is dominated by a minority of splits with unbounded "
                                   "widths; the 2.5th and 97.5th percentiles of every such cell are in the "
                                   "released ledger under the same key with the suffix "
                                   "\\texttt{\\_p025} and \\texttt{\\_p975}")
            + na_note(lines))
    lines += footer(note)
    return "\n".join(lines) + "\n"


def tableC1_levels_a050(S: Summary) -> str:
    return _cov_width_block(
        S, "hyp", "a050",
        "Water-body-averaged coverage and median multiplicative width at $\\alpha = 0.05$, water-body "
        "protocol, hyperspectral features, primary populations. Mean over 20 repeats. The primary level "
        "$\\alpha = 0.10$ is Table~\\ref{tab:coverage} and $\\alpha = 0.20$ is Table~\\ref{tab:levels200}.",
        "tab:levels050", "Nominal coverage is 0.95. ")


def tableC1_levels_a200(S: Summary) -> str:
    return _cov_width_block(
        S, "hyp", "a200",
        "Water-body-averaged coverage and median multiplicative width at $\\alpha = 0.20$, water-body "
        "protocol, hyperspectral features, primary populations. Mean over 20 repeats. The primary level "
        "$\\alpha = 0.10$ is Table~\\ref{tab:coverage} and $\\alpha = 0.05$ is Table~\\ref{tab:levels050}.",
        "tab:levels200", "Nominal coverage is 0.80. ")


def tableC2_sensors_msi(S: Summary) -> str:
    return _cov_width_block(
        S, "msi", ALPHA_TAG,
        "Water-body-averaged coverage and median multiplicative width for the Sentinel-2A MSI band set "
        "(B1 to B6) at $\\alpha = 0.10$, water-body protocol, primary populations. Mean over 20 repeats. "
        "The hyperspectral cell is Table~\\ref{tab:coverage} and Sentinel-3A OLCI is "
        "Table~\\ref{tab:sensorolci}.",
        "tab:sensormsi", "Band sets are defined in Section~\\ref{sec:bands}. ")


def tableC2_sensors_olci(S: Summary) -> str:
    return _cov_width_block(
        S, "olci", ALPHA_TAG,
        "Water-body-averaged coverage and median multiplicative width for the Sentinel-3A OLCI band set "
        "(Oa2 to Oa11) at $\\alpha = 0.10$, water-body protocol, primary populations. Mean over 20 "
        "repeats. The hyperspectral cell is Table~\\ref{tab:coverage} and Sentinel-2A MSI is "
        "Table~\\ref{tab:sensormsi}.",
        "tab:sensorolci", "Band sets are defined in Section~\\ref{sec:bands}. ")


def tableC3_score(S: Summary) -> str:
    """Interval (Winkler) score, moved out of the main coverage table."""
    lines = header(
        "Interval score of Eq.~(\\ref{eq:winkler}) at $\\alpha = 0.10$, water-body protocol, hyperspectral "
        "features, primary populations, in $\\log_{10}$ units. Mean over 20 repeats of the water-body average. "
        "Lower is better, and the score rewards a narrow interval only when it covers.",
        "tab:score", "@{}l" + "r" * len(TARGETS) + "@{}", "6pt")
    lines.append("Model and interval method & " + " & ".join(TARGET_HEAD[t] for t in TARGETS) + r" \\")
    lines.append(r"\midrule")
    for model, method, name in METHOD_ROWS:
        cells = [S.fmt(cell_key(t, "hyp", "waterbody", "primary", model, method, ALPHA_TAG,
                                "winkler_wb_mean")) for t in TARGETS]
        if all(c == NA for c in cells):
            continue
        lines.append(" & ".join([rf"\quad {name}"] + cells) + r" \\")
    lines += footer(
        "The interval score is not comparable across targets, because each is a $\\log_{10}$ concentration on "
        "its own scale; it ranks methods within a column. A cell whose interval is unbounded in some splits "
        "carries an unbounded score in those splits, so the rows of the normalized conformal method on "
        "$a_\\mathrm{CDOM}(440)$ should be read with the widths of Table~\\ref{tab:coverage}. "
        + na_note(lines))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main
TABLES = {"table3_point": table3_point, "table4_coverage": table4_coverage,
          "table5_worstgroup": table5_worstgroup, "table6_sensitivity": table6_sensitivity,
          "table7_budget_local": table7_budget_local,
          "tableC1_levels_a050": tableC1_levels_a050,
          "tableC1_levels_a200": tableC1_levels_a200,
          "tableC2_sensors_msi": tableC2_sensors_msi,
          "tableC2_sensors_olci": tableC2_sensors_olci,
          "tableC3_score": tableC3_score}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(TAB / "summary.csv"))
    ap.add_argument("--out", default=str(TAB), help="output directory for the .tex files")
    ap.add_argument("--table3-sensors", default=",".join(TABLE3_SENSORS),
                    help="comma-separated sensor blocks of Table 3 (drop sensors if the float is too tall)")
    args = ap.parse_args(argv)
    TABLE3_SENSORS[:] = [x for x in args.table3_sensors.split(",") if x]
    S = Summary(Path(args.summary))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        (out / f"{name}.tex").write_text(fn(S), encoding="utf-8", newline="\n")
    n_missing = len(S.missing)
    print(f"[tables] wrote {len(TABLES)} tables to {out} from {args.summary}; "
          f"{n_missing} key lookups unavailable, {len(S.incomplete)} keys from incomplete cells")
    if n_missing:
        uniq = sorted(set(S.missing))
        print(f"[tables] first unavailable keys: {uniq[:5]}")


if __name__ == "__main__":
    main()
