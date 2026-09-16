"""Fig. 1: study workflow diagram (double column). SUPERSEDED, kept for reference only.

Fig. 1 of the paper is now the TikZ diagram src/figures/fig01_workflow_tikz.tex, built by
src/figures/build_fig01.py and checked by src/figures/check_diagram.py. This script writes to the
same file names, so it refuses to run unless FIG01_ALLOW_MATPLOTLIB=1 is set, which prevents it
from silently overwriting the figure used by the paper.

The only numbers drawn are counts computed here from data/processed/gloria.parquet.
Run (not for the paper): python -m src.figures.fig01_workflow
"""
from __future__ import annotations

import os
import sys

if os.environ.get("FIG01_ALLOW_MATPLOTLIB") != "1":
    sys.exit("fig01_workflow.py is superseded by the TikZ diagram (build_fig01.py). "
             "Set FIG01_ALLOW_MATPLOTLIB=1 only if you intend to overwrite figures/fig01_workflow.*")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import style as S

LINE = 0.158  # body line spacing in inches (8 pt text)


def box(ax, x, y_top, w, h, title, lines, fill, cols=None):
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=1.0, edgecolor=S.INK, facecolor=fill))
    ax.text(x + 0.08, y_top - 0.08, title, fontsize=S.SIZE_LABEL, fontweight="bold", va="top", ha="left")
    if cols is None:
        for i, t in enumerate(lines):
            ax.text(x + 0.08, y_top - 0.33 - i * LINE, t, fontsize=S.SIZE_TICK, va="top", ha="left")
    else:
        for (cx, head, items) in cols:
            ax.text(x + cx, y_top - 0.33, head, fontsize=S.SIZE_TICK, va="top", ha="left", style="italic")
            for i, t in enumerate(items):
                ax.text(x + cx, y_top - 0.33 - (i + 1) * LINE, t, fontsize=S.SIZE_TICK, va="top", ha="left")


def arrow(ax, p0, p1, connection="arc3"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=S.LINE_WIDTH,
                                 color=S.INK, connectionstyle=connection, shrinkA=0, shrinkB=0))


def main() -> None:
    S.apply()
    d = S.load_gloria(["GLORIA_ID", "qc_primary"])
    n_all = len(d)
    n_qc = int(d["qc_primary"].astype(bool).sum())

    W, H = S.DOUBLE_COL, 2.92
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    tint = {"data": "#EFEFEF", "split": "#DCEBF5", "model": "#D9F0E9", "eval": "#F7E1D3"}

    top1, h1 = H - 0.05, 1.05
    w1, g1, x0 = 1.64, 0.20, 0.05
    xs = [x0 + i * (w1 + g1) for i in range(4)]
    box(ax, xs[0], top1, w1, h1, "GLORIA in situ data",
        [f"{n_all:,} Rrs spectra", "Chl-a, TSS, aCDOM(440),", "Secchi depth", "Lab and site metadata"], tint["data"])
    box(ax, xs[1], top1, w1, h1, "Quality control, labels",
        [f"Primary QC: {n_qc:,} spectra", "log10 targets, value > 0", "Chl-a: no optical aLH", "Secchi: not bottom-limited"],
        tint["data"])
    box(ax, xs[2], top1, w1, h1, "Band simulation",
        ["Hyperspectral 405 to 745 nm", "S2A MSI B1 to B6", "S3A OLCI Oa2 to Oa11", "SRF-weighted band means"],
        tint["data"])
    box(ax, xs[3], top1, w1, h1, "Grouped splits",
        ["Random (sample)", "Water body (primary)", "Contributor (dataset)", "Region (leave one out)"], tint["split"])
    ymid1 = top1 - h1 / 2
    for i in range(3):
        arrow(ax, (xs[i] + w1, ymid1), (xs[i + 1], ymid1))

    top2, h2 = 1.44, 1.39
    widths = [1.50, 3.30, 1.94]
    g2 = (W - 2 * x0 - sum(widths)) / 2
    x2 = [x0, x0 + widths[0] + g2, x0 + widths[0] + widths[1] + 2 * g2]
    box(ax, x2[0], top2, widths[0], h2, "Point models",
        ["Constant median", "Ridge on log bands", "Empirical band ratios", "LightGBM", "MDN"], tint["model"])
    box(ax, x2[1], top2, widths[1], h2, "Interval methods", [], tint["model"],
        cols=[(0.08, "Model-based", ["MDN native quantiles", "Gaussian residual", "Quantile recalibration"]),
              (1.52, "Conformal", ["Split, pooled calibration", "Split, one sample per water body",
                                   "Normalized split (MDN)", "CQR (LightGBM)", "Group CV+ (K = 10)"])])
    box(ax, x2[2], top2, widths[2], h2, "Evaluation",
        ["Water-body-averaged", "  coverage and width", "Point error vs baselines", "Hypotheses H1 to H4",
         "Calibration budget"], tint["eval"])
    ymid2 = top2 - h2 / 2
    arrow(ax, (x2[0] + widths[0], ymid2), (x2[1], ymid2))
    arrow(ax, (x2[1] + widths[1], ymid2), (x2[2], ymid2))

    # Elbow connector: Grouped splits (row 1) to Point models (row 2), routed in the gap.
    xa = xs[3] + w1 / 2
    yb = top1 - h1
    ygap = (yb + top2) / 2
    xm = x2[0] + widths[0] / 2
    ax.plot([xa, xa, xm], [yb, ygap, ygap], color=S.INK, linewidth=S.LINE_WIDTH, solid_capstyle="butt")
    arrow(ax, (xm, ygap), (xm, top2))
    S.save(fig, "fig01_workflow")


if __name__ == "__main__":
    main()
