"""Fig. A.1: target distributions per primary population (single column, appendix).

Data: data/processed/gloria.parquet; population column per target (TARGET_POP), values from the
raw target column (all > 0 in the populations); log-spaced bins. Panel text gives the number
of samples (n) and of water bodies (wb_group) in the population.
Run: python -m src.figures.figA1_targets
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from . import style as S


def fmt(v, _pos):
    return f"{v:g}"


def main() -> None:
    S.apply()
    cols = ["wb_group"] + S.TARGETS + list(S.TARGET_POP.values())
    d = S.load_gloria(cols)
    fig, axes = plt.subplots(2, 2, figsize=(S.SINGLE_COL, 3.35))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.14, hspace=0.75, wspace=0.28)
    for ax, t, letter in zip(axes.ravel(), S.TARGETS, "abcd"):
        sub = d[d[S.TARGET_POP[t]]]
        v = sub[t].to_numpy(float)
        assert (v > 0).all()
        lo, hi = np.floor(np.log10(v.min())), np.ceil(np.log10(v.max()))
        bins = np.logspace(lo, hi, int((hi - lo) * 8) + 1)
        ax.hist(v, bins=bins, color=S.OKABE_ITO["blue"], edgecolor="white", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlim(v.min() / 1.6, v.max() * 1.6)
        span = np.log10(v.max() / v.min())
        ax.xaxis.set_major_locator(LogLocator(base=100 if span > 4.2 else 10, numticks=12))
        ax.xaxis.set_major_formatter(FuncFormatter(fmt))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(S.TARGET_LABEL[t], labelpad=2)
        if letter in "ac":
            ax.set_ylabel("Samples")
        ax.text(0.99, 0.99, f"n = {len(v):,}\nwb = {sub['wb_group'].nunique():,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=S.SIZE_TICK)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        S.panel_letter(ax, letter, x=-0.04, y=1.04)
    S.save(fig, "figA1_targets")


if __name__ == "__main__":
    main()
