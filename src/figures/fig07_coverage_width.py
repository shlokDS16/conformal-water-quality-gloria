"""Fig. 7: coverage against interval width (double column).

One panel per target, water-body protocol, hyperspectral features, alpha = 0.10. Each marker is
one interval method: x is the mean over repeats of the median multiplicative width
10^(upper - lower), y the mean over repeats of the water-body-averaged coverage. Whiskers give
the 2.5th to 97.5th percentile across repeats on both axes. Methods are encoded by Okabe-Ito
colour AND marker shape. The grey lines mark the nominal coverage 1 - alpha and the tolerance
1 - alpha - delta; the shaded band is the theoretical Beta interval for the realised number of
calibration units of the subsampled conformal methods. A method is preferable when it sits on or
above the nominal line and as far left as possible.

Data: per-split metrics rows; CV+ from results/cvplus_v3 only (Amendment 5).
Run: python -m src.figures.fig07_coverage_width [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import resultsio as R
from . import style as S

NAME = "fig07_coverage_width"
SENSOR = "hyp"
PROTOCOL = "waterbody"
ALPHA = R.ALPHA_PRIMARY


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    met = R.metrics()
    d = R.select(met, sensor=SENSOR, protocol=PROTOCOL, alpha=ALPHA)
    d = d[[(m, h) in R.MAIN_METHODS for m, h in zip(d["model"], d["method"])]]
    a = R.agg(d, ["cov_wb", "width_wb", "beta_q_lo", "beta_q_hi"], ["target", "model", "method"])
    a["mm"] = a["model"] + ":" + a["method"]

    fig, axes = plt.subplots(1, len(R.TARGETS), figsize=(S.DOUBLE_COL, 3.1))
    fig.subplots_adjust(left=0.075, right=0.993, top=0.91, bottom=0.44, wspace=0.26)
    fig.supylabel("Water-body-averaged coverage", fontsize=S.SIZE_LABEL, x=0.012)
    fig.supxlabel("Median multiplicative width $10^{u-l}$", fontsize=S.SIZE_LABEL, y=0.30)
    for j, target in enumerate(R.TARGETS):
        ax = axes[j]
        sub = a[a["target"] == target].set_index("mm")
        g = sub[sub.index.str.endswith("_gsub")]
        if len(g):
            lo, hi = float(np.nanmean(g["beta_q_lo_mean"])), float(np.nanmean(g["beta_q_hi_mean"]))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.axhspan(lo, hi, color=S.LIGHT_GREY, zorder=0,
                           label="Beta interval, subsampled calibration" if j == 0 else None)
        R.nominal_line(ax, ALPHA)
        for model, method in R.MAIN_METHODS:
            mm = f"{model}:{method}"
            if mm not in sub.index:
                continue
            r = sub.loc[mm]
            if not np.isfinite(r["width_wb_mean"]) or not np.isfinite(r["cov_wb_mean"]):
                continue
            e = R.METHOD_ENC[mm]
            ax.errorbar([r["width_wb_mean"]], [r["cov_wb_mean"]],
                        xerr=[[abs(r["width_wb_mean"] - r["width_wb_p025"])],
                              [abs(r["width_wb_p975"] - r["width_wb_mean"])]],
                        yerr=[[abs(r["cov_wb_mean"] - r["cov_wb_p025"])],
                              [abs(r["cov_wb_p975"] - r["cov_wb_mean"])]],
                        fmt=e["marker"], color=e["color"], ecolor=e["color"], elinewidth=0.7, capsize=1.2,
                        linestyle="none", label=f"{R.MODEL_LABEL[model]}, {R.METHOD_LABEL[method]}" if j == 0
                        else None, **R.MARKER_KW)
        ax.set_xscale("log")
        R.log_plain(ax, "x", minor=False)
        ax.set_title(S.TARGET_SHORT[target], pad=4)
        ax.grid(True)
        S.panel_letter(ax, "abcd"[j], x=-0.02, y=1.01)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005), handletextpad=0.4,
               columnspacing=1.2)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
