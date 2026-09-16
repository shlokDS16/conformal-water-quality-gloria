"""Fig. 8: realised coverage across the three nominal levels (double column).

One panel per target, water-body protocol, hyperspectral features. The x axis is the nominal
coverage 1 - alpha at the preregistered levels 0.80, 0.90 and 0.95; the y axis is the mean over
repeats of the water-body-averaged coverage, with the 2.5th to 97.5th percentile across repeats
as a whisker. Each interval method is one line, encoded by Okabe-Ito colour AND marker shape AND
line style. The diagonal grey line is perfect calibration; a method above it over-covers and a
method below it under-covers. The grey band is the theoretical Beta interval for the realised
number of calibration units of the subsampled conformal methods at each level.

Data: per-split metrics rows at alpha in {0.20, 0.10, 0.05}; CV+ from results/cvplus_v3 only.
Run: python -m src.figures.fig08_nominal_levels [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import resultsio as R
from . import style as S

NAME = "fig08_nominal_levels"
SENSOR = "hyp"
PROTOCOL = "waterbody"


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    met = R.metrics()
    d = R.select(met, sensor=SENSOR, protocol=PROTOCOL)
    d = d[d["alpha"].isin(R.NOMINAL_LEVELS)]
    d = d[[(m, h) in R.MAIN_METHODS for m, h in zip(d["model"], d["method"])]]
    a = R.agg(d, ["cov_wb", "beta_q_lo", "beta_q_hi"], ["target", "model", "method", "alpha"])
    a["mm"] = a["model"] + ":" + a["method"]
    a["nominal"] = 1.0 - a["alpha"].astype(float)
    levels = sorted(1.0 - np.array(R.NOMINAL_LEVELS, float))

    fig, axes = plt.subplots(1, len(R.TARGETS), figsize=(S.DOUBLE_COL, 2.9), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.42, wspace=0.08)
    for j, target in enumerate(R.TARGETS):
        ax = axes[j]
        sub = a[a["target"] == target]
        g = sub[sub["method"].str.endswith("_gsub")]
        for lev in levels:
            gl = g[np.isclose(g["nominal"], lev)]
            if not len(gl):
                continue
            lo, hi = float(np.nanmean(gl["beta_q_lo_mean"])), float(np.nanmean(gl["beta_q_hi_mean"]))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.add_patch(plt.Rectangle((lev - 0.018, lo), 0.036, hi - lo, color=S.LIGHT_GREY, zorder=0,
                                           label="Beta interval, subsampled" if (j == 0 and lev == levels[0])
                                           else None))
        ax.plot([0.78, 0.97], [0.78, 0.97], color=S.GREY, linewidth=0.9, zorder=0,
                label="Perfect calibration" if j == 0 else None)
        for model, method in R.MAIN_METHODS:
            mm = f"{model}:{method}"
            s = sub[sub["mm"] == mm].sort_values("nominal")
            if not len(s):
                continue
            e = R.METHOD_ENC[mm]
            y = s["cov_wb_mean"].to_numpy(float)
            ax.errorbar(s["nominal"], y,
                        yerr=np.vstack([np.abs(y - s["cov_wb_p025"].to_numpy(float)),
                                        np.abs(s["cov_wb_p975"].to_numpy(float) - y)]),
                        marker=e["marker"], color=e["color"], ecolor=e["color"], linestyle=e["linestyle"],
                        elinewidth=0.7, capsize=1.2, linewidth=0.9,
                        label=f"{R.MODEL_LABEL[model]}, {R.METHOD_LABEL[method]}" if j == 0 else None,
                        **R.MARKER_KW)
        ax.set_title(S.TARGET_SHORT[target], pad=4)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{v:.2f}" for v in levels])
        ax.set_xlim(0.765, 0.985)
        ax.set_xlabel("Nominal coverage")
        if j == 0:
            ax.set_ylabel("Realized coverage")
        ax.grid(True, axis="y")
        S.panel_letter(ax, "abcd"[j], x=-0.02, y=1.01)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005), handletextpad=0.4,
               columnspacing=1.2)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
