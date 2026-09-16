"""Fig. 6: coverage by interval method and split protocol (double column).

One panel per target. The x axis carries the interval methods in reporting order (model-based
first, then conformal); the series are the split protocols, encoded by Okabe-Ito colour AND
marker shape. Points are the mean over repeats of the water-body-averaged coverage; the whisker
spans the 2.5th to the 97.5th percentile across repeats. Two grey reference lines mark the
nominal coverage 1 - alpha and the tolerance 1 - alpha - delta. A grey band behind the
subsampled conformal methods shows the theoretical Beta interval of realised coverage for the
realised number of calibration units (mean over repeats of beta_q_lo and beta_q_hi), which is the
finite-sample law only for those methods.

Data: per-split metrics rows, hyperspectral, primary population, alpha = 0.10, no noise;
CV+ rows come from results/cvplus_v3 only (Amendment 5).
Run: python -m src.figures.fig06_coverage_protocol [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import resultsio as R
from . import style as S

NAME = "fig06_coverage_protocol"
SENSOR = "hyp"
ALPHA = R.ALPHA_PRIMARY


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    met = R.metrics()
    d = R.select(met, sensor=SENSOR, protocol=R.PROTOCOL_ORDER, alpha=ALPHA)
    d = d[[(m, h) in R.MAIN_METHODS for m, h in zip(d["model"], d["method"])]]
    a = R.agg(d, ["cov_wb", "beta_q_lo", "beta_q_hi"], ["target", "protocol", "model", "method"])
    a["mm"] = a["model"] + ":" + a["method"]

    order = [f"{m}:{h}" for m, h in R.MAIN_METHODS if (a["mm"] == f"{m}:{h}").any()]
    fig, axes = plt.subplots(2, 2, figsize=(S.DOUBLE_COL, 4.9), sharey=True)
    fig.subplots_adjust(left=0.078, right=0.995, top=0.945, bottom=0.31, wspace=0.05, hspace=0.30)
    fig.supylabel("Water-body-averaged coverage", fontsize=S.SIZE_LABEL, x=0.012)
    for j, target in enumerate(R.TARGETS):
        ax = axes[j // 2, j % 2]
        x = np.arange(len(order), dtype=float)
        # Beta band for the subsampled conformal methods (the only ones the law applies to)
        b = a[(a["target"] == target) & a["mm"].isin(order) & a["method"].str.endswith("_gsub")]
        if len(b):
            lo, hi = float(np.nanmean(b["beta_q_lo_mean"])), float(np.nanmean(b["beta_q_hi_mean"]))
            if np.isfinite(lo) and np.isfinite(hi):
                ax.axhspan(lo, hi, color=S.LIGHT_GREY, zorder=0,
                           label="Beta interval, subsampled calibration" if j == 0 else None)
        R.nominal_line(ax, ALPHA)
        for i, proto in enumerate(R.PROTOCOL_ORDER):
            sub = a[(a["target"] == target) & (a["protocol"] == proto)].set_index("mm")
            xs, ys, lo_, hi_ = [], [], [], []
            for k, mm in enumerate(order):
                if mm not in sub.index:
                    continue
                xs.append(x[k] + (i - 1.5) * 0.17)
                ys.append(sub.loc[mm, "cov_wb_mean"])
                lo_.append(sub.loc[mm, "cov_wb_p025"])
                hi_.append(sub.loc[mm, "cov_wb_p975"])
            if not xs:
                continue
            e = R.PROTOCOL_ENC[proto]
            ys, lo_, hi_ = np.array(ys, float), np.array(lo_, float), np.array(hi_, float)
            ax.errorbar(xs, ys, yerr=np.vstack([np.abs(ys - lo_), np.abs(hi_ - ys)]), fmt=e["marker"],
                        color=e["color"], ecolor=e["color"], elinewidth=0.8, capsize=1.5, linestyle="none",
                        label=R.PROTOCOL_LABEL[proto] if j == 0 else None, **R.MARKER_KW)
        ax.set_title(S.TARGET_SHORT[target], pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([R.METHOD_LABEL[mm.split(":")[1]] for mm in order], rotation=40, ha="right")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(0.45, 1.02)
        if j < 2:
            ax.set_xticklabels([])
        ax.grid(True, axis="y")
        S.panel_letter(ax, "abcd"[j], x=-0.02, y=1.01)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005), handletextpad=0.4,
               columnspacing=1.4)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
