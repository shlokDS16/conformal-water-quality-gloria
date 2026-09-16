"""Fig. 5: point accuracy by model and split protocol (double column).

One panel per target. Within a panel the x axis carries the point models in a fixed order
(trivial baseline first, then the refitted empirical algorithms, then the learned models) and the
series are the split protocols, encoded by Okabe-Ito colour AND marker shape. The y axis is the
median symmetric accuracy MdSA in per cent on a log scale; the whisker spans the 2.5th to the
97.5th percentile across repeats. A second row of panels repeats the layout for the symmetric
signed percentage bias SSPB, on a symmetric linear scale with a zero line.

Data: per-split metrics rows with `method == "point"` (point accuracy is never read from interval
rows, AUDIT_3 A3-7), hyperspectral features, primary population, no noise.
Run: python -m src.figures.fig05_point_accuracy [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import resultsio as R
from . import style as S

NAME = "fig05_point_accuracy"
SENSOR = "hyp"


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    met = R.metrics()
    d = R.select(met, sensor=SENSOR, method="point", protocol=R.PROTOCOL_ORDER)
    a = R.agg(d, ["mdsa", "sspb"], ["target", "protocol", "model"])

    fig, axes = plt.subplots(2, len(R.TARGETS), figsize=(S.DOUBLE_COL, 4.0), sharex="col")
    fig.subplots_adjust(left=0.105, right=0.995, top=0.93, bottom=0.27, wspace=0.40, hspace=0.18)
    models_present = [m for m in R.MODEL_ORDER if (a["model"] == m).any()]
    for j, target in enumerate(R.TARGETS):
        models = [m for m in models_present if ((a["target"] == target) & (a["model"] == m)).any()]
        x = np.arange(len(models), dtype=float)
        for row, (col, label) in enumerate((("mdsa", "MdSA (%)"), ("sspb", "SSPB (%)"))):
            ax = axes[row, j]
            for i, proto in enumerate(R.PROTOCOL_ORDER):
                sub = a[(a["target"] == target) & (a["protocol"] == proto)].set_index("model")
                xs, ys, lo, hi = [], [], [], []
                for k, m in enumerate(models):
                    if m not in sub.index:
                        continue
                    xs.append(x[k] + (i - 1.5) * 0.17)
                    ys.append(sub.loc[m, f"{col}_mean"])
                    lo.append(sub.loc[m, f"{col}_p025"])
                    hi.append(sub.loc[m, f"{col}_p975"])
                if not xs:
                    continue
                e = R.PROTOCOL_ENC[proto]
                ys, lo, hi = np.array(ys, float), np.array(lo, float), np.array(hi, float)
                ax.errorbar(xs, ys, yerr=np.vstack([np.abs(ys - lo), np.abs(hi - ys)]), fmt=e["marker"],
                            color=e["color"], ecolor=e["color"], elinewidth=0.8, capsize=1.5, linestyle="none",
                            label=R.PROTOCOL_LABEL[proto] if (row == 0 and j == 0) else None, **R.MARKER_KW)
            if col == "mdsa":
                ax.set_yscale("log")
                R.log_plain(ax, "y")
                ax.set_title(S.TARGET_SHORT[target], pad=4)
            else:
                ax.axhline(0.0, color=S.GREY, linewidth=0.8, zorder=0)
            if j == 0:
                ax.set_ylabel(label)
            ax.set_xticks(x)
            ax.set_xticklabels([R.MODEL_LABEL[m] for m in models], rotation=35, ha="right")
            ax.set_xlim(-0.6, len(models) - 0.4)
            ax.grid(True, axis="y")
            S.panel_letter(ax, "abcdefgh"[row * len(R.TARGETS) + j], x=-0.02, y=1.01)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, 0.005), title="Split protocol",
               handletextpad=0.4, columnspacing=1.4)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
