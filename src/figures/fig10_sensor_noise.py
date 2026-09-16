"""Fig. 10: sensor comparison and simulated atmospheric-correction noise (double column).

Twelve panels, one column per target.
  Row 1: water-body-averaged coverage for hyperspectral, S2A MSI and S3A OLCI features under the
         water-body protocol at alpha = 0.10, for LightGBM with subsampled split conformal and
         with subsampled CQR. Grey lines mark the nominal level and the tolerance.
  Row 2: the median multiplicative width of the same cells, on a log axis.
  Row 3: the noise sweep. The x axis is the multiplicative noise level m of the simulated
         atmospheric-correction error (Rrs' = Rrs exp(m N1) + a N2, Amendment 3, levels fixed
         from ACIX-Aqua); the series are the sensor and interval method. Until results/noise_v2
         exists these panels are drawn as placeholders.
Whiskers give the 2.5th to 97.5th percentile across repeats. Series carry an Okabe-Ito colour
AND a marker shape AND a line style.

Data: per-split metrics rows (core_v2 for the sensors, noise_v2 for the sweep).
Run: python -m src.figures.fig10_sensor_noise [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import resultsio as R
from . import style as S

NAME = "fig10_sensor_noise"
ALPHA = R.ALPHA_PRIMARY
PROTOCOL = "waterbody"
METHODS = [("lgbm", "scp_gsub"), ("lgbm", "cqr_gsub")]
METHOD_ENC = R.encoding([f"{m}:{h}" for m, h in METHODS])
NOISE_KEYS = [(s, m, h) for s in R.SENSOR_ORDER[1:] for m, h in METHODS]
NOISE_ENC = R.encoding([f"{s}:{m}:{h}" for s, m, h in NOISE_KEYS], offset=2)


def sensor_panel(ax, a, target, col: str, first: bool) -> None:
    """col = 'cov_wb' (row 1) or 'width_wb' (row 2)."""
    sub = a[a["target"] == target]
    if not len(sub):
        R.empty_panel(ax)
        return []
    sensors = [s for s in R.SENSOR_ORDER if (sub["sensor"] == s).any()]
    x = np.arange(len(sensors), dtype=float)
    if col == "cov_wb":
        R.nominal_line(ax, ALPHA)
    for i, (model, method) in enumerate(METHODS):
        s = sub[(sub["model"] == model) & (sub["method"] == method)].set_index("sensor").reindex(sensors)
        e = METHOD_ENC[f"{model}:{method}"]
        y = s[f"{col}_mean"].to_numpy(float)
        ax.errorbar(x + (i - 0.5) * 0.16, y,
                    yerr=np.vstack([np.abs(y - s[f"{col}_p025"].to_numpy(float)),
                                    np.abs(s[f"{col}_p975"].to_numpy(float) - y)]),
                    marker=e["marker"], color=e["color"], ecolor=e["color"], linestyle="none",
                    elinewidth=0.8, capsize=1.5,
                    label=f"LightGBM, {R.METHOD_LABEL[method]}" if first else None, **R.MARKER_KW)
    ax.set_xticks(x)
    ax.set_xticklabels([R.SENSOR_LABEL[s] for s in sensors], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(sensors) - 0.4)
    ax.grid(True, axis="y")
    return sensors


def noise_panel(ax, met, target, first: bool) -> None:
    d = R.select(met, target=target, protocol=PROTOCOL, alpha=ALPHA, noise=True)
    d = d[d["noise_mult"] > 0]
    d = d[[(m, h) in METHODS for m, h in zip(d["model"], d["method"])]]
    if not len(d):
        R.empty_panel(ax, "noise sweep\nnot available yet")
        return
    a = R.agg(d, ["cov_wb"], ["sensor", "model", "method", "noise_mult"])
    R.nominal_line(ax, ALPHA)
    for s, model, method in NOISE_KEYS:
        sub = a[(a["sensor"] == s) & (a["model"] == model) & (a["method"] == method)].sort_values("noise_mult")
        if not len(sub):
            continue
        e = NOISE_ENC[f"{s}:{model}:{method}"]
        ax.plot(sub["noise_mult"], sub["cov_wb_mean"], marker=e["marker"], color=e["color"],
                linestyle=e["linestyle"], linewidth=0.9,
                label=f"{R.SENSOR_LABEL[s]}, {R.METHOD_LABEL[method]}" if first else None, **R.MARKER_KW)
    ax.grid(True, axis="y")


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    met = R.metrics()
    d = R.select(met, protocol=PROTOCOL, sensor=R.SENSOR_ORDER, alpha=ALPHA)
    d = d[[(m, h) in METHODS for m, h in zip(d["model"], d["method"])]]
    a = R.agg(d, ["cov_wb", "width_wb"], ["target", "sensor", "model", "method"])

    fig, axes = plt.subplots(3, len(R.TARGETS), figsize=(S.DOUBLE_COL, 6.4))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.135, wspace=0.30, hspace=0.55)
    for j, target in enumerate(R.TARGETS):
        sensor_panel(axes[0, j], a, target, "cov_wb", first=(j == 0))
        axes[0, j].set_title(S.TARGET_SHORT[target], pad=4)
        axes[0, j].set_ylim(0.6, 1.02)
        if j:
            axes[0, j].set_yticklabels([])
        else:
            axes[0, j].set_ylabel("Coverage")
        sensor_panel(axes[1, j], a, target, "width_wb", first=False)
        axes[1, j].set_yscale("log")
        R.log_plain(axes[1, j], "y")
        if j == 0:
            axes[1, j].set_ylabel("Median width")
        noise_panel(axes[2, j], met, target, first=(j == 0))
        axes[2, j].set_xlabel("Noise level $m$")
        if j == 0:
            axes[2, j].set_ylabel("Coverage")
        for row in range(3):
            S.panel_letter(axes[row, j], "abcdefghijkl"[row * len(R.TARGETS) + j], x=-0.02, y=1.01)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h3, l3 = axes[2, 0].get_legend_handles_labels()
    fig.legend(h1 + h3, l1 + l3, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.4, columnspacing=1.2)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
