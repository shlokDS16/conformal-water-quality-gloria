"""Fig. 9: calibration budget (double column).

Top row, one panel per target: the group-level budget. The x axis is the number of calibration
water bodies drawn (10, 15, 20, 30, 40 and all); the marker is the mean over split seeds of the
per-seed mean water-body-averaged coverage, and the whisker is the 2.5th to 97.5th percentile
across split seeds of ONE draw per seed (draw 0), because the 20 draws inside a split share the
fitted model, the calibration pool and the test set and are not independent (AUDIT_3 A3-11).
The grey band is the theoretical Beta interval of realised coverage for the realised number of
calibration units, which applies to the subsampled calibration scheme.

Bottom row, Chl-a and TSS: the local time-forward budget. The x axis is the number k of earliest
dated samples of each test water body added to calibration; the series are the calibration
variants (global scores only, global plus local, local only with a Mondrian rule, and the hybrid
that switches when k is large enough), encoded by colour AND marker AND line style.

Data: results/budget_v2/budget_group_*.csv and budget_local_earliest_*.csv, alpha = 0.10.
Run: python -m src.figures.fig09_calibration_budget [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import resultsio as R
from . import style as S

NAME = "fig09_calibration_budget"
ALPHA = R.ALPHA_PRIMARY
GROUP_METHODS = ["scp_gsub", "cqr_gsub"]
GROUP_LABEL = {"scp_gsub": "Split conformal, subsampled", "cqr_gsub": "CQR, subsampled"}
GROUP_ENC = R.encoding(GROUP_METHODS)
VARIANTS = ["global", "augment", "local", "hybrid"]
VARIANT_LABEL = {"global": "Global scores only", "augment": "Global plus local", "local": "Local only (Mondrian)",
                 "hybrid": "Hybrid"}
VARIANT_ENC = R.encoding(VARIANTS, offset=2)
NCAL_ORDER = ["10", "15", "20", "30", "40", "all"]


def group_panel(ax, bud, target, first: bool) -> None:
    d = bud[(bud["target"] == target) & np.isclose(bud["alpha"].astype(float), ALPHA)]
    if not len(d):
        R.empty_panel(ax)
        return
    order = [n for n in NCAL_ORDER if (d["n_cal_wb_target"].astype(str) == n).any()]
    x = np.arange(len(order), dtype=float)
    b = d[d["method"] == "scp_gsub"]
    if len(b):
        lo = b.groupby(b["n_cal_wb_target"].astype(str))["beta_q_lo"].mean().reindex(order)
        hi = b.groupby(b["n_cal_wb_target"].astype(str))["beta_q_hi"].mean().reindex(order)
        ax.fill_between(x, lo.to_numpy(float), hi.to_numpy(float), color=S.LIGHT_GREY, zorder=0, step="mid",
                        label="Beta interval, subsampled" if first else None)
    R.nominal_line(ax, ALPHA)
    for i, meth in enumerate(GROUP_METHODS):
        m = d[d["method"] == meth]
        if not len(m):
            continue
        key = m["n_cal_wb_target"].astype(str)
        per_seed = m.groupby([key, "seed"])["cov_wb"].mean()
        mean_cov = per_seed.groupby(level=0).mean().reindex(order)
        d0 = m[m["draw"] == 0].groupby(key.loc[m[m["draw"] == 0].index])["cov_wb"]
        lo = d0.quantile(0.025).reindex(order)
        hi = d0.quantile(0.975).reindex(order)
        e = GROUP_ENC[meth]
        y = mean_cov.to_numpy(float)
        ax.errorbar(x + (i - 0.5) * 0.12, y,
                    yerr=np.vstack([np.abs(y - lo.to_numpy(float)), np.abs(hi.to_numpy(float) - y)]),
                    marker=e["marker"], color=e["color"], ecolor=e["color"], linestyle=e["linestyle"],
                    elinewidth=0.8, capsize=1.5, linewidth=0.9,
                    label=GROUP_LABEL[meth] if first else None, **R.MARKER_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.grid(True, axis="y")


def local_panel(ax, loc, target, first: bool) -> None:
    d = loc[(loc["target"] == target) & (loc["eval_set"] == "common") & (loc["score"] == "scp")
            & np.isclose(loc["alpha"].astype(float), ALPHA)]
    if not len(d):
        R.empty_panel(ax)
        return
    R.nominal_line(ax, ALPHA)
    for variant in VARIANTS:
        m = d[d["variant"] == variant]
        if not len(m):
            continue
        per_seed = m.groupby(["k", "seed"])["coverage"].mean()
        mean_cov = per_seed.groupby(level=0).mean()
        lo = per_seed.groupby(level=0).quantile(0.025)
        hi = per_seed.groupby(level=0).quantile(0.975)
        e = VARIANT_ENC[variant]
        ks = mean_cov.index.to_numpy(float)
        y = mean_cov.to_numpy(float)
        ax.errorbar(ks, y, yerr=np.vstack([np.abs(y - lo.to_numpy(float)), np.abs(hi.to_numpy(float) - y)]),
                    marker=e["marker"], color=e["color"], ecolor=e["color"], linestyle=e["linestyle"],
                    elinewidth=0.8, capsize=1.5, linewidth=0.9,
                    label=VARIANT_LABEL[variant] if first else None, **R.MARKER_KW)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0", "1", "2", "5", "10", "20"])
    ax.grid(True, axis="y")


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    bud = R.budget("group")
    loc = R.budget("local")
    if len(loc):
        loc = loc[loc["order"] == "earliest"]

    fig = plt.figure(figsize=(S.DOUBLE_COL, 4.6))
    gs = fig.add_gridspec(2, 4, left=0.075, right=0.995, top=0.94, bottom=0.20, wspace=0.16, hspace=0.50)
    top = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for j, target in enumerate(R.TARGETS):
        group_panel(top[j], bud, target, first=(j == 0))
        top[j].set_title(S.TARGET_SHORT[target], pad=4)
        top[j].set_ylim(0.6, 1.02)
        if j:
            top[j].set_yticklabels([])
        else:
            top[j].set_ylabel("Coverage")
        top[j].set_xlabel("Calibration water bodies")
        S.panel_letter(top[j], "abcd"[j], x=-0.02, y=1.01)
    bottom = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])]
    for j, target in enumerate(["Chla", "TSS"]):
        local_panel(bottom[j], loc, target, first=(j == 0))
        bottom[j].set_title(f"{S.TARGET_SHORT[target]}, local calibration", pad=4)
        bottom[j].set_xlabel("Earliest dated samples added per test water body, $k$")
        if j == 0:
            bottom[j].set_ylabel("Coverage")
        S.panel_letter(bottom[j], "ef"[j], x=-0.02, y=1.01)
    h1, l1 = top[0].get_legend_handles_labels()
    h2, l2 = bottom[0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005), handletextpad=0.4,
               columnspacing=1.2)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
