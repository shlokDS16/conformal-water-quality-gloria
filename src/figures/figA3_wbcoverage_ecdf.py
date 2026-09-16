"""Fig. A.3: distribution of per-water-body coverage (double column, appendix).

Marginal coverage is an average, so a method can hit the nominal level while individual water
bodies are far from it. One panel per target shows the empirical cumulative distribution of
within-water-body coverage over every test water body of every available repeat, under the
water-body protocol with hyperspectral features at alpha = 0.10. Each interval method is one
step curve, encoded by Okabe-Ito colour AND line style. The vertical grey lines mark the nominal
level and the tolerance; the height of a curve at the nominal level is the share of test water
bodies whose own coverage falls below it.

Data: results/core_v2/pred_*.parquet (CV+ from results/cvplus_v3, Amendment 5).
Run: python -m src.figures.figA3_wbcoverage_ecdf [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import resultsio as R
from . import style as S

NAME = "figA3_wbcoverage_ecdf"
SENSOR, PROTOCOL, ALPHA = "hyp", "waterbody", R.ALPHA_PRIMARY
SEEDS = range(20)
SHOWN = [("mdn", "native", "results/core_v2"), ("lgbm", "gauss", "results/core_v2"),
         ("lgbm", "scp_pool", "results/core_v2"), ("lgbm", "scp_gsub", "results/core_v2"),
         ("lgbm", "cqr_gsub", "results/core_v2"), ("lgbm", "cvplus", "results/cvplus_v3")]
ENC = R.encoding([f"{m}:{h}" for m, h, _ in SHOWN])
MIN_N = 5  # a water body needs at least this many test samples for its own coverage to be shown


def body_coverage(target) -> pd.DataFrame:
    rows = []
    for model, method, tag in SHOWN:
        p = R.predictions(target, SENSOR, PROTOCOL, SEEDS, tag=tag, methods=[method])
        if not len(p):
            continue
        p = p[(p["model"] == model) & np.isclose(p["alpha"].astype(float), ALPHA)]
        if not len(p):
            continue
        p = p.assign(hit=((p["y"] >= p["lower"]) & (p["y"] <= p["upper"])).astype(float))
        g = p.groupby(["seed", "wb_group"], observed=True)["hit"].agg(["mean", "size"])
        g = g[g["size"] >= MIN_N]
        rows.append(pd.DataFrame({"mm": f"{model}:{method}", "coverage": g["mean"].to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["mm", "coverage"])


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    fig, axes = plt.subplots(1, len(R.TARGETS), figsize=(S.DOUBLE_COL, 2.7), sharey=True)
    fig.subplots_adjust(left=0.072, right=0.995, top=0.90, bottom=0.37, wspace=0.08)
    fig.supylabel("Share of test water bodies", fontsize=S.SIZE_LABEL, x=0.012)
    fig.supxlabel("Within-water-body coverage", fontsize=S.SIZE_LABEL, y=0.225)
    for j, target in enumerate(R.TARGETS):
        ax = axes[j]
        d = body_coverage(target)
        if not len(d):
            R.empty_panel(ax)
            continue
        R.nominal_line(ax, ALPHA, horizontal=False)
        for model, method, _tag in SHOWN:
            mm = f"{model}:{method}"
            v = np.sort(d.loc[d["mm"] == mm, "coverage"].to_numpy(float))
            if not len(v):
                continue
            e = ENC[mm]
            ax.step(np.r_[0.0, v], np.r_[0.0, np.arange(1, len(v) + 1) / len(v)], where="post",
                    color=e["color"], linestyle=e["linestyle"], linewidth=1.0,
                    label=f"{R.MODEL_LABEL[model]}, {R.METHOD_LABEL[method]}" if j == 0 else None)
        ax.set_title(S.TARGET_SHORT[target], pad=3)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00" if j == len(R.TARGETS) - 1 else ""])
        ax.set_ylim(0, 1.02)
        ax.grid(True)
        S.panel_letter(ax, "abcd"[j], x=0.03, y=0.97, ha="left", va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), handlelength=2.6,
               handletextpad=0.5, columnspacing=1.4)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
