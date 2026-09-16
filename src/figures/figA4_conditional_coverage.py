"""Fig. A.4: conditional coverage by spectral display class (double column, appendix).

Marginal coverage says nothing about any subgroup. One panel per target shows the coverage of each
interval method (rows) within each spectral display class (columns), pooled over the test rows of
every available repeat of the water-body protocol with hyperspectral features at alpha = 0.10.
The display class is the wavelength of the maximum of the 5 nm spectrum over 405 to 745 nm
(blue-peaked below 500 nm, green-peaked 500 to below 600 nm, red-peaked from 600 nm), the same
rule as Fig. 4; it is not an optical water type and no model uses it.

Colour is a diverging scale centred on the nominal coverage 0.90, from Okabe-Ito blue
(over-coverage) through a neutral tone to Okabe-Ito vermillion (under-coverage); every cell also
carries its value as text, so the panel is readable without colour, and the number of test rows
per class is written to the ledger. This analysis is exploratory: no Mondrian conformal method
conditioned on these classes was fitted unless the optional analysis was released after the
go/no-go.

Data: results/core_v2/pred_*.parquet (CV+ from results/cvplus_v3, Amendment 5) joined to
data/processed/gloria.parquet for the class rule.
Run: python -m src.figures.figA4_conditional_coverage [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import resultsio as R
from . import style as S

NAME = "figA4_conditional_coverage"
SENSOR, PROTOCOL, ALPHA = "hyp", "waterbody", R.ALPHA_PRIMARY
SEEDS = range(20)
SHOWN = [("mdn", "native", "results/core_v2"), ("lgbm", "gauss", "results/core_v2"),
         ("lgbm", "scp_pool", "results/core_v2"), ("lgbm", "scp_gsub", "results/core_v2"),
         ("lgbm", "cqr_gsub", "results/core_v2"), ("lgbm", "cvplus", "results/cvplus_v3")]
CLASSES = ["Blue-peaked", "Green-peaked", "Red-peaked"]
CMAP = LinearSegmentedColormap.from_list("cov", ["#D55E00", "#F2F2F2", "#0072B2"])


def panel_matrix(target, cls_of) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows, counts, labels = [], [], []
    for model, method, tag in SHOWN:
        p = R.predictions(target, SENSOR, PROTOCOL, SEEDS, tag=tag, methods=[method])
        if not len(p):
            continue
        p = p[(p["model"] == model) & np.isclose(p["alpha"].astype(float), ALPHA)]
        if not len(p):
            continue
        cls = cls_of.reindex(p["GLORIA_ID"].to_numpy()).to_numpy()
        hit = ((p["y"] >= p["lower"]) & (p["y"] <= p["upper"])).to_numpy(float)
        d = pd.DataFrame({"cls": cls, "hit": hit}).dropna(subset=["cls"])
        g = d.groupby("cls", observed=False)["hit"].agg(["mean", "size"]).reindex(CLASSES)
        rows.append(g["mean"].to_numpy(float))
        counts.append(g["size"].to_numpy(float))
        labels.append(f"{R.MODEL_LABEL[model]}, {R.METHOD_LABEL[method]}")
    if not rows:
        return np.empty((0, len(CLASSES))), np.empty((0, len(CLASSES))), []
    return np.vstack(rows), np.vstack(counts), labels


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    ids = S.load_gloria(["GLORIA_ID"])["GLORIA_ID"].to_numpy()
    cls_of = R.spectral_class(ids)
    norm = TwoSlopeNorm(vmin=0.70, vcenter=1 - ALPHA, vmax=1.0)

    fig, axes = plt.subplots(1, len(R.TARGETS), figsize=(S.DOUBLE_COL, 3.3))
    fig.subplots_adjust(left=0.225, right=0.995, top=0.90, bottom=0.34, wspace=0.10)
    im = None
    for j, target in enumerate(R.TARGETS):
        ax = axes[j]
        m, n, labels = panel_matrix(target, cls_of)
        if not len(m):
            R.empty_panel(ax)
            continue
        im = ax.imshow(m, cmap=CMAP, norm=norm, aspect="auto")
        for r in range(m.shape[0]):
            for c in range(m.shape[1]):
                if not np.isfinite(m[r, c]):
                    continue
                ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center", fontsize=S.SIZE_TICK - 1,
                        color=S.INK)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels([c.replace("-peaked", "") for c in CLASSES], rotation=30, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if j == 0 else [])
        ax.set_title(S.TARGET_SHORT[target], pad=4)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        S.panel_letter(ax, "abcd"[j], x=(0.09 if j == 0 else -0.02), y=1.02)
    if im is not None:
        cax = fig.add_axes([0.32, 0.145, 0.40, 0.030])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                          ticks=[0.70, 0.80, 1 - ALPHA, 0.95, 1.0])
        cb.set_label("Coverage within the spectral display class (nominal 0.90)", fontsize=S.SIZE_LEGEND)
        cb.ax.tick_params(labelsize=S.SIZE_TICK, length=2)
        cb.outline.set_visible(False)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
