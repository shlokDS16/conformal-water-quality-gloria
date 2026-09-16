"""Fig. A.5: coverage and width against spectral distance to the training set (double column, appendix).

Exchangeability is what makes the conformal guarantee hold on average; it says nothing about test
spectra that sit far from anything seen in training. For each split of the water-body protocol
with hyperspectral features, every test row is given the mean Euclidean distance to its five
nearest training rows in the standardised log10 reflectance space (standardisation fitted on that
split's training rows only, so nothing from the test rows enters the distance). Test rows are then
binned into quintiles of that distance, pooled over the available repeats.

Row 1: coverage within each distance bin, one panel per target, one line per interval method,
encoded by Okabe-Ito colour AND marker AND line style, with the nominal level and the tolerance
as grey lines. Row 2: the median multiplicative width within the same bins, on a log axis. A
method whose coverage falls with distance while its width stays flat is failing exactly where a
user most needs a warning.

Data: results/core_v2/pred_*.parquet, data/processed/splits_<target>.parquet (roles) and
data/processed/gloria.parquet (spectra). Only the seeds listed in SEEDS are used, to keep the
distance computation light while the experiments are still running.
Run: python -m src.figures.figA5_knn_distance [--outdir figures/_preview]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import resultsio as R
from . import style as S

NAME = "figA5_knn_distance"
SENSOR, PROTOCOL, ALPHA = "hyp", "waterbody", R.ALPHA_PRIMARY
SEEDS = (0, 1, 2, 3, 4)
K_NEIGHBOURS = 5
N_BINS = 5
SHOWN = [("mdn", "native", "results/core_v2"), ("lgbm", "gauss", "results/core_v2"),
         ("lgbm", "scp_gsub", "results/core_v2"), ("lgbm", "cqr_gsub", "results/core_v2"),
         ("lgbm", "cvplus", "results/cvplus_v3")]
ENC = R.encoding([f"{m}:{h}" for m, h, _ in SHOWN])
BANDS = [f"hyp_{w}" for w in range(405, 746, 5)]
FLOOR = 1e-5  # the reflectance floor used by every model (Amendment 2 item 2)


def knn_distance(target: str) -> pd.DataFrame:
    """Mean distance to the K nearest training spectra, per (seed, GLORIA_ID) test row."""
    spec = S.load_gloria(["GLORIA_ID"] + BANDS).set_index("GLORIA_ID")
    spec = spec.dropna()
    X = np.log10(np.maximum(spec.to_numpy(float), FLOOR))
    pos = {g: i for i, g in enumerate(spec.index)}
    sp = pd.read_parquet(S.DATA / "processed" / f"splits_{target}.parquet",
                         columns=["GLORIA_ID", "protocol", "seed", "fold", "role"])
    sp = sp[(sp["protocol"].astype(str) == PROTOCOL) & sp["seed"].isin(SEEDS) & (sp["fold"] == 0)]
    out = []
    for seed, d in sp.groupby("seed"):
        tr = [pos[g] for g in d.loc[d["role"].astype(str) == "train", "GLORIA_ID"] if g in pos]
        te = [g for g in d.loc[d["role"].astype(str) == "test", "GLORIA_ID"] if g in pos]
        if not tr or not te:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        A = (X[tr] - mu) / sd
        B = (X[[pos[g] for g in te]] - mu) / sd
        # chunked squared-distance matrix, K smallest per test row
        dist = np.empty(len(B))
        for s in range(0, len(B), 512):
            b = B[s:s + 512]
            d2 = ((b[:, None, :] - A[None, :, :]) ** 2).sum(-1)
            k = min(K_NEIGHBOURS, d2.shape[1])
            dist[s:s + len(b)] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, :k]).mean(1)
        out.append(pd.DataFrame({"seed": int(seed), "GLORIA_ID": te, "dist": dist}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["seed", "GLORIA_ID", "dist"])


def binned(target: str) -> dict:
    dist = knn_distance(target)
    if not len(dist):
        return {}
    edges = np.quantile(dist["dist"], np.linspace(0, 1, N_BINS + 1))
    edges[0] -= 1e-9
    dist["bin"] = pd.cut(dist["dist"], edges, labels=False)
    centres = dist.groupby("bin", observed=True)["dist"].median()
    out = {}
    for model, method, tag in SHOWN:
        p = R.predictions(target, SENSOR, PROTOCOL, SEEDS, tag=tag, methods=[method])
        if not len(p):
            continue
        p = p[(p["model"] == model) & np.isclose(p["alpha"].astype(float), ALPHA)]
        if not len(p):
            continue
        p = p.merge(dist, on=["seed", "GLORIA_ID"], how="inner")
        if not len(p):
            continue
        with np.errstate(over="ignore"):
            p = p.assign(hit=((p["y"] >= p["lower"]) & (p["y"] <= p["upper"])).astype(float),
                         width=10.0 ** (p["upper"].to_numpy(float) - p["lower"].to_numpy(float)))
        g = p.groupby("bin", observed=True).agg(cov=("hit", "mean"), width=("width", "median"))
        out[f"{model}:{method}"] = g.join(centres.rename("center"))
    return out


def main() -> None:
    args = R.cli(__doc__)
    S.apply()
    fig, axes = plt.subplots(2, len(R.TARGETS), figsize=(S.DOUBLE_COL, 4.2))
    fig.subplots_adjust(left=0.10, right=0.993, top=0.94, bottom=0.29, wspace=0.32, hspace=0.32)
    fig.supxlabel("Mean distance to the five nearest training spectra", fontsize=S.SIZE_LABEL, y=0.175)
    for j, target in enumerate(R.TARGETS):
        b = binned(target)
        ax, axw = axes[0, j], axes[1, j]
        if not b:
            R.empty_panel(ax)
            R.empty_panel(axw)
            continue
        R.nominal_line(ax, ALPHA)
        for model, method, _tag in SHOWN:
            mm = f"{model}:{method}"
            if mm not in b:
                continue
            g, e = b[mm], ENC[f"{model}:{method}"]
            ax.plot(g["center"], g["cov"], marker=e["marker"], color=e["color"], linestyle=e["linestyle"],
                    linewidth=0.9, label=f"{R.MODEL_LABEL[model]}, {R.METHOD_LABEL[method]}" if j == 0 else None,
                    **R.MARKER_KW)
            axw.plot(g["center"], g["width"], marker=e["marker"], color=e["color"], linestyle=e["linestyle"],
                     linewidth=0.9, **R.MARKER_KW)
        axw.set_yscale("log")
        R.log_plain(axw, "y")
        ax.set_title(S.TARGET_SHORT[target], pad=4)
        ax.set_ylim(0.4, 1.02)
        ax.grid(True, axis="y")
        axw.grid(True, axis="y")
        if j == 0:
            ax.set_ylabel("Coverage")
            axw.set_ylabel("Median width")
        S.panel_letter(ax, "abcd"[j], x=0.0, y=1.02, ha="left", va="bottom")
        S.panel_letter(axw, "efgh"[j], x=0.0, y=1.02, ha="left", va="bottom")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), handletextpad=0.4,
               columnspacing=1.2)
    S.save(fig, NAME, R.outdir_of(args))


if __name__ == "__main__":
    main()
