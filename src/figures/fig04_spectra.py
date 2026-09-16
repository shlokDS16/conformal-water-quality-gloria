"""Fig. 4: Rrs spectra envelopes by a display-only spectral class, with SRFs (double column).

Population: rows in the union of the four primary target populations with complete
hyperspectral features (complete_hyp), data/processed/gloria.parquet, columns hyp_405..hyp_745.
Display class (defined here from the data only, used for this figure only, not in any model):
wavelength of the maximum of the 5 nm Rrs spectrum over 405 to 745 nm,
  blue-peaked  < 500 nm; green-peaked 500 to < 600 nm; red-peaked >= 600 nm.
Envelopes: per-wavelength median and 10th to 90th percentile.
SRFs: S2A MSI B1-B6 (S2_MSI_SRF_v5.0.xlsx) and S3A OLCI Oa2-Oa11
(S3A_OL_SRF_20160713_mean_rsr.nc4, 1 nm bin-averaged), loaded with src/data/srf.py,
each band scaled to a peak of 1.
Run: python -m src.figures.fig04_spectra
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.data.srf import load_msi, load_olci

from . import style as S

CLASSES = [("Blue-peaked", 0, 500, S.OKABE_ITO["blue"]),
           ("Green-peaked", 500, 600, S.OKABE_ITO["green"]),
           ("Red-peaked", 600, 1000, S.OKABE_ITO["vermillion"])]
MSI_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6"]
OLCI_BANDS = [f"Oa{i}" for i in range(2, 12)]


def spectral_classes():
    d = S.load_gloria(None)
    pop = d[list(S.TARGET_POP.values())].any(axis=1) & d["complete_hyp"]
    d = d[pop]
    wl = np.arange(405, 746, 5)
    X = d[[f"hyp_{w}" for w in wl]].to_numpy(float)
    lam_max = wl[np.argmax(X, axis=1)]
    return wl, X, lam_max


def srf_panel(ax, srf, bands, color, prefix, label_y):
    for i, b in enumerate(bands):
        s = srf[b]
        s = s[(s.index >= 395) & (s.index <= 760)]
        ax.fill_between(s.index, 0, s.values, color=color, alpha=0.18, linewidth=0)
        ax.plot(s.index, s.values, color=color, linewidth=1.0)
        peak = float((s.index.to_numpy() * s.values).sum() / s.values.sum())
        ax.text(peak, label_y[i % len(label_y)], b.replace(prefix, ""), fontsize=S.SIZE_TICK, ha="center",
                va="bottom", color=S.INK)
    ax.set_xlim(400, 750)
    ax.set_ylim(0, 1.75)
    ax.set_yticks([0, 1])
    ax.set_ylabel("SRF")


def main() -> None:
    S.apply()
    wl, X, lam_max = spectral_classes()
    msi = load_msi("S2A")
    olci = load_olci("S3A")

    fig = plt.figure(figsize=(S.DOUBLE_COL, 4.55))
    gs = fig.add_gridspec(1, 3, left=0.085, right=0.975, top=0.885, bottom=0.50, wspace=0.28)
    gs2 = fig.add_gridspec(2, 1, left=0.085, right=0.975, top=0.39, bottom=0.10, hspace=0.38)
    letters = iter("abcde")
    for j, (name, lo, hi, col) in enumerate(CLASSES):
        ax = fig.add_subplot(gs[0, j])
        m = (lam_max >= lo) & (lam_max < hi)
        q10, q50, q90 = 100 * np.nanpercentile(X[m], [10, 50, 90], axis=0)
        ax.fill_between(wl, q10, q90, color=col, alpha=0.25, linewidth=0, label="10th to 90th percentile")
        ax.plot(wl, q50, color=col, linewidth=1.6, label="Median")
        ax.set_xlim(400, 750)
        ax.set_ylim(bottom=0)
        ax.set_xticks([400, 500, 600, 700])
        ax.set_title(f"{name}, n = {int(m.sum()):,}", fontsize=S.SIZE_LABEL, loc="left", pad=5)
        ax.set_xlabel("Wavelength (nm)")
        if j == 0:
            ax.set_ylabel(r"$R_\mathrm{rs}$ ($10^{-2}\ \mathrm{sr}^{-1}$)")
        ax.grid(True, axis="y")
        S.panel_letter(ax, next(letters), x=-0.03, y=1.01)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color=S.GREY, linewidth=1.6), Patch(facecolor=S.GREY, alpha=0.3, linewidth=0)]
    lab = ["Median", "10th to 90th percentile"]
    fig.legend(h, lab, loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2, handlelength=1.6)
    ax_m = fig.add_subplot(gs2[0, 0])
    srf_panel(ax_m, msi, MSI_BANDS, S.OKABE_ITO["blue"], "", [1.05])
    ax_m.tick_params(labelbottom=False)
    ax_m.text(0.004, 0.93, "S2A MSI", transform=ax_m.transAxes, ha="left", va="top", fontsize=S.SIZE_TICK)
    S.panel_letter(ax_m, next(letters), x=-0.01, y=0.98)
    ax_o = fig.add_subplot(gs2[1, 0], sharex=ax_m)
    srf_panel(ax_o, olci, OLCI_BANDS, S.OKABE_ITO["vermillion"], "Oa", [1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05])
    ax_o.text(0.998, 0.93, "S3A OLCI", transform=ax_o.transAxes, ha="right", va="top", fontsize=S.SIZE_TICK)
    ax_o.set_xlabel("Wavelength (nm)")
    S.panel_letter(ax_o, next(letters), x=-0.01, y=0.98)
    S.save(fig, "fig04_spectra")
    counts = {c[0]: int(((lam_max >= c[1]) & (lam_max < c[2])).sum()) for c in CLASSES}
    print("n spectra", len(lam_max), counts)


if __name__ == "__main__":
    main()
