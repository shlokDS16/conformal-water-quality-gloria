"""Fig. 3: evaluation protocol schematic (double column).

Illustrative toy layout, not data. The rules drawn follow PREREGISTRATION section 2 and
Amendments 1, 2 and 4: 60/20/20 splits; contributor folds on Dataset_ID with removal of
training and calibration rows whose water body occurs in the test fold; region
leave-one-out with calibration water bodies from non-test regions; single subsampling (one
sample per calibration water body); group CV+ with |P| mod K water bodies removed and K equal
folds of water bodies (K = 5 drawn for space; K = 10 used).
Role encoding: colour plus marker shape for samples, colour plus hatch for units.
Run: python -m src.figures.fig03_protocols
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from . import style as S

ROLE_MARK = {"train": "o", "cal": "s", "test": "^", "dropped": "x"}
TINT = {"train": "#CCE3F0", "cal": "#CCECE3", "test": "#F7DFCC", "dropped": "#EDEDED", None: "white"}


def unit(ax, x, y, w, h, role=None, n=4, sample_roles=None, label=None, rng=None, highlight=None):
    """Water-body box at lower-left (x, y) with n sample markers."""
    edge = S.ROLE_COLOR[role] if role in ("train", "cal", "test") else ("#8C8C8C" if role == "dropped" else S.GREY)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03", facecolor=TINT[role],
                                edgecolor=edge, linewidth=0.9, hatch=S.ROLE_HATCH.get(role, "") if role else ""))
    if role == "dropped":
        ax.plot([x + 0.03, x + w - 0.03], [y + 0.03, y + h - 0.03], color="#8C8C8C", lw=0.8)
    cols = 2
    rows = int(np.ceil(n / cols))
    for i in range(n):
        r = sample_roles[i] if sample_roles is not None else role
        cx = x + w * (0.3 + 0.4 * (i % cols))
        cy = y + h - (i // cols + 0.5) * (h / max(rows, 1))
        if r == "dropped":
            ax.plot(cx, cy, marker="o", ms=3.2, mfc="#BDBDBD", mec="#8C8C8C", mew=0.4, ls="none", zorder=3)
        elif r is None:
            ax.plot(cx, cy, marker="o", ms=3.2, mfc="white", mec=S.GREY, mew=0.6, ls="none", zorder=3)
        else:
            ax.plot(cx, cy, marker=ROLE_MARK[r], ms=3.6, mfc=S.ROLE_COLOR[r], mec="white", mew=0.3, ls="none",
                    zorder=3)
        if highlight is not None and i == highlight:
            ax.plot(cx, cy, marker="o", ms=7, mfc="none", mec=S.INK, mew=0.9, ls="none", zorder=4)
    if label:
        ax.text(x + w / 2, y - 0.04, label, fontsize=S.SIZE_TICK, ha="center", va="top")


def panel_frame(ax, x, y, w, h, letter, title):
    ax.text(x, y + h + 0.04, f"({letter})", fontsize=S.SIZE_PANEL, fontweight="bold", ha="left", va="bottom")
    ax.text(x + 0.30, y + h + 0.05, title, fontsize=S.SIZE_LABEL, ha="left", va="bottom")


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=S.INK,
                                 shrinkA=0, shrinkB=0))


def main() -> None:
    S.apply()
    W, H = S.DOUBLE_COL, 4.20
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    rng = np.random.default_rng(7)

    pw, ph = 2.25, 1.45
    xs = [0.08, 0.08 + pw + 0.19, 0.08 + 2 * (pw + 0.19)]
    y_top, y_bot = 2.21, 0.16
    sizes = [4, 6, 2, 4, 6, 4, 2, 4, 6, 4]
    wb_roles = ["train", "test", "train", "cal", "train", "train", "test", "train", "cal", "train"]
    uw, uh, gx = 0.36, 0.52, 0.09

    # (a) random: samples assigned individually, same water body in several roles
    panel_frame(ax, xs[0], y_top, pw, ph, "a", "Random (sample unit)")
    for j in range(10):
        n = sizes[j]
        roles = rng.choice(["train", "train", "train", "cal", "test"], size=n)
        unit(ax, xs[0] + 0.05 + (j % 5) * (uw + gx), y_top + ph - 0.62 - (j // 5) * (uh + 0.18), uw, uh,
             None, n=n, sample_roles=list(roles))
    ax.text(xs[0] + 0.05, y_top - 0.02, "Water bodies span roles (optimistic reference)", fontsize=S.SIZE_TICK,
            ha="left", va="bottom")

    # (b) water body: whole water bodies assigned, 60/20/20 by groups
    panel_frame(ax, xs[1], y_top, pw, ph, "b", "Water body (primary)")
    for j in range(10):
        unit(ax, xs[1] + 0.05 + (j % 5) * (uw + gx), y_top + ph - 0.62 - (j // 5) * (uh + 0.18), uw, uh,
             wb_roles[j], n=sizes[j])
    ax.text(xs[1] + 0.05, y_top - 0.02, "60/20/20 of water bodies, 20 repeats", fontsize=S.SIZE_TICK,
            ha="left", va="bottom")

    # (c) contributor: folds on Dataset_ID; shared water body removed from train/cal
    panel_frame(ax, xs[2], y_top, pw, ph, "c", "Contributor (dataset folds)")
    ds_roles = ["train", "cal", "test", "train"]
    ds_units = [["A", "B"], ["C", "E"], ["E", "F"], ["G", "H"]]
    cw = 0.50
    for k in range(4):
        x = xs[2] + 0.04 + k * (cw + 0.06)
        ax.add_patch(Rectangle((x, y_top + 0.14), cw, ph - 0.40, facecolor="none", edgecolor=S.GREY, lw=0.6,
                               ls=(0, (2, 1.5))))
        ax.text(x + cw / 2, y_top + ph - 0.24, f"D{k + 1}", fontsize=S.SIZE_TICK, ha="center", va="bottom")
        for m, u in enumerate(ds_units[k]):
            role = ds_roles[k]
            if u == "E" and role != "test":
                role = "dropped"
            yy = y_top + ph - 0.68 - m * 0.50
            unit(ax, x + 0.07, yy, uw, 0.36, role, n=0)
            ax.text(x + 0.07 + uw / 2, yy + 0.18, u, fontsize=S.SIZE_LABEL, ha="center", va="center",
                    fontweight="bold", color=S.INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    ax.text(xs[2] + 0.04, y_top - 0.02, "Water body E is in test fold D3:", fontsize=S.SIZE_TICK, ha="left",
            va="bottom")
    ax.text(xs[2] + 0.04, y_top - 0.19, "its rows in D2 are removed", fontsize=S.SIZE_TICK, ha="left", va="bottom")

    # (d) region leave-one-out
    panel_frame(ax, xs[0], y_bot, pw, ph, "d", "Region (leave one out)")
    reg_roles = [["train", "cal", "train", "train"], ["train", "train", "cal", "train"], ["test"] * 4]
    rw = 0.70
    for r in range(3):
        x = xs[0] + 0.04 + r * (rw + 0.05)
        ax.add_patch(Rectangle((x, y_bot + 0.16), rw, ph - 0.26, facecolor="none", edgecolor=S.GREY, lw=0.6,
                               ls=(0, (2, 1.5))))
        ax.text(x + rw / 2, y_bot + ph - 0.14, f"Region {r + 1}", fontsize=S.SIZE_TICK, ha="center", va="top")
        for m in range(4):
            unit(ax, x + 0.06 + (m % 2) * 0.32, y_bot + ph - 0.72 - (m // 2) * 0.50, 0.26, 0.40, reg_roles[r][m], n=2)
    ax.text(xs[0] + 0.04, y_bot - 0.02, "Calibration water bodies from non-test regions", fontsize=S.SIZE_TICK,
            ha="left", va="bottom")

    # (e) single subsampling
    panel_frame(ax, xs[1], y_bot, pw, ph, "e", "Single subsampling (calibration)")
    picks = [1, 0, 3, 2, 0]
    ns = [4, 2, 6, 4, 2]
    for j in range(5):
        unit(ax, xs[1] + 0.05 + j * (uw + gx), y_bot + ph - 0.64, uw, 0.52, "cal", n=ns[j], highlight=picks[j])
    yb = y_bot + 0.16
    arrow(ax, (xs[1] + pw / 2, y_bot + ph - 0.70), (xs[1] + pw / 2, yb + 0.42))
    scores = np.sort(np.array([0.30, 0.62, 0.45, 0.85, 0.20]))
    for j, s in enumerate(scores):
        bx = xs[1] + 0.35 + j * 0.16
        ax.add_patch(Rectangle((bx, yb), 0.11, 0.36 * s, facecolor=S.ROLE_COLOR["cal"], edgecolor="white", lw=0.4))
    ax.text(xs[1] + 1.22, yb + 0.30, "k scores, sorted", fontsize=S.SIZE_TICK, ha="left", va="center")
    ax.text(xs[1] + 1.22, yb + 0.12, "quantile at rank", fontsize=S.SIZE_TICK, ha="left", va="center")
    ax.text(xs[1] + 0.05, y_bot - 0.02, "rank = ceil((1 − α)(k + 1)); infinite if > k", fontsize=S.SIZE_TICK,
            ha="left", va="bottom")

    # (f) group CV+ with equal folds of water bodies
    panel_frame(ax, xs[2], y_bot, pw, ph, "f", "Group CV+ (equal folds)")
    K, per = 5, 4
    sw, sh = 0.26, 0.17
    fold_x0 = xs[2] + 0.06
    ytop_f = y_bot + ph - 0.30
    for k in range(K):
        fx = fold_x0 + k * (sw + 0.10)
        held = k == 1
        for m in range(per):
            unit(ax, fx, ytop_f - m * (sh + 0.04), sw, sh, "test" if held else "train", n=0)
        ax.text(fx + sw / 2, ytop_f - (per - 1) * (sh + 0.04) - 0.04, f"{k + 1}", fontsize=S.SIZE_TICK,
                ha="center", va="top")
    dx = fold_x0 + K * (sw + 0.10) - 0.02
    for m in range(3):
        unit(ax, dx, ytop_f - m * (sh + 0.04), sw, sh, "dropped", n=0)
    ax.text(dx + sw / 2, ytop_f - (per - 1) * (sh + 0.04) - 0.04, "$|\\mathcal{P}|$ mod K\nremoved",
            fontsize=S.SIZE_TICK, ha="center", va="top", linespacing=1.1)
    ax.text(xs[2] + 0.04, y_bot + 0.15, "Folds of water bodies; fold 2 held out", fontsize=S.SIZE_TICK,
            ha="left", va="bottom")
    ax.text(xs[2] + 0.04, y_bot - 0.02, "K = 5 drawn; K = 10 used", fontsize=S.SIZE_TICK, ha="left", va="bottom")

    # legend
    hs, labs = [], []
    for r, lab in (("train", "Training"), ("cal", "Calibration"), ("test", "Test or held out"),
                   ("dropped", "Removed")):
        p = Patch(facecolor=TINT[r], edgecolor=S.ROLE_COLOR[r] if r != "dropped" else "#8C8C8C",
                  hatch=S.ROLE_HATCH[r], linewidth=0.9)
        mk = Line2D([], [], ls="none", marker=ROLE_MARK[r] if r != "dropped" else "o", ms=4.5,
                    mfc=S.ROLE_COLOR[r] if r != "dropped" else "#BDBDBD", mec="white" if r != "dropped" else "#8C8C8C",
                    mew=0.4)
        hs.append((p, mk))
        labs.append(lab)
    hs.append(Line2D([], [], ls="none", marker="o", ms=7, mfc="none", mec=S.INK, mew=0.9))
    labs.append("Drawn sample")
    ax.legend(hs, labs, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)}, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), ncol=5, handlelength=2.6, columnspacing=1.4, borderaxespad=0.15)
    S.save(fig, "fig03_protocols")


if __name__ == "__main__":
    main()
