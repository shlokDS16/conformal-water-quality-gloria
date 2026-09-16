"""Graphical abstract for the ISPRS JPRS submission (13 x 5 cm vector PDF plus PNG).

Guide for Authors, section "Graphical abstract", read live on 2026-09-16 at
https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing/publish/guide-for-authors
  "Image size: Please provide an image with a minimum of 531 x 1328 pixels (h x w) or
   proportionally more. The image should be readable at a size of 5 x 13 cm using a regular
   screen resolution of 96 dpi. Preferred file types: TIFF, EPS, PDF or MS Office files."

The page is therefore drawn at exactly 13 x 5 cm so that every font size set here is the
printed size at the size the Guide names, and the PNG is rendered at 600 dpi, which is far
above the 531 x 1328 px minimum.

Every printed number comes from a key of tables/summary.csv (LEDGER below); nothing is read
off a figure and nothing is rounded beyond the rounding recorded in the ledger.

Story, left to right (research/FINDINGS.md section 0 items 2 and 3):
  1 Problem  random splits leave the same water body in training and test;
  2 Method   split by water body and calibrate conformal intervals on held-out water bodies;
  3 Result   coverage averaged over water bodies reaches the nominal level, while coverage
             for the worst water body does not.

Run:
    python -m src.figures.graphical_abstract            # draw, check, copy to the upload folder
    python -m src.figures.graphical_abstract --no-copy  # draw and check only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch  # noqa: E402

from . import style as S  # noqa: E402

CM_PER_IN = 2.54
W_CM, H_CM = 13.0, 5.0
FIG_W, FIG_H = W_CM / CM_PER_IN, H_CM / CM_PER_IN          # 5.1181 x 1.9685 in
W_BP, H_BP = FIG_W * 72.0, FIG_H * 72.0                     # 368.504 x 141.732 big points
MIN_PT = S.MIN_PRINT_PT                                     # 7.0
MIN_PAD = 2.0                                               # pt of clearance inside a box
PNG_DPI = 600
NAME = "graphical_abstract"
UPLOAD = S.ROOT / "submission" / "upload_ISPRS" / "05_figures"

# ---------------------------------------------------------------------------------------
# Ledger keys. key -> (rounding as recorded in tables/summary.csv, what it is used for)
# ---------------------------------------------------------------------------------------
LEDGER = {
    "cell.Chla.hyp.random.primary.lgbm.point.point.mdsa_mean":
        ("percent, 1 decimal", "zone 1, MdSA under the random protocol"),
    "cell.Chla.hyp.waterbody.primary.lgbm.point.point.mdsa_mean":
        ("percent, 1 decimal", "zone 1, MdSA under the water-body protocol"),
    "cell.Chla.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_wb_mean":
        ("3 decimals", "zone 3, water-body-averaged coverage, Chl-a"),
    "cell.TSS.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_wb_mean":
        ("3 decimals", "zone 3, water-body-averaged coverage, TSS"),
    "cell.aCDOM440.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_wb_mean":
        ("3 decimals", "zone 3, water-body-averaged coverage, aCDOM(440)"),
    "cell.Secchi_depth.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_wb_mean":
        ("3 decimals", "zone 3, water-body-averaged coverage, Secchi depth"),
    "cell.Chla.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_worst_ge10_mean":
        ("3 decimals", "zone 3, worst-water-body coverage, Chl-a"),
    "cell.TSS.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_worst_ge10_mean":
        ("3 decimals", "zone 3, worst-water-body coverage, TSS"),
    "cell.aCDOM440.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_worst_ge10_mean":
        ("3 decimals", "zone 3, worst-water-body coverage, aCDOM(440)"),
    "cell.Secchi_depth.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_worst_ge10_mean":
        ("3 decimals", "zone 3, worst-water-body coverage, Secchi depth"),
    "cell.Chla.hyp.waterbody.primary.lgbm.scp_gsub.a100.n_seeds":
        ("integer", "provenance line, number of split seeds"),
}
TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
ROW_LABEL = {"Chla": "Chl-a", "TSS": "TSS", "aCDOM440": "aCDOM(440)", "Secchi_depth": "Secchi"}


def read_ledger() -> dict[str, float]:
    df = pd.read_csv(S.TAB_DIR / "summary.csv", low_memory=False)
    want = df[df["key"].isin(LEDGER)]
    bad = want[want["status"] != "complete"]
    if len(bad):
        raise SystemExit(f"ledger keys not complete: {list(bad['key'])}")
    missing = set(LEDGER) - set(want["key"])
    if missing:
        raise SystemExit(f"ledger keys absent from tables/summary.csv: {sorted(missing)}")
    return {r.key: float(r.value) for r in want.itertuples()}


# ---------------------------------------------------------------------------------------
# Layout. Vertical positions are given in printed points measured from the top of the page,
# so the leading between text lines is set directly and can be audited.
# ---------------------------------------------------------------------------------------
def ytop(pt: float) -> float:
    """Figure fraction of a position `pt` printed points below the top edge."""
    return 1.0 - pt / H_BP


ZONES = {                       # (x0, x1) in figure fraction; y from ZONE_TOP to ZONE_BOT
    "A": (0.006, 0.310),
    "B": (0.336, 0.610),
    "C": (0.636, 0.994),
}
ZONE_TOP_PT, ZONE_BOT_PT = 2.0, 94.0
BANNER = (0.006, 0.994, 95.0, 141.0)     # x0, x1, top pt, bottom pt of the box
BANNER_FIRST_PT, BANNER_LEAD = 99.0, 9.8
ARROWS = [(0.313, 0.333), (0.613, 0.633)]

TITLE_PT = 6.0          # top of the zone title text
SCHEM_TOP_PT = 17.0     # top of the schematic axes
SCHEM_BOT_PT = 44.0
ROLE_PT = 45.5          # role key / role labels under the schematic
CAP_PT = 56.0           # first caption line
CAP_LEAD = 9.0
CHART_TOP_PT, CHART_BOT_PT = 44.0, 80.0
LEGEND_PT = 17.0        # first of the three key lines in zone 3

SIZE_TITLE = 8.5
SIZE_BODY = 7.0
SIZE_TAKE = 7.2
CHART_L, CHART_R = 0.766, 0.9886   # chart axes, figure fraction; the left edge clears the
#                                    widest y tick label, "aCDOM(440)" (41.6 printed pt)

FACE = "#F2F2F2"
LAKE_FACE = "#E3E3E3"
LAKE_EDGE = "#8C8C8C"
BLUE, GREEN, VERM = S.ROLE_COLOR["train"], S.ROLE_COLOR["cal"], S.ROLE_COLOR["test"]

# Marker offsets inside one water-body ellipse (axes-fraction of the ellipse half axes).
PTS = [(-0.52, 0.34), (0.10, 0.46), (0.55, 0.18), (-0.46, -0.30), (0.08, -0.46), (0.58, -0.22)]


def zone_rect(key: str) -> tuple[float, float, float, float]:
    x0, x1 = ZONES[key]
    return x0, ytop(ZONE_BOT_PT), x1 - x0, (ZONE_BOT_PT - ZONE_TOP_PT) / H_BP


def draw_lake(ax, cx: float, roles: list[str]) -> None:
    """One water body: a light ellipse holding six sample markers with the given roles."""
    rx, ry = 0.135, 0.40
    ax.add_patch(Ellipse((cx, 0.50), 2 * rx, 2 * ry, facecolor=LAKE_FACE,
                         edgecolor=LAKE_EDGE, linewidth=0.6, zorder=1))
    style = {"train": ("o", BLUE, BLUE), "cal": ("s", GREEN, GREEN),
             "test": ("^", "#FFFFFF", VERM)}
    for (dx, dy), role in zip(PTS, roles):
        m, face, edge = style[role]
        ax.plot(cx + dx * rx, 0.50 + dy * ry, marker=m, markersize=2.6, markerfacecolor=face,
                markeredgecolor=edge, markeredgewidth=0.55, linestyle="none", zorder=3)


def schematic(fig, key: str, lakes: list[list[str]]) -> None:
    x0, x1 = ZONES[key]
    pad = 0.010
    ax = fig.add_axes([x0 + pad, ytop(SCHEM_BOT_PT), (x1 - x0) - 2 * pad,
                       (SCHEM_BOT_PT - SCHEM_TOP_PT) / H_BP])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.patch.set_alpha(0.0)
    for cx, roles in zip((0.18, 0.50, 0.82), lakes):
        draw_lake(ax, cx, roles)


def zone_text(fig, key: str, lines: list[str], top_pt: float, size: float = SIZE_BODY,
              weight: str = "normal", color: str = S.INK, lead: float = CAP_LEAD) -> None:
    x0, x1 = ZONES[key]
    xc = 0.5 * (x0 + x1)
    for i, t in enumerate(lines):
        fig.text(xc, ytop(top_pt + i * lead), t, ha="center", va="top", fontsize=size,
                 fontweight=weight, color=color)


def key_marker(fig, xfrac: float, ypt: float, marker: str, face: str, edge: str) -> None:
    """One key marker drawn in figure coordinates, vertically centred on a 7 pt text line
    whose top is at `ypt`."""
    fig.add_artist(plt.Line2D([xfrac], [ytop(ypt + 3.9)], marker=marker, markersize=3.0,
                              markerfacecolor=face, markeredgecolor=edge, markeredgewidth=0.7,
                              linestyle="none", transform=fig.transFigure, zorder=4))


def build(v: dict[str, float]) -> plt.Figure:
    S.apply()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    # Zone panels. Figure-level patches are drawn after the axes when their zorder ties, so
    # the panels are pushed behind everything explicitly.
    for k in ZONES:
        x, y, w, h = zone_rect(k)
        fig.patches.append(FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                                          boxstyle="round,pad=0,rounding_size=0.006",
                                          facecolor=FACE, edgecolor="none", zorder=-10))
    for xa, xb in ARROWS:
        fig.patches.append(plt.Arrow(xa, ytop(0.5 * (ZONE_TOP_PT + ZONE_BOT_PT)), xb - xa, 0,
                                     width=0.050, transform=fig.transFigure,
                                     facecolor=S.GREY, edgecolor="none", zorder=-5))

    # ---- zone 1: the problem -----------------------------------------------------------
    zone_text(fig, "A", ["1   Problem"], TITLE_PT, size=SIZE_TITLE, weight="bold")
    schematic(fig, "A", [["train", "test", "train", "test", "train", "test"]] * 3)
    for xm, xt, lab, m, face, edge in (
            (0.075, 0.087, "train", "o", BLUE, BLUE),
            (0.185, 0.197, "test", "^", "#FFFFFF", VERM)):
        key_marker(fig, xm, ROLE_PT, m, face, edge)
        fig.text(xt, ytop(ROLE_PT), lab, ha="left", va="top", fontsize=SIZE_BODY, color=edge)
    mdsa_r = v["cell.Chla.hyp.random.primary.lgbm.point.point.mdsa_mean"]
    mdsa_w = v["cell.Chla.hyp.waterbody.primary.lgbm.point.point.mdsa_mean"]
    # "error (MdSA)", not "accuracy": the median symmetric accuracy is an error measure and a
    # rise from 33.2 to 43.5 % is a degradation. Saying "accuracy rises" would invert it.
    zone_text(fig, "A", [
        "Random splits put the same water",
        "body in training and in test. Chl-a",
        f"error (MdSA) is {mdsa_r:.1f} % there and",
        f"{mdsa_w:.1f} % under a water-body split.",
    ], CAP_PT)

    # ---- zone 2: the method ------------------------------------------------------------
    zone_text(fig, "B", ["2   Method"], TITLE_PT, size=SIZE_TITLE, weight="bold")
    schematic(fig, "B", [["train"] * 6, ["cal"] * 6, ["test"] * 6])
    x0, x1 = ZONES["B"]
    pad = 0.010
    for frac, lab, col in zip((0.18, 0.50, 0.82), ("train", "calibrate", "test"),
                              (BLUE, GREEN, VERM)):
        fig.text(x0 + pad + frac * ((x1 - x0) - 2 * pad), ytop(ROLE_PT), lab, ha="center",
                 va="top", fontsize=SIZE_BODY, color=col)
    zone_text(fig, "B", [
        "Hold out whole water bodies,",
        "then calibrate conformal",
        "intervals on the held-out",
        "calibration water bodies.",
    ], CAP_PT)

    # ---- zone 3: the result ------------------------------------------------------------
    zone_text(fig, "C", ["3   Result"], TITLE_PT, size=SIZE_TITLE, weight="bold")
    x0, x1 = ZONES["C"]
    fig.text(x0 + 0.009, ytop(LEGEND_PT), "coverage (nominal 0.90 dashed)", ha="left",
             va="top", fontsize=SIZE_BODY, color=S.GREY)
    for i, (lab, m, face, edge) in enumerate((
            ("mean over water bodies", "o", BLUE, BLUE),
            ("worst water body", "v", "#FFFFFF", VERM)), start=1):
        ypt = LEGEND_PT + i * CAP_LEAD
        key_marker(fig, x0 + 0.011, ypt, m, face, edge)
        fig.text(x0 + 0.022, ytop(ypt), lab, ha="left", va="top", fontsize=SIZE_BODY,
                 color=S.INK)

    ax = fig.add_axes([CHART_L, ytop(CHART_BOT_PT), CHART_R - CHART_L,
                       (CHART_BOT_PT - CHART_TOP_PT) / H_BP])
    ax.set_xlim(0.42, 1.00)
    ax.set_ylim(-0.55, 3.55)
    ax.axvline(0.90, color=S.GREY, linestyle=(0, (2.5, 2)), linewidth=0.8, zorder=1)
    for i, t in enumerate(TARGETS):
        y = 3 - i
        base = f"cell.{t}.hyp.waterbody.primary.lgbm.scp_gsub.a100"
        mean = v[f"{base}.cov_wb_mean"]
        worst = v[f"{base}.cov_worst_ge10_mean"]
        ax.plot([worst, mean], [y, y], color=S.LIGHT_GREY, linewidth=0.9, zorder=2)
        ax.plot(worst, y, marker="v", markersize=3.4, markerfacecolor="#FFFFFF",
                markeredgecolor=VERM, markeredgewidth=0.8, linestyle="none", zorder=3)
        ax.plot(mean, y, marker="o", markersize=3.2, color=BLUE, linestyle="none", zorder=3)
    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels([ROW_LABEL[t] for t in TARGETS], fontsize=SIZE_BODY)
    ax.set_xticks([0.5, 0.7, 0.9])
    ax.set_xticklabels(["0.5", "0.7", "0.9"], fontsize=SIZE_BODY)
    ax.tick_params(axis="x", length=2, width=0.6, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2.0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.patch.set_alpha(0.0)

    # ---- banner ------------------------------------------------------------------------
    bx0, bx1 = BANNER[0], BANNER[1]
    xc = 0.5 * (bx0 + bx1)
    cov = [v[f"cell.{t}.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_wb_mean"] for t in TARGETS]
    wor = [v[f"cell.{t}.hyp.waterbody.primary.lgbm.scp_gsub.a100.cov_worst_ge10_mean"]
           for t in TARGETS]
    n_seeds = int(round(v["cell.Chla.hyp.waterbody.primary.lgbm.scp_gsub.a100.n_seeds"]))
    lines = [
        ("Calibration grouped by water body reaches the nominal 0.90 averaged over water bodies",
         SIZE_TAKE, "bold", S.INK),
        (f"({min(cov):.3f} to {max(cov):.3f}); the worst test water body is covered "
         f"{min(wor):.3f} to {max(wor):.3f} of the time.", SIZE_TAKE, "bold", S.INK),
        (f"LightGBM, GLORIA hyperspectral reflectance, water-body hold-out, alpha = 0.10, "
         f"{n_seeds} split seeds,", SIZE_BODY, "normal", S.GREY),
        ("group-subsampled split conformal; worst taken over test water bodies with at least "
         "10 samples.", SIZE_BODY, "normal", S.GREY),
    ]
    for i, (t, size, weight, color) in enumerate(lines):
        fig.text(xc, ytop(BANNER_FIRST_PT + i * BANNER_LEAD), t, ha="center", va="top",
                 fontsize=size, fontweight=weight, color=color)
    return fig


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------
def pdf_spans(pdf: Path):
    import fitz

    doc = fitz.open(pdf)
    page = doc[0]
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for sp in line["spans"]:
                if sp["text"].strip():
                    spans.append({"bbox": tuple(sp["bbox"]), "size": sp["size"],
                                  "text": sp["text"].strip(), "font": sp["font"]})
    fonts = sorted({f[3].split("+")[-1] for f in page.get_fonts(full=True)})
    rect = (page.rect.width, page.rect.height)
    doc.close()
    spans.sort(key=lambda s: (round(s["bbox"][1], 1), s["bbox"][0]))
    return spans, fonts, rect


def inter(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if (w > 0 and h > 0) else 0.0


def boxes_bp() -> list[tuple[str, tuple[float, float, float, float]]]:
    """Declared text boxes in PDF coordinates (origin top-left, big points)."""
    out = []
    for k, (x0, x1) in ZONES.items():
        out.append((f"zone {k}", (x0 * W_BP, ZONE_TOP_PT, x1 * W_BP, ZONE_BOT_PT)))
    bx0, bx1, btop, bbot = BANNER
    out.append(("banner", (bx0 * W_BP, btop, bx1 * W_BP, bbot)))
    return out


def check(pdf: Path, png: Path, latex: bool = True) -> tuple[list[str], list[str]]:
    from PIL import Image

    v: list[str] = []
    notes: list[str] = []

    spans, fonts, (pw, ph) = pdf_spans(pdf)
    notes.append(f"PDF page {pw:.3f} x {ph:.3f} bp "
                 f"({pw / 72 * CM_PER_IN:.4f} x {ph / 72 * CM_PER_IN:.4f} cm); "
                 f"target {W_BP:.3f} x {H_BP:.3f} bp (13.0000 x 5.0000 cm).")
    if abs(pw - W_BP) > 0.05 or abs(ph - H_BP) > 0.05:
        v.append(f"page size {pw:.3f} x {ph:.3f} bp is not {W_BP:.3f} x {H_BP:.3f} bp")

    with Image.open(png) as im:
        w_px, h_px = im.size
    notes.append(f"PNG {w_px} x {h_px} px (w x h) at {PNG_DPI} dpi; "
                 f"Guide minimum 1328 x 531 px (w x h).")
    if w_px < 1328 or h_px < 531:
        v.append(f"PNG {w_px} x {h_px} px is below the 1328 x 531 px minimum")

    notes.append(f"embedded fonts: {', '.join(fonts) if fonts else '(none)'}")
    for f in fonts:
        if "Arial" not in f:
            v.append(f"non-Arial font embedded: {f}")

    sm = min(spans, key=lambda s: s["size"])
    n_below = sum(1 for s in spans if s["size"] < MIN_PT - 1e-6)
    notes.append(f"{len(spans)} text spans; smallest {sm['size']:.2f} pt ({sm['text']!r}); "
                 f"{n_below} below {MIN_PT:g} pt")
    if n_below:
        v.append(f"{n_below} text span(s) below {MIN_PT:g} pt (smallest {sm['size']:.2f} pt, "
                 f"{sm['text']!r})")

    boxes = boxes_bp()
    worst = None
    for s in spans:
        b = s["bbox"]
        holders = [(n, r) for n, r in boxes
                   if b[0] >= r[0] - 1e-6 and b[1] >= r[1] - 1e-6
                   and b[2] <= r[2] + 1e-6 and b[3] <= r[3] + 1e-6]
        touched = [n for n, r in boxes if inter(r, b) > 1e-4]
        if len(holders) != 1 or len(touched) != 1:
            v.append(f"text {s['text']!r} at {b} is not wholly inside exactly one declared box "
                     f"(inside {[n for n, _ in holders]}, touches {touched})")
            continue
        n, r = holders[0]
        pads = (b[0] - r[0], b[1] - r[1], r[2] - b[2], r[3] - b[3])
        if worst is None or min(pads) < worst[0]:
            worst = (min(pads), s["text"], n)
        if min(pads) < MIN_PAD - 1e-6:
            v.append(f"text {s['text']!r} has {min(pads):.2f} pt clearance inside {n} "
                     f"(need {MIN_PAD:g} pt)")
    if worst:
        notes.append(f"smallest clearance inside a declared box {worst[0]:.2f} pt "
                     f"(text {worst[1]!r} in {worst[2]})")

    tmin = None
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a = inter(spans[i]["bbox"], spans[j]["bbox"])
            if a > 1e-4:
                v.append(f"text {spans[i]['text']!r} overlaps {spans[j]['text']!r} "
                         f"by {a:.3f} pt^2")
            bi, bj = spans[i]["bbox"], spans[j]["bbox"]
            if not (bi[2] <= bj[0] or bj[2] <= bi[0]):
                gap = max(bj[1] - bi[3], bi[1] - bj[3])
                if tmin is None or gap < tmin:
                    tmin = gap
    if tmin is not None:
        notes.append(f"smallest vertical gap between spans sharing an x range {tmin:.2f} pt")

    if latex:
        notes += latex_check(pdf, v)
    return v, notes


def latex_check(pdf: Path, v: list[str]) -> list[str]:
    """Method of src/figures/check_legibility.py: place the PDF at 13 cm in an elsarticle
    document, compile with pdflatex, and read the printed span sizes back out."""
    import fitz

    if shutil.which("pdflatex") is None:
        return ["pdflatex not on PATH; the LaTeX placement check was skipped"]
    work = Path(tempfile.mkdtemp(prefix="ga_legib_"))
    try:
        shutil.copy(pdf, work / f"{NAME}.pdf")
        out = []
        for width, tag, expect in (("13cm", "full", 1.0), ("6.5cm", "half", 0.5)):
            tex = ("\\documentclass[final,5p,times,twocolumn]{elsarticle}\n"
                   "\\usepackage{graphicx}\n\\pagestyle{empty}\n\\begin{document}\n"
                   "\\begin{figure*}\\centering\n"
                   f"\\includegraphics[width={width}]{{{NAME}.pdf}}\n"
                   "\\end{figure*}\n\\end{document}\n")
            stem = f"{NAME}_{tag}"
            (work / f"{stem}.tex").write_text(tex, encoding="utf-8")
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                f"{stem}.tex"], cwd=work, capture_output=True, text=True,
                               timeout=600)
            if r.returncode != 0 or not (work / f"{stem}.pdf").exists():
                v.append(f"pdflatex failed for the {tag}-width placement test")
                return out
            doc = fitz.open(work / f"{stem}.pdf")
            sizes = [sp["size"] for page in doc for b in page.get_text("dict")["blocks"]
                     for line in b.get("lines", []) for sp in line["spans"]
                     if sp["text"].strip()]
            doc.close()
            out.append(f"LaTeX placement at {width}: {len(sizes)} spans, smallest "
                       f"{min(sizes):.2f} pt (expected scale {expect:g})")
            if tag == "full":
                full_min = min(sizes)
                if full_min < MIN_PT - 1e-6:
                    v.append(f"placed at 13 cm the smallest printed span is {full_min:.2f} pt")
            else:
                ratio = min(sizes) / full_min
                out.append(f"control: half width measured {ratio:.3f} of full width, so span "
                           f"sizes carry the placement scale")
                if abs(ratio - 0.5) > 0.01:
                    v.append(f"control failed: half-width ratio {ratio:.3f}, expected 0.500")
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def grayscale(png: Path, out: Path) -> None:
    from PIL import Image

    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(png) as im:
        im.convert("L").resize((1328, 511)).save(out)


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-copy", action="store_true", help="do not copy into the upload folder")
    ap.add_argument("--no-latex", action="store_true", help="skip the pdflatex placement check")
    ap.add_argument("--gray", default=None, help="also write a grayscale PNG here")
    args = ap.parse_args(argv)

    v = read_ledger()
    for k, (rounding, use) in LEDGER.items():
        print(f"  ledger {k} = {v[k]!r} [{rounding}] -> {use}")

    fig = build(v)
    S.FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = S.FIG_DIR / f"{NAME}.pdf"
    png = S.FIG_DIR / f"{NAME}.png"
    fig.savefig(pdf, metadata={"Creator": None, "Producer": None})
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    print("written", pdf, png)

    viol, notes = check(pdf, png, latex=not args.no_latex)
    for n in notes:
        print("  ", n)
    for x in viol:
        print("VIOLATION:", x)
    if args.gray:
        grayscale(png, Path(args.gray))
        print("grayscale written", args.gray)

    if not viol and not args.no_copy:
        UPLOAD.mkdir(parents=True, exist_ok=True)
        for p in (pdf, png):
            shutil.copy(p, UPLOAD / p.name)
        print("copied into", UPLOAD)
    print(f"violations: {len(viol)}")
    raise SystemExit(1 if viol else 0)


if __name__ == "__main__":
    main()
