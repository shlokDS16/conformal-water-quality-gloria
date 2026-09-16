"""Shared figure style for the ISPRS JPRS paper (phase 08).

Widths were measured on 2026-09-16 by compiling elsarticle v3.5 test documents with
pdflatex (MiKTeX) and printing the lengths (TeX points, 72.27 pt = 1 in):
  final layout  [final,5p,times,twocolumn]: \\textwidth 522.0pt, \\columnwidth 252.0pt
  submission    [preprint,12pt,authoryear]: \\textwidth = \\columnwidth 390.0pt
Figures are drawn at their final printed size, so a font size set here is the printed
size in the two-column layout. In the preprint build a DOUBLE_COL figure scaled to
\\textwidth prints at 390/522 = 0.747 of its size (see research/FIGURE_LEGIBILITY.md).

Font: Arial (Guide: "Preferred fonts: Arial (or Helvetica), Times New Roman (or Times)";
Arial is installed at C:/Windows/Fonts/arial.ttf, Helvetica is not). TrueType fonts are
embedded (pdf.fonttype 42). Mathtext uses Arial glyphs; its script shrink factor is raised
from 0.7 to 0.8 so that sub- and superscripts in 9 pt labels print at 7.2 pt.

Palette: Okabe and Ito (2008). Yellow #F0E442 fails the dataviz lightness band on a light
surface (validate_palette.js, 2026-09-16), so it is never used for marks. Orange, sky blue
and reddish purple are below 3:1 contrast, so marks in those colours get a dark edge and a
secondary encoding (marker shape, hatch or direct label).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib._mathtext as _mathtext  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

TEX_PT_PER_IN = 72.27
TEXTWIDTH_5P_PT = 522.0
COLUMNWIDTH_5P_PT = 252.0
TEXTWIDTH_PREPRINT_PT = 390.0

SINGLE_COL = COLUMNWIDTH_5P_PT / TEX_PT_PER_IN  # 3.487 in, 88.6 mm
DOUBLE_COL = TEXTWIDTH_5P_PT / TEX_PT_PER_IN  # 7.223 in, 183.5 mm
ONEHALF_COL = TEXTWIDTH_PREPRINT_PT / TEX_PT_PER_IN  # 5.396 in, used in figure* in 5p

FONT = "Arial"
SIZE_LABEL = 9
SIZE_TICK = 8
SIZE_LEGEND = 8
SIZE_PANEL = 10
LINE_WIDTH = 1.2
MIN_PRINT_PT = 7.0

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
# Fixed categorical order (never cycled); yellow excluded.
CATEGORICAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#6E6E6E"
LIGHT_GREY = "#D9D9D9"
INK = "#1A1A1A"

REGION_ORDER = [
    "North America",
    "Europe",
    "East and South-East Asia",
    "South America",
    "Oceania",
    "Africa",
]
REGION_COLOR = dict(zip(REGION_ORDER, CATEGORICAL))
REGION_MARKER = dict(zip(REGION_ORDER, ["o", "s", "^", "D", "v", "P"]))

# Split roles: validated as a 3-colour set (all dataviz checks pass).
ROLE_COLOR = {"train": "#0072B2", "cal": "#009E73", "test": "#D55E00", "dropped": LIGHT_GREY}
ROLE_HATCH = {"train": "", "cal": "////", "test": "xxxx", "dropped": ""}

TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
TARGET_POP = {
    "Chla": "chla_primary",
    "TSS": "tss_primary",
    "aCDOM440": "acdom_primary",
    "Secchi_depth": "secchi_primary",
}
# Units from GLORIA_variables_and_methods.xlsx, sheet "Data headers".
TARGET_LABEL = {
    "Chla": r"Chl-a (mg $\mathrm{m}^{-3}$)",
    "TSS": r"TSS (g $\mathrm{m}^{-3}$)",
    "aCDOM440": r"$a_\mathrm{CDOM}$(440) ($\mathrm{m}^{-1}$)",
    "Secchi_depth": "Secchi depth (m)",
}
TARGET_SHORT = {"Chla": "Chl-a", "TSS": "TSS", "aCDOM440": "aCDOM(440)", "Secchi_depth": "Secchi depth"}


def apply() -> None:
    _mathtext.SHRINK_FACTOR = 0.8
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT,
            "mathtext.it": f"{FONT}:italic",
            "mathtext.bf": f"{FONT}:bold",
            # REVIEW_full M8: with fontset "custom" matplotlib leaves `mathtext.cal` at its default,
            # which resolves through `font.cursive` to Comic Sans MS on Windows. One 8 pt
            # `\mathcal{P}` in Fig. 3 therefore embedded ComicSansMS in a submitted manuscript.
            # Pinning every mathtext alphabet to Arial makes that impossible for any figure.
            "mathtext.cal": f"{FONT}:italic",
            "mathtext.sf": FONT,
            "mathtext.tt": FONT,
            "mathtext.default": "regular",
            "font.size": SIZE_LABEL,
            "axes.labelsize": SIZE_LABEL,
            "axes.titlesize": SIZE_LABEL,
            "xtick.labelsize": SIZE_TICK,
            "ytick.labelsize": SIZE_TICK,
            "legend.fontsize": SIZE_LEGEND,
            "legend.title_fontsize": SIZE_LEGEND,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": LINE_WIDTH,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "hatch.linewidth": 0.8,
        }
    )


def panel_letter(ax, letter: str, x: float = -0.02, y: float = 1.02, ha: str = "right",
                 va: str = "bottom", **kw) -> None:
    """Panel letter in axes coordinates.

    `ha` and `va` are overridable (REVIEW_full M9) so that a letter can be set inside the axes
    where the default position outside the frame would collide with a tick label."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=SIZE_PANEL,
            fontweight="bold", ha=ha, va=va, **kw)


def save(fig, name: str, out_dir: Path | str | None = None) -> list[Path]:
    """Save vector PDF (primary) and 600 dpi PNG to figures/. No bbox cropping, so the
    printed width equals the figure width set in the script.

    `out_dir` writes the pair somewhere else (for example figures/_preview while the runs are
    still going); the default stays figures/ so existing scripts are unchanged."""
    d = FIG_DIR if out_dir is None else Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        p = d / f"{name}.{ext}"
        fig.savefig(p, dpi=600, metadata={"Creator": None, "Producer": None} if ext == "pdf" else None)
        out.append(p)
    plt.close(fig)
    return out


def load_gloria(columns: list[str] | None = None):
    import pandas as pd

    return pd.read_parquet(DATA / "processed" / "gloria.parquet", columns=columns)
