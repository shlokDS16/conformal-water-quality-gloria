"""Fig. 2: global map of GLORIA water bodies by macro-region (double column).

Population: rows in the union of the four primary target populations (primary QC plus the
label rules of PREREGISTRATION section 1), from data/processed/gloria.parquet.
One marker per wb_group at the mean coordinate of its rows; marker area proportional to the
number of samples; colour and marker shape give the macro-region (region_grp).

Coastline: no vector coastline is available offline (cartopy 0.25.0 is installed but has no
Natural Earth shapefiles cached, and no download was made). The land outline is a contour of
a land/water mask derived from the Natural Earth 1 shaded-relief raster bundled with cartopy
(0.5 degree grid, public domain), so small water bodies and narrow coasts are not resolved.
Run: python -m src.figures.fig02_map
"""
from __future__ import annotations

import os

import cartopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from . import style as S

NE_PNG = os.path.join(os.path.dirname(cartopy.__file__), "data", "raster", "natural_earth",
                      "50-natural-earth-1-downsampled.png")
EXTENT = {"world": (-180, 180, -58, 80), "na": (-127, -62, 14, 56), "eu": (-12, 31, 35, 69)}


def land_mask():
    a = np.asarray(Image.open(NE_PNG).convert("RGB")).astype(float)
    water = (a[..., 2] - a[..., 0]) > 25  # water pixels are blue (B >> R)
    land = (~water).astype(float)
    land = zoom(land, 4, order=1)  # 0.125 degree grid for a smoother contour
    land = gaussian_filter(land, 1.5)
    ny, nx = land.shape
    lon = -180 + (np.arange(nx) + 0.5) * 360 / nx
    lat = 90 - (np.arange(ny) + 0.5) * 180 / ny
    return lon, lat, land


def draw_base(ax, lon, lat, land, extent, grid_step):
    ax.contourf(lon, lat, land, levels=[0.5, 2], colors=["#E6E6E6"], zorder=0)
    ax.contour(lon, lat, land, levels=[0.5], colors=["#9A9A9A"], linewidths=0.4, zorder=1)
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    xt = np.arange(np.ceil(x0 / grid_step) * grid_step, x1 + 1e-9, grid_step)
    yt = np.arange(np.ceil(y0 / grid_step) * grid_step, y1 + 1e-9, grid_step)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{abs(v):.0f}°{'W' if v < 0 else ('E' if v > 0 else '')}" for v in xt])
    ax.set_yticklabels([f"{abs(v):.0f}°{'S' if v < 0 else ('N' if v > 0 else '')}" for v in yt])
    ax.grid(True, color="#CFCFCF", linewidth=0.4, zorder=0.5)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
    ax.tick_params(length=2)


def marker_area(n):
    return 5 + 0.32 * np.asarray(n, dtype=float)


def scatter(ax, wb):
    for reg in S.REGION_ORDER:
        sub = wb[wb["region_grp"] == reg].sort_values("n", ascending=False)
        ax.scatter(sub["lon"], sub["lat"], s=marker_area(sub["n"]), c=S.REGION_COLOR[reg],
                   marker=S.REGION_MARKER[reg], edgecolors="#333333", linewidths=0.4, alpha=0.85, zorder=3)


def main() -> None:
    S.apply()
    cols = ["GLORIA_ID", "Latitude", "Longitude", "wb_group", "region_grp"] + list(S.TARGET_POP.values())
    d = S.load_gloria(cols)
    pop = d[list(S.TARGET_POP.values())].astype(bool).any(axis=1)
    d = d[pop]
    wb = d.groupby("wb_group").agg(n=("GLORIA_ID", "size"), lat=("Latitude", "mean"), lon=("Longitude", "mean"),
                                   region_grp=("region_grp", "first")).reset_index()
    n_nocoord = int(wb["lat"].isna().sum())
    counts = wb.groupby("region_grp").size()  # all water bodies, including any without coordinates
    wb = wb.dropna(subset=["lat", "lon"])
    lon, lat, land = land_mask()

    W = S.DOUBLE_COL
    fig = plt.figure(figsize=(W, 5.25))
    ax_w = fig.add_axes((0.075, 0.455, 0.89, 0.52))
    draw_base(ax_w, lon, lat, land, EXTENT["world"], 30)
    ax_w.set_aspect("equal", adjustable="box")
    scatter(ax_w, wb)
    for key, lab in (("na", "b"), ("eu", "c")):
        x0, x1, y0, y1 = EXTENT[key]
        ax_w.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=S.INK, linewidth=0.9, zorder=4))
        ax_w.text(x0 + 1.5, y1 - 1.5, f"({lab})", fontsize=S.SIZE_TICK, ha="left", va="top", zorder=5)
    S.panel_letter(ax_w, "a", x=-0.005)

    ax_na = fig.add_axes((0.075, 0.06, 0.40, 0.33))
    draw_base(ax_na, lon, lat, land, EXTENT["na"], 20)
    ax_na.set_aspect(1 / np.cos(np.deg2rad(35)), adjustable="box")
    scatter(ax_na, wb)
    S.panel_letter(ax_na, "b", x=-0.01)

    ax_eu = fig.add_axes((0.51, 0.06, 0.25, 0.33))
    draw_base(ax_eu, lon, lat, land, EXTENT["eu"], 10)
    ax_eu.set_aspect(1 / np.cos(np.deg2rad(51)), adjustable="box")
    scatter(ax_eu, wb)
    S.panel_letter(ax_eu, "c", x=-0.01)

    ax_l = fig.add_axes((0.78, 0.06, 0.21, 0.33))
    ax_l.axis("off")
    handles = [Line2D([], [], linestyle="none", marker=S.REGION_MARKER[r], markersize=6,
                      markerfacecolor=S.REGION_COLOR[r], markeredgecolor="#333333", markeredgewidth=0.4,
                      label=f"{r} ({int(counts.get(r, 0))})") for r in S.REGION_ORDER]
    labels = [h.get_label().replace("East and South-East Asia", "East and SE Asia") for h in handles]
    leg1 = ax_l.legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.02, 1.02), title="Macro-region",
                       handletextpad=0.3, borderaxespad=0, alignment="left")
    ax_l.add_artist(leg1)
    ref = [1, 100, 600]
    sh = [plt.scatter([], [], s=marker_area(v), marker="o", facecolor="white", edgecolor="#333333", linewidths=0.6,
                      label=f"{v}") for v in ref]
    ax_l.legend(handles=sh, loc="lower left", bbox_to_anchor=(-0.02, -0.02), title="Samples per water body",
                ncol=3, columnspacing=0.9, handletextpad=0.1, borderaxespad=0, alignment="left")
    S.save(fig, "fig02_map")
    print(f"water bodies plotted {len(wb)}, without coordinates {n_nocoord}, samples {len(d)}")


if __name__ == "__main__":
    main()
