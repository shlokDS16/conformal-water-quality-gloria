"""Contact sheet of the finished data-only figures: figures/contact_sheet_data.png.

With --gray, also writes a grayscale copy to the given path for the grayscale legibility review.
Run: python -m src.figures.contact_sheet [--gray PATH]
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageDraw, ImageFont

from . import style as S

NAMES = ["fig01_workflow", "fig02_map", "fig03_protocols", "fig04_spectra", "figA1_targets", "figA2_wbsize"]
CELL_W = 1400
PAD = 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gray", default=None)
    args = ap.parse_args()
    font = ImageFont.truetype("arial.ttf", 28)
    thumbs = []
    for n in NAMES:
        im = Image.open(S.FIG_DIR / f"{n}.png").convert("RGB")
        w = CELL_W if n.startswith("fig0") else CELL_W // 2
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
        thumbs.append((n, im))
    # layout: two columns of double-width figures, then the two single-column figures side by side
    rows = [thumbs[0:2], thumbs[2:4], thumbs[4:6]]
    col_w = CELL_W + PAD
    heights = [max(t[1].height for t in r) + 50 for r in rows]
    sheet = Image.new("RGB", (2 * col_w + PAD, sum(heights) + PAD), "white")
    draw = ImageDraw.Draw(sheet)
    y = PAD
    for r, h in zip(rows, heights):
        x = PAD
        for name, im in r:
            draw.text((x, y), name, fill="black", font=font)
            sheet.paste(im, (x, y + 40))
            draw.rectangle([x - 1, y + 39, x + im.width, y + 40 + im.height], outline="#BBBBBB")
            x += (col_w if im.width == CELL_W else col_w)
        y += h
    out = S.FIG_DIR / "contact_sheet_data.png"
    sheet.save(out)
    print("written", out, sheet.size)
    if args.gray:
        sheet.convert("L").save(args.gray)
        print("written", args.gray)


if __name__ == "__main__":
    main()
