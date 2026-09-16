#!/usr/bin/env python
"""Render every page of a PDF to PNG at a given resolution, for visual inspection.

Usage:
    python src/render_pages.py --pdf paper/main.pdf --out <dir> --dpi 110
    python src/render_pages.py --pdf paper/main.pdf --out <dir> --pages 12,14,16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--pages", default=None, help="comma-separated 1-based page numbers")
    ap.add_argument("--prefix", default="p")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    wanted = (sorted(int(x) for x in args.pages.split(",")) if args.pages
              else range(1, doc.page_count + 1))
    zoom = args.dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for n in wanted:
        pix = doc[n - 1].get_pixmap(matrix=mat)
        f = out / f"{args.prefix}{n:03d}.png"
        pix.save(str(f))
    print(f"{len(list(wanted))} pages rendered at {args.dpi} dpi into {out}")
    doc.close()


if __name__ == "__main__":
    main()
