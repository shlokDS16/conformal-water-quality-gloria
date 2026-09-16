#!/usr/bin/env python
"""Table and margin audit of the manuscript source and the rendered PDF.

Source checks (paper/main.tex with every \\input inlined, and tables/*.tex):
  - no \\resizebox, \\scalebox or \\adjustbox around a tabular (Guide: tables must be
    editable text, and CLAUDE.md section 7 forbids \\resizebox);
  - no vertical rules and no \\hline in a booktabs table, no cell shading;
  - a caption above the table body.

The gap between a caption and the top rule is not read from the source, because it is set
once by \\belowcaptionskip in the preamble; it is measured from the PDF instead.

PDF checks (paper/main.pdf):
  - nothing printed outside the text block: every text span and every drawing is inside
    the left and right text margins, and above the page number. The marginal line numbers
    of the lineno package are the one allowed exception.

Usage:  python src/check_tables.py
Exit status 1 on any violation.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]

# elsarticle [preprint,12pt] on US letter: textwidth 390 pt centred on a 612 pt page.
TEXT_LEFT, TEXT_RIGHT = 111.0, 501.0
TEXT_TOP, TEXT_BOTTOM = 100.0, 682.0
LINENO_LEFT = 88.0          # the lineno margin numbers sit here
TOL = 1.5                   # pt, allows for glyph side bearings


def flatten(path: Path, depth: int = 0) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if depth > 5:
        return text
    out, last = [], 0
    for m in re.finditer(r"\\input\{([^}]*)\}", text):
        inc = (path.parent / m.group(1)).resolve()
        out.append(text[last:m.start()])
        out.append(flatten(inc, depth + 1) if inc.exists() else m.group(0))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def source_checks(tex: Path) -> list[str]:
    s = flatten(tex)
    problems = []
    for cmd in (r"\resizebox", r"\scalebox", r"\adjustbox", r"\cellcolor", r"\rowcolor"):
        n = s.count(cmd)
        if n:
            problems.append(f"{cmd} used {n} time(s)")
    # vertical rules in a tabular preamble
    for m in re.finditer(r"\\begin\{tabular\}(\[[^\]]*\])?\{([^}]*)\}", s):
        spec = m.group(2)
        if "|" in spec:
            problems.append(f"vertical rule in tabular preamble {spec!r}")
    n_hline = len(re.findall(r"\\hline", s))
    if n_hline:
        problems.append(f"\\hline used {n_hline} time(s) in a booktabs document")
    # caption above the body, and a gap after it
    tables = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", s, re.S)
    for t in tables:
        icap = t.find(r"\caption")
        itab = t.find(r"\begin{tabular}")
        lab = re.search(r"\\label\{([^}]*)\}", t)
        name = lab.group(1) if lab else "?"
        if icap == -1:
            problems.append(f"table {name}: no caption")
            continue
        if itab != -1 and icap > itab:
            problems.append(f"table {name}: caption is below the body")
    return problems


def pdf_checks(pdf: Path) -> list[str]:
    doc = fitz.open(str(pdf))
    problems = []
    for i in range(doc.page_count):
        page = doc[i]
        d = page.get_text("dict")
        for b in d["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for s_ in line["spans"]:
                    t = s_["text"].strip()
                    if not t:
                        continue
                    x0, y0, x1, y1 = s_["bbox"]
                    if t.isdigit() and round(s_["size"]) == 6 and x0 < 105:
                        continue          # lineno margin number
                    if t.isdigit() and y0 > 680:
                        continue          # page number
                    if x1 > TEXT_RIGHT + TOL:
                        problems.append(
                            f"p{i+1}: text {t[:28]!r} ends at x={x1:.1f}, past the right"
                            f" margin {TEXT_RIGHT}")
                    if x0 < TEXT_LEFT - TOL:
                        problems.append(
                            f"p{i+1}: text {t[:28]!r} starts at x={x0:.1f}, left of the"
                            f" text block {TEXT_LEFT}")
                    if y1 > TEXT_BOTTOM + TOL:
                        problems.append(
                            f"p{i+1}: text {t[:28]!r} reaches y={y1:.1f}, below the text"
                            f" block {TEXT_BOTTOM}")
        # Rules drawn by the page itself (booktabs rules, the abstract rule) are checked.
        # A page that places a vector figure is skipped, because PyMuPDF reports the paths
        # inside that figure's Form XObject in the figure's own coordinate space and most of
        # them are clipped away by its bounding box, so their coordinates are meaningless
        # here. Those pages are covered by the 110 dpi visual review instead.
        if page.get_xobjects():
            continue
        for dr in page.get_drawings():
            r = dr["rect"]
            if r.is_empty or r.is_infinite:
                continue
            if r.x1 > TEXT_RIGHT + TOL or r.x0 < LINENO_LEFT - TOL:
                problems.append(
                    f"p{i+1}: rule or shape spans x={r.x0:.1f} to {r.x1:.1f}, outside the"
                    f" text block")
    doc.close()
    return problems


def caption_gaps(pdf: Path) -> list[tuple[int, str, float]]:
    """Measure, for every table caption, the gap from its last line to the top rule."""
    doc = fitz.open(str(pdf))
    out = []
    for i in range(doc.page_count):
        page = doc[i]
        d = page.get_text("dict")
        caps = []
        for b in d["blocks"]:
            if b["type"] != 0:
                continue
            txt = "".join(s["text"] for line in b["lines"] for s in line["spans"])
            m = re.match(r"\s*Table\s+([A-C]?\.?\d+):", txt)
            if m:
                caps.append((m.group(1), b["bbox"][3]))
        if not caps:
            continue
        rules = [dr["rect"] for dr in page.get_drawings()
                 if dr["rect"].height < 2 and dr["rect"].width > 100]
        for num, ybot in caps:
            below = [r.y0 for r in rules if r.y0 > ybot]
            if below:
                out.append((i + 1, num, round(min(below) - ybot, 2)))
    doc.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=str(ROOT / "paper" / "main.tex"))
    ap.add_argument("--pdf", default=str(ROOT / "paper" / "main.pdf"))
    args = ap.parse_args()

    print("## Table and margin audit\n")
    src = source_checks(Path(args.tex))
    print(f"Source violations: {len(src)}")
    for p in src:
        print(f"  - {p}")
    pdfp = pdf_checks(Path(args.pdf))
    print(f"\nPDF margin violations: {len(pdfp)}")
    for p in pdfp[:40]:
        print(f"  - {p}")
    if len(pdfp) > 40:
        print(f"  ... {len(pdfp) - 40} more")

    gaps = caption_gaps(Path(args.pdf))
    print(f"\nMeasured gap from the last caption line to the top rule, per table (pt):")
    vals = []
    for page, num, gap in gaps:
        vals.append(gap)
        print(f"  - p{page} Table {num}: {gap}")
    if vals:
        print(f"  range {min(vals)} to {max(vals)} pt over {len(vals)} tables."
              f" The skip itself is one value (\\belowcaptionskip); the spread is the"
              f" depth of the caption's last line, which is larger when that line"
              f" carries a descender.")
        print(f"  tables with no gap at all: {sum(1 for v in vals if v < 1.0)}")
    return 1 if (src or pdfp) else 0


if __name__ == "__main__":
    sys.exit(main())
