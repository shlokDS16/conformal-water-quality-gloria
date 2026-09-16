"""Geometry checker for the Fig. 1 TikZ architecture diagram (phase 08 rework).

Everything is read back out of the rendered PDF with PyMuPDF, not from the TikZ source, so
the check tests what a reader actually sees.

Classification of the vector paths returned by `page.get_drawings()`:
  node rectangle : fill+stroke path whose single item is a PDF `re` operator (TikZ draws
                   every `shape=rectangle` node this way);
  container      : fill-only path of four line segments with a large area (the stage
                   background panels, declared in CONTAINERS below);
  arrow          : everything else (stroked line segments and the filled arrow heads).

Checks (all must pass; the script exits 1 on any violation):
  1 containers    the detected containers match the declared list, and every node rectangle
                  lies wholly inside exactly one container (no partial overlap);
  2 node overlap  no two node rectangles have a positive intersection area;
  3 text in box   every text span lies inside its innermost enclosing rectangle with at
                  least MIN_PAD printed pt of clearance on each of the four sides;
  4 text overlap  no text span intersects another text span;
  5 font size     the smallest printed font size is at least MIN_PT (same criterion and
                  placement scale as src/figures/check_legibility.py);
  6 arrows        no arrow path bounding box intersects a text span.

Geometry is converted from PDF big points to printed TeX points with the placement scale
the figure gets in the final 5p layout: the PDF is placed at \\textwidth = 522.0 TeX pt.

Findings are written into research/FIG1_TIKZ_LOG.md between the CHECK markers.
Run: python -m src.figures.check_diagram
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "figures" / "fig01_workflow.pdf"
LOG = ROOT / "research" / "FIG1_TIKZ_LOG.md"
BEGIN, END = "<!-- CHECK:BEGIN -->", "<!-- CHECK:END -->"

TARGET_W_PT = 522.0  # DOUBLE_COL, src/figures/style.py
MIN_PT = 7.0         # CLAUDE.md section 7, src/figures/style.py MIN_PRINT_PT
MIN_PAD = 2.0        # printed pt of clearance between a text span and its box
MAX_H_IN = 3.6
TEX_PT_PER_IN = 72.27

# Explicit list of background panels (containers). Order: top to bottom.
CONTAINERS = ["Stage 1 panel", "Stage 2 panel", "Stage 3 panel"]
MIN_CONTAINER_AREA = 20000.0  # printed pt^2; the panels are >= 522 x 41


def rect_inter(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if (w > 0 and h > 0) else 0.0


def contains(outer, inner, pad: float = 0.0) -> bool:
    return (inner[0] >= outer[0] + pad - 1e-6 and inner[1] >= outer[1] + pad - 1e-6
            and inner[2] <= outer[2] - pad + 1e-6 and inner[3] <= outer[3] - pad + 1e-6)


def area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def fmt(r) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in r) + "]"


def collect(pdf: Path):
    doc = fitz.open(pdf)
    page = doc[0]
    k = TARGET_W_PT / page.rect.width          # printed TeX pt per PDF big point
    h_pt = page.rect.height * k

    def sc(r):
        return (r[0] * k, r[1] * k, r[2] * k, r[3] * k)

    nodes, containers, arrows, cont_outlines = [], [], [], []
    for d in page.get_drawings():
        items = d["items"]
        r = sc(tuple(d["rect"]))
        is_re = len(items) == 1 and items[0][0] == "re"
        is_big_quad = len(items) == 4 and all(it[0] == "l" for it in items) \
            and area(r) >= MIN_CONTAINER_AREA
        if d["type"] == "fs" and is_re:
            nodes.append(r)
        elif d["type"] == "f" and is_big_quad:
            containers.append(r)            # panel background fill
        elif d["type"] == "s" and is_big_quad:
            cont_outlines.append(r)         # panel outline, same rectangle, not an arrow
        else:
            # a horizontal or vertical arrow has a zero-thickness bounding box; inflate it
            # by the stroke half-width so the intersection test is not degenerate
            hw = max((d.get("width") or 0.0) * k / 2.0, 0.35)
            arrows.append((r[0] - hw, r[1] - hw, r[2] + hw, r[3] + hw))

    spans = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for sp in line["spans"]:
                if sp["text"].strip():
                    spans.append({"bbox": sc(tuple(sp["bbox"])), "size": sp["size"] * k,
                                  "text": sp["text"].strip(), "font": sp["font"]})
    fonts = sorted({f[3].split("+")[-1] for f in page.get_fonts(full=True)})
    doc.close()
    containers.sort(key=lambda r: r[1])
    nodes.sort(key=lambda r: (round(r[1], 1), r[0]))
    spans.sort(key=lambda s: (round(s["bbox"][1], 1), s["bbox"][0]))
    return dict(k=k, h_pt=h_pt, nodes=nodes, containers=containers, arrows=arrows,
                cont_outlines=cont_outlines, spans=spans, fonts=fonts)


def run(pdf: Path = PDF) -> tuple[list[str], list[str]]:
    if not pdf.exists():
        return [f"{pdf} does not exist"], []
    g = collect(pdf)
    v: list[str] = []
    notes: list[str] = []
    nodes, conts, spans, arrows = g["nodes"], g["containers"], g["spans"], g["arrows"]

    notes.append(f"PDF `{pdf.name}`: natural width "
                 f"{TARGET_W_PT:.2f} TeX pt, height {g['h_pt']:.2f} TeX pt "
                 f"({g['h_pt'] / TEX_PT_PER_IN:.3f} in), placement scale 1.000 at DOUBLE_COL.")
    notes.append(f"Extracted: {len(conts)} container fills, {len(g['cont_outlines'])} container "
                 f"outlines, {len(nodes)} node rectangles, "
                 f"{len(arrows)} arrow paths, {len(spans)} text spans. "
                 f"Embedded fonts: {', '.join(g['fonts'])}.")

    if g["h_pt"] / TEX_PT_PER_IN > MAX_H_IN:
        v.append(f"height {g['h_pt'] / TEX_PT_PER_IN:.3f} in exceeds the {MAX_H_IN} in budget")

    # ---- 1 containers -----------------------------------------------------------------
    if len(conts) != len(CONTAINERS):
        v.append(f"check 1: {len(conts)} containers detected, {len(CONTAINERS)} declared "
                 f"({', '.join(CONTAINERS)})")
    for name, r in zip(CONTAINERS, conts):
        notes.append(f"container `{name}` {fmt(r)}")
    for i, n in enumerate(nodes):
        holders = [j for j, c in enumerate(conts) if contains(c, n)]
        partial = [j for j, c in enumerate(conts) if rect_inter(c, n) > 1e-4 and j not in holders]
        if partial:
            v.append(f"check 1: node {fmt(n)} only partly inside container(s) "
                     f"{[CONTAINERS[j] for j in partial]}")
        elif len(holders) != 1:
            v.append(f"check 1: node {fmt(n)} lies inside {len(holders)} containers, expected 1")
    for i in range(len(conts)):
        for j in range(i + 1, len(conts)):
            if rect_inter(conts[i], conts[j]) > 1e-4:
                v.append(f"check 1: containers {CONTAINERS[i]} and {CONTAINERS[j]} overlap")

    # ---- 2 node overlap ---------------------------------------------------------------
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = rect_inter(nodes[i], nodes[j])
            if a > 1e-4:
                v.append(f"check 2: node rectangles {fmt(nodes[i])} and {fmt(nodes[j])} "
                         f"overlap by {a:.3f} pt^2")

    # ---- 3 text inside its rectangle --------------------------------------------------
    boxes = [("node", r) for r in nodes] + [("container", r) for r in conts]
    worst = None
    for s in spans:
        b = s["bbox"]
        # a span that pokes out of any rectangle it touches has escaped that rectangle,
        # even if a larger panel still encloses it
        for kind, r in boxes:
            if rect_inter(r, b) > 1e-4 and not contains(r, b):
                v.append(f"check 3: text {s['text']!r} {fmt(b)} crosses the edge of its "
                         f"{kind} {fmt(r)}")
        holding = [(area(r), kind, r) for kind, r in boxes if contains(r, b)]
        if not holding:
            v.append(f"check 3: text {s['text']!r} {fmt(b)} is not inside any rectangle")
            continue
        _, kind, r = min(holding, key=lambda t: t[0])
        pads = (b[0] - r[0], b[1] - r[1], r[2] - b[2], r[3] - b[3])
        if worst is None or min(pads) < worst[0]:
            worst = (min(pads), s["text"], kind)
        if min(pads) < MIN_PAD - 1e-6:
            v.append(f"check 3: text {s['text']!r} has {min(pads):.2f} pt clearance inside its "
                     f"{kind} {fmt(r)} (need {MIN_PAD:g} pt); pads l/t/r/b "
                     + "/".join(f"{p:.2f}" for p in pads))
    if worst:
        notes.append(f"check 3: smallest clearance {worst[0]:.2f} pt "
                     f"(text {worst[1]!r}, inside its {worst[2]})")

    # ---- 4 text against text ----------------------------------------------------------
    tmin = None
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            a = rect_inter(spans[i]["bbox"], spans[j]["bbox"])
            if a > 1e-4:
                v.append(f"check 4: text {spans[i]['text']!r} overlaps {spans[j]['text']!r} "
                         f"by {a:.3f} pt^2")
            bi, bj = spans[i]["bbox"], spans[j]["bbox"]
            if not (bi[2] <= bj[0] or bj[2] <= bi[0]):  # share an x range: vertical gap
                gap = max(bj[1] - bi[3], bi[1] - bj[3])
                if tmin is None or gap < tmin:
                    tmin = gap
    if tmin is not None:
        notes.append(f"check 4: smallest vertical gap between text spans sharing an x range "
                     f"{tmin:.2f} pt")

    # ---- 5 printed font size ----------------------------------------------------------
    if spans:
        sm = min(spans, key=lambda s: s["size"])
        n_below = sum(1 for s in spans if s["size"] < MIN_PT - 1e-6)
        notes.append(f"check 5: smallest printed font {sm['size']:.2f} pt (text {sm['text']!r}), "
                     f"{n_below} span(s) below {MIN_PT:g} pt")
        if n_below:
            v.append(f"check 5: {n_below} text span(s) below {MIN_PT:g} pt, smallest "
                     f"{sm['size']:.2f} pt ({sm['text']!r})")

    # ---- 6 arrows against text --------------------------------------------------------
    hits = 0
    for a in arrows:
        for s in spans:
            ia = rect_inter(a, s["bbox"])
            if ia > 1e-4:
                hits += 1
                v.append(f"check 6: arrow path {fmt(a)} crosses text {s['text']!r} "
                         f"{fmt(s['bbox'])} ({ia:.3f} pt^2)")
    notes.append(f"check 6: {len(arrows)} arrow paths tested against {len(spans)} text spans, "
                 f"{hits} intersection(s)")
    return v, notes


def write_log(violations: list[str], notes: list[str]) -> None:
    block = [BEGIN, "", f"### Checker output (`src/figures/check_diagram.py`, {date.today().isoformat()})",
             ""]
    block += [f"- {n}" for n in notes]
    block += ["", f"**Violations: {len(violations)}**"]
    block += [f"- {x}" for x in violations] if violations else ["- none"]
    block += ["", f"CHECK VERDICT: {'PASS' if not violations else 'FAIL'}", "", END]
    text = "\n".join(block)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        old = LOG.read_text(encoding="utf-8")
        if BEGIN in old and END in old:
            head, rest = old.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
            LOG.write_text(head + text + tail, encoding="utf-8")
            return
        LOG.write_text(old.rstrip() + "\n\n" + text + "\n", encoding="utf-8")
        return
    LOG.write_text(text + "\n", encoding="utf-8")


def main(argv=None) -> None:
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=str(PDF), help="diagram PDF to check")
    ap.add_argument("--no-log", action="store_true",
                    help="print findings only (used for the negative control)")
    args = ap.parse_args(argv)
    violations, notes = run(Path(args.pdf))
    for n in notes:
        print("  ", n)
    for x in violations:
        print("VIOLATION:", x)
    if not args.no_log:
        write_log(violations, notes)
        print(f"log written to {LOG.relative_to(ROOT).as_posix()}")
    print(f"violations: {len(violations)}")
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
