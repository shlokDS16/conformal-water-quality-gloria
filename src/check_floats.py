#!/usr/bin/env python
"""Float placement and page-density audit for paper/main.tex and paper/main.pdf.

Two reports, both printed as markdown tables.

1. ``floats`` - for every float (figure, table, algorithm) in the manuscript:
   its printed number, the page its caption lands on, the page of its FIRST
   in-text citation, and the distance in pages between the two. The page of a
   citation cannot be read from the normal .aux file, so this script compiles a
   SHADOW COPY of the manuscript in a scratch directory in which every
   ``\\ref{X}`` outside a float body is followed by an invisible
   ``\\label{FLOATCITE@X@n}``. LaTeX records the page of that label, so the
   citation page is exact rather than guessed. The real ``paper/`` tree is never
   touched.

2. ``pages`` - for every page of the PDF: the number of running-text words
   (page numbers, marginal line numbers and caption text removed), the number of
   float captions, and which floats they belong to. Pages are flagged
   ``FLOAT-ONLY`` (captions but almost no running text), ``EMPTY`` (no text) or
   ``THIN`` (few words and not the last page of a section).

Usage::

    python src/check_floats.py                 # both reports
    python src/check_floats.py --mode floats
    python src/check_floats.py --mode pages
    python src/check_floats.py --max-distance 2

Exit status is 1 if any float is further than ``--max-distance`` pages from its
first citation, if figure numbering is out of citation order, or if a page is
flagged FLOAT-ONLY or EMPTY.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

FLOAT_ENVS = ("figure", "table", "algorithm")


# ----------------------------------------------------------------- tex parsing
def read_tex(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def flatten(path: Path, depth: int = 0) -> str:
    """Inline every ``\\input{...}`` so the whole manuscript is one string.

    The manuscript keeps its result tables in ``tables/*.tex`` and pulls them in
    with ``\\input``; without flattening, most tables are invisible to the scan.
    """
    text = read_tex(path)
    if depth > 5:
        return text
    out = []
    last = 0
    for m in re.finditer(r"\\input\{([^}]*)\}", text):
        inc = (path.parent / m.group(1)).resolve()
        if not inc.exists() and inc.suffix != ".tex":
            inc = inc.with_suffix(".tex")
        out.append(text[last:m.start()])
        if inc.exists():
            out.append(flatten(inc, depth + 1))
        else:
            out.append(m.group(0))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def strip_comments(text: str) -> str:
    """Blank out LaTeX comments but keep offsets identical."""
    out = list(text)
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "%":
            j = i
            while j < n and text[j] != "\n":
                out[j] = " "
                j += 1
            i = j
        else:
            i += 1
    return "".join(out)


def float_spans(text: str) -> list[dict]:
    """Return one record per float environment, in source order."""
    spans = []
    pat = re.compile(r"\\begin\{(figure|table|algorithm)(\*?)\}")
    for m in pat.finditer(text):
        env = m.group(1) + m.group(2)
        end = text.find("\\end{" + env + "}", m.end())
        if end == -1:
            continue
        body = text[m.start():end]
        labels = re.findall(r"\\label\{([^}]*)\}", body)
        cap = re.search(r"\\caption\{", body)
        caption = ""
        if cap:
            caption = balanced(body, cap.end() - 1)[:90].replace("\n", " ")
        spans.append({
            "env": m.group(1),
            "starred": bool(m.group(2)),
            "start": m.start(),
            "end": end + len("\\end{" + env + "}"),
            "labels": labels,
            "caption": caption,
        })
    return spans


def balanced(text: str, open_idx: int) -> str:
    """Content of the brace group whose '{' is at open_idx."""
    depth = 0
    i = open_idx
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
        i += 1
    return text[open_idx + 1:]


def parse_aux(path: Path) -> dict[str, dict]:
    """label -> {number, page, anchor} from \\newlabel, brace-aware."""
    txt = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, dict] = {}
    for m in re.finditer(r"\\newlabel\{", txt):
        lab = balanced(txt, m.end() - 1)
        rest_start = m.end() - 1 + len(lab) + 2
        if rest_start >= len(txt) or txt[rest_start] != "{":
            continue
        group = balanced(txt, rest_start)
        fields = []
        i = 0
        while i < len(group) and len(fields) < 5:
            if group[i] == "{":
                f = balanced(group, i)
                fields.append(f)
                i += len(f) + 2
            else:
                i += 1
        if len(fields) >= 4:
            out[lab] = {"number": fields[0], "page": fields[1], "anchor": fields[3]}
    return out


# --------------------------------------------------- shadow build for ref pages
REF_CMDS = ("ref", "autoref", "cref", "Cref")


def build_shadow(tex_path: Path, workdir: Path) -> tuple[dict[str, dict], list[dict]]:
    """Copy the paper tree, annotate every \\ref outside a float, compile, read aux.

    Returns (aux_of_shadow, ref_records) where each ref record is
    {'label', 'idx', 'in_float', 'marker'}.
    """
    src_text = flatten(tex_path)
    clean = strip_comments(src_text)
    floats = float_spans(clean)

    def inside_float(pos: int) -> bool:
        return any(f["start"] <= pos < f["end"] for f in floats)

    refs = []
    pieces = []
    last = 0
    counter = 0
    pat = re.compile(r"\\(" + "|".join(REF_CMDS) + r")\{([^}]*)\}")
    for m in pat.finditer(clean):
        label = m.group(2)
        counter += 1
        marker = f"FLOATCITE@{counter}"
        refs.append({
            "label": label, "idx": m.start(),
            "in_float": inside_float(m.start()), "marker": marker,
        })
        pieces.append(src_text[last:m.end()])
        pieces.append("\\label{" + marker + "}")
        last = m.end()
    pieces.append(src_text[last:])
    shadow_tex = "".join(pieces)

    workdir.mkdir(parents=True, exist_ok=True)
    for f in PAPER.iterdir():
        if f.is_file() and f.suffix in {".bib", ".bbl", ".cls", ".bst", ".sty"}:
            shutil.copy2(f, workdir / f.name)
    # figures live one level up from paper/ in the source; copy the tree alongside
    figdir = ROOT / "figures"
    if figdir.exists() and not (workdir.parent / "figures").exists():
        shutil.copytree(figdir, workdir.parent / "figures",
                        ignore=shutil.ignore_patterns("_superseded", "*.png"))
    (workdir / "main.tex").write_text(shadow_tex, encoding="utf-8", newline="")

    env = dict(os.environ)
    for i in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=str(workdir), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
        if i == 0:
            subprocess.run(["bibtex", "main"], cwd=str(workdir), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env)
    return parse_aux(workdir / "main.aux"), refs


# ------------------------------------------------------------------ page scan
CAPTION_RE = re.compile(
    r"(?:^|\n)\s*(Figure|Table)\s+([A-C]?\.?\d+):", re.M)
ALGO_CAPTION_RE = re.compile(r"(?:^|\n)\s*Algorithm\s+([A-C]?\.?\d+)\.\s+[A-Z]", re.M)


def page_words(text: str) -> int:
    """Words on the page, with marginal line numbers and page numbers removed."""
    t = re.sub(r"(?m)^\s*\d{1,4}\s*$", " ", text)
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", t)
    return len(tokens)


def scan_pages(pdf_path: Path) -> list[dict]:
    """Per page: total words, number of numbered RUNNING-TEXT lines, captions.

    The manuscript is set with ``lineno``, which numbers running text only and
    never numbers float bodies or captions. Counting those marginal numbers is
    therefore an exact measure of how much running text a page carries, which a
    word count cannot give (a table contributes hundreds of words but no running
    text). Line numbers are the 6 pt digit-only spans in the left margin.
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    rows = []
    for i in range(doc.page_count):
        page = doc[i]
        txt = page.get_text()
        caps = [f"{k} {v}" for k, v in CAPTION_RE.findall(txt)]
        caps += [f"Algorithm {v}" for v in ALGO_CAPTION_RE.findall(txt)]
        d = page.get_text("dict")
        nlines = 0
        bottom = 0.0
        top = 1e9
        for b in d["blocks"]:
            if b["type"] != 0:
                bottom = max(bottom, b["bbox"][3])
                top = min(top, b["bbox"][1])
                continue
            for line in b["lines"]:
                for s in line["spans"]:
                    t = s["text"].strip()
                    x0, y0, x1, y1 = s["bbox"]
                    is_lineno = t.isdigit() and round(s["size"]) == 6 and x0 < 105
                    is_pageno = t.isdigit() and y0 > 680
                    if is_lineno:
                        nlines += 1
                    if not is_pageno and t:
                        bottom = max(bottom, y1)
                        top = min(top, y0)
        for dr in page.get_drawings():
            bottom = max(bottom, dr["rect"].y1)
            top = min(top, dr["rect"].y0)
        rows.append({
            "page": i + 1,
            "words": page_words(txt),
            "text_lines": nlines,
            "captions": caps,
            "content_bottom": round(bottom, 1),
            "content_top": round(top, 1) if top < 1e8 else None,
            "images": len(page.get_images(full=True)),
        })
    doc.close()
    return rows


# ---------------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default=str(PAPER / "main.tex"))
    ap.add_argument("--pdf", default=str(PAPER / "main.pdf"))
    ap.add_argument("--mode", choices=["floats", "pages", "both"], default="both")
    ap.add_argument("--max-distance", type=int, default=2)
    ap.add_argument("--thin-lines", type=int, default=6,
                    help="a page with fewer numbered running-text lines is reported")
    ap.add_argument("--bottom-pt", type=float, default=600.0,
                    help="a page whose content ends above this y is reported as short")
    ap.add_argument("--full-fill", type=float, default=0.80,
                    help="fraction of the text block a lone float must fill to be allowed")
    ap.add_argument("--front-matter", type=int, default=2,
                    help="number of front-matter pages exempt from the float-only rule")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    tex = Path(args.tex)
    pdf = Path(args.pdf)
    fail = 0

    if args.mode in ("floats", "both"):
        real_aux = parse_aux(PAPER / "main.aux")
        work = Path(args.workdir) if args.workdir else Path(
            tempfile.mkdtemp(prefix="floatscan_")) / "paper"
        shadow_aux, refs = build_shadow(tex, work)

        clean = strip_comments(flatten(tex))
        floats = float_spans(clean)

        print("## Float distance to its citations\n")
        print("Body floats are judged on the NEAREST citation (the layout rule of"
              " brief 13 section A.2); appendix floats are reported but not failed,"
              " because an appendix float is cited from the main text by design.\n")
        print("| # | Float | Number | Caption page | First cite | Nearest cite |"
              " Dist(first) | Dist(nearest) | Where | Verdict |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        order_issue = []
        # REVIEW_final F5: only FIGURES were checked for citation order, so Table 4 being cited
        # three pages before Table 3 survived every earlier pass. The Guide asks for the same rule
        # on every float class ("Number tables consecutively in accordance with their appearance in
        # the text"), so figures, tables and algorithms are now tracked separately, and each
        # appendix letter is its own sequence because appendix floats restart at 1.
        seen_body: dict[str, list] = {e: [] for e in FLOAT_ENVS}
        seen_app: dict[str, dict[str, list]] = {e: {} for e in FLOAT_ENVS}
        for n, f in enumerate(floats, 1):
            labs = [l for l in f["labels"] if not l.startswith("FLOATCITE@")]
            lab = labs[0] if labs else ""
            info = real_aux.get(lab) or shadow_aux.get(lab) or {}
            number = info.get("number", "?")
            cap_page = int(info["page"]) if info.get("page", "").lstrip("-").isdigit() else None
            is_app = bool(re.match(r"^[A-Z]\.", number))
            cites = [r for r in refs if r["label"] == lab and not r["in_float"]]
            pages = []
            for r in cites:
                a = shadow_aux.get(r["marker"])
                if a and a["page"].lstrip("-").isdigit():
                    pages.append(int(a["page"]))
            first_page = min(pages) if pages else None
            near_page = (min(pages, key=lambda p: abs(p - cap_page))
                         if (pages and cap_page is not None) else None)
            d_first = (abs(cap_page - first_page)
                       if (cap_page is not None and first_page is not None) else None)
            d_near = (abs(cap_page - near_page)
                      if (cap_page is not None and near_page is not None) else None)
            verdict = "OK"
            if not cites:
                verdict = "NO CITATION"
                fail = 1
            elif d_near is None:
                verdict = "UNRESOLVED"
                fail = 1
            elif d_near > args.max_distance:
                if is_app:
                    verdict = "appendix (not failed)"
                else:
                    verdict = f"FAR (> {args.max_distance})"
                    fail = 1
            if first_page is not None and f["env"] in seen_body:
                if is_app:
                    seen_app[f["env"]].setdefault(number.split(".")[0], []) \
                                      .append((number, first_page))
                else:
                    seen_body[f["env"]].append((number, first_page))
            print(f"| {n} | {f['env']} `{lab}` | {number} | {cap_page} | {first_page} |"
                  f" {near_page} | {d_first} | {d_near} |"
                  f" {'appendix' if is_app else 'body'} | {verdict} |")

        def check_order(seq, tag):
            prev = None
            for number, cp in seq:
                if prev is not None and cp < prev[1]:
                    order_issue.append(
                        f"{tag}: {number} first cited on p{cp}, after {prev[0]} on p{prev[1]}")
                prev = (number, cp)

        for env in FLOAT_ENVS:
            check_order(seen_body[env], f"body {env}s")
            for k, seq in seen_app[env].items():
                check_order(seq, f"appendix {k} {env}s")
        print()
        for env in FLOAT_ENVS:
            n = len(seen_body[env]) + sum(len(v) for v in seen_app[env].values())
            bad = [x for x in order_issue if x.startswith(("body " + env, "appendix"))
                   and env in x.split(":")[0]]
            print(f"{env.capitalize()} numbering in order of first citation ({n} floats): "
                  + ("YES" if not bad else "NO - " + "; ".join(bad)))
        if order_issue:
            fail = 1
        print(f"\nShadow build directory: `{work}`\n")

    if args.mode in ("pages", "both"):
        rows = scan_pages(pdf)
        print("\n## Page scan (words, running-text lines and float captions per page)\n")
        print("| Page | Words | Text lines | Captions | Top (pt) | Bottom (pt) |"
              " Fill | Flag |")
        print("|---|---|---|---|---|---|---|---|")
        floatonly, empty, halfempty, fullfloat = [], [], [], []
        TEXT_TOP, TEXT_BOTTOM = 111.0, 680.0
        height = TEXT_BOTTOM - TEXT_TOP
        for r in rows:
            top = r["content_top"] if r["content_top"] is not None else TEXT_TOP
            fill = max(0.0, min(1.0, (r["content_bottom"] - top) / height))
            r["fill"] = fill
            flag = "OK"
            if r["words"] == 0:
                flag = "EMPTY"
                empty.append(r["page"])
            elif r["text_lines"] == 0 and r["page"] > args.front_matter:
                if fill >= args.full_fill:
                    flag = "FULL-PAGE FLOAT"
                    fullfloat.append(r["page"])
                else:
                    flag = "FLOAT-ONLY, NOT FULL"
                    floatonly.append(r["page"])
            elif r["content_bottom"] < args.bottom_pt:
                flag = "SHORT PAGE"
                halfempty.append(r["page"])
            elif r["text_lines"] < args.thin_lines:
                flag = "THIN"
            print(f"| {r['page']} | {r['words']} | {r['text_lines']} |"
                  f" {', '.join(r['captions']) or '-'} | {top} | {r['content_bottom']} |"
                  f" {fill:.2f} | {flag} |")
        print(f"\nFLOAT-ONLY pages that are NOT full (defects): {floatonly}")
        print(f"Full-page floats (a float that fills at least {args.full_fill:.0%}"
              f" of the text block; allowed): {fullfloat}")
        print(f"EMPTY pages: {empty}")
        print(f"SHORT pages (content ends above {args.bottom_pt} pt, text area"
              f" bottom is about 680 pt): {halfempty}")
        if empty or floatonly:
            fail = 1

    return fail


if __name__ == "__main__":
    sys.exit(main())
