#!/usr/bin/env python
"""Verify that every internal hyperlink of the PDF lands on the object it names.

The manuscript resets the table and figure counters in each appendix. hyperref names
its destinations from the raw counter, so without the ``\\theHtable`` fix in the preamble
"Table A.1" and "Table 1" would share one destination and an appendix link would jump to
the main-text float. This script checks the result rather than trusting the fix.

Method, for every link annotation in ``paper/main.pdf``:

1. read the visible text under the link rectangle (for example "A.1", "5", "3.4");
2. read the link destination: its target page and, where the PDF stores one, its named
   destination (for example ``table.A.1``, ``figure.12``, ``equation.3.7``);
3. look up that destination name in ``paper/main.aux``, which gives the number and page
   LaTeX assigned to it;
4. fail if the number printed in the text differs from the number of the destination, or
   if the destination page differs from the page recorded in the .aux.

Usage:  python src/check_links.py [--pdf paper/main.pdf] [--aux paper/main.aux]
Exit status 1 if any link is inconsistent or unresolved.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]


def balanced(text: str, open_idx: int) -> str:
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


def parse_aux(path: Path) -> dict[str, list[dict]]:
    """anchor name -> list of {label, number, page}, built from \\newlabel.

    One anchor can carry several labels when an environment creates no destination of
    its own; that case is reported separately as an ambiguous anchor.
    """
    txt = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, list[dict]] = {}
    for m in re.finditer(r"\\newlabel\{", txt):
        lab = balanced(txt, m.end() - 1)
        rest = m.end() - 1 + len(lab) + 2
        if rest >= len(txt) or txt[rest] != "{":
            continue
        group = balanced(txt, rest)
        fields = []
        i = 0
        while i < len(group) and len(fields) < 5:
            if group[i] == "{":
                f = balanced(group, i)
                fields.append(f)
                i += len(f) + 2
            else:
                i += 1
        if len(fields) >= 4 and fields[3]:
            out.setdefault(fields[3], []).append(
                {"label": lab, "number": fields[0], "page": fields[1]})
    return out


def clean(num: str) -> str:
    """Strip LaTeX markup a number field may carry."""
    n = re.sub(r"\\[A-Za-z]+\s*", "", num)
    return n.replace("{", "").replace("}", "").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", default=str(ROOT / "paper" / "main.pdf"))
    ap.add_argument("--aux", default=str(ROOT / "paper" / "main.aux"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    aux = parse_aux(Path(args.aux))
    doc = fitz.open(args.pdf)
    # page label offset: LaTeX page 1 is the PDF page that carries the title, and the
    # highlights page sits in front of it, so the .aux page number and the PDF page index
    # differ by a constant. Determine it from a destination we can see.
    total, checked, bad, unnamed = 0, 0, [], 0
    kinds = Counter()
    offset_votes = Counter()

    links = []
    for pno in range(doc.page_count):
        page = doc[pno]
        for l in page.get_links():
            total += 1
            kinds[l.get("kind")] += 1
            # internal links are stored as named destinations (kind 4) by hyperref;
            # kind 1 is an explicit goto. Both are internal.
            if l.get("kind") not in (fitz.LINK_GOTO, fitz.LINK_NAMED):
                continue
            name = l.get("nameddest") or ""
            rect = fitz.Rect(l["from"])
            text = page.get_textbox(rect).strip()
            tgt = l.get("page", -1)
            links.append({"src_page": pno + 1, "name": name, "text": text,
                          "tgt_page": tgt + 1 if tgt >= 0 else None})
            if name in aux and aux[name][0]["page"].isdigit() and tgt >= 0:
                offset_votes[(tgt + 1) - int(aux[name][0]["page"])] += 1

    offset = offset_votes.most_common(1)[0][0] if offset_votes else 0

    for l in links:
        name, text = l["name"], l["text"]
        if not name:
            unnamed += 1
            continue
        infos = aux.get(name)
        if infos is None:
            # cite.* destinations point at bibliography items, which carry no \newlabel
            if name.startswith(("cite.", "Hfootnote.", "page.", "AlgoLine.")):
                continue
            bad.append((l, "destination name not in .aux"))
            continue
        checked += 1
        norm_text = text.replace("\u2013", "-").replace("\xa0", " ")
        norm_text = re.sub(r"\s+", " ", norm_text)
        # the visible text of a \ref is the number itself; hyperref also links the
        # surrounding word for \autoref, so require containment rather than equality.
        # A shared anchor is consistent if ANY of its labels matches.
        ok_text, ok_page, tried = False, False, []
        for info in infos:
            number = clean(info["number"]).replace("~", " ")
            tried.append(number)
            if number and number in norm_text:
                ok_text = True
                if info["page"].isdigit() and l["tgt_page"] is not None:
                    ok_page = (l["tgt_page"] == int(info["page"]) + offset)
                else:
                    ok_page = True
                if ok_page:
                    break
        if not ok_text:
            bad.append((l, f"text {text!r} matches none of the numbers {tried} of {name}"))
        elif not ok_page:
            bad.append((l, f"jumps to PDF page {l['tgt_page']}, which is not the page"
                           f" of {name}"))

    print("## Hyperlink and cross-reference resolution\n")
    print(f"Link annotations in the PDF: {total}")
    print(f"  by kind: {dict(kinds)}  (1 = internal goto, 2 = URI)")
    print(f"Internal links with a named destination checked against main.aux: {checked}")
    print(f"Internal links without a named destination (not checkable): {unnamed}")
    print(f"PDF page = LaTeX page + {offset}")
    shared = {k: [i["label"] for i in v] for k, v in aux.items() if len(v) > 1}
    print(f"Anchors carrying more than one label (a link to any of them lands on the"
          f" same spot): {len(shared)}")
    for k, v in list(shared.items())[:20]:
        print(f"  - {k}: {v}")
    print(f"Inconsistent links: {len(bad)}")
    for l, why in bad[:40]:
        print(f"  - p{l['src_page']} {l['name']}: {why}")

    # appendix floats specifically
    apat = r"^(table|figure)\.(Appendix.|[A-C]\.)"
    app = [l for l in links if re.match(apat, l["name"] or "")]
    print(f"\nLinks to appendix floats: {len(app)}")
    appbad = [x for x in bad if re.match(apat, x[0]["name"] or "")]
    print(f"  inconsistent: {len(appbad)}")
    if args.verbose:
        for l in app:
            print(f"  p{l['src_page']} {l['name']:16s} text={l['text']!r}"
                  f" -> PDF page {l['tgt_page']}")
    doc.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
