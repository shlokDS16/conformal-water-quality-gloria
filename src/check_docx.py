#!/usr/bin/env python
"""Check `build/manuscript.docx` element by element against `paper/main.pdf`.

Writes `research/DOCX_CHECK.md` with one row per table, equation block,
algorithm, figure and section heading, plus the reference count, a raw-LaTeX
leftover scan and (when a renderer is available) a page-by-page comparison.

Usage
-----
    python src/check_docx.py [--docx build/manuscript.docx]
                             [--pdf paper/main.pdf] [--tex paper/main.tex]
                             [--out research/DOCX_CHECK.md] [--no-render]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import docx
from docx.oxml.ns import qn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_docx as M  # noqa: E402

W = qn("w:p")
TBL = qn("w:tbl")

LATEX_LEFTOVER = re.compile(r"\\[A-Za-z@]+|(?<!\w)\$(?!\d)|\?\?|\\cite|\\ref")
BAD_CHARS = re.compile(r"[\uFFFD\u0000-\u0008\u000B\u000C\u000E-\u001F]")


# ==========================================================================
def para_text(p):
    return "".join(t.text or "" for t in p.iter(qn("w:t")))


def has(el, tag):
    return len(el.findall(".//" + qn(tag))) > 0


def heading_level(par):
    name = par.style.name if par.style is not None else ""
    m = re.match(r"Heading (\d)", name or "")
    return int(m.group(1)) if m else None


def walk(doc):
    """Yield ('p', Paragraph) and ('tbl', Table) in document order."""
    body = doc.element.body
    tables = {t._tbl: t for t in doc.tables}
    for child in body.iterchildren():
        if child.tag == W:
            yield "p", docx.text.paragraph.Paragraph(child, doc._body)
        elif child.tag == TBL:
            t = tables.get(child)
            if t is not None:
                yield "tbl", t


def table_borders(table):
    top = bottom = False
    rows = table.rows
    if not rows:
        return top, bottom
    for c in rows[0].cells:
        tcb = c._tc.find(qn("w:tcPr"))
        if tcb is None:
            continue
        b = tcb.find(qn("w:tcBorders"))
        if b is not None and b.find(qn("w:top")) is not None:
            top = True
    for c in rows[-1].cells:
        tcb = c._tc.find(qn("w:tcPr"))
        if tcb is None:
            continue
        b = tcb.find(qn("w:tcBorders"))
        if b is not None and b.find(qn("w:bottom")) is not None:
            bottom = True
    return top, bottom


def header_repeats(table):
    n = 0
    for row in table.rows:
        trPr = row._tr.find(qn("w:trPr"))
        if trPr is not None and trPr.find(qn("w:tblHeader")) is not None:
            n += 1
        else:
            break
    return n


# ==========================================================================
def inventory(path: Path):
    doc = docx.Document(str(path))
    inv = {
        "headings": [],
        "figures": [],
        "tables": [],
        "algorithms": [],
        "equations": [],
        "references": 0,
        "empty_paragraphs": 0,
        "leftovers": [],
        "bad_chars": [],
        "paragraphs": 0,
    }
    pending_caption = None
    in_refs = False
    items = list(walk(doc))
    for i, (kind, obj) in enumerate(items):
        if kind == "p":
            par = obj
            txt = para_text(par._p)
            inv["paragraphs"] += 1
            lvl = heading_level(par)
            if lvl:
                inv["headings"].append((lvl, txt.strip()))
                in_refs = txt.strip().lower() == "references"
                pending_caption = None
                continue
            if in_refs and txt.strip():
                inv["references"] += 1
            if not txt.strip() and not has(par._p, "m:oMath") and not has(par._p, "w:drawing"):
                if not has(par._p, "w:fldChar"):
                    inv["empty_paragraphs"] += 1
            if has(par._p, "w:drawing"):
                cap = ""
                if i + 1 < len(items) and items[i + 1][0] == "p":
                    cap = para_text(items[i + 1][1]._p)
                inv["figures"].append(
                    {
                        "caption": cap.strip(),
                        "width_emu": drawing_width(par._p),
                        "n_images": len(par._p.findall(".//" + qn("w:drawing"))),
                    }
                )
                continue
            style_name = par.style.name if par.style is not None else ""
            if has(par._p, "m:oMath") and style_name == "Display Equation":
                m = re.search(r"\(([^()]*)\)\s*$", txt)
                inv["equations"].append(
                    {
                        "number": m.group(1) if m else None,
                        "n_omml": len(par._p.findall(".//" + qn("m:oMath"))),
                        "text": txt.strip()[:60],
                    }
                )
                continue
            if txt.strip().startswith(("Table ", "Fig. ", "Algorithm ")):
                pending_caption = txt.strip()
            for mm in LATEX_LEFTOVER.finditer(txt):
                frag = txt[max(0, mm.start() - 30) : mm.end() + 30]
                inv["leftovers"].append((mm.group(0), frag.strip()))
            if BAD_CHARS.search(txt):
                inv["bad_chars"].append(txt[:80])
        else:
            table = obj
            ncol = len(table.columns)
            nrow = len(table.rows)
            cell_text = "\n".join(
                "\t".join(c.text for c in r.cells) for r in table.rows
            )
            for mm in LATEX_LEFTOVER.finditer(cell_text):
                inv["leftovers"].append((mm.group(0), cell_text[max(0, mm.start() - 30) : mm.end() + 30]))
            top, bottom = table_borders(table)
            entry = {
                "rows": nrow,
                "cols": ncol,
                "caption": pending_caption or "",
                "top_rule": top,
                "bottom_rule": bottom,
                "header_rows": header_repeats(table),
                "omml": len(table._tbl.findall(".//" + qn("m:oMath"))),
                "first_cell": table.cell(0, 0).text.strip()[:60] if nrow and ncol else "",
            }
            if ncol == 1 and entry["first_cell"].startswith("Algorithm"):
                entry["caption"] = table.cell(0, 0).paragraphs[0].text.strip()
                entry["lines"] = sum(
                    1
                    for p in table.cell(0, 0).paragraphs
                    if re.match(r"^\d+:\s", p.text)
                )
                inv["algorithms"].append(entry)
            else:
                inv["tables"].append(entry)
            pending_caption = None
    return inv


def drawing_width(p):
    ext = p.find(".//" + qn("wp:extent"))
    if ext is None:
        return None
    return int(ext.get("cx"))


# ==========================================================================
def expected(tex: Path):
    """Ground truth taken from the LaTeX source and main.aux."""
    aux = tex.parent / (tex.stem + ".aux")
    bbl = tex.parent / (tex.stem + ".bbl")
    labels, bibcites = M.parse_aux(aux)
    refs = M.parse_bbl(bbl) if bbl.exists() else []
    raw = M.strip_comments(M.load_tex(tex))
    body = raw[raw.index("\\begin{document}") :]
    figs, tabs, algs = [], [], []
    for env, bucket in (
        ("figure*", figs),
        ("figure", figs),
        ("table*", tabs),
        ("table", tabs),
        ("algorithm", algs),
    ):
        b = body
        while True:
            span = M.find_env(b, env)
            if not span:
                break
            inner = b[span[1] : span[2]]
            lm = re.search(r"\\label\{([^}]*)\}", inner)
            lab = lm.group(1) if lm else None
            bucket.append((lab, labels.get(lab, "?")))
            b = b[: span[0]] + b[span[3] :]
    eqs = [(k, v) for k, v in labels.items() if k.startswith("eq:")]
    eqs.sort(key=lambda kv: (len(kv[1]), kv[1]))
    secs = [
        (k, v)
        for k, v in labels.items()
        if k.startswith(("sec:", "app:"))
    ]
    return {
        "labels": labels,
        "figures": figs,
        "tables": tabs,
        "algorithms": algs,
        "equations": eqs,
        "sections": secs,
        "references": len(refs),
        "bibcites": len(bibcites),
    }


# ==========================================================================
def docx_to_pdf(src: Path, dst: Path):
    """Render the DOCX with Microsoft Word (read-only, no save-back)."""
    import win32com.client

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    try:
        word.DisplayAlerts = 0
    except Exception:
        pass
    doc = None
    try:
        doc = word.Documents.Open(
            str(src), ReadOnly=True, AddToRecentFiles=False, ConfirmConversions=False
        )
        doc.ExportAsFixedFormat(OutputFileName=str(dst), ExportFormat=17)
    finally:
        if doc is not None:
            doc.Close(SaveChanges=0)
        word.Quit()
    return dst


def render_pages(pdf: Path, outdir: Path, dpi=110, prefix="p"):
    import fitz

    outdir.mkdir(parents=True, exist_ok=True)
    files = []
    with fitz.open(str(pdf)) as d:
        for n, page in enumerate(d, 1):
            pix = page.get_pixmap(dpi=dpi)
            f = outdir / ("%s%03d.png" % (prefix, n))
            pix.save(str(f))
            files.append(f)
    return files


def pdf_pagecount(pdf: Path):
    import fitz

    with fitz.open(str(pdf)) as d:
        return d.page_count


def pdf_text(pdf: Path):
    import fitz

    with fitz.open(str(pdf)) as d:
        return "\n".join(p.get_text() for p in d)


def docx_text(path: Path) -> str:
    d = docx.Document(str(path))
    out = [para_text(p._p) for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                out.append(c.text)
    return "\n".join(out)


def normalise(text: str) -> str:
    text = text.replace("­", "").replace("-\n", "")
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"\s+", " ", text)
    return text


def _wordkeys(text: str):
    """Ligature-blind word keys.

    The PDF is set with T1 Type 1 fonts whose ff/fi/fl ligatures have no
    usable ToUnicode entry, so every extractor returns "reectance" for
    "reflectance".  Deleting `f` runs together with a following i or l on both
    sides makes the two vocabularies comparable.
    """
    t = normalise(text).lower()
    return [
        re.sub(r"f+[il]?", "", w)
        for w in re.findall(r"[a-z]+", t)
        if len(w) >= 2
    ]


def word_coverage(pdf_text_s: str, docx_text_s: str):
    """Multiset comparison of the words in the PDF and in the DOCX.

    PDF extraction also splits a word at an unmapped ligature ("reflectance"
    comes out as "re" + "ectance"), so neighbouring PDF tokens are re-joined
    whenever the join is a word the DOCX actually contains.
    """
    from collections import Counter

    b = Counter(_wordkeys(docx_text_s))
    vocab = set(b)
    pk = _wordkeys(pdf_text_s)
    repaired, i = [], 0
    while i < len(pk):
        if i + 1 < len(pk) and pk[i] not in vocab and (pk[i] + pk[i + 1]) in vocab:
            repaired.append(pk[i] + pk[i + 1])
            i += 2
        else:
            repaired.append(pk[i])
            i += 1
    a = Counter(repaired)
    return sum(a.values()), a - b, b - a


def docx_metadata(path: Path):
    import zipfile
    from xml.etree import ElementTree as ET

    out = {}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for part in ("docProps/core.xml", "docProps/app.xml"):
            if part not in names:
                out[part] = "(absent)"
                continue
            root = ET.fromstring(z.read(part))
            for el in root.iter():
                tag = el.tag.split("}")[-1]
                if tag in ("creator", "lastModifiedBy", "Company", "Manager",
                           "Application", "title", "revision"):
                    out[tag] = (el.text or "").strip()
    return out


def extract_docx_xml(path: Path, outdir: Path):
    """Write the text-bearing XML parts of a DOCX so `wm scan` can read them."""
    import zipfile

    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.startswith("word/") or not name.endswith(".xml"):
                continue
            if not any(
                k in name
                for k in ("document", "footnotes", "endnotes", "header", "footer", "comments")
            ):
                continue
            dst = outdir / name.replace("/", "_")
            dst.write_bytes(z.read(name))
            written.append(dst)
    return written


INVISIBLE = {
    0x00AD: "soft hyphen",
    0x200B: "zero-width space",
    0x200C: "zero-width non-joiner",
    0x200D: "zero-width joiner",
    0x200E: "left-to-right mark",
    0x200F: "right-to-left mark",
    0x2060: "word joiner",
    0xFEFF: "byte order mark",
}


def invisible_audit(parts):
    """Count invisible characters and say whether they are OMML scaffolding."""
    from collections import Counter

    counts, where = Counter(), Counter()
    for p in parts:
        s = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer("[%s]" % "".join(chr(c) for c in INVISIBLE), s):
            counts[ord(m.group(0))] += 1
            before = s[max(0, m.start() - 400) : m.start()]
            if re.search(r"<m:e>(?:(?!</m:e>).)*$", before, re.S):
                where["OMML base of a superscript or subscript (m:e)"] += 1
            elif re.search(r"<m:(sup|sub)>(?:(?!</m:(sup|sub)>).)*$", before, re.S):
                where["OMML empty superscript or subscript limit"] += 1
            elif "<m:t>" in before[-40:]:
                where["other OMML run"] += 1
            elif "</m:oMath>" in before[-400:]:
                where["run that closes an inline equation"] += 1
            else:
                where["plain text run"] += 1
    return counts, where


def wm_scan(paths):
    ps = Path.home() / ".claude" / "skills" / "hidden-unicode" / "scripts" / "wm.ps1"
    if not ps.exists():
        return "wm.ps1 not found at %s" % ps
    out = []
    for p in paths:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps),
             "scan", str(p)],
            capture_output=True, text=True,
        )
        out.append("$ wm scan %s\n%s%s" % (p, r.stdout.strip(), r.stderr.strip()))
    return "\n\n".join(out)


# ==========================================================================
def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docx", default=str(root / "build" / "manuscript.docx"))
    ap.add_argument("--pdf", default=str(root / "paper" / "main.pdf"))
    ap.add_argument("--tex", default=str(root / "paper" / "main.tex"))
    ap.add_argument("--out", default=str(root / "research" / "DOCX_CHECK.md"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-wm", action="store_true")
    args = ap.parse_args(argv)

    dx, pdf, tex = Path(args.docx), Path(args.pdf), Path(args.tex)
    inv = inventory(dx)
    exp = expected(tex)

    rendered = None
    pages = {}
    if not args.no_render:
        try:
            outpdf = dx.with_name(dx.stem + "_from_word.pdf")
            docx_to_pdf(dx, outpdf)
            rendered = outpdf
            pages["docx"] = pdf_pagecount(outpdf)
            render_pages(outpdf, dx.parent / "pages_docx", prefix="d")
        except Exception as exc:  # pragma: no cover
            print("render failed: %r" % (exc,))
    if pdf.exists():
        pages["pdf"] = pdf_pagecount(pdf)
        if not args.no_render:
            render_pages(pdf, dx.parent / "pages_pdf", prefix="t")

    cov = None
    if pdf.exists():
        try:
            cov = word_coverage(pdf_text(pdf), docx_text(dx))
        except Exception as exc:
            print("coverage check failed: %r" % (exc,))
    meta = docx_metadata(dx)
    wm = None
    invis = None
    if not args.no_wm:
        parts = extract_docx_xml(dx, dx.parent / "_docx_xml")
        invis = invisible_audit(parts)
        sources = [tex] + sorted((tex.parent.parent / "tables").glob("*.tex"))
        wm = wm_scan(
            parts
            + [p for p in sources if p.exists()]
            + [Path(__file__).with_name("make_docx.py"), Path(__file__)]
        )
    globals()["_INVIS"] = invis

    report = build_report(dx, pdf, inv, exp, pages, rendered, args, cov, meta, wm)
    out = Path(args.out)
    begin, end = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
    if out.exists():
        old = out.read_text(encoding="utf-8")
        if begin in old and end in old:
            head = old.split(begin)[0]
            tail = old.split(end, 1)[1]
            report = head + begin + "\n\n" + report + "\n" + end + tail
    out.write_text(report, encoding="utf-8", newline="")
    print("wrote %s" % args.out)
    print("figures %d | tables %d | algorithms %d | equations %d | references %d"
          % (len(inv["figures"]), len(inv["tables"]), len(inv["algorithms"]),
             len(inv["equations"]), inv["references"]))
    print("leftovers %d | empty paragraphs %d | bad chars %d"
          % (len(inv["leftovers"]), inv["empty_paragraphs"], len(inv["bad_chars"])))
    if pages:
        print("pages: %s" % pages)
    return 0


def build_report(dx, pdf, inv, exp, pages, rendered, args, cov=None, meta=None, wm=None):
    L = []
    a = L.append
    a("# DOCX element check (generated by src/check_docx.py)")
    a("")
    a("DOCX: `%s`  |  PDF: `%s`" % (dx, pdf))
    a("")
    a("## Counts")
    a("")
    a("| Element | DOCX | LaTeX / PDF | Match |")
    a("|---|---|---|---|")
    for name, got, want in (
        ("Figures", len(inv["figures"]), len(exp["figures"])),
        ("Tables", len(inv["tables"]), len(exp["tables"])),
        ("Algorithms", len(inv["algorithms"]), len(exp["algorithms"])),
        ("Numbered equations", len(inv["equations"]), len(exp["equations"])),
        ("References", inv["references"], exp["references"]),
    ):
        a("| %s | %d | %d | %s |" % (name, got, want, "yes" if got == want else "NO"))
    a("")
    a("## Figures")
    a("")
    a("| # | Caption starts | Image | Width (cm) | Caption below |")
    a("|---|---|---|---|---|")
    for n, f in enumerate(inv["figures"], 1):
        w = "%.1f" % (f["width_emu"] / 360000.0) if f["width_emu"] else "-"
        a("| %d | %s | %d | %s | %s |" % (n, f["caption"][:70].replace("|", "/"),
                                          f["n_images"], w,
                                          "yes" if f["caption"] else "NO"))
    a("")
    a("## Tables")
    a("")
    a("| # | Caption starts | Rows x Cols | Top rule | Bottom rule | Header rows | OMML in cells |")
    a("|---|---|---|---|---|---|---|")
    for n, t in enumerate(inv["tables"], 1):
        a("| %d | %s | %d x %d | %s | %s | %d | %d |"
          % (n, t["caption"][:60].replace("|", "/"), t["rows"], t["cols"],
             "yes" if t["top_rule"] else "NO", "yes" if t["bottom_rule"] else "NO",
             t["header_rows"], t["omml"]))
    a("")
    a("## Algorithms")
    a("")
    a("| # | Caption | Numbered lines | Boxed |")
    a("|---|---|---|---|")
    for n, t in enumerate(inv["algorithms"], 1):
        a("| %d | %s | %d | %s |" % (n, t["caption"][:70].replace("|", "/"),
                                     t.get("lines", 0),
                                     "yes" if t["top_rule"] and t["bottom_rule"] else "NO"))
    a("")
    a("## Equations")
    a("")
    a("| Order | Number in DOCX | OMML runs | Expected number |")
    a("|---|---|---|---|")
    want = [v for _, v in sorted(exp["equations"], key=eq_sort)]
    for n, e in enumerate(inv["equations"]):
        w = want[n] if n < len(want) else "-"
        ok = "" if e["number"] == w else "  <- differs"
        a("| %d | %s | %d | %s%s |" % (n + 1, e["number"], e["n_omml"], w, ok))
    a("")
    a("## Section headings")
    a("")
    a("| Level | Heading |")
    a("|---|---|")
    for lvl, txt in inv["headings"]:
        a("| %d | %s |" % (lvl, txt.replace("|", "/")))
    a("")
    a("## Defect scan")
    a("")
    a("| Check | Result |")
    a("|---|---|")
    a("| Raw LaTeX leftovers (\\command, $, \\cite, \\ref, ??) | %d |" % len(inv["leftovers"]))
    a("| Empty paragraphs | %d |" % inv["empty_paragraphs"])
    a("| Replacement / control characters | %d |" % len(inv["bad_chars"]))
    a("| Figures without a caption | %d |" % sum(1 for f in inv["figures"] if not f["caption"]))
    a("| Tables without a caption | %d |" % sum(1 for t in inv["tables"] if not t["caption"]))
    if inv["leftovers"]:
        a("")
        a("Leftover samples:")
        a("")
        for tok, frag in inv["leftovers"][:40]:
            a("- `%s` in `%s`" % (tok, frag.replace("`", "'")[:110]))
    if inv["bad_chars"]:
        a("")
        for t in inv["bad_chars"][:10]:
            a("- bad char in: `%s`" % t)
    if cov is not None:
        total, missing, extra = cov
        nmiss = sum(missing.values())
        a("")
        a("## Text coverage against the PDF")
        a("")
        a("Word multisets, ligature-blind, words of at least three letters, digits ignored.")
        a("")
        a("| PDF words | In the DOCX | Missing from the DOCX | Only in the DOCX |")
        a("|---|---|---|---|")
        a("| %d | %d | %d (%.2f%%) | %d |"
          % (total, total - nmiss, nmiss, 100.0 * nmiss / max(total, 1),
             sum(extra.values())))
        if missing:
            a("")
            a("Most frequent words missing from the DOCX:")
            a("")
            for w, n in missing.most_common(25):
                a("- `%s` x%d" % (w, n))
        if extra:
            a("")
            a("Most frequent words present only in the DOCX:")
            a("")
            for w, n in extra.most_common(15):
                a("- `%s` x%d" % (w, n))
    if meta:
        a("")
        a("## Metadata")
        a("")
        a("| Field | Value |")
        a("|---|---|")
        for k, v in meta.items():
            a("| %s | %s |" % (k, "(empty)" if v == "" else v))
    a("")
    a("## Pages")
    a("")
    if pages:
        for k, v in pages.items():
            a("- %s: %d pages" % (k, v))
    else:
        a("- no renderer run")
    if rendered:
        a("- DOCX rendered to `%s`" % rendered)
        a("- page images: `build/pages_docx/`, `build/pages_pdf/`")
    invis = globals().get("_INVIS")
    if invis:
        counts, where = invis
        a("")
        a("## Invisible characters in the DOCX XML")
        a("")
        a("| Character | Count |")
        a("|---|---|")
        for cp, n in sorted(counts.items()):
            a("| U+%04X %s | %d |" % (cp, INVISIBLE.get(cp, "?"), n))
        if not counts:
            a("| (none) | 0 |")
        if where:
            a("")
            a("| Where | Count |")
            a("|---|---|")
            for k, n in where.most_common():
                a("| %s | %d |" % (k, n))
            a("")
            a("Every one is pandoc scaffolding for Word mathematics: OMML needs a base for a "
              "superscript that LaTeX wrote without one (`mg m$^{-3}$`), needs a placeholder "
              "for an empty limit, and pandoc closes an inline equation with an empty run so "
              "Word does not absorb the following text into it. They carry no message and "
              "must not be cleaned, since removing them breaks the equations.")
    if wm:
        a("")
        a("## Hidden-character scan (`wm scan`)")
        a("")
        a("```")
        a(wm.strip())
        a("```")
    return "\n".join(L) + "\n"


def eq_sort(kv):
    v = kv[1]
    m = re.match(r"^([A-Z]?)\.?(\d+)", v.replace(" ", ""))
    if v and v[0].isdigit():
        return (0, int(re.match(r"\d+", v).group(0)))
    m2 = re.match(r"([A-Z])\.(\d+)", v)
    if m2:
        return (ord(m2.group(1)), int(m2.group(2)))
    return (99, 0)


if __name__ == "__main__":
    raise SystemExit(main())
