#!/usr/bin/env python
"""Build a journal-quality Word manuscript from the LaTeX source.

Phase 13 section B of the project brief.  The script is idempotent: every run
rebuilds `build/manuscript.docx` from scratch from `paper/main.tex`, the LaTeX
auxiliary files (`main.aux`, `main.bbl`) and the figure directory.

Design notes
------------
* Numbering (sections, equations, figures, tables, algorithms, propositions)
  is taken from `paper/main.aux`, so it is identical to the compiled PDF by
  construction rather than by re-derivation.
* Citations are rendered from the natbib `\\bibcite` records in `main.aux`
  and the reference list from `main.bbl`, so both match the PDF and neither
  needs `refs.bib`.
* Pandoc converts the prose and the mathematics (LaTeX -> OMML).  Tables,
  algorithms and figures are extracted first and rebuilt with python-docx so
  that they become native Word objects with booktabs-like rules, header rows,
  captions and text-width sizing.

Usage
-----
    python src/make_docx.py [--tex paper/main.tex] [--out build/manuscript.docx]
                            [--figdir figures]
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import docx
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor

# --------------------------------------------------------------------------
# page geometry (A4, 25 mm margins, single column -- Guide sets no hard rule,
# only "single-column format" and "keep the layout as simple as possible")
# --------------------------------------------------------------------------
PAGE_W = Cm(21.0)
PAGE_H = Cm(29.7)
MARGIN = Cm(2.5)
TEXT_W = Cm(16.0)
LATEX_TEXTWIDTH_PT = 390.0  # elsarticle [preprint,12pt] \textwidth

BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(12)
SMALL_SIZE = Pt(9)
CAPTION_SIZE = Pt(10)

RED = RGBColor(0xC0, 0x00, 0x00)
BLUE = RGBColor(0x00, 0x00, 0xC0)

MARKER_RE = re.compile(
    r"\[(?:RES:[^\[\]]*|cite needed:[^\[\]]*|Drafting pending:[^\[\]]*)\]", re.S
)

PLACEHOLDER_RE = re.compile(r"^PDXBLOCK(\d{4})$")


# ==========================================================================
# small LaTeX utilities
# ==========================================================================
def read_group(s: str, i: int):
    """s[i] must be '{'.  Return (content, index after the closing brace)."""
    assert s[i] == "{", s[i : i + 30]
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
        j += 1
    raise ValueError("unbalanced braces at %d: %r" % (i, s[i : i + 60]))


def read_optional(s: str, i: int):
    """If s[i] == '[', return (content, index after ']'), else (None, i)."""
    if i < len(s) and s[i] == "[":
        depth = 0
        j = i
        while j < len(s):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    return s[i + 1 : j], j + 1
            j += 1
    return None, i


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        j = 0
        cut = None
        while j < len(line):
            if line[j] == "\\":
                j += 2
                continue
            if line[j] == "%":
                cut = j
                break
            j += 1
        if cut is not None:
            line = line[:cut]
            if line.strip() == "":
                continue
        out.append(line)
    return "\n".join(out)


def split_top_level(s: str, sep: str):
    """Split on `sep` outside braces and outside nested environments."""
    parts, depth, env, start, i = [], 0, 0, 0, 0
    n, m = len(s), len(sep)
    while i < n:
        c = s[i]
        if c == "\\":
            if s.startswith("\\begin{", i):
                env += 1
                _, i = read_group(s, i + 6)
                continue
            if s.startswith("\\end{", i):
                env -= 1
                _, i = read_group(s, i + 4)
                continue
            if s[i : i + m] == sep and depth == 0 and env == 0:
                parts.append(s[start:i])
                i += m
                start = i
                continue
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0 and env == 0 and s[i : i + m] == sep:
            parts.append(s[start:i])
            i += m
            start = i
            continue
        i += 1
    parts.append(s[start:])
    return parts


def find_env(text: str, name: str, start: int = 0):
    """Return (begin_index, body_start, body_end, end_index) of the next env."""
    tag = "\\begin{%s}" % name
    endtag = "\\end{%s}" % name
    i = text.find(tag, start)
    if i < 0:
        return None
    depth = 1
    j = i + len(tag)
    while depth:
        nb = text.find(tag, j)
        ne = text.find(endtag, j)
        if ne < 0:
            raise ValueError("unterminated environment %s" % name)
        if 0 <= nb < ne:
            depth += 1
            j = nb + len(tag)
        else:
            depth -= 1
            j = ne + len(endtag)
    return i, i + len(tag), j - len(endtag), j


# ==========================================================================
# LaTeX auxiliary files
# ==========================================================================
def load_tex(path: Path) -> str:
    """Read a .tex file and expand \\input{...} recursively."""
    text = path.read_text(encoding="utf-8")
    out, i = [], 0
    while True:
        m = re.search(r"\\input\{([^}]*)\}", text[i:])
        if not m:
            out.append(text[i:])
            break
        out.append(text[i : i + m.start()])
        sub = m.group(1)
        if not sub.endswith(".tex"):
            sub += ".tex"
        subpath = (path.parent / sub).resolve()
        out.append(load_tex(subpath))
        i += m.end()
    return "".join(out)


def parse_aux(path: Path):
    """Return (labels, bibcites) from main.aux."""
    text = path.read_text(encoding="utf-8", errors="replace")
    labels, bibcites = {}, {}
    for m in re.finditer(r"\\newlabel\{", text):
        i = m.end() - 1
        name, j = read_group(text, i)
        if j >= len(text) or text[j] != "{":
            continue
        payload, _ = read_group(text, j)
        if not payload.startswith("{"):
            continue
        number, _ = read_group(payload, 0)
        labels[name] = number.replace("~", " ").strip()
    for m in re.finditer(r"\\bibcite\{", text):
        i = m.end() - 1
        key, j = read_group(text, i)
        payload, _ = read_group(text, j)
        fields, k = [], 0
        while k < len(payload) and len(fields) < 4:
            if payload[k] == "{":
                g, k = read_group(payload, k)
                fields.append(g)
            else:
                k += 1
        while len(fields) < 4:
            fields.append("")
        short = fields[2].strip()
        while short.startswith("{") and short.endswith("}"):
            short = short[1:-1].strip()
        bibcites[key] = {"num": fields[0], "year": fields[1].strip(), "short": short}
    return labels, bibcites


BBL_PREAMBLE = r"""
\providecommand{\bibinfo}[2]{#2}
\providecommand{\natexlab}[1]{#1}
\providecommand{\path}[1]{\texttt{#1}}
\providecommand{\href}[2]{#2}
\providecommand{\DOIprefix}{doi:}
\providecommand{\ArXivprefix}{arXiv:}
\providecommand{\URLprefix}{URL: }
\providecommand{\Pubmedprefix}{pmid:}
\providecommand{\doi}[1]{doi:#1}
\providecommand{\Pubmed}[1]{pmid:#1}
\providecommand{\urlprefix}{URL: }
\providecommand{\newblock}{ }
"""


def parse_bbl(path: Path):
    """Return an ordered list of (key, latex) reference entries."""
    text = path.read_text(encoding="utf-8", errors="replace")
    span = find_env(text, "thebibliography")
    if span:
        text = text[span[1] : span[2]]
    entries = []
    for m in re.finditer(r"\\bibitem", text):
        i = m.end()
        _, i = read_optional(text, i)
        if i < len(text) and text[i] == "{":
            key, i = read_group(text, i)
        else:
            continue
        nxt = text.find("\\bibitem", i)
        body = text[i : nxt if nxt > 0 else len(text)]
        body = body.replace("\\newblock", " ")
        body = re.sub(r"\s+", " ", body).strip()
        entries.append((key, body))
    return entries


# ==========================================================================
# citation and cross-reference resolution
# ==========================================================================
def _cite_names(keys, bibcites, missing):
    out = []
    for k in [k.strip() for k in keys.split(",") if k.strip()]:
        rec = bibcites.get(k)
        if rec is None:
            missing.append(k)
            out.append((k, "????"))
        else:
            out.append((rec["short"].replace("~", " "), rec["year"]))
    return out


def resolve_citations(text: str, bibcites: dict, missing: list) -> str:
    """Replace natbib author-year commands with the text the PDF shows."""
    pat = re.compile(r"\\(citep|citet|citealp|citealt|citeauthor|citeyearpar|citeyear|cite)\b")
    out, i = [], 0
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i : m.start()])
        j = m.end()
        opt1, j = read_optional(text, j)
        opt2, j = read_optional(text, j)
        if j >= len(text) or text[j] != "{":
            out.append(m.group(0))
            i = m.end()
            continue
        keys, j = read_group(text, j)
        pre = opt1 if opt2 is not None else None
        post = opt2 if opt2 is not None else opt1
        names = _cite_names(keys, bibcites, missing)
        cmd = m.group(1)
        if cmd in ("citet", "citealt"):
            bits = []
            for k, (nm, yr) in enumerate(names):
                tail = yr
                if post and k == len(names) - 1:
                    tail = "%s, %s" % (yr, post)
                if cmd == "citet":
                    bits.append("%s (%s)" % (nm, tail))
                else:
                    bits.append("%s %s" % (nm, tail))
            rep = "; ".join(bits)
        elif cmd == "citeauthor":
            rep = "; ".join(nm for nm, _ in names)
        elif cmd in ("citeyear",):
            rep = ", ".join(yr for _, yr in names)
        elif cmd in ("citeyearpar",):
            rep = "(%s)" % ", ".join(yr for _, yr in names)
        else:  # citep, citealp, cite
            bits = ["%s, %s" % (nm, yr) for nm, yr in names]
            body = "; ".join(bits)
            if pre:
                body = "%s %s" % (pre, body)
            if post:
                body = "%s, %s" % (body, post)
            rep = body if cmd == "citealp" else "(%s)" % body
        out.append(rep)
        i = j
    return "".join(out)


def resolve_refs(text: str, labels: dict, missing: list) -> str:
    def one(m):
        cmd, lab = m.group(1), m.group(2)
        if lab not in labels:
            missing.append(lab)
            return "??"
        num = labels[lab]
        return "(%s)" % num if cmd == "eqref" else num

    return re.sub(r"\\(ref|eqref|autoref)\{([^}]*)\}", one, text)


# ==========================================================================
# pandoc fragment renderer (LaTeX snippet -> Word runs, math as OMML)
# ==========================================================================
def sanitise_fragment(el):
    """Drop relationship-bearing XML that cannot survive a copy between parts.

    Pandoc renders `\\href` as a `w:hyperlink` whose `r:id` points at a
    relationship of the fragment document.  Copying that element into the
    manuscript leaves a dangling relationship and Word then refuses to open
    the file, so the hyperlink wrapper is unwrapped into its runs.
    """
    for hl in list(el.iter(qn("w:hyperlink"))):
        parent = hl.getparent()
        if parent is None:
            continue
        for child in list(hl):
            hl.addprevious(child)
        parent.remove(hl)
    rid = qn("r:id")
    rembed = qn("r:embed")
    for node in list(el.iter()):
        if node.get(rid) is not None or node.get(rembed) is not None:
            p = node.getparent()
            if p is not None:
                p.remove(node)
    # the Hyperlink character style would show a blue underlined string that is
    # no longer a link, so it is dropped
    for st in list(el.iter(qn("w:rStyle"))):
        if st.get(qn("w:val")) in ("Hyperlink", "FollowedHyperlink"):
            st.getparent().remove(st)
    return el


class FragmentRenderer:
    """Convert many small LaTeX snippets to Word paragraphs in one pandoc run."""

    def __init__(self, macros: str, workdir: Path, refdoc: Path):
        self.macros = macros
        self.workdir = workdir
        self.refdoc = refdoc
        self.frags = []

    def add(self, latex: str) -> int:
        self.frags.append(latex if latex is not None else "")
        return len(self.frags) - 1

    def render(self):
        self.rendered = [[] for _ in self.frags]
        if not self.frags:
            return
        lines = [
            "\\documentclass{article}",
            "\\usepackage{amsmath,amssymb}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage[utf8]{inputenc}",
            BBL_PREAMBLE,
            self.macros,
            "\\begin{document}",
        ]
        for n, frag in enumerate(self.frags):
            lines.append("\n@@F%d@@\n" % n)
            lines.append(frag.strip() if frag.strip() else "~")
        lines.append("\\end{document}")
        tex = self.workdir / "_fragments.tex"
        tex.write_text("\n".join(lines), encoding="utf-8", newline="")
        out = self.workdir / "_fragments.docx"
        run_pandoc(tex, out, self.refdoc, self.workdir)
        d = docx.Document(str(out))
        cur = None
        for child in d.element.body.iterchildren():
            if child.tag != qn("w:p"):
                continue
            txt = "".join(t.text or "" for t in child.iter(qn("w:t"))).strip()
            m = re.fullmatch(r"@@F(\d+)@@", txt)
            if m:
                cur = int(m.group(1))
                continue
            if cur is None:
                continue
            if txt == "~" and not list(child.iter(qn("m:oMath"))):
                continue
            self.rendered[cur].append(sanitise_fragment(copy.deepcopy(child)))

    def paragraphs(self, idx: int):
        return self.rendered[idx]

    def runs(self, idx: int):
        """Inline content of a fragment: children of its first paragraph."""
        pars = self.rendered[idx]
        if not pars:
            return []
        kids = []
        for p in pars:
            for child in p.iterchildren():
                if child.tag == qn("w:pPr"):
                    continue
                kids.append(copy.deepcopy(child))
        return kids


def run_pandoc(src: Path, dst: Path, refdoc: Path, resource_dir: Path):
    cmd = [
        "pandoc",
        str(src),
        "-f",
        "latex",
        "-t",
        "docx",
        "--reference-doc",
        str(refdoc),
        "--resource-path",
        str(resource_dir),
        "-o",
        str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
        raise SystemExit("pandoc failed for %s" % src)
    if res.stderr.strip():
        for line in res.stderr.strip().split("\n"):
            print("   pandoc: %s" % line)


# ==========================================================================
# tabular parsing
# ==========================================================================
def parse_colspec(spec: str):
    """Return a list of dicts with 'align' and optional 'frac' per column."""
    cols, i = [], 0
    spec = spec.replace("|", "")
    while i < len(spec):
        c = spec[i]
        if c == "@":
            _, i = read_group(spec, i + 1)
            continue
        if c == ">" or c == "<":
            _, i = read_group(spec, i + 1)
            continue
        if c in "lcr":
            cols.append({"align": c, "frac": None})
            i += 1
            continue
        if c in "pmbLRC":
            arg, i = read_group(spec, i + 1)
            frac = None
            fm = re.search(r"([0-9.]+)\s*\\(?:textwidth|columnwidth|linewidth)", arg)
            if fm:
                frac = float(fm.group(1))
            align = {"L": "l", "R": "r", "C": "c"}.get(c, "l")
            cols.append({"align": align, "frac": frac})
            continue
        i += 1
    return cols


def parse_tabular(body: str):
    """Return (colspec, rows) where a row is (cells, rule_below).

    A cell is a dict: text (LaTeX), span, align override, indent level.
    """
    body = body.strip()
    assert body[0] == "{", body[:40]
    spec, i = read_group(body, 0)
    cols = parse_colspec(spec)
    rest = body[i:]
    raw_rows = split_top_level(rest, "\\\\")
    rows = []
    pending_rule = None
    for raw in raw_rows:
        rule_before = pending_rule
        pending_rule = None
        # pull leading rules
        while True:
            m = re.match(r"\s*\\(toprule|midrule|bottomrule|hline)\b(\[[^\]]*\])?", raw)
            if m:
                rule_before = m.group(1)
                raw = raw[m.end() :]
                continue
            m = re.match(r"\s*\\cmidrule\b(\([^)]*\))?(\{[^}]*\})?", raw)
            if m:
                rule_before = "cmidrule"
                raw = raw[m.end() :]
                continue
            break
        # trailing rules belong to the next boundary
        while True:
            m = re.search(r"\\(toprule|midrule|bottomrule|hline)\b\s*$", raw)
            if m:
                pending_rule = m.group(1)
                raw = raw[: m.start()]
                continue
            break
        if raw.strip() == "":
            if rows and rule_before:
                rows[-1][1] = rule_before
            elif rule_before:
                rows.append([None, rule_before])
            continue
        cells = []
        for cell in split_top_level(raw, "&"):
            cell = cell.strip()
            span, align = 1, None
            m = re.match(r"\\multicolumn\s*\{(\d+)\}\s*\{", cell)
            if m:
                span = int(m.group(1))
                sp, k = read_group(cell, m.end() - 1)
                al = parse_colspec(sp)
                align = al[0]["align"] if al else "l"
                cell, _ = read_group(cell, k)
                cell = cell.strip()
            indent = 0
            while True:
                m = re.match(r"\\(quad|qquad|hspace\*?)\b", cell)
                if not m:
                    break
                if m.group(1).startswith("hspace"):
                    _, k = read_group(cell, m.end())
                    cell = cell[k:].strip()
                else:
                    cell = cell[m.end() :].strip()
                indent += 1
            cells.append({"tex": cell, "span": span, "align": align, "indent": indent})
        rows.append([cells, None])
        if rule_before:
            if len(rows) >= 2:
                rows[-2][1] = rule_before
            else:
                rows[-1].append(rule_before)  # top rule marker
        if rule_before == "toprule" and len(rows) == 1:
            rows[0] = [cells, None, "toprule"]
    # normalise: record header boundary = row index whose rule_below is midrule
    return cols, rows


# ==========================================================================
# docx helpers
# ==========================================================================
TCPR_ORDER = [
    "w:cnfStyle", "w:tcW", "w:gridSpan", "w:hMerge", "w:vMerge", "w:tcBorders",
    "w:shd", "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign",
    "w:hideMark", "w:headers", "w:cellIns", "w:cellDel", "w:cellMerge", "w:tcPrChange",
]
TBLPR_ORDER = [
    "w:tblStyle", "w:tblpPr", "w:tblOverlap", "w:bidiVisual", "w:tblStyleRowBandSize",
    "w:tblStyleColBandSize", "w:tblW", "w:jc", "w:tblCellSpacing", "w:tblInd",
    "w:tblBorders", "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
    "w:tblCaption", "w:tblDescription", "w:tblPrChange",
]


def insert_ordered(parent, element, order):
    """Insert `element` into `parent` at the position the schema requires."""
    tag = element.tag
    name = next((n for n in order if qn(n) == tag), None)
    if name is None:
        parent.append(element)
        return element
    idx = order.index(name)
    later = {qn(n) for n in order[idx + 1 :]}
    for child in parent:
        if child.tag in later:
            child.addprevious(element)
            return element
    parent.append(element)
    return element


def set_cell_border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        insert_ordered(tcPr, borders, TCPR_ORDER)
    for edge in ("top", "left", "bottom", "right"):
        if edge in kw:
            tag = "w:%s" % edge
            el = borders.find(qn(tag))
            if el is None:
                el = OxmlElement(tag)
                borders.append(el)
            spec = kw[edge]
            el.set(qn("w:val"), spec.get("val", "single"))
            el.set(qn("w:sz"), str(spec.get("sz", 6)))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), spec.get("color", "000000"))


def clear_table_borders(table):
    tblPr = table._tbl.tblPr
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    insert_ordered(tblPr, borders, TBLPR_ORDER)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    trPr.append(el)


def set_run_size(el, size):
    """Set the font size on every run, including runs inside OMML maths.

    `Paragraph.runs` only reaches direct `w:r` children, so text inside
    `m:oMath` keeps the Normal size unless it is set here as well.
    """
    half = str(int(round(size.pt * 2)))
    for tag in ("w:r", "m:r"):
        for r in el.iter(qn(tag)):
            rPr = r.find(qn("w:rPr"))
            if rPr is None:
                rPr = OxmlElement("w:rPr")
                mrpr = r.find(qn("m:rPr"))
                if mrpr is not None:
                    mrpr.addnext(rPr)
                else:
                    r.insert(0, rPr)
            for t in ("w:sz", "w:szCs"):
                e = rPr.find(qn(t))
                if e is None:
                    e = OxmlElement(t)
                    rPr.append(e)
                e.set(qn("w:val"), half)


def set_paragraph_content(par, children, size=None, bold=False):
    """Replace a paragraph's inline content with copied XML children."""
    for child in list(par._p.iterchildren()):
        if child.tag != qn("w:pPr"):
            par._p.remove(child)
    for child in children:
        par._p.append(copy.deepcopy(child))
    if size is not None or bold:
        for r in par.runs:
            if size is not None:
                r.font.size = size
            if bold:
                r.font.bold = True


def insert_after(ref_element, new_element):
    ref_element.addnext(new_element)
    return new_element


def new_paragraph_after(ref_element, doc, style=None):
    p = OxmlElement("w:p")
    ref_element.addnext(p)
    par = docx.text.paragraph.Paragraph(p, doc._body)
    if style:
        try:
            par.style = doc.styles[style]
        except KeyError:
            pass
    return par


def add_page_number_footer(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    par = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in list(par.runs):
        r._r.getparent().remove(r._r)
    run = par.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)
    run.font.name = BODY_FONT
    run.font.size = Pt(10)


def split_run_at(par, run_idx, offset):
    """Split run `run_idx` of `par` at character `offset`; return new run index."""
    run = par.runs[run_idx]
    text = run.text
    if offset <= 0 or offset >= len(text):
        return run_idx if offset <= 0 else run_idx + 1
    new_r = copy.deepcopy(run._r)
    run.text = text[:offset]
    for t in new_r.iter(qn("w:t")):
        t.text = text[offset:]
        t.set(qn("xml:space"), "preserve")
        break
    run._r.addnext(new_r)
    return run_idx + 1


def color_marker_spans(par):
    """Colour drafting markers ([RES: ...], [Drafting pending: ...])."""
    text = par.text
    if "[RES:" not in text and "[Drafting pending:" not in text and "[cite needed:" not in text:
        return 0
    hits = 0
    for m in list(MARKER_RE.finditer(par.text)):
        start, end = m.start(), m.end()
        colour = BLUE if m.group(0).startswith("[Drafting pending") else RED
        pos = 0
        idx = 0
        while idx < len(par.runs):
            run = par.runs[idx]
            rlen = len(run.text)
            rstart, rend = pos, pos + rlen
            if rend <= start or rstart >= end:
                pos = rend
                idx += 1
                continue
            if rstart < start:
                idx = split_run_at(par, idx, start - rstart)
                pos = start
                continue
            if rend > end:
                split_run_at(par, idx, end - rstart)
                run = par.runs[idx]
            run.font.color.rgb = colour
            pos = rstart + len(run.text)
            idx += 1
        hits += 1
    return hits


# ==========================================================================
# reference document
# ==========================================================================
def build_reference_doc(path: Path):
    """Create a pandoc reference.docx with the manuscript page setup."""
    base = path.parent / "_pandoc_reference_default.docx"
    with open(base, "wb") as fh:
        res = subprocess.run(
            ["pandoc", "--print-default-data-file", "reference.docx"],
            stdout=subprocess.PIPE,
        )
        if res.returncode != 0:
            raise SystemExit("cannot obtain pandoc default reference.docx")
        fh.write(res.stdout)
    d = docx.Document(str(base))
    for section in d.sections:
        section.page_width = PAGE_W
        section.page_height = PAGE_H
        section.left_margin = MARGIN
        section.right_margin = MARGIN
        section.top_margin = MARGIN
        section.bottom_margin = MARGIN
        add_page_number_footer(section)
    normal = d.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = BODY_SIZE
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), BODY_FONT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size, bold in (
        ("Heading 1", Pt(14), True),
        ("Heading 2", Pt(12), True),
        ("Heading 3", Pt(12), True),
        ("Heading 4", Pt(12), True),
    ):
        try:
            st = d.styles[name]
        except KeyError:
            continue
        st.font.name = BODY_FONT
        st.font.size = size
        st.font.bold = bold
        st.font.italic = name == "Heading 3"
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(12)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.line_spacing = 1.15
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for name in ("Image Caption", "Table Caption", "Caption"):
        try:
            st = d.styles[name]
        except KeyError:
            continue
        st.font.name = BODY_FONT
        st.font.size = CAPTION_SIZE
        st.font.italic = False
        st.font.bold = False
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.line_spacing = 1.0
        st.paragraph_format.space_before = Pt(6)
        st.paragraph_format.space_after = Pt(12)
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if "Display Equation" not in [s.name for s in d.styles]:
        eq = d.styles.add_style("Display Equation", WD_STYLE_TYPE.PARAGRAPH)
        eq.base_style = d.styles["Normal"]
        eq.font.name = BODY_FONT
        eq.font.size = BODY_SIZE
        eq.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        eq.paragraph_format.line_spacing = 1.0
        eq.paragraph_format.space_before = Pt(6)
        eq.paragraph_format.space_after = Pt(6)
    d.save(str(path))
    base.unlink(missing_ok=True)


# ==========================================================================
# document assembly
# ==========================================================================
class Builder:
    def __init__(self, tex: Path, out: Path, figdir: Path):
        self.tex = tex
        self.out = out
        self.figdir = figdir
        self.paper_dir = tex.parent
        self.work = out.parent / "_docx_work"
        self.blocks = []          # extracted floats, in document order
        self.eq_numbers = []      # display-equation numbers, in order
        self.missing_refs = []
        self.missing_cites = []
        self.report = {}

    # ------------------------------------------------------------------
    def build(self):
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.parent.mkdir(parents=True, exist_ok=True)

        aux = self.paper_dir / (self.tex.stem + ".aux")
        bbl = self.paper_dir / (self.tex.stem + ".bbl")
        if not aux.exists():
            raise SystemExit("missing %s -- compile the PDF first" % aux)
        self.labels, self.bibcites = parse_aux(aux)
        self.refs = parse_bbl(bbl) if bbl.exists() else []

        raw = load_tex(self.tex)
        raw = strip_comments(raw)
        # pandoc's math reader rejects non-letter macro names such as \1
        raw = re.sub(r"\\1(?![0-9A-Za-z])", r"\\mathbf{1}", raw)
        raw = raw.replace("\\operatorname*{med}", "\\operatorname{med}")
        self.macros = self.extract_macros(raw)
        body = raw[raw.index("\\begin{document}") + len("\\begin{document}") :]
        body = body[: body.rindex("\\end{document}")]

        body = self.resolve_drafting_macros(body)
        body = resolve_citations(body, self.bibcites, self.missing_cites)
        body = resolve_refs(body, self.labels, self.missing_refs)
        body = self.extract_floats(body)
        body = self.transform_equations(body)
        body = self.transform_theorems(body)
        body = self.transform_frontmatter(body)
        body = self.number_sections(body)
        body = self.cleanup(body)

        self.refdoc = self.work / "reference.docx"
        build_reference_doc(self.refdoc)

        pandoc_tex = self.work / "body.tex"
        pandoc_tex.write_text(self.wrap_document(body), encoding="utf-8", newline="")
        staged = self.work / "manuscript_raw.docx"
        run_pandoc(pandoc_tex, staged, self.refdoc, self.work)

        self.render_fragments()
        self.postprocess(staged)
        return self.report

    # ------------------------------------------------------------------
    def extract_macros(self, raw: str) -> str:
        pre = raw[: raw.index("\\begin{document}")]
        keep = []
        for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}", pre):
            i = m.start()
            name, j = read_group(pre, m.start() + len("\\newcommand"))
            opt, j = read_optional(pre, j)
            if pre[j] != "{":
                continue
            defn, j = read_group(pre, j)
            if m.group(1) in ("RES", "CITENEEDED", "PENDING"):
                continue
            if opt:
                keep.append("\\newcommand{\\%s}[%s]{%s}" % (m.group(1), opt, defn))
            else:
                keep.append("\\newcommand{\\%s}{%s}" % (m.group(1), defn))
        return "\n".join(keep)

    # ------------------------------------------------------------------
    def resolve_drafting_macros(self, body: str) -> str:
        """\\RES / \\PENDING / \\CITENEEDED -> literal bracketed markers."""
        for cmd, tpl in (
            ("RES", "[RES: %s]"),
            ("PENDING", "[Drafting pending: %s]"),
            ("CITENEEDED", "[cite needed: %s]"),
        ):
            out, i = [], 0
            tag = "\\" + cmd
            while True:
                k = body.find(tag, i)
                if k < 0 or (k + len(tag) < len(body) and body[k + len(tag)].isalpha()):
                    if k < 0:
                        out.append(body[i:])
                        break
                    out.append(body[i : k + len(tag)])
                    i = k + len(tag)
                    continue
                out.append(body[i:k])
                arg, j = read_group(body, k + len(tag))
                arg = arg.replace("\\_", "_").replace("\\&", "&").replace("\\%", "%")
                out.append(tpl % arg.strip())
                i = j
            body = "".join(out)
        return body

    # ------------------------------------------------------------------
    def _placeholder(self, kind: str, data: dict) -> str:
        data["kind"] = kind
        self.blocks.append(data)
        return "\n\nPDXBLOCK%04d\n\n" % (len(self.blocks) - 1)

    def extract_floats(self, body: str) -> str:
        for env in ("figure*", "figure", "table*", "table", "algorithm"):
            while True:
                span = find_env(body, env)
                if not span:
                    break
                b0, s0, s1, b1 = span
                inner = body[s0:s1]
                _, after = read_optional(body, s0)
                inner = body[after:s1]
                if env.startswith("figure"):
                    data = self.parse_figure(inner)
                    kind = "figure"
                elif env.startswith("table"):
                    data = self.parse_table(inner)
                    kind = "table"
                else:
                    data = self.parse_algorithm(inner)
                    kind = "algorithm"
                body = body[:b0] + self._placeholder(kind, data) + body[b1:]
        return body

    def _caption_and_label(self, inner: str):
        cap, label = "", None
        m = re.search(r"\\caption\s*\*?\s*\{", inner)
        if m:
            cap, j = read_group(inner, m.end() - 1)
            rest = inner[:m.start()] + inner[j:]
        else:
            rest = inner
        lm = re.search(r"\\label\{([^}]*)\}", inner)
        if lm:
            label = lm.group(1)
        cap = re.sub(r"\\label\{[^}]*\}", "", cap).strip()
        rest = re.sub(r"\\label\{[^}]*\}", "", rest)
        return cap, label, rest

    def parse_figure(self, inner: str):
        cap, label, rest = self._caption_and_label(inner)
        m = re.search(r"\\includegraphics", rest)
        path, frac = None, 1.0
        if m:
            opt, j = read_optional(rest, m.end())
            if rest[j] == "{":
                path, _ = read_group(rest, j)
            if opt:
                wm = re.search(r"width\s*=\s*([0-9.]*)\s*\\(textwidth|columnwidth|linewidth)", opt)
                pm = re.search(r"width\s*=\s*([0-9.]+)\s*pt", opt)
                if wm:
                    frac = float(wm.group(1)) if wm.group(1) else 1.0
                elif pm:
                    frac = float(pm.group(1)) / LATEX_TEXTWIDTH_PT
        return {"caption": cap, "label": label, "path": path, "frac": min(frac, 1.0)}

    def parse_table(self, inner: str):
        cap, label, rest = self._caption_and_label(inner)
        span = find_env(rest, "tabular")
        tab = None
        if span:
            tab = rest[span[1] : span[2]]
            note = rest[span[3] :]
        else:
            note = ""
        note = re.sub(r"\\par\b|\\smallskip\b|\\raggedright\b|\\footnotesize\b|\\small\b|\\centering\b", " ", note)
        note = note.strip().strip("{}").strip()
        return {"caption": cap, "label": label, "tabular": tab, "note": note}

    def parse_algorithm(self, inner: str):
        cap, label, rest = self._caption_and_label(inner)
        return {"caption": cap, "label": label, "body": rest}

    # ------------------------------------------------------------------
    def transform_equations(self, body: str) -> str:
        out = []
        i = 0
        pat = re.compile(r"\\begin\{(equation|align)\*?\}")
        while True:
            m = pat.search(body, i)
            if not m:
                out.append(body[i:])
                break
            out.append(body[i : m.start()])
            env = m.group(1)
            span = find_env(body, env, m.start())
            inner = body[span[1] : span[2]]
            i = span[3]
            if env == "equation":
                labs = re.findall(r"\\label\{([^}]*)\}", inner)
                num = self.labels.get(labs[0]) if labs else None
                clean = re.sub(r"\\label\{[^}]*\}", "", inner).strip()
                self.eq_numbers.append(num or "")
                out.append("\n\n$$\n%s\n$$\n\n" % clean)
            else:
                lines = split_top_level(inner, "\\\\")
                labelled = [re.findall(r"\\label\{([^}]*)\}", ln) for ln in lines]
                n_lab = sum(1 for l in labelled if l)
                if n_lab > 1 and not re.search(r"\\nonumber", inner):
                    for ln, labs in zip(lines, labelled):
                        if not ln.strip():
                            continue
                        clean = re.sub(r"\\label\{[^}]*\}", "", ln)
                        clean = clean.replace("&", "").strip().rstrip(",").strip()
                        if clean.endswith("."):
                            clean = clean[:-1]
                        num = self.labels.get(labs[0]) if labs else None
                        self.eq_numbers.append(num or "")
                        out.append("\n\n$$\n%s\n$$\n\n" % clean)
                else:
                    labs = [l for ll in labelled for l in ll]
                    num = self.labels.get(labs[0]) if labs else None
                    clean = re.sub(r"\\label\{[^}]*\}", "", inner)
                    clean = clean.replace("\\nonumber", "").strip()
                    self.eq_numbers.append(num or "")
                    out.append("\n\n$$\n\\begin{aligned}\n%s\n\\end{aligned}\n$$\n\n" % clean)
        return "".join(out)

    # ------------------------------------------------------------------
    THEOREMS = {
        "assumption": "Assumption",
        "proposition": "Proposition",
        "lemma": "Lemma",
        "remark": "Remark",
    }

    def transform_theorems(self, body: str) -> str:
        counters = {k: 0 for k in self.THEOREMS}
        for env, word in self.THEOREMS.items():
            while True:
                span = find_env(body, env)
                if not span:
                    break
                b0, s0, s1, b1 = span
                name, j = read_optional(body, s0)
                inner = body[j:s1]
                labs = re.findall(r"\\label\{([^}]*)\}", inner)
                counters[env] += 1
                num = self.labels.get(labs[0]) if labs else str(counters[env])
                inner = re.sub(r"\\label\{[^}]*\}", "", inner).strip()
                head = "%s %s" % (word, num or counters[env])
                if name:
                    head += " (%s)" % name
                # no \emph wrapper: theorem bodies may contain display maths and
                # several paragraphs, which cannot sit inside an inline command
                rep = "\n\n\\textbf{%s.} %s\n\n" % (head, inner)
                body = body[:b0] + rep + body[b1:]
        while True:
            span = find_env(body, "pf")
            if not span:
                break
            b0, s0, s1, b1 = span
            name, j = read_optional(body, s0)
            inner = body[j:s1].strip()
            head = "Proof" + (" (%s)" % name if name else "")
            body = body[:b0] + "\n\n\\textbf{%s.} %s\n\n" % (head, inner) + body[b1:]
        return body

    # ------------------------------------------------------------------
    def transform_frontmatter(self, body: str) -> str:
        span = find_env(body, "frontmatter")
        if not span:
            return body
        b0, s0, s1, b1 = span
        fm = body[s0:s1]
        parts = []
        m = re.search(r"\\title\{", fm)
        if m:
            title, _ = read_group(fm, m.end() - 1)
            parts.append("\\title{%s}" % title.strip())
        authors = []
        for am in re.finditer(r"\\author(\[[^\]]*\])?\{", fm):
            a, _ = read_group(fm, am.end() - 1)
            authors.append(a.strip())
        for am in re.finditer(r"\\affiliation(\[[^\]]*\])?\{", fm):
            a, _ = read_group(fm, am.end() - 1)
            authors.append(a.strip())
        if authors:
            parts.append("\\author{%s}" % " \\\\ ".join(authors))
        parts.append("\\maketitle")
        nn = re.search(r"\\nonumnote\{", fm)
        if nn:
            note, _ = read_group(fm, nn.end() - 1)
            note = re.sub(r"^\s*Abbreviations\.\s*", "", note.strip())
            parts.append("\n\\section*{Abbreviations}\n\n%s\n" % note)
        ab = find_env(fm, "abstract")
        if ab:
            parts.append("\n\\section*{Abstract}\n\n%s\n" % fm[ab[1] : ab[2]].strip())
        hl = find_env(fm, "highlights")
        if hl:
            items = fm[hl[1] : hl[2]].strip()
            items = re.sub(r"\\item\b", "\\\\item", items)
            parts.append(
                "\n\\section*{Highlights}\n\n\\begin{itemize}\n%s\n\\end{itemize}\n" % items
            )
        kw = find_env(fm, "keyword")
        if kw:
            words = [w.strip() for w in fm[kw[1] : kw[2]].split("\\sep")]
            parts.append("\n\\section*{Keywords}\n\n%s\n" % "; ".join(w for w in words if w))
        return body[:b0] + "\n".join(parts) + body[b1:]

    # ------------------------------------------------------------------
    def number_sections(self, body: str) -> str:
        out, i = [], 0
        pat = re.compile(r"\\(section|subsection|subsubsection)(\*?)\{")
        while True:
            m = pat.search(body, i)
            if not m:
                out.append(body[i:])
                break
            out.append(body[i : m.start()])
            title, j = read_group(body, m.end() - 1)
            lm = re.match(r"\s*\\label\{([^}]*)\}", body[j:])
            label = None
            if lm:
                label = lm.group(1)
                j += lm.end()
            prefix = ""
            if not m.group(2) and label and label in self.labels:
                num = self.labels[label]
                prefix = "%s. " % num
            out.append("\\%s%s{%s%s}" % (m.group(1), m.group(2), prefix, title.strip()))
            i = j
        return "".join(out)

    # ------------------------------------------------------------------
    def cleanup(self, body: str) -> str:
        # a drafting marker straight after \item would be read as the optional
        # description label and the item would come out empty
        body = re.sub(
            r"(\\item)(\s*)\[(RES:|Drafting pending:|cite needed:)",
            r"\1\2{}[\3",
            body,
        )
        body = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", body)
        body = re.sub(r"\\renewcommand\{[^}]*\}\{[^}]*\}", "", body)
        body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
        for cmd in ("FloatBarrier", "linenumbers", "appendix", "clearpage", "newpage",
                    "centering", "small", "footnotesize", "normalsize", "par"):
            body = re.sub(r"\\%s\b" % cmd, "", body)
        body = re.sub(r"\n\{\s*\n", "\n\n", body)
        body = re.sub(r"\n\}\s*\n", "\n\n", body)
        body = re.sub(r"\n{3,}", "\n\n", body)
        return body

    def wrap_document(self, body: str) -> str:
        return "\n".join(
            [
                "\\documentclass{article}",
                "\\usepackage{amsmath,amssymb}",
                "\\usepackage[T1]{fontenc}",
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage{graphicx}",
                self.macros,
                "\\begin{document}",
                body,
                "\n\\section*{References}\n",
                "PDXREFERENCES",
                "\\end{document}",
            ]
        )

    # ==================================================================
    # fragment rendering for tables, algorithms, captions, references
    # ==================================================================
    def render_fragments(self):
        fr = FragmentRenderer(self.macros, self.work, self.refdoc)
        self.fr = fr
        for blk in self.blocks:
            blk["_cap_id"] = fr.add(blk["caption"])
            if blk["kind"] == "table":
                if blk["note"]:
                    blk["_note_id"] = fr.add(blk["note"])
                cols, rows = parse_tabular(blk["tabular"])
                blk["_cols"], blk["_rows"] = cols, rows
                for row in rows:
                    if row[0] is None:
                        continue
                    for cell in row[0]:
                        cell["_ids"] = [fr.add(x) for x in self.cell_lines(cell["tex"])]
            elif blk["kind"] == "algorithm":
                blk["_lines"] = parse_algorithm_body(blk["body"])
                for ln in blk["_lines"]:
                    ln["_id"] = fr.add(ln["tex"])
        self.ref_ids = [fr.add(tex) for _, tex in self.refs]
        fr.render()

    @staticmethod
    def cell_lines(tex: str):
        """Split a cell that contains a stacked (nested tabular) group."""
        span = find_env(tex, "tabular")
        if not span:
            return [tex]
        inner = tex[span[1] : span[2]].strip()
        _, k = read_optional(inner, 0)
        if k < len(inner) and inner[k] == "{":
            _, k = read_group(inner, k)
        return [p.strip() for p in split_top_level(inner[k:], "\\\\") if p.strip()]

    # ==================================================================
    def postprocess(self, staged: Path):
        doc = docx.Document(str(staged))
        body = doc.element.body

        n_eq = self.number_equations(doc)
        n_blocks = self.expand_blocks(doc)
        n_refs = self.insert_references(doc)
        n_mark = 0
        for par in doc.paragraphs:
            n_mark += color_marker_spans(par)
        self.style_body(doc)
        self.drop_empty_paragraphs(doc)

        for section in doc.sections:
            section.page_width = PAGE_W
            section.page_height = PAGE_H
            section.left_margin = MARGIN
            section.right_margin = MARGIN
            section.top_margin = MARGIN
            section.bottom_margin = MARGIN
            add_page_number_footer(section)

        doc.save(str(self.out))
        strip_metadata(self.out)
        self.report.update(
            {
                "equations": n_eq,
                "blocks": n_blocks,
                "references": n_refs,
                "markers": n_mark,
                "missing_refs": sorted(set(self.missing_refs)),
                "missing_cites": sorted(set(self.missing_cites)),
            }
        )

    # ------------------------------------------------------------------
    def number_equations(self, doc):
        """Append the PDF equation number to every display-math paragraph."""
        idx = 0
        for par in doc.paragraphs:
            if not par._p.findall(".//" + qn("m:oMathPara")):
                continue
            num = self.eq_numbers[idx] if idx < len(self.eq_numbers) else ""
            idx += 1
            try:
                par.style = doc.styles["Display Equation"]
            except KeyError:
                pass
            if not num:
                continue
            pf = par.paragraph_format
            pf.tab_stops.clear_all()
            pf.tab_stops.add_tab_stop(TEXT_W, WD_TAB_ALIGNMENT.RIGHT)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            # move the OMML paragraph content out of oMathPara wrapper so the
            # number can sit on the same line
            for mp in par._p.findall(qn("m:oMathPara")):
                for child in list(mp):
                    if child.tag == qn("m:oMathParaPr"):
                        continue
                    mp.addprevious(child)
                mp.getparent().remove(mp)
            run = par.add_run("\t(%s)" % num)
            run.font.name = BODY_FONT
            run.font.size = BODY_SIZE
        return idx

    # ------------------------------------------------------------------
    def expand_blocks(self, doc):
        count = 0
        for par in list(doc.paragraphs):
            m = PLACEHOLDER_RE.match(par.text.strip())
            if not m:
                continue
            blk = self.blocks[int(m.group(1))]
            anchor = par._p
            if blk["kind"] == "figure":
                self.emit_figure(doc, anchor, blk)
            elif blk["kind"] == "table":
                self.emit_table(doc, anchor, blk)
            else:
                self.emit_algorithm(doc, anchor, blk)
            anchor.getparent().remove(anchor)
            count += 1
        return count

    def caption_label(self, blk):
        lab = blk.get("label") or ""
        num = self.labels.get(lab, "?")
        word = {"figure": "Fig.", "table": "Table", "algorithm": "Algorithm"}[blk["kind"]]
        return "%s %s. " % (word, num)

    def emit_figure(self, doc, anchor, blk):
        src = blk["path"] or ""
        name = Path(src).name
        png = self.figdir / (Path(name).stem + ".png")
        pic_par = new_paragraph_after(anchor, doc)
        pic_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pic_par.paragraph_format.space_before = Pt(12)
        pic_par.paragraph_format.space_after = Pt(2)
        if png.exists():
            run = pic_par.add_run()
            run.add_picture(str(png), width=Emu(int(TEXT_W * blk["frac"])))
            blk["_image"] = str(png)
        else:
            r = pic_par.add_run("[missing figure file: %s]" % png)
            r.font.color.rgb = RED
            blk["_image"] = None
        cap_par = new_paragraph_after(pic_par._p, doc, style="Image Caption")
        self.fill_caption(cap_par, blk)

    def fill_caption(self, par, blk):
        run = par.add_run(self.caption_label(blk))
        run.font.bold = True
        run.font.size = CAPTION_SIZE
        run.font.name = BODY_FONT
        for child in self.fr.runs(blk["_cap_id"]):
            par._p.append(child)
        for r in par.runs:
            if r.font.name is None:
                r.font.name = BODY_FONT
        set_run_size(par._p, CAPTION_SIZE)
        par.paragraph_format.line_spacing = 1.0
        par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def emit_table(self, doc, anchor, blk):
        cap_par = new_paragraph_after(anchor, doc, style="Table Caption")
        self.fill_caption(cap_par, blk)
        cap_par.paragraph_format.space_before = Pt(12)
        cap_par.paragraph_format.space_after = Pt(4)
        cols, rows = blk["_cols"], blk["_rows"]
        data_rows = [r for r in rows if r[0] is not None]
        ncol = max((sum(c["span"] for c in r[0]) for r in data_rows), default=1)
        ncol = max(ncol, len(cols))
        table = doc.add_table(rows=len(data_rows), cols=ncol)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        clear_table_borders(table)
        cap_par._p.addnext(table._tbl)

        # column widths: explicit only when the LaTeX colspec gives them,
        # otherwise let Word size the columns to their content
        fracs = [c.get("frac") for c in cols] + [None] * (ncol - len(cols))
        known = [f for f in fracs if f]
        if known and len(known) == ncol:
            total = sum(known)
            widths = [Emu(int(TEXT_W * f / total)) for f in fracs]
        else:
            widths = None
            table.autofit = True
            tblPr = table._tbl.tblPr
            for tag in ("w:tblW", "w:tblLayout"):
                old = tblPr.find(qn(tag))
                if old is not None:
                    tblPr.remove(old)
            tblW = OxmlElement("w:tblW")
            tblW.set(qn("w:w"), "5000")
            tblW.set(qn("w:type"), "pct")
            insert_ordered(tblPr, tblW, TBLPR_ORDER)
            layout = OxmlElement("w:tblLayout")
            layout.set(qn("w:type"), "autofit")
            insert_ordered(tblPr, layout, TBLPR_ORDER)

        header_rows = 0
        seen_mid = False
        for ridx, row in enumerate(data_rows):
            if not seen_mid:
                header_rows = ridx + 1
            if row[1] == "midrule":
                seen_mid = True
        if not seen_mid:
            header_rows = 1

        for ridx, row in enumerate(data_rows):
            cells, rule = row[0], row[1]
            col = 0
            for cell in cells:
                if col >= ncol:
                    break
                tc = table.cell(ridx, col)
                if cell["span"] > 1:
                    last = min(col + cell["span"] - 1, ncol - 1)
                    tc = tc.merge(table.cell(ridx, last))
                align = cell["align"] or (cols[col]["align"] if col < len(cols) else "l")
                self.fill_cell(tc, cell, align, ridx < header_rows)
                col += cell["span"]
            for c in range(ncol):
                spec = {}
                if ridx == 0:
                    spec["top"] = {"val": "single", "sz": 12}
                if rule in ("midrule", "hline", "cmidrule"):
                    spec["bottom"] = {"val": "single", "sz": 6}
                if ridx == len(data_rows) - 1:
                    spec["bottom"] = {"val": "single", "sz": 12}
                if spec:
                    set_cell_border(table.cell(ridx, c), **spec)
            if ridx < header_rows:
                repeat_header(table.rows[ridx])
        if widths is not None:
            for ridx in range(len(data_rows)):
                for c in range(ncol):
                    table.cell(ridx, c).width = widths[c]
        # table note
        if blk.get("_note_id") is not None:
            note_par = new_paragraph_after(table._tbl, doc)
            for child in self.fr.runs(blk["_note_id"]):
                note_par._p.append(child)
            for r in note_par.runs:
                if r.font.name is None:
                    r.font.name = BODY_FONT
            set_run_size(note_par._p, SMALL_SIZE)
            note_par.paragraph_format.line_spacing = 1.0
            note_par.paragraph_format.space_after = Pt(12)
            note_par.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def fill_cell(self, tc, cell, align, header):
        tc.paragraphs[0]._p.getparent().remove(tc.paragraphs[0]._p)
        for n, fid in enumerate(cell["_ids"]):
            p = tc.add_paragraph()
            for child in self.fr.runs(fid):
                p._p.append(child)
            for r in p.runs:
                if r.font.name is None:
                    r.font.name = BODY_FONT
                if header:
                    r.font.bold = True
            set_run_size(p._p, SMALL_SIZE)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.left_indent = Cm(0.3 * cell["indent"])
            p.alignment = {
                "l": WD_ALIGN_PARAGRAPH.LEFT,
                "c": WD_ALIGN_PARAGRAPH.CENTER,
                "r": WD_ALIGN_PARAGRAPH.RIGHT,
            }[align]
        if not cell["_ids"]:
            tc.add_paragraph()

    def emit_algorithm(self, doc, anchor, blk):
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        anchor.addnext(table._tbl)
        tc = table.cell(0, 0)
        tc.width = TEXT_W
        set_cell_border(
            tc,
            top={"val": "single", "sz": 12},
            bottom={"val": "single", "sz": 12},
            left={"val": "single", "sz": 6},
            right={"val": "single", "sz": 6},
        )
        tc.paragraphs[0]._p.getparent().remove(tc.paragraphs[0]._p)
        cap_par = tc.add_paragraph()
        run = cap_par.add_run(self.caption_label(blk))
        run.font.bold = True
        run.font.size = CAPTION_SIZE
        run.font.name = BODY_FONT
        for child in self.fr.runs(blk["_cap_id"]):
            cap_par._p.append(child)
        for r in cap_par.runs:
            if r.font.name is None:
                r.font.name = BODY_FONT
        set_run_size(cap_par._p, CAPTION_SIZE)
        cap_par.paragraph_format.line_spacing = 1.0
        cap_par.paragraph_format.space_after = Pt(4)
        set_cell_border(tc, bottom={"val": "single", "sz": 12})
        lineno = 0
        for ln in blk["_lines"]:
            p = tc.add_paragraph()
            if ln["numbered"]:
                lineno += 1
                pre = p.add_run("%d: " % lineno)
                pre.font.size = SMALL_SIZE
                pre.font.name = BODY_FONT
            elif ln["kind"] in ("in", "out"):
                pre = p.add_run(ln["label"])
                pre.font.bold = True
                pre.font.size = SMALL_SIZE
                pre.font.name = BODY_FONT
            for child in self.fr.runs(ln["_id"]):
                p._p.append(child)
            for r in p.runs:
                if r.font.name is None:
                    r.font.name = BODY_FONT
            set_run_size(p._p, SMALL_SIZE)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.left_indent = Cm(0.5 + 0.4 * ln["depth"])
            p.paragraph_format.first_line_indent = Cm(-0.5)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        blk["_nlines"] = lineno

    # ------------------------------------------------------------------
    def insert_references(self, doc):
        target = None
        for par in doc.paragraphs:
            if par.text.strip() == "PDXREFERENCES":
                target = par
                break
        if target is None:
            return 0
        anchor = target._p
        for fid in self.ref_ids:
            p = new_paragraph_after(anchor, doc)
            for child in self.fr.runs(fid):
                p._p.append(child)
            for r in p.runs:
                if r.font.name is None:
                    r.font.name = BODY_FONT
            set_run_size(p._p, Pt(11))
            pf = p.paragraph_format
            pf.left_indent = Cm(0.75)
            pf.first_line_indent = Cm(-0.75)
            pf.line_spacing = 1.0
            pf.space_after = Pt(4)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            anchor = p._p
        target._p.getparent().remove(target._p)
        return len(self.ref_ids)

    # ------------------------------------------------------------------
    def style_body(self, doc):
        for par in doc.paragraphs:
            for r in par.runs:
                if r.font.name is None:
                    r.font.name = BODY_FONT

    def drop_empty_paragraphs(self, doc):
        removed = 0
        body = doc.element.body
        for p in list(body.findall(qn("w:p"))):
            par = docx.text.paragraph.Paragraph(p, doc._body)
            if par.text.strip():
                continue
            if list(p.iter(qn("m:oMath"))) or list(p.iter(qn("w:drawing"))):
                continue
            if list(p.iter(qn("w:fldChar"))):
                continue
            body.remove(p)
            removed += 1
        return removed


# ==========================================================================
# algorithm2e body parser
# ==========================================================================
def parse_algorithm_body(body: str):
    """Flatten an algorithm2e body into numbered / labelled lines."""
    lines = []

    def emit(tex, depth, numbered=True, kind="stmt", label=""):
        tex = tex.strip()
        if not tex and kind == "stmt":
            return
        lines.append(
            {"tex": tex, "depth": depth, "numbered": numbered, "kind": kind, "label": label}
        )

    def walk(src, depth):
        i = 0
        buf = []

        def flush():
            txt = "".join(buf).strip()
            buf.clear()
            if txt:
                emit(txt, depth)

        while i < len(src):
            m = re.compile(
                r"\\(KwIn|KwOut|KwData|KwResult|ForEach|ForAll|For|While|If|ElseIf|Else|"
                r"lIf|lElse|lForEach|Return|KwRet|tcp|tcc|Repeat|Until)\b"
            ).match(src, i)
            if not m:
                if src[i] == "\\" and src[i : i + 2] == "\\;":
                    flush()
                    i += 2
                    continue
                buf.append(src[i])
                i += 1
                continue
            cmd = m.group(1)
            j = m.end()
            _, j = read_optional(src, j)
            args = []
            while j < len(src) and src[j] == "{":
                a, j = read_group(src, j)
                args.append(a)
                if cmd in ("KwIn", "KwOut", "KwData", "KwResult", "Return", "KwRet",
                           "tcp", "tcc", "Else", "lElse") and len(args) >= 1:
                    break
                if cmd in ("ForEach", "ForAll", "For", "While", "If", "ElseIf",
                           "lIf", "lForEach", "Repeat", "Until") and len(args) >= 2:
                    break
            flush()
            if cmd in ("KwIn", "KwData"):
                emit(args[0] if args else "", depth, numbered=False, kind="in", label="Input: ")
            elif cmd in ("KwOut", "KwResult"):
                emit(args[0] if args else "", depth, numbered=False, kind="out", label="Output: ")
            elif cmd in ("Return", "KwRet"):
                emit("return " + (args[0] if args else ""), depth)
            elif cmd in ("tcp", "tcc"):
                emit("// " + (args[0] if args else ""), depth, numbered=False, kind="cmt")
            elif cmd in ("lIf",):
                emit("if %s then %s" % (args[0], args[1] if len(args) > 1 else ""), depth)
            elif cmd in ("lForEach",):
                emit("foreach %s do %s" % (args[0], args[1] if len(args) > 1 else ""), depth)
            elif cmd in ("lElse",):
                emit("else %s" % (args[0] if args else ""), depth)
            elif cmd in ("ForEach", "ForAll"):
                emit("foreach %s do" % args[0], depth)
                walk(args[1] if len(args) > 1 else "", depth + 1)
            elif cmd == "For":
                emit("for %s do" % args[0], depth)
                walk(args[1] if len(args) > 1 else "", depth + 1)
            elif cmd == "While":
                emit("while %s do" % args[0], depth)
                walk(args[1] if len(args) > 1 else "", depth + 1)
            elif cmd == "Repeat":
                emit("repeat", depth)
                walk(args[1] if len(args) > 1 else "", depth + 1)
                emit("until %s" % args[0], depth)
            elif cmd == "Until":
                emit("until %s" % args[0], depth)
            elif cmd in ("If", "ElseIf"):
                word = "if" if cmd == "If" else "else if"
                emit("%s %s then" % (word, args[0]), depth)
                walk(args[1] if len(args) > 1 else "", depth + 1)
            elif cmd == "Else":
                emit("else", depth)
                walk(args[0] if args else "", depth + 1)
            i = j
        flush()

    walk(body, 0)
    return lines


# ==========================================================================
# metadata
# ==========================================================================
CORE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" \
xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" \
xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\
<dc:title></dc:title><dc:subject></dc:subject><dc:creator></dc:creator><cp:keywords></cp:keywords>\
<dc:description></dc:description><cp:lastModifiedBy></cp:lastModifiedBy><cp:revision>1</cp:revision>\
<dcterms:created xsi:type="dcterms:W3CDTF">{ts}</dcterms:created>\
<dcterms:modified xsi:type="dcterms:W3CDTF">{ts}</dcterms:modified><cp:category></cp:category>\
</cp:coreProperties>"""

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" \
xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">\
<Application></Application><Company></Company><Manager></Manager></Properties>"""


def strip_metadata(path: Path, timestamp="2026-01-01T00:00:00Z"):
    """Remove author, company and last-modified-by fields; fix timestamps."""
    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        names = set(zin.namelist())
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = CORE_XML.format(ts=timestamp).encode("utf-8")
            elif item.filename == "docProps/app.xml":
                data = APP_XML.encode("utf-8")
            info = zipfile.ZipInfo(item.filename, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = item.external_attr
            zout.writestr(info, data)
        if "docProps/core.xml" not in names:
            zout.writestr("docProps/core.xml", CORE_XML.format(ts=timestamp))
    shutil.move(str(tmp), str(path))


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--tex", default=str(root / "paper" / "main.tex"))
    ap.add_argument("--out", default=str(root / "build" / "manuscript.docx"))
    ap.add_argument("--figdir", default=str(root / "figures"))
    args = ap.parse_args(argv)

    tex = Path(args.tex).resolve()
    out = Path(args.out).resolve()
    figdir = Path(args.figdir).resolve()
    print("make_docx: %s -> %s" % (tex, out))
    b = Builder(tex, out, figdir)
    rep = b.build()
    print("  display equations numbered : %d" % rep["equations"])
    print("  floats rebuilt             : %d" % rep["blocks"])
    print("  references written         : %d" % rep["references"])
    print("  drafting markers coloured  : %d" % rep["markers"])
    if rep["missing_refs"]:
        print("  UNRESOLVED \\ref            : %s" % ", ".join(rep["missing_refs"]))
    if rep["missing_cites"]:
        print("  UNRESOLVED \\cite           : %s" % ", ".join(rep["missing_cites"]))
    print("  wrote %s (%.1f kB)" % (out, out.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
