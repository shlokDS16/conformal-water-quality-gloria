"""Prose checks for LaTeX manuscripts (CLAUDE.md section 7).

Checks, per \\section / \\subsection block of the given .tex file:
  1. sentences longer than 45 words (prose only; math, tables, algorithms and comments removed);
  2. em dashes, en dashes and LaTeX dash ligatures (-- or ---) in prose;
  3. abbreviations (tokens with two or more capital letters, plus a fixed list of mixed-case
     water-colour abbreviations) whose first use in the document precedes their definition
     "(ABBR" or that are never defined. Heuristic: review every hit by eye.
Also reports words per block (blocks containing only \\PENDING{...} are skipped).

Usage: python src/check_text.py paper/main.tex [--max-words 45] [--all]
Exit code 0 when no sentence is too long and no dash is found; abbreviation hits do not change it.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_ENVS = ["equation", "equation*", "align", "align*", "table", "tabular", "algorithm", "figure",
             "keyword", "thebibliography"]
ALLOW = {"GLORIA", "LightGBM", "NumPy", "SciPy", "PyTorch", "NVIDIA", "RTX", "GB", "SHA256", "ACIX-Aqua",
         "MDN-STREAM", "Sentinel-2A", "Sentinel-3A", "Sentinel-2", "Sentinel-3", "Landsat-8", "ISPRS",
         "CRediT", "PANGAEA", "MultiSpectral", "AI", "WM", "II", "III", "IV",
         "GNU", "GitHub", "Sentinel-3B", "Sentinel-2C", "SentiWiki", "Oa1", "Oa2", "Oa10",
         "Oa11", "Oa12", "B1", "B4", "B6"}  # proper nouns that are not abbreviations of a defined term
MIXED = ["Chl-a", "aCDOM", "aLH", "MdSA", "Rrs", "CV+"]
ABBREV_RE = re.compile(r"(?<![\w\\])([A-Za-z][A-Za-z0-9]*(?:[+\-][A-Za-z0-9]+)*\+?)")
NO_SPLIT = ["et al.", "e.g.", "i.e.", "Eq.", "Eqs.", "Fig.", "Figs.", "Prop.", "Sect.", "vs.", "cf.", "approx."]


def strip_comments(t: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", t)


def remove_env(t: str, env: str) -> str:
    e = re.escape(env)
    return re.sub(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", " ", t, flags=re.S)


DISPLAY_ENVS = ["equation", "equation*", "align", "align*"]


def delatex(t: str) -> str:
    for env in DISPLAY_ENVS:  # a display equation ends the clause that introduces it
        e = re.escape(env)
        t = re.sub(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", " MATH. ", t, flags=re.S)
    for env in SKIP_ENVS:
        t = remove_env(t, env)
    t = re.sub(r"\\\[.*?\\\]", " MATH ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " MATH ", t)
    t = re.sub(r"\\PENDING\{(?:[^{}]|\{[^{}]*\})*\}", " ", t)
    # REVIEW_final F7: an \input{...} left its literal path in the prose stream, so the sentence
    # before it was joined to the paragraph after it (the splitter needs [A-Z(\[] after the stop)
    # and was reported as one long sentence. Layout edit L6 moved such an \input into the middle of
    # a subsection, which is when the join started to matter. File names are not prose; drop them.
    t = re.sub(r"\\(?:input|include|includegraphics)(\[[^\]]*\])?\{[^{}]*\}", " ", t)
    # title-page metadata is not prose: it carries no sentences and no abbreviation definitions
    for macro in ("title", "author", "ead", "cortext", "affiliation", "journal"):
        t = re.sub(r"\\" + macro + r"(\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}", " ", t)
    t = re.sub(r"\\url\{[^{}]*\}", " URL ", t)            # URLs are not prose: no abbreviation or dash checks
    t = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r" \1 ", t)  # keep the visible link text only
    t = re.sub(r"\\CITENEEDED\{[^{}]*\}", " CITATION ", t)
    t = re.sub(r"\\RES\{[^{}]*\}", " RESULT ", t)
    t = re.sub(r"\\cite[a-z]*\*?(\[[^\]]*\])*\{[^{}]*\}", " CITATION ", t)
    t = re.sub(r"\\(eq)?ref\{[^{}]*\}", "1", t)
    t = re.sub(r"\\label\{[^{}]*\}", " ", t)
    t = re.sub(r"\\(begin|end)\{[^{}]*\}(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\(emph|textit|textbf|texttt|mathrm|text)\{([^{}]*)\}", r"\2", t)
    t = re.sub(r"\\(item)\b", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = t.replace("~", " ").replace("{", "").replace("}", "").replace("\\_", "_").replace("\\%", "%")
    return t


def blocks(tex: str):
    """Split on \\section and \\subsection; yield (heading, raw body)."""
    parts = re.split(r"(\\(?:sub)?section\*?\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", tex)
    head = "front matter"
    for p in parts:
        m = re.match(r"\\((?:sub)?section)\*?\{(.*)\}$", p, flags=re.S)
        if m:
            head = ("  " if m.group(1) == "subsection" else "") + m.group(2)
        else:
            yield head, p


def sentences(text: str):
    t = " ".join(text.split())
    for i, abbr in enumerate(NO_SPLIT):
        t = t.replace(abbr, abbr.replace(".", f"<DOT{i}>"))
    t = re.sub(r"(\d)\.(\d)", r"\1<DEC>\2", t)
    raw = re.split(r"(?<=[.?!])\s+(?=[A-Z(\[])", t)
    out = []
    for s in raw:
        s = re.sub(r"<DOT\d+>", ".", s).replace("<DEC>", ".").strip()
        if s:
            out.append(s)
    return out


def words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'+\-]*", s))


# REVIEW_full_v2 V10: British spellings had returned in generated table notes and in one prose
# sentence, against the manuscript's American rule. Each pattern maps to the required form. The
# check runs over the given file AND every file it \input's, because the offenders were in
# tables/*.tex, which check_text.py did not previously see.
# Three forms are deliberately absent because they are correct here:
#   "centre"   only inside the verbatim variable name \texttt{srf_centre_wavelength};
#   "analyses" the American plural of "analysis" (the verb forms analyse/analysed are checked);
#   "Colour"   the official instrument name "Ocean and Land Colour Instrument" (OLCI).
#
# REVIEW_final F4: the V10 fix reached main.tex and its \input files only, so two British words
# survived inside figure IMAGES ("Normalised split (MDN)" in Fig. 1, "Realised coverage" on the
# y axis of Fig. 9), where no text check could see them, and six "grey"/"greyscale" and one
# "licence" survived in captions because they were not on the word list. The list below is the
# whole mix the Guide forbids ("American or British usage is accepted, but not a mixture"), and the
# scan now covers three populations: main.tex and its \input files, the FIGURE SOURCES that draw
# the labels, and the RENDERED TEXT of every figure PDF.
SPELLING = [
    (r"\brealis(e|ed|es|ing|ation)\b", "realiz..."),
    (r"\bsummaris(e|ed|es|ing|ation)\b", "summariz..."),
    (r"\bnormalis(e|ed|es|ing|ation)\b", "normaliz..."),
    (r"\bjudgement\b", "judgment"),
    (r"\banalys(e|ed|ing)\b", "analyz..."),
    (r"\bbehaviour\b", "behavior"),
    (r"\blabelled\b", "labeled"),
    (r"\bmodelling\b", "modeling"),
    (r"\bcolours?\b(?! Instrument)", "color"),
    (r"\bcoloured\b", "colored"),
    (r"\bgrey(s|ish)?\b", "gray"),
    (r"\bgreyscale\b", "grayscale"),
    (r"\blicence\b", "license"),
    (r"\bcentres?\b", "center"),
    (r"\bfibre\b", "fiber"),
    (r"\bmetres?\b", "meter"),
    (r"\bfavour(s|ed|ing|able)?\b", "favor..."),
    (r"\bneighbour(s|ing|hood)?\b", "neighbor..."),
    (r"\bcatalogu(e|ed|es|ing)\b", "catalog..."),
    (r"\bparameteris(e|ed|es|ing|ation)\b", "parameteriz..."),
    (r"\bcharacteris(e|ed|es|ing|ation)\b", "characteriz..."),
    (r"\bstandardis(e|ed|es|ing|ation)\b", "standardiz..."),
    (r"\bminimis(e|ed|es|ing|ation)\b", "minimiz..."),
    (r"\bmaximis(e|ed|es|ing|ation)\b", "maximiz..."),
    (r"\bemphasis(e|ed|es|ing)\b", "emphasiz..."),
    (r"\brecognis(e|ed|es|ing)\b", "recogniz..."),
    (r"\bgeneralis(e|ed|es|ing|ation)\b", "generaliz..."),
    (r"\bmodelled\b", "modeled"),
    (r"\btravelled\b", "traveled"),
]
# Proper names and verbatim identifiers that are correct as they stand and must not be "fixed".
# "Organisation" is EUMETSAT's registered name; "Colour" is the instrument name behind OLCI;
# "srf_centre_wavelength" is a column name in the released tables; "analyses" is the American
# plural of "analysis" and is not matched by the verb pattern above.
SPELLING_EXEMPT = [
    r"European Organisation",
    r"Ocean and Land Colour Instrument",
    r"srf_centre_wavelength",
]
INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
FIGURE_SRC_DIR = "src/figures"
FIGURE_PDF_DIR = "figures"


def _mask(body: str) -> str:
    """Blank the exempt proper names, keeping offsets, so the scan cannot hit them."""
    for pat in SPELLING_EXEMPT:
        body = re.sub(pat, lambda m: " " * len(m.group(0)), body)
    return body


def _drawable_python(src: str) -> str:
    """Only the string literals a figure script can actually DRAW.

    Docstrings and ``#`` comments describe the figure; they never reach the page, so scanning them
    would report British words that no reader of the article can see. This mirrors the LaTeX pass,
    which has always stripped ``%`` comments before looking. Everything else that is a string
    constant (axis labels, titles, legend entries, CLI flags) is returned for scanning.
    """
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = getattr(node, "body", None)
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                docstrings.add(id(b[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            out.append(node.value)
    return "\n".join(out)


def _figure_pdf_text(pdf: Path) -> str:
    import fitz
    doc = fitz.open(str(pdf))
    try:
        return "\n".join(doc[i].get_text() for i in range(doc.page_count))
    finally:
        doc.close()


def _scan(label: str, body: str) -> int:
    body = _mask(body)
    hits = 0
    for pat, want in SPELLING:
        for m in re.finditer(pat, body, re.I):
            ctx = body[max(0, m.start() - 45): m.end() + 45].replace("\n", " ")
            print(f"  SPELLING {label}: {m.group(0)!r} -> {want}   ...{ctx}...")
            hits += 1
    return hits


def spelling_check(tex_path: Path, figures: bool = True) -> int:
    """Scan the manuscript, the figure sources and the rendered figure PDFs. Returns the hits."""
    root = tex_path.resolve().parents[1]
    files = [tex_path]
    raw = tex_path.read_text(encoding="utf-8")
    for m in INPUT_RE.finditer(strip_comments(raw)):
        name = m.group(1)
        if not name.lower().endswith(".tex"):
            name += ".tex"
        for cand in (tex_path.parent / name, root / name):
            if cand.exists():
                files.append(cand)
                break
    hits = 0
    for f in files:
        body = strip_comments(f.read_text(encoding="utf-8"))
        body = re.sub(r"\\texttt\{[^}]*\}", " ", body)     # verbatim variable names
        hits += _scan(f.name, body)
    n_src = n_pdf = 0
    if figures:
        for f in sorted((root / FIGURE_SRC_DIR).glob("*")):
            if f.suffix == ".py":
                hits += _scan(f.name, _drawable_python(f.read_text(encoding="utf-8")))
            elif f.suffix == ".tex":
                hits += _scan(f.name, strip_comments(f.read_text(encoding="utf-8")))
            else:
                continue
            n_src += 1
        for f in sorted((root / FIGURE_PDF_DIR).glob("*.pdf")):
            try:
                hits += _scan(f.name, _figure_pdf_text(f))
            except ImportError:
                print("  WARN PyMuPDF not installed; the rendered-figure pass was skipped")
                break
            n_pdf += 1
    print(f"== spelling: {hits} hit(s) over {len(files)} manuscript file(s), "
          f"{n_src} figure source(s) and {n_pdf} rendered figure(s) ==")
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--max-words", type=int, default=45)
    ap.add_argument("--all", action="store_true", help="list every abbreviation found")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the figure-source and rendered-figure spelling passes")
    a = ap.parse_args(argv)
    tex = strip_comments(Path(a.tex).read_text(encoding="utf-8"))
    body_start = tex.find("\\begin{document}")
    tex_body = tex[body_start:]
    bib_cut = tex_body.find("\\bibliographystyle")
    if bib_cut > 0:
        tex_body = tex_body[:bib_cut]

    n_long = n_dash = 0
    total = 0
    print("== words per block (prose only; PENDING-only blocks skipped) ==")
    prose_all = []
    for head, raw in blocks(tex_body):
        prose = delatex(raw)
        w = words(prose)
        if w < 5:
            continue
        total += w
        print(f"{w:6d}  {head.strip() if not head.startswith('  ') else head}")
        for s in sentences(prose):
            nw = words(s)
            if nw > a.max_words:
                n_long += 1
                print(f"    LONG ({nw} words): {s[:160]}...")
        prose_all.append(prose)
        # dashes in prose (raw body minus math and skipped environments)
        raw_nomath = raw
        for env in SKIP_ENVS:
            raw_nomath = remove_env(raw_nomath, env)
        raw_nomath = re.sub(r"\$[^$]*\$", " ", raw_nomath)
        for m in re.finditer(r"\u2014|\u2013|(?<!-)---?(?!-)", raw_nomath):
            n_dash += 1
            ctx = raw_nomath[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
            print(f"    DASH: ...{ctx}...")
    print(f"{total:6d}  TOTAL (prose words in non-pending blocks)")

    # dashes inside the environments skipped above (tables, captions, algorithms, keywords), outside math
    for env in SKIP_ENVS:
        e = re.escape(env)
        for chunk in re.findall(r"\\begin\{" + e + r"\}.*?\\end\{" + e + r"\}", tex_body, flags=re.S):
            chunk = re.sub(r"\$[^$]*\$", " ", chunk)
            for m in re.finditer(r"\u2014|\u2013|(?<![-!])---?(?!-)", chunk):
                ctx = chunk[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
                print(f"    DASH ({env}): ...{ctx}...")
                n_dash += 1

    # abbreviations
    print("== abbreviation first use vs definition (heuristic) ==")
    prose = " ".join(prose_all)
    seen = {}
    for m in ABBREV_RE.finditer(prose):
        tok = re.sub(r"(-[a-z][a-z0-9]*)+$", "", m.group(1))  # SRF-weighted -> SRF
        caps = sum(ch.isupper() for ch in tok)
        if (caps >= 2 or tok in MIXED) and tok not in ALLOW and not re.fullmatch(r"[A-Z][a-z]+[A-Z]?", tok):
            if tok in {"MATH", "RESULT", "CITATION"} or re.search(r"\d{2,}", tok) or tok.count("-") >= 3:
                continue
            seen.setdefault(tok, m.start())
    n_abbr = 0
    for tok, first in sorted(seen.items(), key=lambda kv: kv[1]):
        d = re.search(r"\(\s*" + re.escape(tok) + r"(?![A-Za-z0-9])", prose)
        status = "ok"
        if d is None:
            status = "NEVER DEFINED"
        elif d.start() > first + len(tok) + 400:
            status = f"USED BEFORE DEFINITION (first use at char {first}, definition at {d.start()})"
        if status != "ok" or a.all:
            ctx = prose[max(0, first - 50): first + 30].replace("\n", " ")
            print(f"  {tok:12s} {status}   ...{ctx}...")
            n_abbr += status != "ok"
    print("== American spelling (this file, every \\input file, the figure sources "
          "and the rendered figures) ==")
    n_spell = spelling_check(Path(a.tex), figures=not a.no_figures)
    print(f"== summary: long sentences {n_long}; dashes {n_dash}; abbreviation hits {n_abbr}; "
          f"spelling hits {n_spell} ==")
    return 0 if (n_long == 0 and n_dash == 0 and n_spell == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
