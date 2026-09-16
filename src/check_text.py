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
         "GNU", "GitHub"}  # proper nouns that are not abbreviations of a defined term
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--max-words", type=int, default=45)
    ap.add_argument("--all", action="store_true", help="list every abbreviation found")
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
    print(f"== summary: long sentences {n_long}; dashes {n_dash}; abbreviation hits {n_abbr} ==")
    return 0 if (n_long == 0 and n_dash == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
