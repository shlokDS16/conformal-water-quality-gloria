"""Section E checks on submission/upload_ISPRS/.

Reports, and never fixes:
  * every file with its size in bytes and MB, against the Guide's only stated limits
    (10 MB per individual figure file; 150 MB per video, none here);
  * PDF metadata fields (any author-bearing field must be empty);
  * DOCX core properties for every .docx in the package;
  * abstract words, highlights bullet count and per-bullet characters, keyword count;
  * placeholders left for a human.

Usage: python src/check_upload_package.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "upload_ISPRS"
TEX = ROOT / "paper" / "main.tex"

FIG_LIMIT = 10 * 1024 * 1024  # Guide: "individual figure files larger than 10 MB must be
#                               uploaded separately" (a routing rule, quoted as the limit)

PDF_META_KEYS = ["title", "author", "subject", "keywords", "creator", "producer"]

# The journal runs a SINGLE anonymized review, so the manuscript PDF is expected to name the
# authors. These are the only author-bearing values allowed anywhere in the package; anything
# else in an author field is an unintended leak.
INTENDED_PDF_AUTHOR = "Shlok Kumar Goenka; R. Manjula;"
INTENDED_PDF_TITLE = ("Group-aware conformal calibration for water-quality retrieval: "
                      "marginal validity under water-body shift, poor conditional coverage")
DOCX_META_KEYS = ["author", "last_modified_by", "title", "subject", "keywords",
                  "category", "comments", "content_status", "identifier", "language"]


def human(n: int) -> str:
    return f"{n/1024/1024:.2f} MB" if n >= 1024 * 1024 else f"{n/1024:.1f} KB"


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def delatex_text(s: str) -> str:
    """The plain text a human pastes into the submission form: escapes resolved, ties are spaces."""
    s = s.replace("~", " ")
    s = re.sub(r"\\(%|&|_|\$|#|\{|\})", r"\1", s)
    return s


def frontmatter_stats() -> dict:
    src = strip_comments(TEX.read_text(encoding="utf-8", errors="replace"))
    out = {}

    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S)
    # REVIEW_final F12: this used to count the RAW LaTeX, so "440~nm" was one token where the
    # pasted plain text is two, and "90\%" was one character longer than "90%". The checker
    # therefore printed 292 words and 1,953 characters where HOW_TO_SUBMIT.md section 3.2 and
    # SUBMISSION_PACKAGE_LOG.md section 5.4 both record 293 and 1,952. The artefact was right.
    # Counting the de-LaTeXed abstract, which is what an author pastes into the submission form,
    # makes all four records agree.
    abst = re.sub(r"\s+", " ", delatex_text(m.group(1))).strip() if m else ""
    out["abstract_words"] = len(abst.split())
    out["abstract_chars"] = len(abst)

    m = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", src, re.S)
    hl = [re.sub(r"\s+", " ", x).strip()
          for x in re.split(r"\\item\b", m.group(1))if x.strip()] if m else []
    out["highlights"] = hl

    m = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", src, re.S)
    kw = [k.strip() for k in re.split(r"\\sep", m.group(1))if k.strip()] if m else []
    out["keywords"] = [re.sub(r"\s+", " ", k) for k in kw]
    return out


def check_pdf(p: Path) -> list[str]:
    import fitz
    problems = []
    d = fitz.open(str(p))
    meta = d.metadata or {}
    for k in PDF_META_KEYS:
        v = (meta.get(k) or "").strip()
        if k in ("creator", "producer"):
            # a tool name is not an author field; report it, do not fail it
            print(f"      {k:9s} = {v!r}")
            continue
        intended = {"author": INTENDED_PDF_AUTHOR, "title": INTENDED_PDF_TITLE}.get(k, "")
        mark = ""
        if v and v == intended:
            mark = "   [intended: single-anonymized review]"
        print(f"      {k:9s} = {v!r}{mark}")
        if v and v != intended:
            problems.append(f"{p.name}: PDF {k} carries an unintended value ({v!r})")
    print(f"      pages     = {d.page_count}")
    d.close()
    return problems


def check_docx(p: Path) -> list[str]:
    import docx
    problems = []
    d = docx.Document(str(p))
    cp = d.core_properties
    for k in DOCX_META_KEYS:
        v = getattr(cp, k, None)
        v = "" if v is None else str(v).strip()
        print(f"      {k:17s} = {v!r}")
        if v:
            problems.append(f"{p.name}: DOCX {k} is not empty ({v!r})")
    # app.xml carries Company / Manager / Application
    with zipfile.ZipFile(p) as z:
        if "docProps/app.xml" in z.namelist():
            app = z.read("docProps/app.xml").decode("utf-8", "replace")
            for tag in ("Company", "Manager", "Application"):
                m = re.search(rf"<{tag}>(.*?)</{tag}>", app)
                v = m.group(1) if m else ""
                print(f"      app:{tag:13s} = {v!r}")
                if tag in ("Company", "Manager") and v:
                    problems.append(f"{p.name}: DOCX app.xml {tag} is not empty ({v!r})")
    return problems


def main() -> None:
    problems: list[str] = []
    total = 0
    print("=" * 78)
    print("FILE INVENTORY AND SIZES")
    print("=" * 78)
    for p in sorted(OUT.rglob("*")):
        if p.is_dir():
            continue
        n = p.stat().st_size
        total += n
        rel = p.relative_to(OUT).as_posix()
        flag = ""
        if p.suffix.lower() in (".pdf", ".png", ".tif", ".tiff", ".eps") and "Figures" in rel:
            if n > FIG_LIMIT:
                flag = "  <-- OVER THE 10 MB FIGURE LIMIT"
                problems.append(f"{rel}: {human(n)} exceeds the 10 MB figure limit")
        print(f"{n:>10d}  {human(n):>9s}  {rel}{flag}")
    print(f"\nTOTAL {total} bytes = {human(total)}")

    print()
    print("=" * 78)
    print("METADATA")
    print("=" * 78)
    for p in sorted(OUT.rglob("*.pdf")):
        if p.parent.name == "05_Figures" or p.parent.name == "06_LaTeX_Source":
            continue
        print(f"  {p.relative_to(OUT).as_posix()}")
        problems += check_pdf(p)
    for p in sorted(OUT.rglob("*.docx")):
        print(f"  {p.relative_to(OUT).as_posix()}")
        problems += check_docx(p)

    print()
    print("=" * 78)
    print("FIGURE PDF METADATA (author fields only)")
    print("=" * 78)
    import fitz
    fig_bad = []
    for p in sorted((OUT / "05_Figures").glob("*.pdf")):
        d = fitz.open(str(p))
        meta = d.metadata or {}
        bad = {k: meta.get(k) for k in ("author", "title", "subject", "keywords")
               if (meta.get(k) or "").strip()}
        d.close()
        if bad:
            fig_bad.append((p.name, bad))
    if fig_bad:
        for name, bad in fig_bad:
            print(f"  {name}: {bad}")
            problems.append(f"{name}: figure PDF carries {sorted(bad)}")
    else:
        print("  all figure PDFs: author, title, subject, keywords empty")

    print()
    print("=" * 78)
    print("ABSTRACT, HIGHLIGHTS, KEYWORDS")
    print("=" * 78)
    st = frontmatter_stats()
    print(f"  abstract: {st['abstract_words']} words, {st['abstract_chars']} characters")
    print(f"  highlights: {len(st['highlights'])} bullets")
    for i, h in enumerate(st["highlights"], 1):
        print(f"    {i}. {len(h):>3d} chars  {h}")
    print(f"  keywords: {len(st['keywords'])}")
    for k in st["keywords"]:
        print(f"    - {k}")
    if len(st["keywords"]) > 6:
        problems.append("more than 6 keywords (Guide limit)")
    if not (3 <= len(st["highlights"]) <= 5):
        problems.append("highlights outside the 3 to 5 bullet convention")
    for h in st["highlights"]:
        if len(h) > 85:
            problems.append(f"highlight over 85 characters: {h}")

    print()
    print("=" * 78)
    print("PLACEHOLDERS LEFT FOR A HUMAN")
    print("=" * 78)
    # case sensitive on purpose: the BibTeX .bbl is full of \bibinfo{title}{...}
    pat = re.compile(r"\[TO BE FILLED[^\]]*\]|\{JOURNAL\}|\{TOPIC\}|\{TITLE\}|XXXX")
    hits = 0
    for p in sorted(OUT.rglob("*")):
        if p.is_dir() or p.suffix.lower() in (".pdf", ".png", ".zip", ".docx"):
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in pat.finditer(t):
            line = t[:m.start()].count("\n") + 1
            print(f"  {p.relative_to(OUT).as_posix()}:{line}  {m.group(0)}")
            hits += 1
    # docx placeholders
    for p in sorted(OUT.rglob("*.docx")):
        with zipfile.ZipFile(p) as z:
            t = z.read("word/document.xml").decode("utf-8", "replace")
        for m in pat.finditer(t):
            print(f"  {p.relative_to(OUT).as_posix()}  {m.group(0)}")
            hits += 1
    if hits == 0:
        print("  none")

    print()
    print("=" * 78)
    print(f"PROBLEMS: {len(problems)}")
    for x in problems:
        print("  -", x)
    print("=" * 78)


if __name__ == "__main__":
    main()
