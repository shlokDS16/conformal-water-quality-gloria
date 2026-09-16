"""Build submission/upload_ISPRS/: one clearly named item per Editorial Manager upload slot.

Phase 14 section D. Nothing here is invented: the manuscript PDF, Word file, highlights,
figures and flat LaTeX source are copied from submission/, which
src/make_submission_package.py already built from paper/main.tex. This script adds the two
items that folder does not hold (title page, cover letter), gives every item a slot-numbered
folder, and writes nothing into paper/, results/, tables/ or src/analysis/.

Usage:
    python src/make_upload_package.py            # build
    python src/make_upload_package.py --check    # build, then re-run the section E checks
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submission"
OUT = SUB / "upload_ISPRS"
TEX = ROOT / "paper" / "main.tex"

# ---------------------------------------------------------------- author facts
# Source: paper/main.tex frontmatter (read only) and progress.md user step R1.
TITLE = ("Group-aware conformal calibration for water-quality retrieval: "
         "marginal validity under water-body shift, poor conditional coverage")
AFFIL = ("School of Computer Science and Engineering, Vellore Institute of Technology, "
         "Vellore, 632014, Tamil Nadu, India")
AUTHORS = [
    ("Shlok Kumar Goenka", "shlok.goenka2023@vitstudent.ac.in", False),
    ("R. Manjula", "rmanjula@vit.ac.in", True),
]
REPO = "https://github.com/shlokDS16/conformal-water-quality-gloria"
ZENODO = "10.5281/zenodo.22791749"

COVER_LETTER = """\
16 September 2026

Clément Mallet and Qihao Weng
Editors-in-Chief
ISPRS Journal of Photogrammetry and Remote Sensing

Dear Editors-in-Chief,

We submit the manuscript "Group-aware conformal calibration for water-quality retrieval: \
marginal validity under water-body shift, poor conditional coverage" for consideration as a \
Paper in the ISPRS Journal of Photogrammetry and Remote Sensing.

The study audits prediction intervals for chlorophyll-a, total suspended solids, colored \
dissolved organic matter absorption at 440 nm and Secchi disk depth, retrieved from in situ \
hyperspectral remote-sensing reflectance in the GLORIA compilation. Match-up databases hold \
many samples per water body, so a random split trains, calibrates and tests on the same lakes. \
We therefore hold training, calibration and test water bodies disjoint, compare eleven interval \
methods, and prove a finite-sample marginal coverage result for split conformal calibration \
that draws one conformity score per calibration water body.

The work fits this journal because it is about the reliability of a remote sensing retrieval \
rather than about a new retrieval. It reports the evaluation protocol, the Sentinel-2 MSI and \
Sentinel-3 OLCI band simulations, and simulated atmospheric-correction noise calibrated on \
published ACIX-Aqua error levels, so the results speak to operational products built on those \
sensors.

Three findings matter. Native mixture density network intervals under-cover: 0.676 to 0.748 \
water-body-averaged coverage against a nominal 0.90. Group-aware conformal calibration held \
the nominal level under water-body hold-out, at 0.895 to 0.905, and the cost is width, a factor \
of 15.5 for chlorophyll-a. Conditional coverage is poor for every method we tested, including \
the valid ones: the worst test water body of a split received 0.468 to 0.594, and no variant \
repaired it. Marginal validity is achievable and per-water-body validity is not, so a \
validation report should state both.

The analysis was preregistered before the confirmatory runs. None of the 16 preregistered \
tests was rejected after multiplicity correction, and the non-inferiority test of conformal \
validity had power 0.14 to 0.17, so the findings rest on effect sizes and descriptive \
components rather than on confirmatory rejections. The manuscript says so in the abstract and \
in the Results, and it keeps the negative results that went against our own preregistered \
expectations.

The code that reproduces every reported number, the water-body grouping table, the split \
definitions and the result ledgers are public on GitHub and archived on Zenodo under the \
concept DOI 10.5281/zenodo.22791749, which is published on acceptance. GLORIA itself is \
publicly available from PANGAEA and is cited as a dataset.

We confirm that this work is original, that it has not been published previously, and that it \
is not under consideration for publication elsewhere. Both authors have read and approved the \
submitted manuscript and agree to its submission to this journal. The research received no \
specific grant, and we declare no competing interests.

Yours sincerely,

R. Manjula
Corresponding author, on behalf of both authors
School of Computer Science and Engineering
Vellore Institute of Technology
Vellore, 632014, Tamil Nadu, India
rmanjula@vit.ac.in
"""


SUPPLEMENTARY_NOTE = """\
# Supplementary material

One file is uploaded as supplementary material at the Attach/Upload Files step:

| File | What it holds | Where the manuscript calls for it |
|---|---|---|
| `Supplementary_Table_S1_point_errors.csv` | log-MAE, log-bias and log-RMSE, with MdSA, SSPB and the repeat count, for every cell Table 5 prints: water-body protocol, primary population, three sensor configurations, eight point models, four targets (63 rows) | the note under Table 5 |

Produced by `src/make_supplementary.py` from `tables/summary.csv`. Values are copied at full
stored precision; nothing is recomputed.

## Everything else stays in the manuscript or in the deposit

| Material | Where it is |
|---|---|
| Notation tables | Appendix A, Tables A.1 and A.2 |
| Band simulation, spectral response functions, irradiance | Appendix A, Tables A.3 and A.4, Fig. A.1 |
| Proofs | Appendix B, Lemmas B.1 and B.2, Eqs. (B.1) to (B.3) |
| Results at the other two nominal levels, both multispectral sensors, interval score | Appendix C, Tables C.1 to C.5 |
| Per-water-body conditional coverage and nearest-neighbor distance | Appendix C, Figs. C.1 and C.2 |
| The water-body grouping table named in Section 2.2 | the deposit, `data/interim/wb_groups.csv` |
| Code that reproduces every reported number | https://github.com/shlokDS16/conformal-water-quality-gloria |
| Code, derived data, splits, per-split endpoint files, result ledgers | Zenodo, concept DOI 10.5281/zenodo.22791749 |
| The raw GLORIA compilation (not redistributed) | PANGAEA, doi:10.1594/PANGAEA.948492 |

The Zenodo deposit is about 1.36 GiB across 13 files, far larger than a submission system is
meant to carry, which is why it is a citation and not an upload.

## Status of the Zenodo record at submission time

The record is a **reserved draft**. The DOI 10.5281/zenodo.22791749 is reserved but does not
resolve until the record is published. The manuscript says so in the Data availability section,
so a reviewer who tries the DOI and gets a 404 has been told why. The record is published, and
the DOI made resolvable, on acceptance. See `research/DATA_AVAILABILITY.md` and
`submission/ZENODO_SETUP.md`.

## If the editor asks for more

Do not build a new file. Point the editor at the GitHub repository and the Zenodo record, or
upload `submission/upload_ISPRS/06_LaTeX_Source/LaTeX_Source.zip`, which holds the complete
typesetting source.
"""


# ------------------------------------------------------------------- utilities
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_docx_metadata(path: Path) -> None:
    """Rewrite docProps/core.xml and docProps/app.xml so no name, company or tool is left.

    Reuses src/make_docx.py, which already does exactly this for the Word manuscript, so the
    whole package carries one metadata convention.
    """
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.make_docx import strip_metadata
    strip_metadata(path)


def clear_docx_metadata(d) -> None:
    """Leave no author field in the file. Checked afterwards by --check."""
    cp = d.core_properties
    cp.author = ""
    cp.last_modified_by = ""
    cp.title = ""
    cp.subject = ""
    cp.keywords = ""
    cp.category = ""
    cp.comments = ""
    cp.content_status = ""
    cp.identifier = ""
    cp.language = ""
    cp.version = ""


def new_document():
    import docx
    from docx.shared import Cm, Pt
    d = docx.Document()
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(2.5)
        s.top_margin = s.bottom_margin = Cm(2.5)
    st = d.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    return d


def para(d, text: str, bold: bool = False, size: int = 12, space_after: int = 8):
    from docx.shared import Pt
    p = d.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.bold = bold
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    return p


def author_line(d, name: str, mark: str, space_after: int = 2):
    """One author, with the affiliation letter set as a true superscript run."""
    from docx.shared import Pt
    p = d.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(name)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    s = p.add_run(mark)
    s.font.name = "Times New Roman"
    s.font.size = Pt(12)
    s.font.superscript = True
    return p


# ------------------------------------------------------------------ title page
def write_title_page_docx(path: Path) -> None:
    d = new_document()
    para(d, "Title page", bold=True, size=14)
    para(d, TITLE, bold=True, size=13, space_after=14)

    # The Guide: "Indicate all affiliations with a lower-case superscript letter immediately
    # after the author's name", so the marker is a real superscript run, not a bare letter.
    para(d, "Authors", bold=True)
    author_line(d, "Shlok Kumar Goenka", "a")
    author_line(d, "R. Manjula", "a,*", space_after=10)

    para(d, "Affiliation", bold=True)
    aff = d.add_paragraph()
    from docx.shared import Pt as _Pt
    aff.paragraph_format.space_after = _Pt(10)
    m = aff.add_run("a")
    m.font.name = "Times New Roman"
    m.font.size = _Pt(12)
    m.font.superscript = True
    t = aff.add_run(" " + AFFIL)
    t.font.name = "Times New Roman"
    t.font.size = _Pt(12)

    para(d, "Corresponding author", bold=True)
    para(d, "* R. Manjula", space_after=2)
    para(d, AFFIL, space_after=2)
    para(d, "E-mail: rmanjula@vit.ac.in", space_after=10)

    para(d, "E-mail addresses", bold=True)
    para(d, "Shlok Kumar Goenka: shlok.goenka2023@vitstudent.ac.in", space_after=2)
    para(d, "R. Manjula: rmanjula@vit.ac.in", space_after=10)

    para(d, "ORCID", bold=True)
    para(d, "Shlok Kumar Goenka: [TO BE FILLED BY THE AUTHOR]", space_after=2)
    para(d, "R. Manjula: [TO BE FILLED BY THE AUTHOR]", space_after=10)

    para(d, "Declarations", bold=True)
    para(d, "Funding: This research did not receive any specific grant from funding agencies "
            "in the public, commercial, or not-for-profit sectors.", space_after=6)
    para(d, "Declaration of competing interest: The authors declare that they have no known "
            "competing financial interests or personal relationships that could have appeared "
            "to influence the work reported in this paper.", space_after=6)
    para(d, "CRediT authorship contribution statement: Shlok Kumar Goenka: Conceptualization, "
            "Methodology, Software, Formal analysis, Investigation, Data curation, "
            "Visualization, Writing - original draft. R. Manjula: Conceptualization, "
            "Supervision, Project administration, Writing - review and editing.", space_after=6)
    para(d, "Declaration of generative AI and AI-assisted technologies in the manuscript "
            "preparation process: During the preparation of this work the authors used Claude "
            "(Anthropic) in order to assist with drafting and to improve the language of the "
            "manuscript. After using this tool/service, the authors reviewed and edited the "
            "content as needed and take full responsibility for the content of the published "
            "article.", space_after=6)
    para(d, "Acknowledgements: The authors thank the data contributors and the compilation team "
            "of the GLORIA dataset. They also thank the European Space Agency and the European "
            "Organisation for the Exploitation of Meteorological Satellites (EUMETSAT) for "
            "making the Sentinel-2 MSI and Sentinel-3 OLCI spectral response functions publicly "
            "available.", space_after=6)
    para(d, "Data availability: The GLORIA compilation is publicly available from PANGAEA. The "
            "code that reproduces every result reported here, the water-body grouping table, "
            "the split definitions, the per-split endpoint files and the derived result ledgers "
            "are available on GitHub at " + REPO + " and are archived on Zenodo under the "
            "concept DOI " + ZENODO + ". The Zenodo record is a reserved draft; it is "
            "published, and the DOI made resolvable, on acceptance.", space_after=6)

    clear_docx_metadata(d)
    d.save(str(path))


TITLE_PAGE_TEX = r"""%% Title page with author details, ISPRS Journal of Photogrammetry and Remote Sensing.
%% Compiles on its own: pdflatex Title_Page.tex
%% The journal operates a single anonymized review process, so the author names also appear on
%% the manuscript itself; this file is the editable title page the submission system asks for.
\documentclass[preprint,12pt,authoryear]{elsarticle}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{xurl}

\journal{ISPRS Journal of Photogrammetry and Remote Sensing}

\begin{document}
\begin{frontmatter}

\title{TITLE_GOES_HERE}

\author[a]{Shlok Kumar Goenka}
\ead{shlok.goenka2023@vitstudent.ac.in}
%% ORCID, first author (field present, to be filled by the author):
%% ORCID:

\author[a]{R. Manjula\corref{cor1}}
\ead{rmanjula@vit.ac.in}
%% ORCID, second author (field present, to be filled by the author):
%% ORCID:

\cortext[cor1]{Corresponding author.}

\affiliation[a]{organization={School of Computer Science and Engineering, Vellore Institute of Technology},
  city={Vellore},
  postcode={632014},
  state={Tamil Nadu},
  country={India}}

\end{frontmatter}

\section*{Corresponding author}
R. Manjula, School of Computer Science and Engineering, Vellore Institute of Technology,
Vellore, 632014, Tamil Nadu, India. E-mail: \texttt{rmanjula@vit.ac.in}.

\section*{Funding}
This research did not receive any specific grant from funding agencies in the public,
commercial, or not-for-profit sectors.

\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this paper.

\section*{CRediT authorship contribution statement}
\textbf{Shlok Kumar Goenka:} Conceptualization, Methodology, Software, Formal analysis,
Investigation, Data curation, Visualization, Writing - original draft.
\textbf{R. Manjula:} Conceptualization, Supervision, Project administration,
Writing - review and editing.

\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}
During the preparation of this work the authors used Claude (Anthropic) in order to assist with
drafting and to improve the language of the manuscript. After using this tool/service, the
authors reviewed and edited the content as needed and take full responsibility for the content
of the published article.

\section*{Acknowledgements}
The authors thank the data contributors and the compilation team of the GLORIA dataset. They
also thank the European Space Agency and the European Organisation for the Exploitation of
Meteorological Satellites (EUMETSAT) for making the Sentinel-2 MSI and Sentinel-3 OLCI spectral
response functions publicly available.

\section*{Data availability}
The GLORIA compilation of in situ hyperspectral remote-sensing reflectance and paired
water-quality measurements is publicly available from PANGAEA. The code that reproduces every
result reported here, the water-body grouping table, the split definitions, the per-split
endpoint files and the derived result ledgers are available on GitHub at
\url{REPO_GOES_HERE} and are archived on Zenodo under the concept digital object identifier
(DOI) \texttt{ZENODO_GOES_HERE}. The Zenodo record is a reserved draft; it is published, and
the DOI made resolvable, on acceptance.

\end{document}
"""


def write_title_page_tex(path: Path) -> None:
    src = (TITLE_PAGE_TEX
           .replace("TITLE_GOES_HERE", TITLE)
           .replace("REPO_GOES_HERE", REPO)
           .replace("ZENODO_GOES_HERE", ZENODO))
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(src)


# ----------------------------------------------------------------- cover letter
def write_cover_letter_txt(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(COVER_LETTER)


def write_cover_letter_docx(path: Path) -> None:
    d = new_document()
    blocks = [b for b in COVER_LETTER.split("\n\n")]
    for b in blocks:
        b = b.strip("\n")
        if not b:
            continue
        if "\n" in b:
            # address blocks and the signature: one paragraph per line, tight spacing
            for i, line in enumerate(b.split("\n")):
                para(d, line, space_after=(8 if i == len(b.split("\n")) - 1 else 0))
        else:
            para(d, " ".join(b.split()), space_after=10)
    clear_docx_metadata(d)
    d.save(str(path))


def cover_letter_word_count() -> int:
    """Words of the letter body only: the date, address block and signature are excluded."""
    body = COVER_LETTER.split("Dear Editors-in-Chief,", 1)[1]
    body = body.split("Yours sincerely,", 1)[0]
    return len(body.split())


# ------------------------------------------------------------------- the build
def copy_tree(src: Path, dst: Path) -> list[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(src.iterdir()):
        if p.is_file():
            shutil.copy2(p, dst / p.name)
            out.append(dst / p.name)
    return out


def build() -> None:
    # Only the numbered upload-slot folders are rebuilt. HOW_TO_SUBMIT.md and the two notes at
    # the top level are hand-written prose and are never deleted by this script.
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("01_Cover_Letter", "02_Title_Page", "03_Manuscript", "04_Highlights",
                 "05_Figures", "06_LaTeX_Source", "07_Supplementary"):
        if (OUT / name).exists():
            shutil.rmtree(OUT / name)
    # REVIEW_final F3: the package used to carry a pointer folder saying there was nothing to
    # upload, while the note under Table 5 promised a supplementary CSV. The file is produced now,
    # so the pointer folder is replaced by a real supplementary slot.
    if (OUT / "07_Supplementary_pointer").exists():
        shutil.rmtree(OUT / "07_Supplementary_pointer")

    # 01 cover letter
    d = OUT / "01_Cover_Letter"
    d.mkdir()
    write_cover_letter_docx(d / "Cover_Letter.docx")
    write_cover_letter_txt(d / "Cover_Letter.txt")

    # 02 title page
    d = OUT / "02_Title_Page"
    d.mkdir()
    write_title_page_docx(d / "Title_Page.docx")
    write_title_page_tex(d / "Title_Page.tex")

    # 03 manuscript
    d = OUT / "03_Manuscript"
    d.mkdir()
    shutil.copy2(SUB / "Manuscript_review.pdf", d / "Manuscript.pdf")
    shutil.copy2(SUB / "Manuscript.docx", d / "Manuscript.docx")

    # 04 highlights
    d = OUT / "04_Highlights"
    d.mkdir()
    shutil.copy2(SUB / "Highlights.docx", d / "Highlights.docx")
    shutil.copy2(SUB / "Highlights.txt", d / "Highlights.txt")

    # every .docx written or copied above gets the same empty core.xml and app.xml
    for p in sorted(OUT.rglob("*.docx")):
        strip_docx_metadata(p)

    # 05 figures. The numbered Figure_N files come from submission/figures/, which
    # make_submission_package.py builds by mapping each figure to its PRINTED number. The
    # graphical abstract has no printed number - it is a separate Editorial Manager upload slot -
    # so it is not in that folder and is copied straight from figures/. Without this it would be
    # deleted on every rebuild of the package.
    copy_tree(SUB / "figures", OUT / "05_Figures")
    for name in ("graphical_abstract.pdf", "graphical_abstract.png"):
        src = ROOT / "figures" / name
        if src.exists():
            shutil.copy2(src, OUT / "05_Figures" / name)
        else:
            print(f"  note: {name} not found in figures/; the graphical-abstract slot is empty")

    # 06 LaTeX source, flat, plus a single archive for the LaTeX upload item
    d = OUT / "06_LaTeX_Source"
    files = copy_tree(SUB / "latex_source", d)
    zpath = OUT / "06_LaTeX_Source" / "LaTeX_Source.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)

    # 07 supplementary material
    d = OUT / "07_Supplementary"
    d.mkdir()
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.make_supplementary import OUT as SUPP_CSV, build as build_supp
    build_supp(ROOT / "tables" / "summary.csv", SUPP_CSV)
    shutil.copy2(SUPP_CSV, d / SUPP_CSV.name)
    (d / "Supplementary_Material.md").write_text(SUPPLEMENTARY_NOTE, encoding="utf-8", newline="")

    print("built", OUT)
    print("cover letter body words:", cover_letter_word_count())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    build()
    if a.check:
        import subprocess
        subprocess.run(["python", str(ROOT / "src" / "check_upload_package.py")], cwd=ROOT)


if __name__ == "__main__":
    main()
