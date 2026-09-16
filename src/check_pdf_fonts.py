"""Embedded-font check for the compiled manuscript and the figure PDFs.

REVIEW_full_v2 V2. `\\usepackage[T1]{fontenc}` without a scalable T1 Computer Modern makes pdfTeX
fall back to bitmapped EC fonts, and the whole manuscript is then set in Type 3. Elsevier does not
accept Type 3 fonts, and nothing in the build reported it: the log had no error, the page count was
right and the PDF looked correct on screen. `\\usepackage{lmodern}` before `fontenc` fixes it, and
this checker makes the fix impossible to lose again.

Every font object in every checked PDF must

  * be of type Type1, TrueType, Type0 or MMType1 (never Type3), and
  * be embedded, that is, PyMuPDF must report a non-empty font-file extension
    (pfa/pfb/ttf/cff/otf/ttc), not "n/a".

Run:
    python src/check_pdf_fonts.py                       # paper/main.pdf + every figures/*.pdf
    python src/check_pdf_fonts.py --pdf paper/main.pdf  # one file
Exit code 0 when every checked PDF passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF

FORBIDDEN_TYPES = {"Type3"}
ALLOWED_TYPES = {"Type1", "TrueType", "Type0", "MMType1"}
NOT_EMBEDDED = {"", "n/a", "na", "none"}


def audit_pdf(path: Path) -> tuple[list[str], list[str]]:
    """Return (failures, font description lines) for one PDF."""
    fails: list[str] = []
    doc = fitz.open(path)
    fonts: dict[tuple[str, str, str], set[int]] = {}
    for pno in range(doc.page_count):
        for item in doc.get_page_fonts(pno, full=True):
            # (xref, ext, type, basefont, name, encoding[, referencer]) across PyMuPDF versions
            _xref, ext, ftype, basefont = item[0], item[1], item[2], item[3]
            fonts.setdefault((ftype, basefont, ext), set()).add(pno + 1)
    lines = []
    for (ftype, basefont, ext), pages in sorted(fonts.items()):
        pp = sorted(pages)
        span = f"{len(pp)} page(s), first p.{pp[0]}"
        lines.append(f"    {ftype:<9} {ext or 'n/a':<5} {basefont:<38} {span}")
        if ftype in FORBIDDEN_TYPES:
            fails.append(f"{path.name}: {basefont} is {ftype} (bitmap) on {len(pp)} page(s), "
                         f"first p.{pp[0]}")
        elif ftype not in ALLOWED_TYPES:
            fails.append(f"{path.name}: {basefont} has unexpected font type {ftype!r}")
        if (ext or "").lower() in NOT_EMBEDDED:
            fails.append(f"{path.name}: {basefont} ({ftype}) is not embedded")
    if not fonts:
        lines.append("    (no font objects)")
    doc.close()
    return fails, lines


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="append", default=None,
                    help="PDF to check; repeatable. Default: paper/main.pdf and figures/*.pdf")
    args = ap.parse_args()

    if args.pdf:
        targets = [Path(p) for p in args.pdf]
    else:
        targets = [root / "paper" / "main.pdf"]
        targets += sorted(p for p in (root / "figures").glob("*.pdf")
                          if not p.name.startswith("_"))

    all_fails: list[str] = []
    for p in targets:
        if not p.exists():
            all_fails.append(f"{p} does not exist")
            print(f"{p}: MISSING")
            continue
        fails, lines = audit_pdf(p)
        print(f"{p.relative_to(root) if root in p.parents or p.is_relative_to(root) else p}: "
              f"{'FAIL' if fails else 'pass'}")
        for ln in lines:
            print(ln)
        all_fails += fails

    print(f"\nchecked {len(targets)} PDF(s); failures {len(all_fails)}")
    for f in all_fails:
        print(f"  FAIL {f}")
    return 1 if all_fails else 0


if __name__ == "__main__":
    sys.exit(main())
