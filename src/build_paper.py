#!/usr/bin/env python
"""Build paper/main.pdf with the pdflatex/bibtex sequence and report the log summary.

latexmk is unusable on this machine (MiKTeX cannot find a Perl script engine), so the
equivalent manual sequence is run: pdflatex, bibtex, pdflatex, pdflatex.

Usage:  python src/build_paper.py [--tex paper/main.tex]
Prints: exit codes, page count, and the counts of errors, overfull boxes, undefined
references and "Float too large" messages found in main.log.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> int:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode


def summarize(logpath: Path) -> dict:
    txt = logpath.read_text(encoding="utf-8", errors="replace")
    out = {
        "errors": len(re.findall(r"^! ", txt, re.M)),
        "overfull": len(re.findall(r"^Overfull \\[hv]box", txt, re.M)),
        "underfull": len(re.findall(r"^Underfull \\[hv]box", txt, re.M)),
        "undefined_ref": len(re.findall(r"Reference `[^']*' on page .* undefined", txt)),
        "undefined_cite": len(re.findall(r"Citation `[^']*' .* undefined", txt)),
        "float_too_large": len(re.findall(r"Float too large", txt)),
        "multiply_defined": len(re.findall(r"multiply.defined", txt, re.I)),
        "dest_dup": len(re.findall(r"has been already used", txt)),
    }
    m = re.search(r"Output written on \S+ \((\d+) pages?, (\d+) bytes\)", txt)
    out["pages"] = int(m.group(1)) if m else None
    out["bytes"] = int(m.group(2)) if m else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "paper"))
    ap.add_argument("--jobname", default="main")
    args = ap.parse_args()
    d = Path(args.dir)
    j = args.jobname
    codes = []
    codes.append(("pdflatex-1", run(["pdflatex", "-interaction=nonstopmode", f"{j}.tex"], d)))
    codes.append(("bibtex", run(["bibtex", j], d)))
    codes.append(("pdflatex-2", run(["pdflatex", "-interaction=nonstopmode", f"{j}.tex"], d)))
    codes.append(("pdflatex-3", run(["pdflatex", "-interaction=nonstopmode", f"{j}.tex"], d)))
    for name, c in codes:
        print(f"{name}: exit {c}")
    s = summarize(d / f"{j}.log")
    for k, v in s.items():
        print(f"{k}: {v}")
    blg = d / f"{j}.blg"
    if blg.exists():
        t = blg.read_text(encoding="utf-8", errors="replace")
        print("bibtex_errors:", len(re.findall(r"^I couldn't|^Repeated entry|error message", t, re.M)))
        print("bibtex_warnings:", len(re.findall(r"^Warning--", t, re.M)))
    bad = s["errors"] or s["overfull"] or s["undefined_ref"] or s["undefined_cite"] or s["float_too_large"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
