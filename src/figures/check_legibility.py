"""Legibility check for figure PDFs (phase 08).

For each figure PDF, a LaTeX test document places the figure at its intended width and is
compiled with pdflatex. PyMuPDF then reads every text span on the compiled page; span sizes
include the placement scale, so they are the printed sizes. A figure fails if its smallest
span is below MIN_PT in the final layout [final,5p,times,twocolumn]. The submission layout
[preprint,12pt,authoryear] is measured too (double-column figures scaled to \\textwidth,
single-column figures at natural size), reported for information.
A control compiles one figure at half width and asserts the measured size halves, which
verifies that span sizes carry the placement scale.
Output: research/FIGURE_LEGIBILITY.md.
Run: python -m src.figures.check_legibility
Preview run on the scratch renders while the experiments are still going:
  python -m src.figures.check_legibility --figdir figures/_preview --out research/FIGURE_LEGIBILITY_preview.md
A figure whose PDF is absent from the directory is reported as "not rendered" and does not count
as a failure, so the check runs before every result figure exists.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF

from . import style as S

MIN_PT = S.MIN_PRINT_PT
FIGS = {  # name: intended width in the final layout
    "fig01_workflow": "double",
    "fig02_map": "double",
    "fig03_protocols": "double",
    "fig04_spectra": "double",
    "fig05_point_accuracy": "double",
    "fig06_coverage_protocol": "double",
    "fig07_coverage_width": "double",
    "fig08_nominal_levels": "double",
    "fig09_calibration_budget": "double",
    "fig10_sensor_noise": "double",
    "figA1_targets": "single",
    "figA2_wbsize": "single",
    "figA3_wbcoverage_ecdf": "double",
    "figA4_conditional_coverage": "double",
    "figA5_knn_distance": "double",
}
OUT = S.ROOT / "research" / "FIGURE_LEGIBILITY.md"


def compile_doc(workdir: Path, name: str, classopts: str, env: str, width: str) -> Path:
    tex = (
        f"\\documentclass[{classopts}]{{elsarticle}}\n\\usepackage{{graphicx}}\n\\pagestyle{{empty}}\n"
        f"\\begin{{document}}\n\\begin{{{env}}}\\centering\n\\includegraphics[width={width}]{{{name}.pdf}}\n"
        f"\\end{{{env}}}\n\\end{{document}}\n"
    )
    stem = f"{name}_{classopts.split(',')[0]}_{width.replace('.', 'p').replace(chr(92), '')}"
    (workdir / f"{stem}.tex").write_text(tex, encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"], cwd=workdir,
                       capture_output=True, text=True, timeout=600)
    pdf = workdir / f"{stem}.pdf"
    if r.returncode != 0 or not pdf.exists():
        raise RuntimeError(f"pdflatex failed for {stem}:\n{r.stdout[-2000:]}")
    return pdf


def spans(pdf: Path):
    doc = fitz.open(pdf)
    out, fonts = [], set()
    for page in doc:
        for f in page.get_fonts(full=True):
            fonts.add(f[3])
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line["spans"]:
                    if sp["text"].strip():
                        out.append((sp["size"], sp["text"].strip(), sp["font"]))
    return out, fonts


def main(argv=None) -> None:
    import sys

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figdir", default=str(S.FIG_DIR), help="directory holding the figure PDFs")
    ap.add_argument("--out", default=str(OUT), help="markdown report to write")
    ap.add_argument("--only", default=None, help="comma-separated subset of figure names")
    args = ap.parse_args(argv)
    figdir, out_path = Path(args.figdir), Path(args.out)
    names = [n for n in FIGS if args.only is None or n in args.only.split(",")]
    present = [n for n in names if (figdir / f"{n}.pdf").exists()]
    missing = [n for n in names if n not in present]

    sys.stdout.reconfigure(encoding="utf-8")
    work = Path(tempfile.mkdtemp(prefix="legib_"))
    rows = []
    try:
        for name in present:
            shutil.copy(figdir / f"{name}.pdf", work / f"{name}.pdf")
        # control: half width must halve the measured size (any present figure serves)
        control = "figA1_targets" if "figA1_targets" in present else present[0]
        full = compile_doc(work, control, "final,5p,times,twocolumn", "figure", "252pt")
        half = compile_doc(work, control, "final,5p,times,twocolumn", "figure", "126pt")
        s_full = min(s for s, _, _ in spans(full)[0])
        s_half = min(s for s, _, _ in spans(half)[0])
        ratio = s_half / s_full
        assert abs(ratio - 0.5) < 0.01, f"span sizes do not carry placement scale (ratio {ratio:.3f})"

        for name in present:
            kind = FIGS[name]
            if kind == "double":
                fin = compile_doc(work, name, "final,5p,times,twocolumn", "figure*", f"{S.TEXTWIDTH_5P_PT}pt")
                pre = compile_doc(work, name, "preprint,12pt,authoryear", "figure", "\\textwidth")
                pre_scale = S.TEXTWIDTH_PREPRINT_PT / S.TEXTWIDTH_5P_PT
            else:
                fin = compile_doc(work, name, "final,5p,times,twocolumn", "figure", f"{S.COLUMNWIDTH_5P_PT}pt")
                pre = compile_doc(work, name, "preprint,12pt,authoryear", "figure", f"{S.COLUMNWIDTH_5P_PT}pt")
                pre_scale = 1.0
            sp_f, fonts = spans(fin)
            sp_p, _ = spans(pre)
            mf = min(sp_f, key=lambda x: x[0])
            mp = min(sp_p, key=lambda x: x[0])
            n_below = sum(1 for s, _, _ in sp_f if s < MIN_PT - 1e-6)
            rows.append({
                "name": name, "kind": kind, "min_final": mf[0], "text_final": mf[1], "n_spans": len(sp_f),
                "n_below": n_below, "min_pre": mp[0], "pre_scale": pre_scale,
                "fonts": sorted({f.split("+")[-1] for f in fonts}),
                "pass": mf[0] >= MIN_PT - 1e-6,
            })
            print(name, kind, f"final min {mf[0]:.2f} pt ({mf[1]!r})", f"preprint min {mp[0]:.2f} pt",
                  "PASS" if rows[-1]["pass"] else "FAIL")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    n_fail = sum(not r["pass"] for r in rows)
    for name in missing:
        print(name, FIGS[name], "not rendered (no PDF in the figure directory)")
    lines = [
        "# Figure legibility",
        "",
        f"Date: {date.today().isoformat()}. Script: `src/figures/check_legibility.py` (PyMuPDF {fitz.VersionBind}). "
        f"Figure directory: `{figdir.as_posix()}`.",
        "",
        "Method: each figure PDF is placed with `\\includegraphics` at its intended width in an elsarticle test "
        "document, compiled with pdflatex, and every text span on the page is read with PyMuPDF. Span sizes include "
        f"the placement scale (control: {control} at half width measured {ratio:.3f} of full width). "
        f"Fail criterion: smallest span < {MIN_PT:g} pt in the final layout.",
        "",
        "Layouts (measured with pdflatex, 2026-09-16): final `[final,5p,times,twocolumn]` textwidth 522.0pt, "
        "columnwidth 252.0pt; submission `[preprint,12pt,authoryear]` textwidth 390.0pt.",
        "",
        "| Figure | Width | Spans | Smallest span, final (pt) | Smallest text | Spans < 7 pt | Preprint min (pt), scale | "
        "Embedded fonts | Result |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['kind']} | {r['n_spans']} | {r['min_final']:.2f} | `{r['text_final']}` | {r['n_below']} | "
            f"{r['min_pre']:.2f}, {r['pre_scale']:.3f} | {', '.join(r['fonts'])} | {'PASS' if r['pass'] else 'FAIL'} |")
    if missing:
        lines += ["", "Not rendered yet (no PDF in the figure directory, not a failure): "
                  + ", ".join(f"`{m}`" for m in missing) + "."]
    lines += [
        "",
        "Notes:",
        "- Sizes are font sizes as set in the PDF text matrix; sub- and superscripts use a mathtext shrink factor "
        "of 0.8 (style.py), so a 9 pt label carries 7.2 pt scripts.",
        "- In the preprint build a double-column figure scaled to \\textwidth prints at 0.747 of its final size, "
        "so 8 pt text prints near 6 pt there. The final two-column layout is the criterion (CLAUDE.md section 7); "
        "the preprint values are informational.",
        "",
        f"VERDICT: {'PASS' if n_fail == 0 else f'FAIL - {n_fail} figure(s) below {MIN_PT:g} pt'}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("written", out_path, "failures", n_fail, "not rendered", len(missing))


if __name__ == "__main__":
    main()
