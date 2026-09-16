"""Build Fig. 1 from the TikZ source (phase 08 rework).

Compiles src/figures/fig01_workflow_tikz.tex with pdflatex in a scratch directory and
copies the result to figures/fig01_workflow.pdf, then renders figures/fig01_workflow.png
at 600 dpi with PyMuPDF. The output names are unchanged, so paper/main.tex needs no edit.

Run: python -m src.figures.build_fig01
Single process, no GPU, nothing in results/ or logs/ is touched.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "src" / "figures" / "fig01_workflow_tikz.tex"
OUT_PDF = ROOT / "figures" / "fig01_workflow.pdf"
OUT_PNG = ROOT / "figures" / "fig01_workflow.png"
DPI = 600
TEX_PT_PER_IN = 72.27
TARGET_W_PT = 522.0  # DOUBLE_COL in TeX pt (src/figures/style.py)


def build() -> tuple[float, float]:
    work = Path(tempfile.mkdtemp(prefix="fig01_tikz_"))
    try:
        shutil.copy(TEX, work / TEX.name)
        # lualatex embeds the system Arial through fontspec, so Fig. 1 matches the font of the
        # matplotlib figures; pdflatex (Helvetica metrics) stays as the fallback engine.
        engine = "lualatex" if shutil.which("lualatex") else "pdflatex"
        r = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", TEX.name],
            cwd=work, capture_output=True, text=True, timeout=900)
        pdf = work / (TEX.stem + ".pdf")
        if r.returncode != 0 or not pdf.exists():
            sys.stdout.write(r.stdout[-4000:])
            raise SystemExit(f"{engine} failed")
        print(f"[fig01] engine: {engine}")
        OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, OUT_PDF)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    doc = fitz.open(OUT_PDF)
    page = doc[0]
    w_bp, h_bp = page.rect.width, page.rect.height
    page.get_pixmap(dpi=DPI).save(OUT_PNG)
    doc.close()
    # bp -> TeX pt
    return w_bp * 72.27 / 72.0, h_bp * 72.27 / 72.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    w_pt, h_pt = build()
    print(f"{OUT_PDF.name}: {w_pt:.2f} x {h_pt:.2f} TeX pt "
          f"= {w_pt / TEX_PT_PER_IN:.3f} x {h_pt / TEX_PT_PER_IN:.3f} in")
    print(f"placement scale at DOUBLE_COL: {TARGET_W_PT / w_pt:.5f}")
    print(f"{OUT_PNG.name}: written at {DPI} dpi")
    if abs(w_pt - TARGET_W_PT) > 0.5:
        raise SystemExit(f"natural width {w_pt:.2f} pt differs from DOUBLE_COL {TARGET_W_PT} pt")
    if h_pt / TEX_PT_PER_IN > 3.6:
        raise SystemExit(f"height {h_pt / TEX_PT_PER_IN:.3f} in exceeds the 3.6 in budget")


if __name__ == "__main__":
    main()
