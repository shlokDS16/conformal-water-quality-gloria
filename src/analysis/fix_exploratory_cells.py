"""Rebuild tables/exploratory_cells.csv so that CV+ rows obey PREREGISTRATION Amendment 5 item 2.

AUDIT_4 (BLOCKING) found that running the frozen src/analysis/confirmatory.py with both result
directories concatenates them without a tag filter, so its exploratory cell table averaged the
withdrawn unpermuted core_v2 CV+ rows with the corrected cvplus_v3 rows. The confirmatory tests,
tables/summary.csv, the result tables and the figures were unaffected (verified in AUDIT_4).

The frozen script is not modified. It is run twice into temporary directories, and the CV+ rows are
taken from the cvplus_v3 run only, every other row from the core_v2 run. Run:

    python -m src.analysis.fix_exploratory_cells
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tables" / "exploratory_cells.csv"


def run(results: str, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.analysis.confirmatory", "--results", results,
           "--budget", "results/budget_v2", "--out", str(out_dir)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        sys.stdout.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"confirmatory failed for {results}")
    return pd.read_csv(out_dir / "exploratory_cells.csv", low_memory=False)


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="exploratory_fix_"))
    try:
        core = run("results/core_v2", work / "core")
        cvp = run("results/cvplus_v3", work / "cvplus")
    finally:
        keep = work
    base = core[core["method"] != "cvplus"].copy()
    cv = cvp[cvp["method"] == "cvplus"].copy()
    out = pd.concat([base, cv], ignore_index=True).sort_values(
        ["target", "sensor", "protocol", "population", "model", "method", "alpha"]).reset_index(drop=True)
    assert (out[out["method"] == "cvplus"].shape[0] == cv.shape[0]), "cvplus rows lost"
    out.to_csv(OUT, index=False)
    shutil.rmtree(keep, ignore_errors=True)
    print(f"[fix] wrote {OUT}: {len(out)} rows "
          f"({len(cv)} cvplus rows from cvplus_v3, {len(base)} other rows from core_v2)")


if __name__ == "__main__":
    main()
