"""Build the one supplementary file the manuscript promises (REVIEW_final F3).

The note under Table 5 states that log-MAE, log-bias and log-RMSE for the same cells are in a
supplementary CSV. Those numbers exist in `tables/summary.csv` but were never emitted as a file,
so the manuscript promised something the submission did not carry, against the Guide's rule that
"If your article includes any Supplementary material, this should be included in your initial
submission for peer review purposes."

This script writes exactly the cells Table 5 prints - water-body protocol, primary population,
the three sensor configurations, the eight point models, the four targets - with the three
logarithmic error statistics the note names, plus the MdSA and SSPB that Table 5 itself shows so
the file stands alone, and the repeat count of each cell. Values are copied from
`tables/summary.csv` at full stored precision rather than at the rounding the printed table uses,
because a data file should not throw away digits; nothing is recomputed here and
`tables/summary.csv` is only read.

Output: `submission/supplementary/Supplementary_Table_S1_point_errors.csv`
Run:    python -m src.make_supplementary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.make_tables_results import (TAB, POINT_MODELS, SENSOR_NAME, TABLE3_SENSORS,
                                     TARGETS, MODEL_NAME, Summary, cell_key)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "supplementary" / "Supplementary_Table_S1_point_errors.csv"
STATS = [("mdsa_mean", "MdSA (%)"), ("sspb_mean", "SSPB (%)"),
         ("logmae_mean", "log-MAE"), ("logbias_mean", "log-bias"), ("logrmse_mean", "log-RMSE"),
         ("n_splits", "repeats")]
TARGET_NAME = {"Chla": "Chl-a", "TSS": "TSS", "aCDOM440": "aCDOM(440)",
               "Secchi_depth": "Secchi depth"}


def build(summary: Path, out: Path) -> pd.DataFrame:
    S = Summary(summary)
    rows = []
    for sensor in TABLE3_SENSORS:
        for model in POINT_MODELS:
            for target in TARGETS:
                base = cell_key(target, sensor, "waterbody", "primary", model, "point", "point", "")
                vals = {label: S.num(base + stat) for stat, label in STATS}
                if all(v is None for v in vals.values()):
                    continue          # this model does not apply to this target
                rows.append({"sensor": SENSOR_NAME[sensor], "model": MODEL_NAME[model],
                             "target": TARGET_NAME[target], **vals})
    d = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(d.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="")
    return d


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(TAB / "summary.csv"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    d = build(Path(a.summary), Path(a.out))
    print(f"[supplementary] wrote {len(d)} rows to {a.out}")


if __name__ == "__main__":
    main()
