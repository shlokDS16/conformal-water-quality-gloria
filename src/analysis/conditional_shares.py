"""Ledger of per-water-body coverage shares (AUDIT_4 finding A4-7).

AUDIT_4 A4-7 makes conditional coverage the substantive result of the study and requires the
manuscript to lead with it. The worst-water-body statistic is already in `tables/summary.csv`, but
the share of test water bodies whose own coverage falls below the nominal level is only drawn, in
Fig. A.3, and CLAUDE.md section 2 item 1 forbids reading a number off a figure. This module
recomputes that distribution from the per-split prediction files and writes
`tables/conditional_shares.csv` in the ledger schema (key, value, rounding, source_file, filter).

Cell: water-body protocol, hyperspectral features, primary population, alpha = 0.10, seeds 0 to 19.
A test water body enters the distribution when it has at least `MIN_N` test samples in that split,
the same rule as Fig. A.3, and each (split, water body) pair counts once. Every quantity here is
exploratory (PREREGISTRATION section 5) and descriptive: no method claims conditional validity, and
distribution-free conditional coverage is impossible in general (Foygel Barber et al., 2021).

Run:  python -m src.analysis.conditional_shares
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.figures import resultsio as R
from src.figures import style as S

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["Blue-peaked", "Green-peaked", "Red-peaked"]
CLASS_TAG = {"Blue-peaked": "blue", "Green-peaked": "green", "Red-peaked": "red"}
SENSOR, PROTOCOL, ALPHA = "hyp", "waterbody", 0.10
SEEDS = range(20)
MIN_N = 5
SHOWN = [("mdn", "native", "results/core_v2"), ("lgbm", "gauss", "results/core_v2"),
         ("lgbm", "scp_pool", "results/core_v2"), ("lgbm", "scp_gsub", "results/core_v2"),
         ("lgbm", "cqr_gsub", "results/core_v2"), ("lgbm", "cvplus", "results/cvplus_v3")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "tables" / "conditional_shares.csv"))
    args = ap.parse_args(argv)

    rows = []
    cls_of = R.spectral_class(S.load_gloria(["GLORIA_ID"])["GLORIA_ID"].to_numpy())
    for target in R.TARGETS:
        for model, method, tag in SHOWN:
            p = R.predictions(target, SENSOR, PROTOCOL, SEEDS, tag=tag, methods=[method])
            if not len(p):
                continue
            p = p[(p["model"] == model) & np.isclose(p["alpha"].astype(float), ALPHA)]
            if not len(p):
                continue
            hit = ((p["y"] >= p["lower"]) & (p["y"] <= p["upper"])).astype(float)
            # Coverage within the spectral display class of Fig. A.4 (exploratory; the class is a
            # display rule from the wavelength of the spectral maximum, not an optical water type,
            # and no model uses it).
            cls = cls_of.reindex(p["GLORIA_ID"].to_numpy()).to_numpy()
            dcls = pd.DataFrame({"cls": cls, "hit": hit.to_numpy()}).dropna(subset=["cls"])
            gcls = dcls.groupby("cls", observed=False)["hit"].agg(["mean", "size"]).reindex(CLASSES)
            for cname in CLASSES:
                if not np.isfinite(gcls.loc[cname, "mean"]):
                    continue
                kb = f"cond.{target}.{model}_{method}.class_{CLASS_TAG[cname]}."
                cf = (f"{SENSOR}, {PROTOCOL}, primary, alpha={ALPHA}, seeds 0-19, pooled test rows "
                      f"of the {cname.lower()} display class; EXPLORATORY")
                rows.append(dict(key=kb + "cov", value=float(gcls.loc[cname, "mean"]),
                                 rounding="3 decimals", source_file=f"{tag}/pred_*.parquet",
                                 filter=cf))
                rows.append(dict(key=kb + "n_rows", value=float(gcls.loc[cname, "size"]),
                                 rounding="integer", source_file=f"{tag}/pred_*.parquet",
                                 filter=cf))
            g = p.assign(hit=hit).groupby(["seed", "wb_group"], observed=True)["hit"] \
                 .agg(["mean", "size"])
            g = g[g["size"] >= MIN_N]
            v = g["mean"].to_numpy(float)
            base = f"cond.{target}.{model}_{method}."
            src = f"{tag}/pred_*.parquet"
            filt = (f"{SENSOR}, {PROTOCOL}, primary, alpha={ALPHA}, seeds 0-19, "
                    f"water bodies with at least {MIN_N} test samples; one entry per split and "
                    f"water body; EXPLORATORY")
            out = {"n_body_splits": (float(len(v)), "integer"),
                   "median_body_cov": (float(np.median(v)), "3 decimals"),
                   "share_below_090": (float(np.mean(v < 0.90)), "3 decimals"),
                   "share_below_080": (float(np.mean(v < 0.80)), "3 decimals"),
                   "share_below_050": (float(np.mean(v < 0.50)), "3 decimals"),
                   "share_zero": (float(np.mean(v <= 0.0)), "3 decimals")}
            for stat, (value, rounding) in out.items():
                rows.append(dict(key=base + stat, value=value, rounding=rounding,
                                 source_file=src, filter=filt))

    d = pd.DataFrame(rows, columns=["key", "value", "rounding", "source_file", "filter"])
    if d["key"].duplicated().any():
        raise SystemExit("duplicate key in conditional-shares ledger")
    Path(args.out).write_text(d.to_csv(index=False, lineterminator="\n"), encoding="utf-8",
                              newline="")
    print(f"[conditional] wrote {len(d)} ledger rows to {args.out}")


if __name__ == "__main__":
    main()
