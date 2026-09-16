"""Power ledger for the preregistered non-inferiority test H2 (AUDIT_4 finding A4-2).

AUDIT_4 A4-2 requires the Results section to state the H2 outcome as "non-inferiority was not
established" and to put the power of the test, its cause and the supporting descriptive evidence
next to that statement. The audit computed those quantities in `audit/audit4/power_check.py`, which
prints most of them to the console instead of writing a file, so none of them carries a ledger key.
CLAUDE.md section 2 item 1 forbids quoting a number that is not in a ledger file, so this module
recomputes them from the per-split metrics files and writes `tables/h2_power.csv` in the ledger
schema (key, value, rounding, source_file, filter).

Nothing here is a hypothesis test of its own. The Nadeau and Bengio (2003) test itself stays in the
frozen `src/analysis/confirmatory.py`; this module only reads its output and describes what that
test could and could not have detected. Every quantity written here is exploratory and must be
labeled as such in the manuscript, except the estimate, interval and p-values that are copied from
`tables/confirmatory.csv`.

Definitions
-----------
* `sd_splits`     sample standard deviation of the B per-split values of water-body-averaged
                  coverage, B = 20 seeds of the primary cell.
* `se_naive`      sd_splits / sqrt(B), the uncorrected standard error.
* `se_nb`         the Nadeau-Bengio corrected standard error sqrt[(1/B + n_test/n_train) s^2],
                  read from `tables/confirmatory.csv`.
* `nb_inflation`  se_nb / se_naive.
* `ess`           1 / (1/B + n_test/n_train), the sample size at which the naive standard error
                  would equal the corrected one ("effective number of repeats").
* `reject_threshold`  the smallest mean coverage that would have rejected the null hypothesis of
                  H2 at the one-sided 5 % level, 0.88 + t_{0.95, B-1} se_nb.
* `power_at_X`    Pr(mean coverage > reject_threshold) when the true mean is X and the per-split
                  standard deviation is sd_splits, under a normal approximation with the naive
                  standard error. This is the power the design had, not a claim about the truth.
* `naive_t`, `naive_p`  the same one-sided test with the uncorrected standard error. EXPLORATORY
                  contrast only: it ignores the dependence between overlapping training sets.
* `n_splits_ge_088`, `n_splits_ge_090`, `split_min`, `split_max`  per-split tallies.

Run:  python -m src.analysis.h2_power
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
CORE = "results/core_v2"
CELL = dict(protocol="waterbody", sensor="hyp", population="primary", model="lgbm",
            method="scp_gsub", alpha=0.10)
NULL = 0.88  # 1 - alpha - delta (PREREGISTRATION section 4 and 5)


def per_split_coverage(met: pd.DataFrame, target: str) -> np.ndarray:
    d = met[(met.target == target) & (met.protocol == CELL["protocol"]) & (met.sensor == CELL["sensor"])
            & (met.population == CELL["population"]) & (met.model == CELL["model"])
            & (met.method == CELL["method"]) & (np.abs(met.alpha - CELL["alpha"]) < 1e-9)
            & (~met.descriptive_only.astype(bool))]
    return d.sort_values(["seed", "fold"])["cov_wb"].to_numpy(float)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=CORE)
    ap.add_argument("--out", default=str(ROOT / "tables" / "h2_power.csv"))
    args = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / args.results / "metrics_*.csv")))
    if not files:
        raise SystemExit(f"no metrics files under {args.results}")
    met = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    met["population"] = met["population"].fillna("primary")
    conf = pd.read_csv(ROOT / "tables" / "confirmatory.csv")

    src = f"{args.results}/metrics_*.csv + tables/confirmatory.csv"
    filt = ("target.hyp.waterbody.primary.lgbm.scp_gsub, alpha=0.10, B=20 seeds; "
            "EXPLORATORY: power, effective repeats and naive tests are not preregistered")
    rows = []

    def add(key, value, rounding):
        rows.append(dict(key=key, value=value, rounding=rounding, source_file=src, filter=filt))

    for t in TARGETS:
        v = per_split_coverage(met, t)
        c = conf[(conf.hypothesis == "H2") & (conf.target == t)].iloc[0]
        B = int(len(v))
        s = float(v.std(ddof=1))
        se_naive = s / np.sqrt(B)
        se_nb = float(c["se"])
        n_te, n_tr = float(c["n_test_mean"]), float(c["n_train_mean"])
        tcrit = float(stats.t.ppf(0.95, B - 1))
        thr = NULL + tcrit * se_nb
        b = f"h2.{t}."
        add(b + "n_splits", B, "integer")
        add(b + "mean_cov_wb", float(v.mean()), "3 decimals")
        add(b + "sd_splits", s, "4 decimals")
        add(b + "se_naive", float(se_naive), "4 decimals")
        add(b + "se_nb", se_nb, "4 decimals")
        add(b + "nb_inflation", float(se_nb / se_naive), "2 decimals")
        add(b + "n_test_wb_mean", n_te, "1 decimal")
        add(b + "n_train_wb_mean", n_tr, "1 decimal")
        add(b + "test_train_ratio", float(n_te / n_tr), "4 decimals")
        add(b + "ess_repeats", float(1.0 / (1.0 / B + n_te / n_tr)), "1 decimal")
        add(b + "reject_threshold", float(thr), "4 decimals")
        for mu in (0.90, 0.92):
            p = float(stats.norm.sf(thr, loc=mu, scale=se_naive))
            add(b + f"power_at_{int(round(mu * 100))}", p, "3 significant figures")
        tt, pp = stats.ttest_1samp(v, NULL, alternative="greater")
        add(b + "naive_t", float(tt), "3 significant figures")
        add(b + "naive_p", float(pp), "3 significant figures")
        add(b + "n_splits_ge_088", int((v >= 0.88).sum()), "integer")
        add(b + "n_splits_ge_090", int((v >= 0.90).sum()), "integer")
        add(b + "split_min", float(v.min()), "3 decimals")
        add(b + "split_max", float(v.max()), "3 decimals")

    out = pd.DataFrame(rows, columns=["key", "value", "rounding", "source_file", "filter"])
    if out["key"].duplicated().any():
        raise SystemExit("duplicate key in h2_power ledger")
    Path(args.out).write_text(out.to_csv(index=False, lineterminator="\n"), encoding="utf-8",
                              newline="")
    print(f"[h2_power] wrote {len(out)} ledger rows to {args.out} from {len(files)} metrics files")


if __name__ == "__main__":
    main()
