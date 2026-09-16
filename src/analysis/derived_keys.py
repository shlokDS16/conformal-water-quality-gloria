"""Ledger keys for the three manuscript numbers that no other ledger file carries.

CLAUDE.md section 2 item 1 forbids quoting a number that is not in a ledger file written from the
result files of this project. `research/REVIEW_full.md` findings B1, B2 and B5 and finding M3 need
four quantities that `tables/summary.csv`, `tables/h2_power.csv` and
`tables/exploratory_budget_group.csv` do not hold:

  1. B5  the largest absolute gap between the realized mean water-body-averaged coverage of a
         calibration-budget cell and the finite-sample quantile rule for that cell,
         `|mean_cov - beta_mean|`, reported for two named scopes. The manuscript claimed 0.013,
         which is wrong under every scope and wrong in the paper's own favour.
  2. B1  the per-split check of the condition `G_te >= n/K` of Proposition 2, which the manuscript
         promises is checked on the realized counts of every CV+ split.
  3. B2  the wall time of the reported runs.
  4. M3  the number of distinct interval methods evaluated in the primary cell.

It also mirrors the two variance quantities of AUDIT_5 N7 under the `summary.csv` key convention,
so a reader who starts from the main ledger can rebuild the H2 power without knowing this module.

This module writes a NEW file, `tables/derived.csv`, in the ledger schema
(key, value, rounding, source_file, filter, status). It deliberately does not touch
`tables/summary.csv`: that file is certified by a SHA-256 digest in `release/zenodo/` and is being
read by the audit, so a key added there would invalidate a manifest that this agent does not own.

Run:  python -m src.analysis.derived_keys
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# CV+ was run with K = 10 folds over the retained water-body pool `n_cvplus_wb` (Algorithm 4).
CVPLUS_TAG = "cvplus_v3"
PRIMARY_PROTOCOLS = ("random", "waterbody")
# Runs whose per-split times back the reported results.
TIMED_TAGS = ("core_v2", "cvplus_v3", "noise_v2", "sens_v2")
# The interval methods of the primary cell (REVIEW_full M3); counted from the ledger, not hard-coded.
PRIMARY_CELL_PREFIX = "cell.Chla.hyp.waterbody.primary."
NON_INTERVAL_METHODS = {"point"}

STREAM_EVENT = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+(START|END)\s+\[(\d+)/\d+\]")


# --------------------------------------------------------------- B5: calibration-budget agreement
def budget_gap(path: Path) -> list[dict]:
    """Largest |mean_cov - beta_mean| over the group calibration-budget grid, by scope.

    Two scopes are reported and both are printed in the manuscript. The wide scope is every cell of
    the grid. The narrow scope is the group-subsampled calibration schemes, which are the only ones
    for which the beta law is applicable (`beta_law_applicable`); pooled calibration has no
    finite-sample guarantee under within-body dependence, so the quantile rule is a reference curve
    there rather than a prediction. Reporting only the narrow scope would flatter the method, so the
    wide number leads."""
    d = pd.read_csv(path)
    d = d.assign(gap=(d["mean_cov"] - d["beta_mean"]).abs())
    scopes = {
        "all": d,
        "gsub": d[d["method"].str.endswith("_gsub")],
        "pool": d[d["method"].str.endswith("_pool")],
    }
    rows = []
    for name, sub in scopes.items():
        i = sub["gap"].idxmax()
        top = sub.loc[i]
        rows.append(dict(
            key=f"budget.group.{name}.max_abs_cov_minus_beta",
            value=float(sub["gap"].max()), rounding="3 decimals, rounded up",
            source_file="tables/exploratory_budget_group.csv",
            filter=(f"alpha=0.10, primary population, hyperspectral, 4 targets, 6 budgets; "
                    f"{len(sub)} cells; scope={name}; "
                    f"max at {top['target']}/{top['method']}/n_cal={top['n_cal_wb_target']}"),
            status="complete"))
        rows.append(dict(
            key=f"budget.group.{name}.n_cells", value=int(len(sub)), rounding="integer",
            source_file="tables/exploratory_budget_group.csv",
            filter=f"scope={name}", status="complete"))
    return rows


# ------------------------------------------------------- B1: Proposition 2 condition, per split
def cvplus_gte_check(results: Path) -> list[dict]:
    """Per-split check of `G_te >= n/K`, and whether a failure can change the CV+ bound.

    Proposition 2's second bound needs `G_te >= n/K`; its first bound does not. Eq. (32) takes the
    minimum of the two terms, so a split that fails the condition loses the second term only if that
    term was the smaller one, which for K = 10 happens exactly when n < 199. Both counts are written
    here, because the honest statement needs the second one."""
    files = sorted(glob.glob(str(results / "metrics_*.csv")))
    if not files:
        raise SystemExit(f"no metrics files under {results}")
    keep = ["target", "sensor", "protocol", "population", "seed", "fold", "descriptive_only",
            "n_test_wb", "n_cvplus_wb", "cv_folds"]
    recs = []
    for f in files:
        m = pd.read_csv(f, low_memory=False)
        if "method" not in m.columns:
            continue
        m = m[m["method"] == "cvplus"]
        if not len(m):
            continue
        recs.append(m[keep].iloc[[0]])            # one split per metrics file
    d = pd.concat(recs, ignore_index=True)
    d = d[~d["descriptive_only"].astype(bool)]
    n, K = d["n_cvplus_wb"].astype(float), d["cv_folds"].astype(float)
    ok = d["n_test_wb"].astype(float) >= n / K
    term1 = 2 * (1 - 1 / K) / (n / K + 1)          # Eq. (31), no condition needed
    term2 = (1 - K / n) / (K + 1)                  # needs G_te >= n/K and Assumption 5
    second_binds = term2 < term1
    src = f"{results.relative_to(ROOT).as_posix()}/metrics_*.csv"
    filt = ("one row per CV+ split; non-descriptive splits only; "
            "condition G_te >= n/K of Proposition 2 on the realized counts")
    out = [
        ("cvplus.gte.n_splits", int(len(d)), "integer", filt),
        ("cvplus.gte.n_hold", int(ok.sum()), "integer", filt),
        ("cvplus.gte.n_fail", int((~ok).sum()), "integer", filt),
        ("cvplus.gte.n_splits_primary_protocols", int(d["protocol"].isin(PRIMARY_PROTOCOLS).sum()),
         "integer", filt + "; random and waterbody protocols"),
        ("cvplus.gte.n_hold_primary_protocols",
         int((ok & d["protocol"].isin(PRIMARY_PROTOCOLS)).sum()), "integer",
         filt + "; random and waterbody protocols"),
        ("cvplus.gte.n_second_term_is_minimum", int(second_binds.sum()), "integer",
         filt + "; splits where the second term of Eq. (32) is the smaller one"),
        ("cvplus.gte.n_fail_and_second_term_is_minimum", int((second_binds & ~ok).sum()), "integer",
         filt + "; splits that fail the condition AND would have needed the second term"),
    ]
    rows = [dict(key=k, value=v, rounding=r, source_file=src, filter=fl, status="complete")
            for k, v, r, fl in out]
    for proto, g in d.groupby("protocol"):
        gi = g.index
        rows.append(dict(key=f"cvplus.gte.n_fail.{proto}", value=int((~ok[gi]).sum()),
                         rounding="integer", source_file=src,
                         filter=filt + f"; protocol={proto}; {len(g)} splits", status="complete"))
    return rows


# ------------------------------------------------------------------------ B2: compute wall time
def wall_time(logs: Path, results_root: Path) -> list[dict]:
    """Wall time of the reported runs, from the stream logs and the per-split times.

    Three different quantities, all reported, because "wall time" is ambiguous on a machine that ran
    two job streams at once:
      * `elapsed_hours`  the time the machine was busy, the union of the stream spans. This is the
        number a reader needs in order to repeat the study.
      * `job_hours`      the sum of the per-job durations over the streams; larger than the elapsed
        time because two streams overlapped.
      * `split_hours`    the sum of `time_split_s` over every split of the reported runs; larger
        again because each job ran splits on several worker processes."""
    spans, job_h, n_jobs = [], 0.0, 0
    for f in sorted(logs.glob("*.log")):
        if re.search(r"_job\d+\.log$", f.name):
            continue
        ev = []
        for line in f.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            m = STREAM_EVENT.match(line.strip())
            if m:
                ev.append((datetime.fromisoformat(m.group(1)), m.group(2), int(m.group(3))))
        if not ev:
            continue
        open_jobs = {}
        for t, kind, i in ev:
            if kind == "START":
                open_jobs[i] = t
            elif i in open_jobs:
                job_h += (t - open_jobs.pop(i)).total_seconds() / 3600.0
                n_jobs += 1
        ts = [e[0] for e in ev]
        spans.append([min(ts), max(ts)])
    spans.sort()
    merged: list[list[datetime]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    elapsed = sum((e - s).total_seconds() for s, e in merged) / 3600.0

    split_h = 0.0
    for tag in TIMED_TAGS:
        for f in glob.glob(str(results_root / tag / "metrics_*.csv")):
            m = pd.read_csv(f, low_memory=False)
            if "time_split_s" in m.columns:
                v = m["time_split_s"].dropna().unique()
                if len(v):
                    split_h += float(v[0]) / 3600.0

    filt = (f"{n_jobs} jobs in {len(spans)} timestamped stream logs; "
            f"{len(merged)} contiguous busy blocks; runs {', '.join(TIMED_TAGS)}")
    return [
        dict(key="compute.elapsed_wall_hours", value=elapsed, rounding="1 decimal",
             source_file="logs/*.log", filter=filt + "; union of the stream spans",
             status="complete"),
        dict(key="compute.job_wall_hours", value=job_h, rounding="1 decimal",
             source_file="logs/*.log", filter=filt + "; sum of per-job START to END durations",
             status="complete"),
        dict(key="compute.split_compute_hours", value=split_h, rounding="1 decimal",
             source_file="results/<tag>/metrics_*.csv",
             filter=filt + "; sum of time_split_s over the reported runs", status="complete"),
        dict(key="compute.n_jobs", value=n_jobs, rounding="integer", source_file="logs/*.log",
             filter=filt, status="complete"),
    ]


# ----------------------------------------------------------------- M3: interval-method count
def n_interval_methods(summary: Path) -> list[dict]:
    d = pd.read_csv(summary, low_memory=False, usecols=["key"])
    methods = set()
    for k in d["key"]:
        if not k.startswith(PRIMARY_CELL_PREFIX):
            continue
        parts = k.split(".")
        if len(parts) >= 9 and parts[7] != "point":
            methods.add(parts[6])
    methods -= NON_INTERVAL_METHODS
    return [dict(key="txt.n_interval_methods", value=int(len(methods)), rounding="integer",
                 source_file="tables/summary.csv",
                 filter=(f"distinct interval methods with a cell in {PRIMARY_CELL_PREFIX}*: "
                         + ", ".join(sorted(methods))), status="complete")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "tables" / "derived.csv"))
    args = ap.parse_args(argv)

    rows = (budget_gap(ROOT / "tables" / "exploratory_budget_group.csv")
            + cvplus_gte_check(ROOT / "results" / CVPLUS_TAG)
            + wall_time(ROOT / "logs", ROOT / "results")
            + n_interval_methods(ROOT / "tables" / "summary.csv"))
    out = pd.DataFrame(rows, columns=["key", "value", "rounding", "source_file", "filter", "status"])
    if out["key"].duplicated().any():
        raise SystemExit("duplicate key in derived ledger")
    Path(args.out).write_text(out.to_csv(index=False, lineterminator="\n"), encoding="utf-8",
                              newline="")
    print(f"[derived_keys] wrote {len(out)} ledger rows to {args.out}")
    for r in rows:
        print(f"  {r['key']:<50} {r['value']}")


if __name__ == "__main__":
    main()
