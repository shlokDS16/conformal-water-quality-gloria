"""Reconcile every number printed in the abstract, highlights and conclusions against its ledger key.

CLAUDE.md section 2 item 1: no number may be quoted that is not in a ledger file written from the
project's own result files. This checker closes the loop mechanically for the three places a
reviewer reads first, so a ledger change that is not carried into the prose fails the build.

How it works
------------
The manuscript already carries a `% ledger: <keys>` comment next to every quoted number. This
module does not parse those comments (they name key *patterns*, not values). Instead it holds an
explicit table of CLAIMS: for each claim, the printed string as it must appear in `paper/main.tex`,
the ledger keys it comes from, and the rounding to apply. Each claim is checked by resolving the
keys from `tables/{summary,derived,data_summary,h2_power,confirmatory,conditional_shares}.csv`,
formatting them with the stated rule, and requiring the result to equal the printed string AND that
string to be present in the relevant block of `main.tex`.

Adding a number to the abstract or conclusions therefore requires adding a claim here, which is the
point: an unledgered number cannot pass silently.

Run:  python src/check_numbers.py
Exit 0 when every claim reconciles and is present; 1 otherwise.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "tables"
TEX = ROOT / "paper" / "main.tex"
LEDGERS = ["summary.csv", "derived.csv", "data_summary.csv", "h2_power.csv",
           "confirmatory.csv", "conditional_shares.csv"]
TARGETS = ["Chla", "TSS", "aCDOM440", "Secchi_depth"]
PRIMARY = "hyp.waterbody.primary"


def load_ledger(tab: Path | None = None) -> dict[str, float]:
    tab = tab or TAB
    vals: dict[str, float] = {}
    for name in LEDGERS:
        p = tab / name
        if not p.exists():
            continue
        d = pd.read_csv(p, low_memory=False)
        if "key" not in d.columns or "value" not in d.columns:
            continue          # confirmatory.csv is a wide table, not a key ledger
        for k, v in zip(d["key"], d["value"]):
            try:
                vals[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return vals


def fmt(v: float, rule: str) -> str:
    """Format one value. `rule` is `dN` (N decimals), `iN` (integer with thousands separators),
    `sN` (N significant figures), `pN` (percent, N decimals)."""
    kind, n = rule[0], int(rule[1:])
    if kind == "d":
        return f"{v:.{n}f}"
    if kind == "i":
        return f"{round(v):,}" if n else f"{round(v)}"
    if kind == "s":
        if v == 0:
            return "0"
        d = n - 1 - math.floor(math.log10(abs(v)))
        return f"{round(v, d):.{max(d, 0)}f}"
    if kind == "p":
        return f"{v * 100:.{n}f}"
    raise ValueError(rule)


def rng(vals: dict, keys: list[str], rule: str) -> str:
    """The printed "lo to hi" range over a set of keys."""
    xs = [vals[k] for k in keys]
    return f"{fmt(min(xs), rule)} to {fmt(max(xs), rule)}"


def per_target(pattern: str) -> list[str]:
    return [pattern.replace("<t>", t) for t in TARGETS]


def cell(model: str, method: str, stat: str, alpha: str = "a100") -> list[str]:
    return per_target(f"cell.<t>.{PRIMARY}.{model}.{method}.{alpha}.{stat}")


# --------------------------------------------------------------------------- claims
# (id, block, printed string, builder). `block` selects the region of main.tex the string must
# occur in: "abstract", "highlights" or "conclusions".
def claims(vals: dict) -> list[tuple[str, str, str, str]]:
    out = []

    def add(cid, block, printed, computed):
        out.append((cid, block, printed, computed))

    # ---- abstract
    add("abs.n_rows", "abstract", "3,709 to 4,475",
        rng(vals, per_target("t1.<t>.n_rows"), "i1"))
    add("abs.n_wb", "abstract", "250 to 330",
        # printed as a rounded envelope of the per-target water-body counts
        f"{int(math.floor(min(vals[k] for k in per_target('t1.<t>.n_wb_group')) / 10) * 10)} to "
        f"{int(math.ceil(max(vals[k] for k in per_target('t1.<t>.n_wb_group')) / 10) * 10)}")
    add("abs.n_methods", "abstract", "Eleven",
        {11: "Eleven"}[int(vals["txt.n_interval_methods"])])
    add("abs.power90", "abstract", "0.14 to 0.17",
        rng(vals, per_target("h2.<t>.power_at_90_nb_sd"), "d2"))
    add("abs.cov_gsub", "abstract", "0.895 to 0.905",
        rng(vals, cell("lgbm", "scp_gsub", "cov_wb_mean"), "d3"))
    add("abs.cov_mdn", "abstract", "0.676 to 0.748",
        rng(vals, cell("mdn", "native", "cov_wb_mean"), "d3"))
    add("abs.width_chla", "abstract", "factor of 15.5",
        "factor of " + fmt(vals[f"cell.Chla.{PRIMARY}.lgbm.scp_gsub.a100.width_wb_mean"], "d1"))
    add("abs.budget_gsub", "abstract", "to within 0.016",
        "to within " + fmt(math.ceil(vals["budget.group.gsub.max_abs_cov_minus_beta"] * 1e3) / 1e3, "d3"))
    add("abs.budget_all", "abstract", "to within 0.039",
        "to within " + fmt(math.ceil(vals["budget.group.all.max_abs_cov_minus_beta"] * 1e3) / 1e3, "d3"))
    add("abs.worst", "abstract", "0.468 to 0.594",
        rng(vals, cell("lgbm", "scp_gsub", "cov_worst_ge10_mean"), "d3"))

    # ---- highlights
    add("hl.cov_gsub", "highlights", "0.895 to 0.905",
        rng(vals, cell("lgbm", "scp_gsub", "cov_wb_mean"), "d3"))
    add("hl.worst", "highlights", "0.468 to 0.594",
        rng(vals, cell("lgbm", "scp_gsub", "cov_worst_ge10_mean"), "d3"))

    # ---- conclusions
    # (1) MdSA under the random protocol against the water-body protocol, LightGBM point rows.
    for t, printed in (("Chla", "33 to 44\\%"), ("TSS", "46 to 71\\%"),
                       ("aCDOM440", "47 to 71\\%"), ("Secchi_depth", "28 to 39\\%")):
        add(f"con.mdsa_{t}", "conclusions", printed,
            f"{round(vals[f'cell.{t}.hyp.random.primary.lgbm.point.point.mdsa_mean'])} to "
            f"{round(vals[f'cell.{t}.hyp.waterbody.primary.lgbm.point.point.mdsa_mean'])}\\%")
    # (2) Chl-a interval width, random protocol against water-body protocol.
    add("con.width_chla", "conclusions", "from 8.55 to 15.5",
        "from " + fmt(vals["cell.Chla.hyp.random.primary.lgbm.scp_gsub.a100.width_wb_mean"], "d2")
        + " to " + fmt(vals["cell.Chla.hyp.waterbody.primary.lgbm.scp_gsub.a100.width_wb_mean"], "d1"))
    add("con.h2_thresholds", "conclusions", "0.929 to 0.939",
        rng(vals, per_target("h2.<t>.reject_threshold"), "d3"))
    add("con.power90", "conclusions", "0.14 to 0.17",
        rng(vals, per_target("h2.<t>.power_at_90_nb_sd"), "d2"))
    add("con.power90_naive", "conclusions", "0.0025 to 0.0062",
        rng(vals, per_target("h2.<t>.power_at_90_naive_sd"), "s2"))
    add("con.cov_gsub", "conclusions", "0.895 to 0.905",
        rng(vals, cell("lgbm", "scp_gsub", "cov_wb_mean"), "d3"))
    add("con.h1_gauss", "conclusions", "0.894 to 0.924",
        rng(vals, per_target("conf.H1.<t>.arm.lgbm_gauss.estimate"), "d3"))
    # (5) the native-MDN shortfall, stated at the primary level alpha = 0.10 (nominal 0.90).
    short = [(0.90 - vals[k]) * 100 for k in cell("mdn", "native", "cov_wb_mean")]
    add("con.mdn_shortfall", "conclusions", "15 to 22 points",
        f"{round(min(short))} to {round(max(short))} points")
    shares = [vals[f"cond.{t}.lgbm_scp_gsub.share_below_090"] for t in TARGETS]
    add("con.share_below", "conclusions", "24 to 34\\%",
        f"{round(min(shares) * 100)} to {round(max(shares) * 100)}\\%")

    def signed(v: float, rule: str) -> str:
        """LaTeX-safe signed number: a negative value is wrapped in math mode for the minus sign."""
        s = fmt(v, rule)
        return f"$-{s[1:]}$" if s.startswith("-") else s

    add("con.h4", "conclusions", "0.008, 0.022, $-0.005$ and $-0.002$",
        ", ".join(signed(vals[f"conf.H4.{t}.estimate"], "d3") for t in TARGETS[:3])
        + " and " + signed(vals["conf.H4.Secchi_depth.estimate"], "d3"))
    add("con.recal", "conclusions", "0.840 to 0.866",
        rng(vals, cell("mdn", "recal_cal", "cov_wb_mean"), "d3"))
    add("con.worst", "conclusions", "0.468 to 0.594",
        rng(vals, cell("lgbm", "scp_gsub", "cov_worst_ge10_mean"), "d3"))
    add("con.budget_gsub", "conclusions", "within 0.016",
        "within " + fmt(math.ceil(vals["budget.group.gsub.max_abs_cov_minus_beta"] * 1e3) / 1e3, "d3"))
    add("con.budget_all", "conclusions", "within 0.039",
        "within " + fmt(math.ceil(vals["budget.group.all.max_abs_cov_minus_beta"] * 1e3) / 1e3, "d3"))

    # ---- V3: the corrected H2 cause sentence, in Results and in Limitations
    add("h2.dist_nominal", "body", "0.029 to 0.039",
        rng(vals, per_target("h2.<t>.threshold_minus_nominal"), "d3"))
    add("h2.dist_nominal_se_nb", "body", "1.02 to 1.14",
        rng(vals, per_target("h2.<t>.threshold_minus_nominal_over_se_nb"), "d2"))
    add("h2.dist_nominal_se_naive", "body", "2.50 to 2.81",
        rng(vals, per_target("h2.<t>.threshold_minus_nominal_over_se_naive"), "d2"))
    add("h2.dist_margin_se_nb", "body", "1.73",
        fmt(min(vals[k] for k in per_target("h2.<t>.threshold_minus_margin_over_se_nb")), "d2"))
    add("h2.threshold_secchi", "body", "0.939, 0.929, 0.934 and 0.929",
        ", ".join(fmt(vals[f"h2.{t}.reject_threshold"], "d3") for t in TARGETS[:3])
        + " and " + fmt(vals["h2.Secchi_depth.reject_threshold"], "d3"))

    # ---- V8 / V9: the Appendix A numbers that had no key
    add("appA.qc_strict", "body", "5,793", fmt(vals["txt.all.n_qc_strict"], "i1"))
    add("appA.raw_names", "body", "486 raw site names",
        fmt(vals["txt.all.n_wb_raw_names"], "i0") + " raw site names")
    add("appA.norm_names", "body", "473 normalized names",
        fmt(vals["txt.all.n_wb_norm_names"], "i0") + " normalized names")
    add("appA.wb_max", "body", "largest group holds 614 samples",
        "largest group holds " + fmt(vals["txt.all.wb_group_max_samples"], "i0") + " samples")
    add("appA.wb_median", "body", "the median holds 4",
        "the median holds " + fmt(vals["txt.all.wb_group_median_samples"], "i0"))
    add("appA.oa1", "body", "which only 37\\% of quality-controlled spectra reach",
        "which only " + fmt(vals["txt.srf.frac_spectra.Oa1"], "p0")
        + "\\% of quality-controlled spectra reach")
    add("appA.oa12", "body", "which 88\\% reach",
        "which " + fmt(vals["txt.srf.frac_spectra.Oa12"], "p0") + "\\% reach")
    return out


BLOCK_RE = {
    "abstract": r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
    "highlights": r"\\begin\{highlights\}(.*?)\\end\{highlights\}",
    "conclusions": r"\\section\{Conclusions\}(.*?)(?=\\section\*?\{)",
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default=None,
                    help="directory holding the ledger CSVs (default: tables/). Point it at a "
                         "release tree to prove the deposit can reproduce every quoted number.")
    args = ap.parse_args()
    vals = load_ledger(Path(args.tables) if args.tables else None)
    tex = TEX.read_text(encoding="utf-8")
    blocks = {"body": tex}
    for name, pat in BLOCK_RE.items():
        m = re.search(pat, tex, re.S)
        blocks[name] = m.group(1) if m else ""
        if not m:
            print(f"  WARN could not locate the {name} block in main.tex")

    fails = 0
    for cid, block, printed, computed in claims(vals):
        ok_val = printed == computed
        ok_txt = printed in blocks[block]
        status = "ok" if (ok_val and ok_txt) else "FAIL"
        if status == "FAIL":
            fails += 1
        detail = ""
        if not ok_val:
            detail += f"  ledger gives {computed!r}"
        if not ok_txt:
            detail += f"  not found in the {block} block"
        print(f"  {status:4s} {cid:26s} {printed!r}{detail}")

    print(f"\nchecked {len(claims(vals))} claims; failures {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
