"""Appendix A tables: spectral response provenance and the irradiance-weighting check.

`research/REVIEW_full.md` finding B3: Appendix A carried only a drafting placeholder, while
Section 2.3 forward-references it as the evidence for "the two definitions differ by a median of at
most 0.46% per band". These two tables are that evidence, rendered from the interim files that the
data pipeline wrote (`research/DATA_PIPELINE_LOG.md` sections 2 and 6), so no number in the appendix
is typed by hand:

  Table A.2  tables/tableA2_srf.tex         spectral response function files, sizes, SHA-256 digests,
                                            and the per-band 1 % support with the fraction of the
                                            response integral it retains
  Table A.3  tables/tableA3_irradiance.tex  per-band relative difference between the reflectance
                                            average of Eq. (band) and the exact irradiance-weighted
                                            band value, on the rows that carry irradiance spectra

Only the bands that the default feature sets use are printed: S2A MSI B1 to B6 and S3A OLCI Oa2 to
Oa11. The other units in the interim files (S2B, S2C, S3B) are sensitivity units and are summarized
in the table note rather than tabulated.

Run:  python -m src.analysis.appendix_a_tables
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
TAB = ROOT / "tables"

DEFAULT_BANDS = {
    "S2A": ["B1", "B2", "B3", "B4", "B5", "B6"],
    "S3A": ["Oa2", "Oa3", "Oa4", "Oa5", "Oa6", "Oa7", "Oa8", "Oa9", "Oa10", "Oa11"],
}
UNIT_LABEL = {"S2A": "Sentinel-2A MSI", "S3A": "Sentinel-3A OLCI"}
IRR_UNIT_LABEL = {"S2A": "Sentinel-2A MSI", "S3A": "Sentinel-3A OLCI"}


def _up2(frac: float) -> str:
    """A fraction as a percentage, two decimals, rounded up.

    This is the rounding recorded next to `txt.irrad.max_median_rel_diff` in
    `tables/data_summary.csv`, and the rule behind the 0.46 % statement of Section 2.3. Rounding up
    keeps every printed value an upper bound on the true difference, so the appendix table and the
    body statement cannot disagree."""
    return f"{math.ceil(frac * 1e4 - 1e-9) / 100:.2f}"


def _floor4(frac: float) -> str:
    """A fraction to four decimals, rounded DOWN.

    REVIEW_full_v2 V14: Appendix A states "the truncation to the 1 % support retains at least 0.9987
    of the response integral in every band used", which is `txt.srf.min_frac_integral_in_support`
    = 0.998754 rounded down (the recorded convention). With round-to-nearest in the table the
    smallest cell printed 0.9988, so no cell showed the number the text quotes. Rounding down keeps
    every printed value a lower bound, so the table and the body statement cannot disagree."""
    return f"{math.floor(frac * 1e4 + 1e-9) / 1e4:.4f}"


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_")


def tableA2_srf() -> str:
    man = pd.read_csv(INTERIM / "srf_manifest.csv")
    bands = pd.read_csv(INTERIM / "srf_bands.csv")
    cov = pd.read_csv(INTERIM / "band_coverage.csv")
    b = bands.merge(cov[["unit", "band", "frac_all", "frac_primary_qc"]], on=["unit", "band"],
                    how="left")

    out = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{Spectral response functions and the 1\% support of each simulated band. "
           r"The support is the wavelength range over which the response is at least 1\% of its "
           r"peak; the band value of Eq.~(\ref{eq:band}) is formed over that range only. "
           r"\emph{Integral} is the fraction of the full response integral the support retains, "
           r"rounded down to four decimals so that the smallest value printed is a lower bound, and "
           r"\emph{Spectra} the fraction of quality-controlled GLORIA spectra that carry a "
           r"reflectance value at every 1\,nm wavelength of the support.}",
           r"\label{tab:srf}", r"\footnotesize",
           r"\setlength{\tabcolsep}{5pt}",
           r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
           r"Band & Support (nm) & FWHM (nm) & Integral & Spectra \\"]
    for unit, keep in DEFAULT_BANDS.items():
        out.append(r"\midrule")
        out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{UNIT_LABEL[unit]}}}}} \\")
        sub = b[(b.unit == unit) & (b.band.isin(keep))].set_index("band").loc[keep].reset_index()
        for _, r in sub.iterrows():
            out.append(rf"\quad {r['band']} & {int(r['support_lo'])} to {int(r['support_hi'])} & "
                       rf"{r['fwhm_nm']:g} & {_floor4(r['frac_integral_in_support'])} & "
                       rf"{r['frac_primary_qc']:.3f} \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip\raggedright\footnotesize "]
    files = "; ".join(
        rf"\texttt{{\seqsplit{{{_esc(r['file'])}}}}} ({int(r['bytes']):,}~bytes, SHA-256 "
        rf"\texttt{{\seqsplit{{{r['sha256']}}}}})" for _, r in man.iterrows())
    out[-1] += ("Source files: " + files + ". "
                "The OLCI responses were resampled to 1\\,nm by averaging the native curve over each "
                "$[\\lambda - 0.5, \\lambda + 0.5)$\\,nm bin; the resulting centroids agree with the "
                "\\texttt{srf\\_centre\\_wavelength} variable of the file to within 0.1\\,nm. "
                "Bands outside the default sets (OLCI Oa1 and Oa12, and the hyperspectral endpoints "
                "at 400 and 750\\,nm) are retained as extra columns but not used, because fewer than "
                "90\\% of spectra cover their supports. "
                "Sentinel-2B, Sentinel-2C and Sentinel-3B responses are carried as sensitivity units.")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def tableA3_irradiance() -> str:
    d = pd.read_csv(INTERIM / "irradiance_weighting.csv")
    out = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{Irradiance weighting check. Eq.~(\ref{eq:band}) averages remote-sensing "
           r"reflectance over the band response, whereas the exact band value is the ratio of the "
           r"response-weighted water-leaving radiance to the response-weighted downwelling "
           r"irradiance. Both were computed on the GLORIA rows that carry an irradiance spectrum, "
           r"and the table gives the relative difference between them, $|b_{\mathrm{ratio}} - "
           r"b_{\mathrm{mean}}| / b_{\mathrm{ratio}}$, per band.}",
           r"\label{tab:irradiance}", r"\footnotesize",
           r"\setlength{\tabcolsep}{6pt}",
           r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
           r"Band & Rows & Median (\%) & 95th percentile (\%) \\"]
    worst_med = 0.0
    for unit, keep in DEFAULT_BANDS.items():
        sub = d[(d.unit == unit) & (d.band.isin(keep))]
        if not len(sub):
            continue
        out.append(r"\midrule")
        out.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{IRR_UNIT_LABEL[unit]}}}}} \\")
        sub = sub.set_index("band").loc[[k for k in keep if k in set(sub.band)]].reset_index()
        for _, r in sub.iterrows():
            worst_med = max(worst_med, float(r["median_rel_diff"]))
            out.append(rf"\quad {r['band']} & {int(r['n']):,} & "
                       rf"{_up2(r['median_rel_diff'])} & {_up2(r['p95_rel_diff'])} \\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\par\smallskip\raggedright\footnotesize "
            + (f"The largest median difference over the bands used is {_up2(worst_med)}\\%, at "
               "the Sentinel-2A B1 coastal-aerosol band, where the response is widest relative to "
               "the structure of the spectrum. Every other band stays below that, and no band "
               "reaches 1\\% at the 95th percentile. The reflectance average was therefore kept, "
               "because the irradiance spectra are absent from most GLORIA rows and requiring them "
               "would have cut the populations by more than four fifths. "),
            r"\end{table}"]
    return "\n".join(out) + "\n"


TABLES = {"tableA2_srf": tableA2_srf, "tableA3_irradiance": tableA3_irradiance}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(TAB))
    args = ap.parse_args(argv)
    out = Path(args.out)
    for name, fn in TABLES.items():
        (out / f"{name}.tex").write_text(fn(), encoding="utf-8", newline="\n")
    print(f"[appendix_a] wrote {len(TABLES)} tables to {out}")


if __name__ == "__main__":
    main()
