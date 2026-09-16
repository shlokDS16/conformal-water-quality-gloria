"""Build the public submission archive (phase 14, CLAUDE.md phase-14 brief section B).

Writes `submission/archive/`:

    code/
        src/, scripts/, tests/           verbatim copies (no __pycache__)
        environment/requirements.lock    copy with local editable-install lines stripped
        LICENSE                          generated MIT licence for the code folder
        README.md                        maps every table and figure to its generating
                                          script, its input result files and the exact command
    data_derived/
        data/interim/{wb_groups,wb_overrides,band_coverage,chl_methods}.csv
        data/processed/{splits*.parquet, groups_*.parquet}
        tables/*.csv
        LICENSE                          CC-BY-4.0 (inherited from GLORIA, see README)
    results/                             only with --with-results, and only tags that exist:
        core_v2, cvplus_v3, budget_v2, sens_v2, noise_v2
        LICENSE                          CC-BY-4.0 (derived from GLORIA)
    README.md                            top-level provenance, GLORIA citation, per-folder licence
    MANIFEST.sha256                      sha256  relative/path, one line per file, sorted by path

Raw GLORIA files are never copied; they are cited (doi:10.1594/PANGAEA.948492, CC-BY-4.0).

Idempotent: the output directory is removed and rebuilt from scratch on every run, so two runs
against unchanged inputs produce a byte-identical MANIFEST.sha256 (path list and hashes; file
mtimes are not hashed).

Usage
-----
    python -m src.make_submission                    # code + data_derived only
    python -m src.make_submission --with-results      # also copy result tags that exist
    python -m src.make_submission --out submission/_test   # build to a different directory

This script only ever writes under the `--out` directory (default `submission/archive`) and
never touches `results/`, `logs/`, `data/`, `paper/` or `src/` (other than being itself). It does
not read anything with more than light, single-pass I/O, so it is safe to run while experiments
are using the CPU/GPU.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESULT_TAGS = ["core_v2", "cvplus_v3", "budget_v2", "sens_v2", "noise_v2"]

DATA_INTERIM_FILES = [
    "wb_groups.csv",
    "wb_overrides.csv",
    "band_coverage.csv",
    "chl_methods.csv",
]

TEXT_SUFFIXES = {
    ".py", ".ps1", ".md", ".txt", ".lock", ".cfg", ".ini", ".toml", ".json", ".csv", ".yml", ".yaml",
}

GLORIA_DOI = "10.1594/PANGAEA.948492"
GLORIA_DESCRIPTOR_DOI = "10.1038/s41597-023-01973-y"
# Elements verified in this project (research/DATA_INVENTORY.md, research/DATA_PIPELINE_LOG.md,
# progress.md validation ledger): PANGAEA dataset title and DOI, version tag, licence, and the
# peer-reviewed data descriptor's authors/year/DOI. No author list is asserted for the PANGAEA
# record itself beyond what those files record, per CLAUDE.md section 2 (do not guess citation
# details not confirmed in a project ledger file).
GLORIA_CITATION = (
    f"[dataset] GLORIA - A global dataset of remote sensing reflectance and water quality from "
    f"inland and coastal waters. PANGAEA, version GLORIA-2022 (deposited 2024-04-20). "
    f"https://doi.org/{GLORIA_DOI}\n"
    f"Data descriptor: Lehmann, M.K. et al., 2023. Scientific Data 10:100. "
    f"https://doi.org/{GLORIA_DESCRIPTOR_DOI}"
)
GLORIA_LICENCE = "CC-BY-4.0 (PANGAEA metadata field `license`: \"Creative Commons Attribution 4.0 International\")"

# Identifying strings to scrub from generated copies (never edited in the source tree).
# Deliberately generic (built from the running OS account, never a literal name in this file's
# own source): a scanner that hardcodes the specific name it looks for would leak that name into
# its own text the moment this script is copied into the archive it builds.
_CURRENT_USER = getpass.getuser()

# E-mail addresses that may survive the scrub and the scan. Empty by default, so the behaviour of
# `python -m src.make_submission` is unchanged: everything that looks like an address is redacted.
# `src/make_release.py` sets this to the authors' own institutional addresses, because this
# journal's review is single anonymised and the release may name the authors.
ALLOWED_EMAILS: set[str] = set()

SCRUB_PATTERNS = [
    # Any "<drive>:\Users\<name>\..." absolute path, not just the current account's.
    (re.compile(r"[A-Za-z]:\\[Uu]sers\\[^\\\s\"'\)]+\\?[^\s\"'\)]*"), "<REDACTED_PATH>"),
    (re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}"), "<REDACTED_EMAIL>"),
]
if _CURRENT_USER and len(_CURRENT_USER) > 2:
    SCRUB_PATTERNS.append((re.compile(re.escape(_CURRENT_USER), re.IGNORECASE), "<REDACTED_NAME>"))

SCAN_PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:\\[Uu]sers\\"),
    "email address": re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}"),
    "current OS account name": re.compile(re.escape(_CURRENT_USER), re.IGNORECASE) if _CURRENT_USER and len(_CURRENT_USER) > 2 else re.compile(r"(?!)"),
    ".git metadata": re.compile(r"(^|[\\/])\.git([\\/]|$)"),
}

MIT_LICENSE = """MIT License

Copyright (c) 2026 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

CC_BY_4_LICENSE = f"""Creative Commons Attribution 4.0 International (CC BY 4.0)

The files in this folder are derived from the GLORIA dataset:

    {GLORIA_CITATION}
    Licence: {GLORIA_LICENCE}

GLORIA is distributed under CC BY 4.0. A derivative of a CC BY 4.0 work must itself be offered
under CC BY 4.0 or a compatible licence; these files therefore inherit CC BY 4.0. Full licence
text: https://creativecommons.org/licenses/by/4.0/legalcode

You are free to share and adapt this material for any purpose, even commercially, under the
following terms: give appropriate credit to the GLORIA data contributors (cite the reference
above), provide a link to the licence, and indicate if changes were made. No additional
restrictions may be applied.
"""


def _iter_files(base: Path):
    for p in sorted(base.rglob("*")):
        if p.is_file():
            yield p


def _copytree_clean(src: Path, dst: Path) -> None:
    """Copy a directory tree, dropping __pycache__ and *.pyc."""
    if not src.exists():
        raise FileNotFoundError(f"expected source directory missing: {src}")
    def ignore(_dir, names):
        return {n for n in names if n == "__pycache__" or n.endswith(".pyc")}
    shutil.copytree(src, dst, ignore=ignore)


def _scrub_text_file(path: Path) -> int:
    """Apply SCRUB_PATTERNS to a text file in place. Returns number of replacements made."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0
    total = 0
    new_text = text

    # Protect allowed addresses from every pattern, not just the e-mail one: the OS-account-name
    # pattern would otherwise cut the local part out of an address it is supposed to keep.
    placeholders: dict[str, str] = {}
    if ALLOWED_EMAILS:
        for i, addr in enumerate(sorted(ALLOWED_EMAILS, key=len, reverse=True)):
            token = f"\x00ALLOWED_EMAIL_{i}\x00"
            for variant in {addr, addr.lower()}:
                if variant in new_text:
                    new_text = new_text.replace(variant, token)
                    placeholders[token] = variant

    for pattern, repl in SCRUB_PATTERNS:
        new_text, n = pattern.subn(repl, new_text)
        total += n

    for token, addr in placeholders.items():
        new_text = new_text.replace(token, addr)
    if total:
        path.write_text(new_text, encoding="utf-8", newline="")
    return total


def _filter_requirements_lock(src: Path, dst: Path) -> None:
    """Copy requirements.lock, dropping local editable-install lines (not publicly resolvable
    and, in this environment, the source of an absolute local path)."""
    # utf-8-sig drops a byte-order mark if pip freeze wrote one; a stray U+FEFF in the first line
    # otherwise survives into the release and shows up as a zero-width finding.
    lines = src.read_text(encoding="utf-8-sig").splitlines()
    kept = []
    n_editable = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("-e "):
            n_editable += 1
            # the preceding "# Editable install with no version control (...)" comment belongs
            # to this line; drop it too if present immediately above (already appended to kept).
            if kept and kept[-1].lstrip().startswith("# Editable install"):
                kept.pop()
            i += 1
            continue
        kept.append(line)
        i += 1
    header = [
        f"# environment/{dst.name}",
        "# Full pip freeze of the development environment, pinned versions only.",
        "# This project's own runtime dependencies are a subset (see code/README.md).",
    ]
    if n_editable:
        header.append(
            f"# {n_editable} local editable-install line(s) removed for this public archive: each"
            " referenced an absolute filesystem path outside this repository, not resolvable by"
            " anyone else and not a public package."
        )
    header.append("")
    dst.write_text("\n".join(header + kept) + "\n", encoding="utf-8", newline="")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CODE_README = """# Code archive

This folder is the code companion to the manuscript. It is a verbatim copy of the project's
`src/`, `scripts/` and `tests/` at the point the results in the paper were produced, plus the
pinned environment and licence for this folder (MIT, see `LICENSE`).

## Layout
- `src/` - data pipeline (`src/data/`), models (`src/models/`), conformal methods
  (`src/conformal/`), evaluation (`src/eval/`), experiment runners (`src/run_experiment.py`,
  `src/run_budget.py`, `src/experiment_common.py`), confirmatory analysis
  (`src/analysis/confirmatory.py`), figures (`src/figures/`), table and summary builders
  (`src/make_tables_data.py`, `src/make_tables_results.py`, `src/make_summary.py`), and this
  archive builder (`src/make_submission.py`).
- `scripts/` - PowerShell launchers for the experiment matrix (`run_core.ps1`,
  `run_cvplus_v3.ps1`); read the header comment of each for flags and resumability.
- `tests/` - pytest suite (`test_data.py`, `test_conformal.py`, `test_confirmatory.py`). Run with
  `python -m pytest tests -q` from the repository root.
- `environment/requirements.lock` - pinned pip freeze of the development environment (local
  editable installs removed; see the header comment inside the file).

## How to reproduce the pipeline
1. Data: `python -m src.data.build_all` (raw GLORIA and the SRF files described in the top-level
   README must be present under `data/raw/`; the derived outputs this produces are shipped
   directly in `data_derived/`, so this step is only needed to rebuild them from scratch).
2. Experiments: `scripts/run_core.ps1` (or call `python -m src.run_experiment ...` /
   `python -m src.run_budget ...` directly; see the script header for the full job matrix and
   flags). Results land in `results/<tag>/`.
3. Result ledger: `python -m src.make_summary` reads `results/` and writes `tables/summary.csv`.
4. Tables and figures: see the mapping below.

## Table and figure provenance
Every table and figure is generated by exactly one script, from either `data/processed/` (no
model results needed) or `results/` + `tables/summary.csv` (needs finished experiment runs). The
`Run` column is the exact command, executed from the repository root.

| Output | Script | Inputs | Run |
|---|---|---|---|
| Table 1 (dataset summary) | `src/make_tables_data.py` | `data/processed/gloria.parquet` | `python -m src.make_tables_data` |
| Table 2 (split protocols) | `src/make_tables_data.py` | `data/processed/gloria.parquet`, `data/processed/splits_*.parquet` | `python -m src.make_tables_data` |
| `tables/data_summary.csv` (ledger for Tables 1-2) | `src/make_tables_data.py` | `data/processed/gloria.parquet` | `python -m src.make_tables_data` |
| Fig. 1 (workflow diagram) | `src/figures/fig01_workflow.py` | `data/processed/gloria.parquet` (counts only) | `python -m src.figures.fig01_workflow` |
| Fig. 2 (site map) | `src/figures/fig02_map.py` | `data/processed/gloria.parquet` | `python -m src.figures.fig02_map` |
| Fig. 3 (protocol schematic) | `src/figures/fig03_protocols.py` | illustrative layout, no data file | `python -m src.figures.fig03_protocols` |
| Fig. 4 (Rrs spectra envelopes + SRFs) | `src/figures/fig04_spectra.py` | `data/processed/gloria.parquet`, `data/raw/srf/*` via `src/data/srf.py` | `python -m src.figures.fig04_spectra` |
| Fig. A.1 (target distributions) | `src/figures/figA1_targets.py` | `data/processed/gloria.parquet` | `python -m src.figures.figA1_targets` |
| Fig. A.2 (water-body size distribution) | `src/figures/figA2_wbsize.py` | `data/processed/groups_<target>.parquet` | `python -m src.figures.figA2_wbsize` |

The following are pending on the completion of the experiment runs (`results/core_v2`,
`results/cvplus_v3`, and, for Fig. 10 and Table 6, `results/noise_v2` and `results/sens_v2`).
They are built by the same scripts once those results exist and are not shipped until then; see
`progress.md` for run status.

| Output | Script | Inputs | Run |
|---|---|---|---|
| `tables/summary.csv` (result ledger) | `src/make_summary.py` | `results/core_v2`, `results/cvplus_v3`, `results/budget_v2`, `results/sens_v2`, `results/noise_v2`, `tables/confirmatory*.csv` | `python -m src.make_summary` |
| Tables 3-6 | `src/make_tables_results.py` | `tables/summary.csv` | `python -m src.make_tables_results` |
| Fig. 5 (point accuracy) | `src/figures/fig05_point_accuracy.py` | per-split metrics, `method == "point"` | `python -m src.figures.fig05_point_accuracy` |
| Fig. 6 (coverage by protocol) | `src/figures/fig06_coverage_protocol.py` | per-split metrics; CV+ from `results/cvplus_v3` | `python -m src.figures.fig06_coverage_protocol` |
| Fig. 7 (coverage vs width) | `src/figures/fig07_coverage_width.py` | per-split metrics; CV+ from `results/cvplus_v3` | `python -m src.figures.fig07_coverage_width` |
| Fig. 8 (nominal levels) | `src/figures/fig08_nominal_levels.py` | per-split metrics at alpha in {0.20, 0.10, 0.05} | `python -m src.figures.fig08_nominal_levels` |
| Fig. 9 (calibration budget) | `src/figures/fig09_calibration_budget.py` | `results/budget_v2` via `src.analysis.confirmatory` | `python -m src.figures.fig09_calibration_budget` |
| Fig. 10 (sensor comparison + noise sweep) | `src/figures/fig10_sensor_noise.py` | `results/core_v2` (sensors), `results/noise_v2` (sweep) | `python -m src.figures.fig10_sensor_noise` |
| Fig. A.3 (per-water-body coverage ECDF) | `src/figures/figA3_wbcoverage_ecdf.py` | `results/core_v2/pred_*.parquet`; CV+ from `results/cvplus_v3` | `python -m src.figures.figA3_wbcoverage_ecdf` |
| Fig. A.4 (conditional coverage) | `src/figures/figA4_conditional_coverage.py` | `results/core_v2/pred_*.parquet` | `python -m src.figures.figA4_conditional_coverage` |
| Fig. A.5 (coverage/width vs kNN distance) | `src/figures/figA5_knn_distance.py` | `results/core_v2/pred_*.parquet` | `python -m src.figures.figA5_knn_distance` |

Every script's own module docstring repeats its exact data source and run command; read the
script for the authoritative version if this table and the code ever disagree.

## Licence
MIT, see `LICENSE` in this folder. This covers the code only; data licensing is described in the
top-level README and in `data_derived/LICENSE`.
"""


def _write_top_level_readme(out: Path, with_results: bool, result_tags_present: list[str]) -> None:
    results_section = ""
    if with_results:
        if result_tags_present:
            tag_list = "\n".join(f"- `results/{t}/`" for t in result_tags_present)
            results_section = f"""
## results/ (included: --with-results)
Per-split metrics and prediction files produced by the experiment runners
(`src/run_experiment.py`, `src/run_budget.py`). These are numerical outputs derived from GLORIA
(via the pipeline in `code/src/data/`) and inherit its CC BY 4.0 licence; see `results/LICENSE`.

{tag_list}

Tags requested but not yet present at build time are silently skipped (see `code/src/make_submission.py`
`--with-results`); re-run this script once they exist to add them.
"""
        else:
            results_section = """
## results/ (--with-results was passed, but no result tags existed yet)
None of core_v2, cvplus_v3, budget_v2, sens_v2, noise_v2 existed under `results/` when this
archive was built, so no `results/` folder was written. Re-run `python -m src.make_submission
--with-results` once the experiment runs finish.
"""
    else:
        results_section = """
## results/ (not included in this build)
Run `python -m src.make_submission --with-results` to add `results/core_v2`, `results/cvplus_v3`,
`results/budget_v2`, `results/sens_v2` and `results/noise_v2`, copying only the tags that exist
at build time. Omitted here because the experiment runs were still in progress.
"""

    content = f"""# Submission archive

Built by `code/src/make_submission.py`. Contains the code, the data derived from GLORIA, and
(optionally) the numerical results behind the manuscript. Raw GLORIA files are not included; they
are openly available from their own repository and are cited below.

## Data provenance and citation

{GLORIA_CITATION}

Licence: {GLORIA_LICENCE}

The raw GLORIA files (`GLORIA_Rrs.csv`, `meta_and_lab.csv`, `qc_flags.csv`, and related files) are
not redistributed here. They are downloaded from PANGAEA and processed into `data_derived/` by
the pipeline in `code/src/data/` (entry point `python -m src.data.build_all`, documented in
`code/README.md`). Sentinel-2 MSI and Sentinel-3 OLCI spectral response function files used by the
same pipeline are third-party files from ESA/SentiWiki, also not redistributed; see
`code/src/data/srf.py` for their source URLs.

## Folder licences

| Folder | Licence | Reason |
|---|---|---|
| `code/` | MIT (`code/LICENSE`) | Original software written for this project. |
| `data_derived/` | CC BY 4.0 (`data_derived/LICENSE`) | Row/column subsets, aggregates and transformations of GLORIA (CC BY 4.0); a derivative work inherits the licence of its source under CC BY 4.0's terms. |
{"| `results/` | CC BY 4.0 (`results/LICENSE`) | Model outputs computed from the CC BY 4.0 GLORIA-derived data above. |" if with_results and result_tags_present else ""}

## Contents

- `code/` - source code, tests, pinned environment, and a README mapping every table and figure
  to the script, input files and command that produced it.
- `data_derived/` - the water-body grouping keys, band-coverage and Chl-a method-mapping tables,
  train/calibration/test split definitions and per-water-body group tables, and the data-only
  results ledger (`tables/data_summary.csv`) and its preview companion.
{results_section}
## Integrity

`MANIFEST.sha256` (in this folder) lists every file in this archive with its SHA-256 hash, one
line per file as `<hash>  <relative path>`, generated by `code/src/make_submission.py`. Verify
with, for example, `sha256sum -c MANIFEST.sha256` on Linux/macOS or by recomputing hashes in
PowerShell (`Get-FileHash -Algorithm SHA256`) on Windows.

## What is not here

Personal file paths, author names and institutional identifiers found in the source tree during
the archive build were removed from these generated copies only (never from the project's own
working files); see the archive build log for what was found and changed.
"""
    (out / "README.md").write_text(content, encoding="utf-8", newline="")


def build(out: Path, with_results: bool) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # ---- code/ ----
    code = out / "code"
    _copytree_clean(ROOT / "src", code / "src")
    _copytree_clean(ROOT / "scripts", code / "scripts")
    _copytree_clean(ROOT / "tests", code / "tests")
    (code / "environment").mkdir(parents=True, exist_ok=True)
    _filter_requirements_lock(ROOT / "environment" / "requirements.lock", code / "environment" / "requirements.lock")
    (code / "LICENSE").write_text(MIT_LICENSE, encoding="utf-8", newline="")
    (code / "README.md").write_text(CODE_README, encoding="utf-8", newline="")

    # ---- data_derived/ ----
    dd = out / "data_derived"
    interim_dst = dd / "data" / "interim"
    interim_dst.mkdir(parents=True, exist_ok=True)
    for name in DATA_INTERIM_FILES:
        src = ROOT / "data" / "interim" / name
        if not src.exists():
            raise FileNotFoundError(f"expected data_derived input missing: {src}")
        shutil.copy2(src, interim_dst / name)

    processed_src = ROOT / "data" / "processed"
    processed_dst = dd / "data" / "processed"
    processed_dst.mkdir(parents=True, exist_ok=True)
    processed_files = sorted(processed_src.glob("splits*.parquet")) + sorted(processed_src.glob("groups_*.parquet"))
    if not processed_files:
        raise FileNotFoundError(f"no splits*/groups_* parquet files found under {processed_src}")
    for f in processed_files:
        shutil.copy2(f, processed_dst / f.name)

    tables_src = ROOT / "tables"
    tables_dst = dd / "tables"
    tables_dst.mkdir(parents=True, exist_ok=True)
    # AUDIT_5 N6 / REVIEW_full_v2 V13: the .tex table bodies and DATA_DICTIONARY.md ship alongside
    # the ledgers, so a reader of the released CSVs can resolve the method-dependent columns.
    for f in (sorted(tables_src.glob("*.csv")) + sorted(tables_src.glob("*.tex"))
              + sorted(tables_src.glob("*.md"))):
        shutil.copy2(f, tables_dst / f.name)

    (dd / "LICENSE").write_text(CC_BY_4_LICENSE, encoding="utf-8", newline="")

    # ---- results/ (optional) ----
    result_tags_present: list[str] = []
    if with_results:
        for tag in RESULT_TAGS:
            src = ROOT / "results" / tag
            if src.exists():
                _copytree_clean(src, out / "results" / tag)
                result_tags_present.append(tag)
        if result_tags_present:
            (out / "results" / "LICENSE").write_text(CC_BY_4_LICENSE, encoding="utf-8", newline="")

    # ---- top-level README ----
    _write_top_level_readme(out, with_results, result_tags_present)

    # ---- scrub generated copies of identifying strings ----
    scrub_hits = 0
    for p in _iter_files(out):
        if p.name == "MANIFEST.sha256":
            continue
        if p.suffix.lower() in TEXT_SUFFIXES:
            scrub_hits += _scrub_text_file(p)
    if scrub_hits:
        print(f"[make_submission] scrubbed {scrub_hits} identifying string(s) from generated copies")

    # ---- post-scrub scan (should be 0 hits; report if not) ----
    residual: list[str] = []
    allowed_lower = {e.lower() for e in ALLOWED_EMAILS}
    n_allowed = 0
    for p in _iter_files(out):
        if p.name == "MANIFEST.sha256":
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # Spans of the allowed addresses: a hit that lies inside one of them is that address,
        # not a separate leak (the account name is a substring of the author's own address).
        allowed_spans: list[tuple[int, int]] = []
        if allowed_lower:
            for m in SCAN_PATTERNS["email address"].finditer(text):
                if m.group(0).lower() in allowed_lower:
                    allowed_spans.append(m.span())
                    n_allowed += 1

        def _inside_allowed(span: tuple[int, int]) -> bool:
            return any(a <= span[0] and span[1] <= b for a, b in allowed_spans)

        for label, pattern in SCAN_PATTERNS.items():
            for m in pattern.finditer(text):
                if _inside_allowed(m.span()):
                    continue
                residual.append(f"{p.relative_to(out)}: {label} -> {m.group(0)!r}")
    if n_allowed:
        print(f"[make_submission] {n_allowed} allowed author e-mail occurrence(s) kept (ALLOWED_EMAILS)")
    if residual:
        print(f"[make_submission] WARNING: {len(residual)} residual identifying string(s) after scrub:")
        for r in residual:
            print(f"  {r}")
    else:
        print("[make_submission] scan clean: 0 identifying strings in the archive")

    # ---- manifest ----
    entries = []
    total_bytes = 0
    n_files = 0
    for p in _iter_files(out):
        if p.name == "MANIFEST.sha256":
            continue
        rel = p.relative_to(out).as_posix()
        entries.append((rel, _sha256(p)))
        total_bytes += p.stat().st_size
        n_files += 1
    entries.sort(key=lambda e: e[0])
    manifest_lines = [f"{digest}  {rel}" for rel, digest in entries]
    (out / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8", newline="")

    print(f"[make_submission] built {out}")
    print(f"[make_submission] files: {n_files}  total size: {total_bytes / (1024 * 1024):.2f} MiB")
    print(f"[make_submission] results tags included: {result_tags_present if with_results else 'none (pass --with-results)'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="submission/archive", help="output directory, relative to the repo root unless absolute (default: submission/archive)")
    ap.add_argument("--with-results", action="store_true", help="also copy results/{core_v2,cvplus_v3,budget_v2,sens_v2,noise_v2} for any tag that currently exists")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    build(out, args.with_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
