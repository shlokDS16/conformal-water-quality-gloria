"""Build both public release trees (phase 14, sections B and C).

Two trees, because the full result set (about 1.55 GB) does not belong in a git repository:

    release/github/     a clean, self-contained repository tree, small enough to push to GitHub
    release/zenodo/     the full deposit: zipped code, derived data and every result tag,
                        with README, per-folder LICENSE files and MANIFEST.sha256

Nothing here creates a repository, runs git, pushes, or uploads anything. It only writes files
under the `--out` directory (default `release/`). `submission/HOW_TO_RELEASE.md` holds the steps
the user performs by hand.

Usage
-----
    python -m src.make_release                       # build both trees
    python -m src.make_release --github-only         # skip the 1.5 GB zenodo build
    python -m src.make_release --zenodo-only
    python -m src.make_release --out release --keep-staging

Idempotence
-----------
Every output directory is removed and rebuilt from scratch, generated text carries no timestamp
(RELEASE_DATE below is a constant, not `today()`), and the zip writer uses a fixed entry
timestamp and a fixed compression level, so two runs over unchanged inputs produce byte-identical
archives and byte-identical manifests. Verified by running twice and diffing:

    release/_manifests/github.sha256
    release/zenodo/MANIFEST.sha256

What this script does NOT touch: `results/`, `tables/`, `data/`, `paper/`, `src/` (other than
reading them), and `src/analysis/confirmatory.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import src.make_submission as _submission
from src.make_submission import (
    CC_BY_4_LICENSE,
    GLORIA_CITATION,
    GLORIA_DOI,
    GLORIA_LICENCE,
    MIT_LICENSE,
    RESULT_TAGS,
    _filter_requirements_lock,
)
from src.make_submission import build as build_submission_archive

ROOT = Path(__file__).resolve().parents[1]

# Constants, not derived from the clock, so the build is reproducible.
RELEASE_DATE = "2026-09-16"
RELEASE_VERSION = "v1.0.0"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
ZIP_LEVEL = 1  # result files are already-compressed parquet/npz; level 1 costs far less time

PAPER_TITLE = (
    "Group-aware conformal calibration for water-quality retrieval: marginal validity under "
    "water-body shift, poor conditional coverage"
)
JOURNAL = "ISPRS Journal of Photogrammetry and Remote Sensing"
AFFILIATION = (
    "School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, "
    "632014, Tamil Nadu, India"
)
AUTHOR_EMAILS = ["shlok.goenka2023@vitstudent.ac.in", "rmanjula@vit.ac.in"]

# `tables/summary.csv` is 212,838 rows. Anything at or above this goes to Zenodo only, because
# GitHub warns above 50 MiB per file and blocks above 100 MiB.
GITHUB_FILE_LIMIT = 50 * 1024 * 1024

# ---------------------------------------------------------------------------
# scrub / scan
#
# Release policy differs from `src/make_submission.py`: this journal's review is single
# anonymised, so the authors' own names and institutional e-mail addresses may appear. Everything
# else that identifies a machine or a person is removed.
# ---------------------------------------------------------------------------
_ALLOWED_EMAILS = {e.lower() for e in AUTHOR_EMAILS}

_PATH_RE = re.compile(r"[A-Za-z]:[\\/]{1,2}[Uu]sers[\\/]{1,2}[^\s\"'\)\],;]*")
_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9_.-]+\.[A-Za-z]{2,}")

TEXT_SUFFIXES = {
    ".py", ".ps1", ".md", ".txt", ".lock", ".cfg", ".ini", ".toml", ".json", ".csv", ".yml",
    ".yaml", ".cff", ".gitignore", ".tex", ".bib", "",
}

CREDENTIAL_PATTERNS = {
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "assigned secret": re.compile(
        r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password)\b\s*[:=]\s*"
        r"[\"']?[A-Za-z0-9_\-/+]{16,}"
    ),
}


def _is_allowed_email(value: str) -> bool:
    return value.lower() in _ALLOWED_EMAILS


def _read_text_keep_newlines(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def _write_text_keep_newlines(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def scrub_tree(base: Path) -> tuple[int, int]:
    """Redact absolute user paths and non-author e-mails in every text file under `base`.

    Line endings are preserved exactly (CLAUDE.md section 6.3): the file is read and written with
    `newline=""`. Returns (files_changed, replacements).
    """
    files_changed = 0
    replacements = 0
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # Never write inside git plumbing: `release/github/.git` is the repository this tree is
        # pushed from, and several of its files (HEAD, config, COMMIT_EDITMSG) carry no suffix and
        # would otherwise be read and rewritten by the redaction below.
        if ".git" in p.relative_to(base).parts:
            continue
        text = _read_text_keep_newlines(p)
        if text is None:
            continue
        new_text, n_path = _PATH_RE.subn("<REDACTED_PATH>", text)
        n_mail = 0

        def _mail_sub(m: re.Match) -> str:
            nonlocal n_mail
            if _is_allowed_email(m.group(0)):
                return m.group(0)
            n_mail += 1
            return "<REDACTED_EMAIL>"

        new_text = _EMAIL_RE.sub(_mail_sub, new_text)
        if n_path or n_mail:
            _write_text_keep_newlines(p, new_text)
            files_changed += 1
            replacements += n_path + n_mail
    return files_changed, replacements


def scan_tree(base: Path, allow_git: bool = False) -> tuple[list[str], int]:
    """Scan for identifying strings that must not survive. Returns (findings, allowed_email_hits).

    `allow_git` is set only for `release/github/`, which IS the working copy of the repository that
    is pushed, so a `.git` directory there is expected rather than a leak. In every other tree
    (the Zenodo staging archive above all) a `.git` directory is still reported."""
    findings: list[str] = []
    allowed_hits = 0
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            if p.name == ".git" and not allow_git:
                findings.append(f"{p.relative_to(base).as_posix()}: git metadata directory present")
            continue
        if not p.is_file():
            continue
        if ".git" in p.relative_to(base).parts:
            continue
        rel = p.relative_to(base).as_posix()
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = _read_text_keep_newlines(p)
        if text is None:
            continue
        for m in _PATH_RE.finditer(text):
            findings.append(f"{rel}: absolute user path -> {m.group(0)!r}")
        for m in _EMAIL_RE.finditer(text):
            if _is_allowed_email(m.group(0)):
                allowed_hits += 1
            else:
                findings.append(f"{rel}: non-author e-mail -> {m.group(0)!r}")
        for label, pattern in CREDENTIAL_PATTERNS.items():
            for m in pattern.finditer(text):
                findings.append(f"{rel}: {label} -> {m.group(0)[:40]!r}")
    return findings, allowed_hits


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(base: Path):
    """Every file of the release tree, excluding git metadata.

    `release/github/` is itself the working copy of the repository that is pushed, so it carries a
    `.git` directory that `_reset_dir` deliberately keeps. That directory is repository plumbing,
    not released content: hashing it into the manifest would make the manifest change with every
    commit and would report hundreds of loose objects as release files."""
    for p in sorted(base.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(base).parts:
            yield p


def _copytree_clean(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"expected source directory missing: {src}")

    def ignore(_dir, names):
        return {
            n for n in names
            if n in {"__pycache__", ".pytest_cache", ".git"} or n.endswith((".pyc", ".pyo"))
        }

    shutil.copytree(src, dst, ignore=ignore)


def _write_manifest(paths: list[tuple[str, Path]], dest: Path) -> tuple[int, int]:
    """Write `<sha256>  <relative path>` lines, sorted by path. Returns (n_files, total_bytes)."""
    rows = sorted(paths, key=lambda t: t[0])
    lines = []
    total = 0
    for rel, p in rows:
        lines.append(f"{_sha256(p)}  {rel}")
        total += p.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write_text_keep_newlines(dest, "\n".join(lines) + "\n")
    return len(rows), total


def _human(n_bytes: int) -> str:
    mib = n_bytes / (1024 * 1024)
    if mib >= 1024:
        return f"{mib / 1024:.2f} GiB"
    return f"{mib:.2f} MiB"


def _write_deterministic_zip(zip_path: Path, base: Path, rel_paths: list[str]) -> None:
    """Zip `rel_paths` (relative to `base`) with fixed entry metadata, so the bytes reproduce."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for rel in sorted(rel_paths):
            info = zipfile.ZipInfo(rel, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = ZIP_LEVEL
            info.create_system = 3          # unix, so the mode below is meaningful
            info.external_attr = 0o644 << 16
            with (base / rel).open("rb") as src, zf.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)


# ---------------------------------------------------------------------------
# release/github
# ---------------------------------------------------------------------------
GITIGNORE = """# Raw third-party data: never committed. GLORIA is cited, not redistributed
# (see README.md, "Data provenance"); the Sentinel SRF files are fetched by src/data/srf.py.
data/raw/

# Experiment output: about 1.55 GB, archived on Zenodo instead (see README.md).
results/
logs/
build/

# Working manuscript build artefacts. Figure PDFs ARE tracked (see the negation below).
*.pdf
!figures/*.pdf
*.aux
*.bbl
*.bcf
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.run.xml
*.synctex.gz
*.toc

# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/
.ipynb_checkpoints/

# Editors and OS
.vscode/
.idea/
.DS_Store
Thumbs.db
"""


def _citation_cff() -> str:
    return f"""cff-version: 1.2.0
message: "If you use this software or the derived data, please cite the article and this software record."
title: "{PAPER_TITLE}: code and derived data"
abstract: >-
  Code, derived data tables and reproduction instructions for the study of group-aware conformal
  calibration for hyperspectral water-quality retrieval (chlorophyll-a, total suspended solids,
  Secchi depth, aCDOM(440)) under water-body and regional shift, using the GLORIA dataset with
  Sentinel-2 MSI and Sentinel-3 OLCI band simulation.
type: software
version: "{RELEASE_VERSION}"
date-released: "{RELEASE_DATE}"
license: MIT
authors:
  - family-names: Goenka
    given-names: "Shlok Kumar"
    email: {AUTHOR_EMAILS[0]}
    affiliation: "{AFFILIATION}"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"   # ORCID pending, add before release
  - family-names: Manjula
    given-names: "R."
    email: {AUTHOR_EMAILS[1]}
    affiliation: "{AFFILIATION}"
    # orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"   # ORCID pending, add before release
keywords:
  - water quality
  - conformal prediction
  - uncertainty quantification
  - domain shift
  - Sentinel-3 OLCI
  - chlorophyll-a
repository-code: "https://github.com/shlokDS16/conformal-water-quality-gloria"
identifiers:
  - type: doi
    value: "10.5281/zenodo.22791749"
    description: "Archived code, derived data and full result set (Zenodo, reserved 2026-09-16)"
references:
  - type: data
    title: "GLORIA - A global dataset of remote sensing reflectance and water quality from inland and coastal waters"
    authors:
      - name: "The GLORIA data contributors"
    year: 2022
    doi: "{GLORIA_DOI}"
    license: CC-BY-4.0
    notes: "Raw data, cited and not redistributed. Data descriptor: Lehmann, M.K. et al., 2023, Scientific Data 10:100, doi:10.1038/s41597-023-01973-y"
"""


def _github_readme(summary_included: bool, summary_bytes: int) -> str:
    if summary_included:
        summary_note = (
            "`tables/summary.csv` is included in this repository."
        )
    else:
        summary_note = (
            f"`tables/summary.csv` (the per-cell result ledger, 212,838 rows, "
            f"{_human(summary_bytes)}) is **not** in this repository: it is above the "
            f"{GITHUB_FILE_LIMIT // (1024 * 1024)} MiB per-file threshold above which GitHub "
            "warns, so it lives in the Zenodo record together with the raw result files. Rebuild "
            "it locally with `python -m src.make_summary` once `results/` is in place, or "
            "download it from Zenodo."
        )

    return f"""# {PAPER_TITLE}

Code and derived data for the manuscript submitted to the {JOURNAL}.

**Authors**: Shlok Kumar Goenka (<{AUTHOR_EMAILS[0]}>) and R. Manjula (<{AUTHOR_EMAILS[1]}>, corresponding author),
{AFFILIATION}.
Version {RELEASE_VERSION}, {RELEASE_DATE}.

## What the paper does

The study asks whether split conformal prediction still delivers its promised coverage when a
water-quality retrieval model is applied to water bodies it never saw in training. Four targets
are retrieved from hyperspectral remote-sensing reflectance in the GLORIA dataset: chlorophyll-a,
total suspended solids (TSS), Secchi depth and the CDOM absorption coefficient at 440 nm
(aCDOM(440)). Reflectance is used at full hyperspectral resolution and also convolved to
Sentinel-2 MSI and Sentinel-3 OLCI bands. Conformal intervals are calibrated marginally and per
water-body group, under random, water-body, contributor-dataset and regional splits. The headline
finding is in the title: marginal validity survives the shift, conditional coverage does not.

Result tables and figures are reproduced exactly by the commands in "Reproduction" below. Every
number quoted in the manuscript has a key in `tables/summary.csv` (see the Zenodo record, and the
note below).

## Repository layout

| Path | Contents |
|---|---|
| `src/data/` | GLORIA ingestion, quality control, water-body grouping, spectral response convolution, split construction |
| `src/models/` | point models (LightGBM, XGBoost, mixture density network) |
| `src/conformal/` | split conformal, group-conditional and CV+ interval construction |
| `src/eval/` | coverage, width and conditional-coverage metrics |
| `src/analysis/` | confirmatory tests, conditional shares, power analysis |
| `src/figures/` | one script per figure, plus the legibility check and contact sheet |
| `src/run_experiment.py`, `src/run_budget.py` | experiment runners |
| `src/make_summary.py`, `src/make_tables_data.py`, `src/make_tables_results.py` | ledger and table builders |
| `src/make_submission.py`, `src/make_release.py` | archive and release builders |
| `scripts/` | PowerShell launchers for the experiment matrix |
| `tests/` | pytest suite |
| `environment/requirements.lock` | pinned versions actually used for the reported runs |
| `data/interim/`, `data/processed/` | small derived tables: water-body grouping keys, band coverage, chlorophyll-a method map, split definitions, per-group tables |
| `tables/` | result ledgers (CSV) and the LaTeX table bodies used in the manuscript |
| `figures/` | every figure as 300 dpi PNG and vector PDF |

## Data provenance and licences

**GLORIA (raw, not redistributed).** All measurements come from the GLORIA dataset:

> {GLORIA_CITATION}

Licence: {GLORIA_LICENCE}. Download it yourself from
<https://doi.org/{GLORIA_DOI}> and unpack it into `data/raw/gloria/`. It is cited rather than
copied here, in line with the journal's data-citation rules and to avoid duplicating an
already-archived, citable, open dataset.

**Spectral response functions (not redistributed).** Sentinel-3 OLCI (S3A, S3B) and Sentinel-2 MSI
spectral response functions are third-party Copernicus files. `src/data/srf.py` holds their exact
download URLs on the Copernicus SentiWiki (ESA; the Sentinel-3 mission is operated jointly with
EUMETSAT) and verifies the Sentinel-2 workbook by SHA-256. The script downloads them into
`data/raw/srf/` on first use. Their redistribution terms were not established for this release, so
they are fetched from source rather than copied here.

**Derived data in this repository** (`data/interim/`, `data/processed/`, `tables/*.csv`) are
subsets, aggregates and transformations of GLORIA. Under CC BY 4.0 a derivative inherits the
licence of its source, so these files are released under **CC BY 4.0** (`data/LICENSE` and
`tables/LICENSE`), and any use must cite GLORIA as above. The figures in `figures/` are drawn from
the same derived data and carry the same terms.

**Code** (`src/`, `scripts/`, `tests/`) is released under the **MIT** licence, see `LICENSE`.

## Heavy files: the Zenodo record

The full experiment output (`results/core_v2`, `results/cvplus_v3`, `results/budget_v2`,
`results/sens_v2`, `results/noise_v2`) is about 1.55 GB across 9,926 files, so it is not in this
repository. It is deposited on Zenodo together with a copy of this code and the derived data:

> Zenodo DOI: **[10.5281/zenodo.22791749](https://doi.org/10.5281/zenodo.22791749)**
> (reserved 2026-09-16; the record is a draft until the deposit is published, so the link starts
> resolving only after publication).

{summary_note}

## Environment

```
python -m venv .venv
.venv\\Scripts\\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r environment/requirements.lock
```

Python 3.14.3. The pinned set includes `torch==2.11.0+cu130`, which is a CUDA build; install the
matching CPU wheel instead if no NVIDIA GPU is available (only the mixture density network uses
the GPU). `environment/requirements_full.lock` is the complete development freeze, for forensic
comparison; `environment/requirements.lock` is what the code needs.

## Reproduction

All commands run from the repository root. Steps 1 and 2 are the expensive ones; steps 3 to 5 take
minutes and are what regenerate every table and figure in the manuscript.

### 1. Build the derived data from raw GLORIA

```
python -m src.data.build_all
```

Needs `data/raw/gloria/` and network access for the spectral response files. It rewrites exactly
the files already shipped in `data/interim/` and `data/processed/`, so this step is only needed to
rebuild them from scratch.

### 2. Run the experiment matrix

```
scripts\\run_core.ps1          # core_v2, budget_v2, sens_v2, noise_v2  (126 jobs)
scripts\\run_cvplus_v3.ps1     # cvplus_v3                              (68 jobs)
```

Read the header comment of each script for the job matrix and the resume behaviour. Jobs write
into `results/<tag>/`. Equivalent direct calls are `python -m src.run_experiment ...` and
`python -m src.run_budget ...`.

### 3. Ledger

```
python -m src.analysis.confirmatory --results results/core_v2 --results results/cvplus_v3 --budget results/budget_v2 --out tables
python -m src.make_summary
```

The confirmatory call must list **both** result directories, because CV+ lives only in
`results/cvplus_v3`; `--results` is an append argument. `src/make_summary.py` must run **after**
it, or the `conf.*` keys are missing from the ledger.

### 4. Tables

```
python -m src.make_tables_data        # Table 1 (dataset summary), Table 2 (split protocols)
python -m src.make_tables_results     # Tables 3 to 7, from tables/summary.csv only
```

| Manuscript table | Output file | Script |
|---|---|---|
| Table 1 | `tables/table1_data.tex` | `src/make_tables_data.py` |
| Table 2 | `tables/table2_splits.tex` | `src/make_tables_data.py` |
| Table 3 | `tables/table3_point.tex` | `src/make_tables_results.py` |
| Table 4 | `tables/table4_coverage.tex` | `src/make_tables_results.py` |
| Table 5 | `tables/table5_worstgroup.tex` | `src/make_tables_results.py` |
| Table 6 | `tables/table6_sensitivity.tex` | `src/make_tables_results.py` |
| Table 7 | `tables/table7_budget_local.tex` | `src/make_tables_results.py` |

### 5. Figures

Data-only figures (no experiment results needed):

```
python -m src.figures.fig01_workflow
python -m src.figures.fig02_map
python -m src.figures.fig03_protocols
python -m src.figures.fig04_spectra
python -m src.figures.figA1_targets
python -m src.figures.figA2_wbsize
```

Result figures (need step 2 and step 3):

```
python -m src.figures.fig05_point_accuracy
python -m src.figures.fig06_coverage_protocol
python -m src.figures.fig07_coverage_width
python -m src.figures.fig08_nominal_levels
python -m src.figures.fig09_calibration_budget
python -m src.figures.fig10_sensor_noise
python -m src.figures.figA3_wbcoverage_ecdf
python -m src.figures.figA4_conditional_coverage
python -m src.figures.figA5_knn_distance
python -m src.figures.check_legibility
python -m src.figures.contact_sheet
```

Each script writes `figures/<name>.png` (300 dpi) and `figures/<name>.pdf` (vector). Each script's
module docstring repeats its own data source and command; read the script if this README and the
code ever disagree.

| Figure | Script | Inputs |
|---|---|---|
| Fig. 1 workflow | `fig01_workflow.py` | `data/processed/gloria.parquet` (counts only) |
| Fig. 2 site map | `fig02_map.py` | `data/processed/gloria.parquet` |
| Fig. 3 protocol schematic | `fig03_protocols.py` | illustrative layout, no data file |
| Fig. 4 Rrs envelopes and SRFs | `fig04_spectra.py` | `data/processed/gloria.parquet`, `data/raw/srf/*` |
| Fig. 5 point accuracy | `fig05_point_accuracy.py` | per-split metrics, `method == "point"` |
| Fig. 6 coverage by protocol | `fig06_coverage_protocol.py` | per-split metrics, CV+ from `results/cvplus_v3` |
| Fig. 7 coverage against width | `fig07_coverage_width.py` | per-split metrics, CV+ from `results/cvplus_v3` |
| Fig. 8 nominal levels | `fig08_nominal_levels.py` | per-split metrics at alpha in 0.20, 0.10, 0.05 |
| Fig. 9 calibration budget | `fig09_calibration_budget.py` | `results/budget_v2` via `src.analysis.confirmatory` |
| Fig. 10 sensor and noise | `fig10_sensor_noise.py` | `results/core_v2`, `results/noise_v2` |
| Fig. A.1 target distributions | `figA1_targets.py` | `data/processed/gloria.parquet` |
| Fig. A.2 water-body size | `figA2_wbsize.py` | `data/processed/groups_<target>.parquet` |
| Fig. A.3 per-water-body coverage ECDF | `figA3_wbcoverage_ecdf.py` | `results/core_v2/pred_*.parquet`, CV+ from `results/cvplus_v3` |
| Fig. A.4 conditional coverage | `figA4_conditional_coverage.py` | `results/core_v2/pred_*.parquet` |
| Fig. A.5 coverage and width against kNN distance | `figA5_knn_distance.py` | `results/core_v2/pred_*.parquet` |

### Tests

```
python -m pytest tests -q
```

## Hardware and runtime

Runs reported in the manuscript were produced on one laptop: Intel Core Ultra 9 275HX (24
threads), 31.4 GB RAM, NVIDIA RTX 5070 Ti Laptop GPU with 12 GB (driver 595.71), Windows 11,
Python 3.14.3, PyTorch 2.11.0+cu130.

The full matrix (194 jobs: 126 in `run_core.ps1`, 68 in `run_cvplus_v3.ps1`) was launched at 02:37
and finished at 11:44 on 2026-09-16, about nine hours wall clock, with zero job failures. The CPU
stream (`budget_v2`, `sens_v2`: 54 jobs) finished in under an hour. The GPU work was run as two
parallel CUDA streams of 36 jobs each, about 8.6 minutes per job effective; a single stream costs
roughly 23 minutes per job, since the mixture density network is latency-bound rather than
memory-bound (it used about 3.5 GB of the 12 GB available). Output: 9,926 files, about 1.55 GB.

## Citing this work

See `CITATION.cff`. Cite the article when it appears, the Zenodo record for the code and result
archive, and GLORIA for the underlying measurements.

## Licence

MIT for the code (`LICENSE`); CC BY 4.0 for the derived data described above. The two are
separate: see "Data provenance and licences".
"""


def _reset_dir(out: Path, keep: tuple[str, ...] = (".git",)) -> list[str]:
    """Empty `out` without removing `out` itself, and return the names kept.

    `shutil.rmtree(out)` fails on Windows whenever any process holds `out` as its working
    directory, and it would also delete a `.git` directory if the release tree has been
    initialised as the repository that is pushed to GitHub. Emptying in place avoids both:
    the tree is still rebuilt from scratch, but the git history and the working-directory
    handle survive. Anything named in `keep` is left untouched; everything else goes.
    """
    kept = []
    if out.exists():
        for p in sorted(out.iterdir()):
            if p.name in keep:
                kept.append(p.name)
                continue
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, onexc=lambda f, path, exc: (os.chmod(path, 0o700), f(path)))
            else:
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    return kept


def build_github(out: Path) -> dict:
    kept = _reset_dir(out)
    if kept:
        print(f"[make_release] kept in place, not rebuilt: {', '.join(kept)}")

    # code
    _copytree_clean(ROOT / "src", out / "src")
    _copytree_clean(ROOT / "scripts", out / "scripts")
    _copytree_clean(ROOT / "tests", out / "tests")

    # environment
    (out / "environment").mkdir(parents=True, exist_ok=True)
    _filter_requirements_lock(
        ROOT / "environment" / "requirements.lock", out / "environment" / "requirements.lock"
    )
    _filter_requirements_lock(
        ROOT / "environment" / "requirements_full.lock",
        out / "environment" / "requirements_full.lock",
    )

    # small derived data
    interim_dst = out / "data" / "interim"
    interim_dst.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT / "data" / "interim").glob("*.csv")):
        shutil.copy2(f, interim_dst / f.name)
    bri = ROOT / "data" / "interim" / "build_report.json"
    if bri.exists():
        shutil.copy2(bri, interim_dst / bri.name)

    processed_dst = out / "data" / "processed"
    processed_dst.mkdir(parents=True, exist_ok=True)
    processed = sorted((ROOT / "data" / "processed").glob("splits*.parquet")) + sorted(
        (ROOT / "data" / "processed").glob("groups_*.parquet")
    )
    if not processed:
        raise FileNotFoundError("no splits*/groups_* parquet files under data/processed")
    for f in processed:
        shutil.copy2(f, processed_dst / f.name)

    # tables: every ledger CSV under the GitHub per-file threshold, the LaTeX table bodies, and the
    # data dictionary (AUDIT_5 N6 / REVIEW_full_v2 V13: without it the method-dependent columns
    # k_cal, calib_scheme, inf_flag, cvplus_fold_units and cvplus_bound are undefined for a reader
    # of the released CSVs).
    tables_dst = out / "tables"
    tables_dst.mkdir(parents=True, exist_ok=True)
    oversized: list[tuple[str, int]] = []
    for f in (sorted((ROOT / "tables").glob("*.csv")) + sorted((ROOT / "tables").glob("*.tex"))
              + sorted((ROOT / "tables").glob("*.md"))):
        size = f.stat().st_size
        if size >= GITHUB_FILE_LIMIT:
            oversized.append((f.name, size))
            continue
        shutil.copy2(f, tables_dst / f.name)

    # figures: final figures only, no superseded copies, no QA contact sheet
    figures_dst = out / "figures"
    figures_dst.mkdir(parents=True, exist_ok=True)
    for f in sorted((ROOT / "figures").glob("fig*")):
        if f.is_file() and f.suffix.lower() in {".png", ".pdf"}:
            shutil.copy2(f, figures_dst / f.name)

    # generated files
    _write_text_keep_newlines(out / "LICENSE", MIT_LICENSE)
    _write_text_keep_newlines(out / ".gitignore", GITIGNORE)
    _write_text_keep_newlines(out / "CITATION.cff", _citation_cff())
    summary = ROOT / "tables" / "summary.csv"
    summary_bytes = summary.stat().st_size if summary.exists() else 0
    summary_included = (tables_dst / "summary.csv").exists()
    _write_text_keep_newlines(out / "README.md", _github_readme(summary_included, summary_bytes))
    # per-folder licence for the two folders that hold GLORIA-derived data
    _write_text_keep_newlines(out / "data" / "LICENSE", CC_BY_4_LICENSE)
    _write_text_keep_newlines(out / "tables" / "LICENSE", CC_BY_4_LICENSE)

    files_changed, replacements = scrub_tree(out)
    findings, allowed_hits = scan_tree(out, allow_git=True)

    rel_files = [(p.relative_to(out).as_posix(), p) for p in _iter_files(out)]
    n_files, total_bytes = _write_manifest(
        rel_files, ROOT / "release" / "_manifests" / "github.sha256"
    )

    print(f"[make_release] github tree: {out}")
    print(f"[make_release]   files {n_files}  size {_human(total_bytes)}")
    print(f"[make_release]   scrub: {replacements} replacement(s) in {files_changed} file(s)")
    print(f"[make_release]   scan: {len(findings)} finding(s), {allowed_hits} allowed author e-mail hit(s)")
    for f in findings:
        print(f"[make_release]     ! {f}")
    for name, size in oversized:
        print(f"[make_release]   excluded (>= {GITHUB_FILE_LIMIT} B): tables/{name} = {size} B")
    return {
        "files": n_files,
        "bytes": total_bytes,
        "findings": findings,
        "allowed_email_hits": allowed_hits,
        "oversized": oversized,
    }


# ---------------------------------------------------------------------------
# release/zenodo
# ---------------------------------------------------------------------------
def _zenodo_readme_addendum(rows: list[tuple[str, int, int, str]], contents_files: int,
                            contents_bytes: int) -> str:
    lines = [
        "",
        "---",
        "",
        "# Deposit layout (Zenodo record)",
        "",
        f"Built by `code/src/make_release.py`, version {RELEASE_VERSION}, {RELEASE_DATE}.",
        "",
        "Zenodo's deposit form keeps a flat file list and accepts at most 100 files, so each",
        "top-level folder of the archive is deposited as one zip. Unzipping every file below into",
        "one directory reconstructs the archive tree exactly, and `MANIFEST_CONTENTS.sha256` then",
        "verifies each of its files individually.",
        "",
        "| Deposited file | Size | Files inside | SHA-256 |",
        "|---|---|---|---|",
    ]
    for name, size, count, digest in rows:
        inside = "-" if count < 0 else str(count)
        lines.append(f"| `{name}` | {_human(size)} | {inside} | `{digest}` |")
    lines += [
        "",
        f"Unzipped, the archive holds {contents_files:,} files totalling {_human(contents_bytes)}.",
        "",
        "## Two manifests, and what each one covers",
        "",
        "- `MANIFEST.sha256` lists the files **as deposited** (the zips and the plain files next to",
        "  them). Check it against the checksums Zenodo shows on the record page.",
        "- `MANIFEST_CONTENTS.sha256` lists every file **inside** the archive, by its path within",
        "  the unzipped tree (`code/...`, `data_derived/...`, `results/<tag>/...`). Check it after",
        "  unzipping, with `sha256sum -c MANIFEST_CONTENTS.sha256` on Linux or macOS, or with",
        "  `Get-FileHash -Algorithm SHA256` on Windows.",
        "",
        "## Per-folder licences",
        "",
        "| Scope | Licence | File |",
        "|---|---|---|",
        "| `code/` | MIT | `LICENSE-code.txt`, and `code/LICENSE` inside `code.zip` |",
        "| `data_derived/` | CC BY 4.0 | `LICENSE-data_derived.txt`, and `data_derived/LICENSE` inside `data_derived.zip` |",
        "| `results/` | CC BY 4.0 | `LICENSE-results.txt`, and `results/LICENSE` inside every `results_*.zip` |",
        "",
        "The derived data and the results inherit CC BY 4.0 from GLORIA; the code is original work",
        "and is MIT. Raw GLORIA files and the Sentinel spectral response files are not in this",
        "record; both are cited and fetched from their own sources.",
        "",
        "## Relation to the source repository",
        "",
        "`code.zip` is the same source tree as the public repository, with one transformation: every",
        "absolute filesystem path from the machine that produced the runs is replaced by",
        "`<REDACTED_PATH>`, and every e-mail address other than the two authors' institutional",
        "addresses is replaced by `<REDACTED_EMAIL>`. No logic is changed. The repository README",
        "carries the full reproduction instructions, the hardware and the runtime.",
        "",
    ]
    return "\n".join(lines)


def build_zenodo(out: Path, staging: Path, keep_staging: bool) -> dict:
    _reset_dir(out, keep=())
    if staging.exists():
        shutil.rmtree(staging)

    # Keep the authors' own institutional addresses in the archive's code copy, so it matches the
    # GitHub tree byte for byte. Single anonymised review, so author identity is not withheld.
    _submission.ALLOWED_EMAILS = set(AUTHOR_EMAILS)

    print("[make_release] building the submission archive into staging (this copies ~1.5 GB) ...")
    build_submission_archive(staging, with_results=True)

    # the archive's own per-file manifest becomes the contents manifest of the deposit
    contents_manifest = staging / "MANIFEST.sha256"
    shutil.copy2(contents_manifest, out / "MANIFEST_CONTENTS.sha256")
    contents_lines = contents_manifest.read_text(encoding="utf-8").splitlines()
    contents_files = len([ln for ln in contents_lines if ln.strip()])
    contents_bytes = sum(
        p.stat().st_size for p in _iter_files(staging) if p.name != "MANIFEST.sha256"
    )

    shutil.copy2(staging / "README.md", out / "README.md")
    _write_text_keep_newlines(out / "LICENSE-code.txt", MIT_LICENSE)
    _write_text_keep_newlines(out / "LICENSE-data_derived.txt", CC_BY_4_LICENSE)

    zip_rows: list[tuple[str, int, int, str]] = []

    def _rel_under(folder: Path) -> list[str]:
        return [p.relative_to(staging).as_posix() for p in _iter_files(folder)]

    # code.zip and data_derived.zip
    for folder_name, zip_name in (("code", "code.zip"), ("data_derived", "data_derived.zip")):
        folder = staging / folder_name
        members = _rel_under(folder)
        zp = out / zip_name
        print(f"[make_release] zipping {folder_name} ({len(members)} files) ...")
        _write_deterministic_zip(zp, staging, members)
        zip_rows.append((zip_name, zp.stat().st_size, len(members), _sha256(zp)))

    # results_<tag>.zip, one per tag, each carrying results/LICENSE
    results_dir = staging / "results"
    tags_present: list[str] = []
    if results_dir.exists():
        _write_text_keep_newlines(out / "LICENSE-results.txt", CC_BY_4_LICENSE)
        licence_rel = (results_dir / "LICENSE").relative_to(staging).as_posix()
        for tag in RESULT_TAGS:
            tag_dir = results_dir / tag
            if not tag_dir.exists():
                continue
            tags_present.append(tag)
            members = _rel_under(tag_dir) + [licence_rel]
            zip_name = f"results_{tag}.zip"
            zp = out / zip_name
            print(f"[make_release] zipping results/{tag} ({len(members)} files) ...")
            _write_deterministic_zip(zp, staging, members)
            zip_rows.append((zip_name, zp.stat().st_size, len(members), _sha256(zp)))

    # README addendum recording what is actually deposited
    readme = out / "README.md"
    text = _read_text_keep_newlines(readme) or ""
    # only the zips are tabulated; the plain files beside them are covered by MANIFEST.sha256
    _write_text_keep_newlines(
        readme, text + _zenodo_readme_addendum(zip_rows, contents_files, contents_bytes)
    )

    # deposit manifest, over everything actually uploaded (itself excluded)
    deposited = [
        (p.relative_to(out).as_posix(), p)
        for p in _iter_files(out)
        if p.name != "MANIFEST.sha256"
    ]
    n_files, total_bytes = _write_manifest(deposited, out / "MANIFEST.sha256")

    if not keep_staging:
        shutil.rmtree(staging)
        staging_parent = staging.parent
        if staging_parent.exists() and not any(staging_parent.iterdir()):
            staging_parent.rmdir()

    print(f"[make_release] zenodo tree: {out}")
    print(f"[make_release]   deposited files {n_files}  size {_human(total_bytes)}")
    print(f"[make_release]   archive contents {contents_files} files  {_human(contents_bytes)}")
    print(f"[make_release]   result tags: {tags_present}")
    return {
        "files": n_files,
        "bytes": total_bytes,
        "contents_files": contents_files,
        "contents_bytes": contents_bytes,
        "tags": tags_present,
        "zips": zip_rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="release", help="output directory (default: release)")
    ap.add_argument("--github-only", action="store_true")
    ap.add_argument("--zenodo-only", action="store_true")
    ap.add_argument(
        "--keep-staging",
        action="store_true",
        help="keep release/_staging/archive after zipping (about 1.6 GB)",
    )
    args = ap.parse_args(argv)

    if args.github_only and args.zenodo_only:
        ap.error("--github-only and --zenodo-only are mutually exclusive")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if not args.zenodo_only:
        build_github(out / "github")
    if not args.github_only:
        build_zenodo(out / "zenodo", out / "_staging" / "archive", args.keep_staging)
    return 0


if __name__ == "__main__":
    sys.exit(main())
