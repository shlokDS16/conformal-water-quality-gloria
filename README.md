# Group-aware conformal calibration for water-quality retrieval: marginal validity under water-body shift, poor conditional coverage

Code and derived data for the manuscript submitted to the ISPRS Journal of Photogrammetry and Remote Sensing.

**Authors**: Shlok Kumar Goenka (<shlok.goenka2023@vitstudent.ac.in>) and R. Manjula (<rmanjula@vit.ac.in>, corresponding author),
School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, 632014, Tamil Nadu, India.
Version v1.0.0, 2026-09-16.

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

> [dataset] GLORIA - A global dataset of remote sensing reflectance and water quality from inland and coastal waters. PANGAEA, version GLORIA-2022 (deposited 2024-04-20). https://doi.org/10.1594/PANGAEA.948492
Data descriptor: Lehmann, M.K. et al., 2023. Scientific Data 10:100. https://doi.org/10.1038/s41597-023-01973-y

Licence: CC-BY-4.0 (PANGAEA metadata field `license`: "Creative Commons Attribution 4.0 International"). Download it yourself from
<https://doi.org/10.1594/PANGAEA.948492> and unpack it into `data/raw/gloria/`. It is cited rather than
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

> Zenodo DOI: **10.5281/zenodo.XXXXXXX** (fill in once the record is published; the reserved DOI
> goes here before submission).

`tables/summary.csv` (the per-cell result ledger, 212,838 rows, 50.22 MiB) is **not** in this repository: it is above the 50 MiB per-file threshold above which GitHub warns, so it lives in the Zenodo record together with the raw result files. Rebuild it locally with `python -m src.make_summary` once `results/` is in place, or download it from Zenodo.

## Environment

```
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
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
scripts\run_core.ps1          # core_v2, budget_v2, sens_v2, noise_v2  (126 jobs)
scripts\run_cvplus_v3.ps1     # cvplus_v3                              (68 jobs)
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
