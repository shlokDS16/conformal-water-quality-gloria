"""Spectral response functions (SRFs) for Sentinel-2 MSI and Sentinel-3 OLCI.

Every SRF is returned on a common 1 nm grid (integer nm, 300-1100 for OLCI,
300-2600 for MSI) as a DataFrame indexed by wavelength with one column per band,
scaled so that each band peaks at 1.

Default units: S2A MSI and S3A OLCI. S2B, S2C and S3B are sensitivity units.
S2A is the default because it is the longest-operating MSI unit and the reference
in the ESA SRF document; S3A likewise for OLCI (launched 2016, S3B 2018).
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .common import SRF_DIR, sha256

OLCI_URLS = {
    "S3A": (
        "https://sentiwiki.copernicus.eu/__attachments/a_1376b0c7651794dd4dc96b9372d6cd9142231ae04d8578f9bfc884eba902cbc3/S3A_OL_SRF_20160713_mean_rsr.nc4",
        "S3A_OL_SRF_20160713_mean_rsr.nc4",
    ),
    "S3B": (
        "https://sentiwiki.copernicus.eu/__attachments/a_a3662d25ceecc2d2c0927107c2dcc9a3299a1a1d15dd28c9b7f9e09daece7c4d/S3B_OL_SRF_0_20180109_mean_rsr.nc4",
        "S3B_OL_SRF_0_20180109_mean_rsr.nc4",
    ),
}
MSI_XLSX = "S2_MSI_SRF_v5.0.xlsx"
MSI_XLSX_SHA256 = "42be3ebd8bfcbde11c37caeb2e2fdda65fefa8d2b6f8329559b7806d02cdc9e2"

MSI_ALL_VNIR = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]
OLCI_ALL = [f"Oa{i}" for i in range(1, 22)]

_PRINTED_NC = False


def download(url: str, dest: Path) -> dict:
    """Resumable download (curl -C -), falls back to urllib Range requests."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl:
        subprocess.run([curl, "-sS", "-L", "--retry", "3", "-C", "-", "-o", str(dest), url], check=True)
    else:
        start = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        with urllib.request.urlopen(req) as r, open(dest, "ab" if start else "wb") as f:
            shutil.copyfileobj(r, f)
    return {"file": dest.name, "url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest)}


def fetch_olci() -> list[dict]:
    out = []
    for unit, (url, name) in OLCI_URLS.items():
        dest = SRF_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            rec = {"file": name, "url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest)}
        else:
            rec = download(url, dest)
        rec["unit"] = unit
        out.append(rec)
    return out


def _bin_to_1nm(wl: np.ndarray, rsr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Average the native SRF over each 1 nm bin [g-0.5, g+0.5) (linear interpolation on a 0.01 nm grid).

    This preserves the band integral; values outside the native range are zero.
    """
    order = np.argsort(wl)
    wl, rsr = wl[order], rsr[order]
    fine = np.arange(grid[0] - 0.5, grid[-1] + 0.5, 0.01) + 0.005
    val = np.interp(fine, wl, rsr, left=0.0, right=0.0)
    idx = np.floor(fine - (grid[0] - 0.5)).astype(int)
    sums = np.bincount(idx, weights=val, minlength=len(grid))[: len(grid)]
    cnt = np.bincount(idx, minlength=len(grid))[: len(grid)]
    return sums / np.maximum(cnt, 1)


def load_olci(unit: str = "S3A") -> pd.DataFrame:
    import netCDF4

    global _PRINTED_NC
    path = SRF_DIR / OLCI_URLS[unit][1]
    ds = netCDF4.Dataset(path)
    try:
        if not _PRINTED_NC:
            print(f"[srf] {path.name} variables: {list(ds.variables)}")
            _PRINTED_NC = True
        rsr = np.asarray(ds.variables["mean_spectral_response_function"][:], dtype=float)
        wl = np.asarray(ds.variables["mean_spectral_response_function_wavelength"][:], dtype=float)
    finally:
        ds.close()
    grid = np.arange(300, 1101)
    cols = {}
    for b in range(rsr.shape[0]):
        ok = np.isfinite(rsr[b]) & np.isfinite(wl[b])
        s = _bin_to_1nm(wl[b][ok], rsr[b][ok], grid)
        cols[f"Oa{b + 1}"] = s / s.max()
    return pd.DataFrame(cols, index=pd.Index(grid, name="wavelength"))


def load_msi(unit: str = "S2A") -> pd.DataFrame:
    path = SRF_DIR / MSI_XLSX
    got = sha256(path)
    if got != MSI_XLSX_SHA256:
        raise ValueError(f"MSI SRF checksum mismatch: {got}")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(path, sheet_name=f"Spectral Responses ({unit})")
    df = df.set_index("SR_WL")
    df.index = df.index.astype(int).rename("wavelength")
    df.columns = [c.replace(f"{unit}_SR_AV_", "") for c in df.columns]
    df = df.astype(float).fillna(0.0)
    return df / df.max()


def srf_summary(srf: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Centroid wavelength, 1%-of-peak support and FWHM per band (for the log)."""
    rows = []
    for b in bands:
        s = srf[b]
        wl = s.index.to_numpy()
        sup = wl[s.to_numpy() >= 0.01]
        half = wl[s.to_numpy() >= 0.5]
        rows.append({
            "band": b,
            "centroid_nm": float((s * wl).sum() / s.sum()),
            "support_lo": int(sup.min()), "support_hi": int(sup.max()),
            "fwhm_nm": int(half.max() - half.min() + 1),
            "frac_integral_in_support": float(s[s >= 0.01].sum() / s.sum()),
        })
    return pd.DataFrame(rows)
